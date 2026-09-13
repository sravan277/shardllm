//! Quinn 0.11 mTLS transport with TOFU fingerprint pinning.
//!
//! Pure Rust (no C++). Builds on the wire primitives in the crate root
//! ([`ALPN`](crate::ALPN), [`ActivationFrame`](crate::ActivationFrame),
//! [`Ack`](crate::Ack), [`encode_frame`](crate::encode_frame),
//! [`decode_frame`](crate::decode_frame)).
//!
//! ## Trust model (TOFU)
//!
//! Both ends hold self-signed certificates ([`Identity`], minted via
//! [`dev_identity`](crate::dev_identity)). There is no CA: each side pins the
//! peer's certificate fingerprint (`sha256(cert_der)`, lowercase hex, see
//! [`Identity::fingerprint`]).
//!
//! * Server: [`server`] takes a TOFU store (`HashSet<String>` of allowed peer
//!   fingerprints). The rustls client-cert verifier rejects the handshake for
//!   unknown peers, so unauthorised nodes never get a connection.
//! * Client: [`connect`] takes the `expected_fingerprint` of the server. On
//!   first pairing the caller accepts whatever the server presents (and stores
//!   it); afterwards the stored value is passed back and the handshake fails on
//!   mismatch (strict mode).
//!
//! ## Stream priorities
//!
//! Quinn transmits locally-buffered data from higher-priority streams first
//! (every stream starts at `0`; larger `i32` = served first). Mapping:
//!
//! | [`StreamKind`] | priority | rationale                              |
//! |----------------|----------|----------------------------------------|
//! | `Control`      | `20`     | plan/commit must pre-empt bulk traffic |
//! | `Ack`          | `10`     | tiny + latency-sensitive               |
//! | `Activation`   | `0`      | bulk tensor bytes (quinn default)      |
//!
//! [`open_stream`] tags each bidirectional stream with a 1-byte
//! [`StreamKind`] header and calls `SendStream::set_priority`; the accepter
//! must use [`accept_stream`] so the tag byte is consumed before framing.
//!
//! ## Backpressure policy
//!
//! [`FRAME_QUEUE_CAPACITY`] (4 frames) bounds the in-memory queue between the
//! QUIC receive loop ([`spawn_frame_recv_loop`]) and the consumer. The loop
//! awaits `tx.send(frame)`; when the queue is full the send parks, the recv
//! task stops reading, and QUIC stream flow-control propagates the pressure to
//! the sender. A slow consumer therefore stalls the peer instead of growing
//! memory unbounded. When the consumer drops the receiver the loop exits.

use std::{collections::HashSet, net::SocketAddr, sync::Arc};

use sha2::{Digest, Sha256};

use crate::{
    ALPN, FRAME_MAGIC, MAX_FRAME_BYTES, Ack, ActivationFrame, NetError, decode_frame,
    dev_identity, encode_frame,
};

/// Highest priority: plan/commit control traffic. Served before all else.
pub const CONTROL_PRIORITY: i32 = 20;
/// Middle priority: small latency-sensitive inference ACKs.
pub const ACK_PRIORITY: i32 = 10;
/// Lowest priority: bulk activation tensors (quinn default `0`).
pub const ACTIVATION_PRIORITY: i32 = 0;

/// Max queued frames between the QUIC recv loop and the consumer.
/// See module docs for the backpressure policy.
pub const FRAME_QUEUE_CAPACITY: usize = 4;

/// Bytes of framing overhead per message on a stream: `MAGIC(5) + u32-LE len`.
const HEADER_LEN: usize = 5 + 4;

/// Errors from the mTLS transport layer.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error(transparent)]
    Net(#[from] NetError),
    #[error("rustls: {0}")]
    Rustls(#[from] rustls::Error),
    #[error("quinn connect: {0}")]
    Connect(#[from] quinn::ConnectError),
    #[error("quinn connection: {0}")]
    Connection(#[from] quinn::ConnectionError),
    #[error("quinn write: {0}")]
    Write(#[from] quinn::WriteError),
    #[error("quinn read: {0}")]
    Read(#[from] quinn::ReadError),
    #[error("quinn read_exact: {0}")]
    ReadExact(#[from] quinn::ReadExactError),
    #[error("quinn closed stream: {0}")]
    ClosedStream(#[from] quinn::ClosedStream),
    #[error("quinn tls config: {0}")]
    QuicTls(#[from] quinn::crypto::rustls::NoInitialCipherSuite),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("unknown stream kind tag: {0}")]
    UnknownStreamKind(u8),
    #[error("TOFU fingerprint mismatch: expected {expected}, got {got}")]
    FingerprintMismatch { expected: String, got: String },
}

/// Self-signed mTLS identity: DER certificate + DER (PKCS#8) private key.
#[derive(Debug, Clone)]
pub struct Identity {
    /// DER-encoded self-signed certificate.
    pub cert_der: Vec<u8>,
    /// DER-encoded PKCS#8 private key.
    pub key_der: Vec<u8>,
}

impl Identity {
    /// Mint a fresh dev identity (wraps [`dev_identity`](crate::dev_identity)).
    pub fn generate() -> Result<Self, TransportError> {
        let (cert_der, key_der) = dev_identity()?;
        Ok(Self { cert_der, key_der })
    }

    /// Build from existing DER bytes (e.g. loaded from SQLite / exe dir).
    pub fn from_der(cert_der: Vec<u8>, key_der: Vec<u8>) -> Self {
        Self { cert_der, key_der }
    }

    /// TOFU fingerprint: lowercase hex of `sha256(cert_der)`.
    /// Display this for out-of-band pairing verification.
    pub fn fingerprint(&self) -> String {
        Self::fingerprint_of(&self.cert_der)
    }

    /// Fingerprint of a raw DER certificate (e.g. a peer's presented cert).
    pub fn fingerprint_of(cert_der: &[u8]) -> String {
        let digest = Sha256::digest(cert_der);
        let mut out = String::with_capacity(digest.len() * 2);
        for b in digest {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }
}

/// Logical bidirectional stream kind. The opener tags the stream with one
/// byte ([`open_stream`]); the accepter reads it via [`accept_stream`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    /// Plan/commit control traffic (highest priority).
    Control,
    /// Bulk activation tensors (lowest priority).
    Activation,
    /// Small inference ACKs (middle priority).
    Ack,
}

impl StreamKind {
    /// Quinn send priority: larger `i32` is transmitted first.
    /// `Control(20) > Ack(10) > Activation(0)`.
    pub const fn priority(self) -> i32 {
        match self {
            StreamKind::Control => CONTROL_PRIORITY,
            StreamKind::Ack => ACK_PRIORITY,
            StreamKind::Activation => ACTIVATION_PRIORITY,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            StreamKind::Control => 0,
            StreamKind::Activation => 1,
            StreamKind::Ack => 2,
        }
    }

    const fn from_tag(tag: u8) -> Result<Self, TransportError> {
        match tag {
            0 => Ok(StreamKind::Control),
            1 => Ok(StreamKind::Activation),
            2 => Ok(StreamKind::Ack),
            other => Err(TransportError::UnknownStreamKind(other)),
        }
    }
}

/// Server-side TOFU verifier: only peers whose cert fingerprint is in the
/// pinned `allowed` set pass the handshake; anything else is rejected.
///
/// Note: `HandshakeSignatureValid` is only re-exported under
/// `rustls::client::danger`, but it names the same `verify::HandshakeSignatureValid`
/// type the server trait expects, so it is used below for both verifiers.
#[derive(Debug)]
struct PinnedClients {
    allowed: HashSet<String>,
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl rustls::server::danger::ClientCertVerifier for PinnedClients {
    fn root_hint_subjects(&self) -> &[rustls::DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::server::danger::ClientCertVerified, rustls::Error> {
        let fp = Identity::fingerprint_of(end_entity.as_ref());
        if self.allowed.contains(&fp) {
            Ok(rustls::server::danger::ClientCertVerified::assertion())
        } else {
            Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::UnknownIssuer,
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Client-side TOFU verifier: the server cert fingerprint must equal the
/// expected (previously paired) value; anything else fails the handshake.
#[derive(Debug)]
struct PinnedServer {
    expected: String,
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl rustls::client::danger::ServerCertVerifier for PinnedServer {
    fn verify_server_cert(
        &self,
        end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let fp = Identity::fingerprint_of(end_entity.as_ref());
        if fp == self.expected {
            Ok(rustls::client::danger::ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::UnknownIssuer,
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &rustls::pki_types::CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn cert_chain(identity: &Identity) -> Vec<rustls::pki_types::CertificateDer<'static>> {
    vec![rustls::pki_types::CertificateDer::from(
        identity.cert_der.clone(),
    )]
}

fn private_key(identity: &Identity) -> rustls::pki_types::PrivateKeyDer<'static> {
    rustls::pki_types::PrivatePkcs8KeyDer::from(identity.key_der.clone()).into()
}

fn crypto_provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

/// Start a QUIC endpoint with mTLS + TOFU client verification.
///
/// Binds `0.0.0.0:port` (pass `0` for an OS-assigned port, then read
/// [`quinn::Endpoint::local_addr`]). Only peers whose cert fingerprint is in
/// `allowed_peers` complete the handshake; unknown peers are rejected during
/// the TLS handshake. ALPN is [`ALPN`](crate::ALPN) (`dllm/1`).
pub fn server(
    identity: &Identity,
    port: u16,
    allowed_peers: HashSet<String>,
) -> Result<quinn::Endpoint, TransportError> {
    let provider = crypto_provider();
    let verifier = Arc::new(PinnedClients {
        allowed: allowed_peers,
        provider: provider.clone(),
    });
    let mut tls = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_client_cert_verifier(verifier)
        .with_single_cert(cert_chain(identity), private_key(identity))?;
    tls.alpn_protocols = vec![ALPN.to_vec()];

    let quic_tls = quinn::crypto::rustls::QuicServerConfig::try_from(tls)?;
    let server_config = quinn::ServerConfig::with_crypto(Arc::new(quic_tls));
    let addr = SocketAddr::from((std::net::Ipv4Addr::UNSPECIFIED, port));
    let endpoint = quinn::Endpoint::server(server_config, addr)?;
    Ok(endpoint)
}

/// Connect to a peer, verifying its cert fingerprint against the TOFU-pinned
/// `expected_fingerprint` (strict after first pairing). Presents our own
/// client certificate for the server's TOFU check (mTLS).
///
/// The returned [`quinn::Connection`] keeps the underlying endpoint driver
/// alive; no separate endpoint handle needs to be retained.
pub async fn connect(
    identity: &Identity,
    addr: SocketAddr,
    expected_fingerprint: &str,
) -> Result<quinn::Connection, TransportError> {
    let provider = crypto_provider();
    let verifier = Arc::new(PinnedServer {
        expected: expected_fingerprint.to_string(),
        provider: provider.clone(),
    });
    let mut tls = rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .dangerous()
        .with_custom_certificate_verifier(verifier)
        .with_client_auth_cert(cert_chain(identity), private_key(identity))?;
    tls.alpn_protocols = vec![ALPN.to_vec()];

    let quic_tls = quinn::crypto::rustls::QuicClientConfig::try_from(tls)?;
    let client_config = quinn::ClientConfig::new(Arc::new(quic_tls));
    let mut endpoint = quinn::Endpoint::client(SocketAddr::from((
        std::net::Ipv4Addr::UNSPECIFIED,
        0,
    )))?;
    endpoint.set_default_client_config(client_config);
    // SAN minted by dev_identity(); must match `server_name` here.
    let conn = endpoint.connect(addr, "dllm.local")?.await?;
    Ok(conn)
}

/// Open a tagged bidirectional stream of `kind` and set its Quinn send
/// priority ([`StreamKind::priority`]). The opener should transmit first
/// (quinn only notifies the peer of a stream once it carries data).
pub async fn open_stream(
    conn: &quinn::Connection,
    kind: StreamKind,
) -> Result<(quinn::SendStream, quinn::RecvStream), TransportError> {
    let (mut send, recv) = conn.open_bi().await?;
    send.set_priority(kind.priority())?;
    send.write_all(&[kind.tag()]).await?;
    Ok((send, recv))
}

/// Accept the next inbound bidirectional stream and read its [`StreamKind`]
/// tag. Must pair with [`open_stream`]; the tag byte is consumed so the
/// remainder of the stream is pure frame data.
pub async fn accept_stream(
    conn: &quinn::Connection,
) -> Result<(StreamKind, quinn::SendStream, quinn::RecvStream), TransportError> {
    let (send, mut recv) = conn.accept_bi().await?;
    let mut tag = [0u8; 1];
    recv.read_exact(&mut tag).await?;
    Ok((StreamKind::from_tag(tag[0])?, send, recv))
}

/// Send one [`ActivationFrame`] on `send` using [`encode_frame`]
/// (8 MiB cap enforced at encode time; multiple frames may share a stream).
pub async fn send_frame(
    send: &mut quinn::SendStream,
    frame: &ActivationFrame,
) -> Result<(), TransportError> {
    let bytes = encode_frame(frame)?;
    send.write_all(&bytes).await?;
    Ok(())
}

/// Receive one [`ActivationFrame`] framed by [`send_frame`].
/// The declared length is checked against the 8 MiB cap *before* allocating;
/// violations error without unbounded allocation.
pub async fn recv_frame(
    recv: &mut quinn::RecvStream,
) -> Result<ActivationFrame, TransportError> {
    let mut header = [0u8; HEADER_LEN];
    recv.read_exact(&mut header).await?;
    if header[..FRAME_MAGIC.len()] != FRAME_MAGIC[..] {
        return Err(TransportError::Net(NetError::BadMagic));
    }
    let mut len_buf = [0u8; 4];
    len_buf.copy_from_slice(&header[FRAME_MAGIC.len()..HEADER_LEN]);
    let declared = u32::from_le_bytes(len_buf) as usize;
    if HEADER_LEN + declared > MAX_FRAME_BYTES {
        return Err(TransportError::Net(NetError::TooLarge(
            HEADER_LEN + declared,
        )));
    }
    let mut full = vec![0u8; HEADER_LEN + declared];
    full[..HEADER_LEN].copy_from_slice(&header);
    if declared > 0 {
        recv.read_exact(&mut full[HEADER_LEN..]).await?;
    }
    Ok(decode_frame(&full)?)
}

/// Send one [`Ack`] with the same `MAGIC + u32-LE len + postcard` framing
/// (8 MiB cap enforced; ACKs are tiny in practice).
pub async fn send_ack(send: &mut quinn::SendStream, ack: &Ack) -> Result<(), TransportError> {
    let body = postcard::to_allocvec(ack).map_err(NetError::Postcard)?;
    if HEADER_LEN + body.len() > MAX_FRAME_BYTES {
        return Err(TransportError::Net(NetError::TooLarge(
            HEADER_LEN + body.len(),
        )));
    }
    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(FRAME_MAGIC);
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    send.write_all(&out).await?;
    Ok(())
}

/// Receive one [`Ack`] framed by [`send_ack`]; length is cap-checked before
/// allocating, mirroring [`recv_frame`].
pub async fn recv_ack(recv: &mut quinn::RecvStream) -> Result<Ack, TransportError> {
    let mut header = [0u8; HEADER_LEN];
    recv.read_exact(&mut header).await?;
    if header[..FRAME_MAGIC.len()] != FRAME_MAGIC[..] {
        return Err(TransportError::Net(NetError::BadMagic));
    }
    let mut len_buf = [0u8; 4];
    len_buf.copy_from_slice(&header[FRAME_MAGIC.len()..HEADER_LEN]);
    let declared = u32::from_le_bytes(len_buf) as usize;
    if HEADER_LEN + declared > MAX_FRAME_BYTES {
        return Err(TransportError::Net(NetError::TooLarge(
            HEADER_LEN + declared,
        )));
    }
    let mut body = vec![0u8; declared];
    if declared > 0 {
        recv.read_exact(&mut body).await?;
    }
    postcard::from_bytes::<Ack>(&body).map_err(|e| TransportError::Net(NetError::Postcard(e)))
}

/// Bounded queue between the QUIC recv loop and the frame consumer.
/// Capacity is [`FRAME_QUEUE_CAPACITY`]; sends park when full (backpressure).
pub fn frame_channel() -> (
    tokio::sync::mpsc::Sender<ActivationFrame>,
    tokio::sync::mpsc::Receiver<ActivationFrame>,
) {
    tokio::sync::mpsc::channel(FRAME_QUEUE_CAPACITY)
}

/// Pump [`recv_frame`] into `tx` until the stream ends or the consumer drops
/// the receiver. `tx.send().await` parks when the bounded queue is full, so a
/// slow consumer stalls this task and QUIC flow-control stalls the peer —
/// memory never grows unbounded. Exits silently on stream close/reset.
pub fn spawn_frame_recv_loop(
    mut recv: quinn::RecvStream,
    tx: tokio::sync::mpsc::Sender<ActivationFrame>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match recv_frame(&mut recv).await {
                Ok(frame) => {
                    if tx.send(frame).await.is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn test_frame(tag: u8) -> ActivationFrame {
        ActivationFrame {
            session_id: 0xA5A5,
            plan_id: 7,
            token_position: 3,
            source_stage: 0,
            target_stage: 1,
            tensor_format: crate::tensor_format::F16,
            payload: vec![tag; 64],
        }
    }

    async fn paired() -> (quinn::Endpoint, quinn::Connection, quinn::Connection) {
        let server_id = Identity::generate().unwrap();
        let client_id = Identity::generate().unwrap();
        let server_fp = server_id.fingerprint();

        let mut allowed = HashSet::new();
        allowed.insert(client_id.fingerprint());
        let server_ep = server(&server_id, 0, allowed).unwrap();
        let port = server_ep.local_addr().unwrap().port();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);

        let (client_conn, server_conn) = tokio::join!(
            connect(&client_id, addr, &server_fp),
            async {
                server_ep
                    .accept()
                    .await
                    .expect("incoming")
                    .await
                    .expect("handshake")
            }
        );
        (server_ep, client_conn.unwrap(), server_conn)
    }

    #[tokio::test]
    async fn loopback_frame_both_ways_plus_ack() {
        let (server_ep, client_conn, server_conn) = paired().await;

        // client -> server: one activation frame
        let frame_a = test_frame(0x11);
        let (mut c_send, mut c_recv) = open_stream(&client_conn, StreamKind::Activation)
            .await
            .unwrap();
        assert_eq!(c_send.priority().unwrap(), ACTIVATION_PRIORITY);
        send_frame(&mut c_send, &frame_a).await.unwrap();

        let (kind, mut s_send, mut s_recv) = accept_stream(&server_conn).await.unwrap();
        assert_eq!(kind, StreamKind::Activation);
        assert_eq!(recv_frame(&mut s_recv).await.unwrap(), frame_a);

        // server -> client on the same bidirectional stream
        let frame_b = test_frame(0x22);
        send_frame(&mut s_send, &frame_b).await.unwrap();
        assert_eq!(recv_frame(&mut c_recv).await.unwrap(), frame_b);

        // one ack on a dedicated ack-kind stream (priority mapping check)
        let ack = Ack::Committed { pos: 3, token: 42 };
        let (mut c_ack_send, _c_ack_recv) =
            open_stream(&client_conn, StreamKind::Ack).await.unwrap();
        assert_eq!(c_ack_send.priority().unwrap(), ACK_PRIORITY);
        send_ack(&mut c_ack_send, &ack).await.unwrap();

        let (ack_kind, _s_ack_send, mut s_ack_recv) =
            accept_stream(&server_conn).await.unwrap();
        assert_eq!(ack_kind, StreamKind::Ack);
        assert_eq!(recv_ack(&mut s_ack_recv).await.unwrap(), ack);

        // control priority mapping sanity
        assert_eq!(StreamKind::Control.priority(), CONTROL_PRIORITY);
        assert!(CONTROL_PRIORITY > ACK_PRIORITY && ACK_PRIORITY > ACTIVATION_PRIORITY);

        client_conn.close(0u32.into(), b"done");
        server_ep.close(0u32.into(), b"done");
    }

    #[tokio::test]
    async fn tofu_rejects_wrong_server_fingerprint() {
        let server_id = Identity::generate().unwrap();
        let client_id = Identity::generate().unwrap();

        let mut allowed = HashSet::new();
        allowed.insert(client_id.fingerprint());
        let server_ep = server(&server_id, 0, allowed).unwrap();
        let port = server_ep.local_addr().unwrap().port();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);

        // Wrong expected fingerprint: client-side TOFU check must fail.
        let wrong_fp = Identity::generate().unwrap().fingerprint();
        assert!(connect(&client_id, addr, &wrong_fp).await.is_err());

        server_ep.close(0u32.into(), b"done");
    }

    #[tokio::test]
    async fn tofu_rejects_unknown_client() {
        let server_id = Identity::generate().unwrap();
        let rogue_id = Identity::generate().unwrap();

        // Server allows nobody: rogue client handshake must fail.
        let server_ep = server(&server_id, 0, HashSet::new()).unwrap();
        let port = server_ep.local_addr().unwrap().port();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);

        assert!(
            connect(&rogue_id, addr, &server_id.fingerprint())
                .await
                .is_err()
        );

        server_ep.close(0u32.into(), b"done");
    }

    #[tokio::test]
    async fn backpressure_queue_is_bounded() {
        // Channel capacity is exactly FRAME_QUEUE_CAPACITY: filling it parks
        // the next sender instead of growing memory.
        let (tx, mut rx) = frame_channel();
        assert_eq!(FRAME_QUEUE_CAPACITY, 4);
        assert_eq!(tx.capacity(), FRAME_QUEUE_CAPACITY);
        for _ in 0..FRAME_QUEUE_CAPACITY {
            tx.try_send(test_frame(0x33)).unwrap();
        }
        assert!(tx.try_send(test_frame(0x33)).is_err());
        for _ in 0..FRAME_QUEUE_CAPACITY {
            rx.recv().await.unwrap();
        }
    }
}
