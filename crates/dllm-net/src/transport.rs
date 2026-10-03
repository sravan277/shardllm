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
    ACK_HEADER_LEN, ACK_MAGIC, ALPN, Ack, ActivationFrame, FRAME_HEADER_LEN, MAX_ACK_BYTES,
    MAX_PAYLOAD_BYTES, NetError, decode_ack, decode_frame_header, dev_identity, encode_ack,
    encode_frame,
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

/// TOFU fingerprint (`sha256(cert_der)`, lowercase hex) of the certificate
/// the peer actually presented on `conn`.
///
/// [`PinnedClients`] compares exactly this value against its allow-list, so a
/// server that only receives `incoming.await` knows *that* a peer passed but
/// not *who* it was. [`quinn::Connection::peer_identity`] carries the rustls
/// session's negotiated chain (`Vec<CertificateDer>`, end-entity first), which
/// lets the accept loop resolve the accepted connection back to a registry
/// row without re-reading the wire.
///
/// Returns `None` when the crypto session exposes no certificate chain (a
/// non-rustls session) or the chain is empty. Callers MUST treat `None` as
/// "unknown peer" and close the connection — never as a match.
pub fn peer_fingerprint(conn: &quinn::Connection) -> Option<String> {
    let identity = conn.peer_identity()?;
    let certs = identity
        .downcast::<Vec<rustls::pki_types::CertificateDer<'static>>>()
        .ok()?;
    Some(Identity::fingerprint_of(certs.first()?.as_ref()))
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
///
/// Reads the fixed 40-byte header, validates magic + the edge invariants +
/// the declared payload length against the 8 MiB cap, and only then allocates
/// the payload. A hostile or corrupt peer therefore cannot force an
/// unbounded allocation.
pub async fn recv_frame(
    recv: &mut quinn::RecvStream,
) -> Result<ActivationFrame, TransportError> {
    let mut header = [0u8; FRAME_HEADER_LEN];
    recv.read_exact(&mut header).await?;
    let hdr = decode_frame_header(&header)?;
    let declared = hdr.payload_length as usize;
    // decode_frame_header already bounds `declared`; re-checked here so the
    // invariant is local to the allocation.
    debug_assert!(declared <= MAX_PAYLOAD_BYTES);
    let mut payload = vec![0u8; declared];
    if declared > 0 {
        recv.read_exact(&mut payload).await?;
    }
    let mut frame = hdr.as_frame();
    frame.payload = payload;
    Ok(frame)
}

/// Send one [`Ack`] as `ACK1 || u32-LE len || {"ack":…}`.
///
/// ACKs carry their own magic (`ACK1`), so they can never be confused with an
/// activation frame even if both appear on the same stream.
pub async fn send_ack(send: &mut quinn::SendStream, ack: &Ack) -> Result<(), TransportError> {
    let bytes = encode_ack(ack)?;
    send.write_all(&bytes).await?;
    Ok(())
}

/// Receive one [`Ack`] framed by [`send_ack`]; the declared length is
/// cap-checked against [`MAX_ACK_BYTES`] before the body is read.
pub async fn recv_ack(recv: &mut quinn::RecvStream) -> Result<Ack, TransportError> {
    let mut header = [0u8; ACK_HEADER_LEN];
    recv.read_exact(&mut header).await?;
    if header[..ACK_MAGIC.len()] != ACK_MAGIC[..] {
        return Err(TransportError::Net(NetError::BadMagic));
    }
    let declared =
        u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    if declared > MAX_ACK_BYTES {
        return Err(TransportError::Net(NetError::AckTooLarge(declared)));
    }
    let mut body = vec![0u8; declared];
    if declared > 0 {
        recv.read_exact(&mut body).await?;
    }
    let mut full = Vec::with_capacity(ACK_HEADER_LEN + body.len());
    full.extend_from_slice(&header);
    full.extend_from_slice(&body);
    decode_ack(&full).map_err(TransportError::Net)
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

    async fn paired_with_ids() -> (
        quinn::Endpoint,
        quinn::Connection,
        quinn::Connection,
        Identity,
        Identity,
    ) {
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
        (
            server_ep,
            client_conn.unwrap(),
            server_conn,
            server_id,
            client_id,
        )
    }

async fn paired() -> (quinn::Endpoint, quinn::Connection, quinn::Connection) {
        let (server_ep, client_conn, server_conn, _, _) = paired_with_ids().await;
        (server_ep, client_conn, server_conn)
    }

    /// Drive one server-side handshake attempt the way production does
    /// (`dllm_serve::mesh::run_mesh_server`): take the attempt off
    /// [`quinn::Endpoint::accept`], then await the yielded
    /// [`quinn::Incoming`].
    ///
    /// Both halves are mandatory. `Endpoint::accept()` only *queues* the
    /// attempt (quinn pushes it onto `RecvState::incoming`); the server's first
    /// flight is not produced until the `Incoming` is consumed, because that is
    /// where `EndpointInner::accept` — the call that runs the rustls handshake
    /// and therefore [`PinnedClients`] — actually happens. A test that merely
    /// holds an `Incoming` open without awaiting it never gets a response, so
    /// its peer blocks until quinn's 30 s default idle timeout fires and the
    /// handshake dies with `ConnectionError::TimedOut`. That is worse than a
    /// hang for a rejection test, since a timeout is an `Err` as well: the
    /// assertion would pass without `PinnedClients` ever being consulted.
    async fn accept_once(ep: &quinn::Endpoint) -> Result<quinn::Connection, quinn::ConnectionError> {
        ep.accept().await.expect("endpoint open").await
    }

    /// Assert the server refused the attempt in its certificate verifier.
    ///
    /// Only [`quinn::ConnectionError::TransportError`] (a rustls alert) counts.
    /// In particular [`quinn::ConnectionError::TimedOut`] does not: that is what
    /// a peer sees when the server never consumed its `Incoming`, so tolerating
    /// it would let a broken accept path masquerade as a working TOFU refusal.
    fn assert_rejected_by_verifier(
        who: &str,
        outcome: Result<quinn::Connection, quinn::ConnectionError>,
    ) {
        match outcome {
            Err(quinn::ConnectionError::TransportError(_)) => {}
            Err(other) => panic!(
                "{who} was refused, but not by the certificate verifier: {other:?} \
                 (a timeout here means the server never consumed its Incoming)"
            ),
            Ok(_) => panic!("{who} completed a handshake against a server that must refuse it"),
        }
    }

    /// Assert the refusing side is observable from the rejected peer as well.
    ///
    /// TLS 1.3 orders messages so the client can finish its own handshake
    /// before the server has validated the client certificate, so `connect`
    /// legitimately resolves `Ok` here even though the peer is being turned
    /// away. The refusal lands a moment later as a `CONNECTION_CLOSE`; waiting
    /// for it (bounded) is what distinguishes "refused" from "quietly dropped",
    /// without letting a regression park the test on a 30 s idle timeout.
    async fn await_peer_rejection(who: &str, conn: &quinn::Connection) {
        let closed = tokio::time::timeout(std::time::Duration::from_secs(5), conn.closed())
            .await
            .unwrap_or_else(|_| {
                panic!("{who} kept its connection open: it was admitted, not rejected")
            });
        assert!(
            !matches!(closed, quinn::ConnectionError::TimedOut),
            "{who} was rejected only by an idle timeout, not by the server: {closed:?}"
        );
    }

    #[tokio::test]
    async fn peer_fingerprint_reports_the_negotiated_peer_cert() {
        let (server_ep, client_conn, server_conn, server_id, client_id) = paired_with_ids().await;

        // Each side must recover exactly the value PinnedClients/PinnedServer
        // pinned, so an accept loop can map a connection back to a device row.
        assert_eq!(peer_fingerprint(&server_conn).as_deref(), Some(client_id.fingerprint().as_str()));
        assert_eq!(peer_fingerprint(&client_conn).as_deref(), Some(server_id.fingerprint().as_str()));
        // Fingerprints are 64 lowercase hex chars (sha256).
        let fp = peer_fingerprint(&server_conn).unwrap();
        assert_eq!(fp.len(), 64);
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));

        client_conn.close(0u32.into(), b"done");
        server_ep.close(0u32.into(), b"done");
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

        // Wrong expected fingerprint: client-side TOFU check must fail. The
        // server must consume the `Incoming` concurrently, otherwise it never
        // sends its certificate and `connect` fails with `TimedOut` rather than
        // with the `PinnedServer` rejection this test is about.
        let wrong_fp = Identity::generate().unwrap().fingerprint();
        let (client, server_side) = tokio::join!(
            connect(&client_id, addr, &wrong_fp),
            accept_once(&server_ep)
        );
        let client_err = match client {
            Err(e) => e,
            Ok(_) => panic!("client accepted a server whose fingerprint is not the pinned one"),
        };
        assert!(
            !matches!(
                client_err,
                TransportError::Connection(quinn::ConnectionError::TimedOut)
            ),
            "client refusal must come from the pinned-fingerprint check, not a \
             missing server flight: {client_err:?}"
        );
        // The server learns of the refusal only as the client's TLS alert.
        assert!(server_side.is_err(), "server handshake should not succeed");

        server_ep.close(0u32.into(), b"done");
    }

    #[tokio::test]
    async fn tofu_rejects_unknown_client() {
        let server_id = Identity::generate().unwrap();
        let rogue_id = Identity::generate().unwrap();

        // Server allows nobody: rogue client handshake must fail. Both ends are
        // driven, so the failure is `PinnedClients` rejecting the certificate
        // rather than the rogue simply never getting a reply.
        let server_ep = server(&server_id, 0, HashSet::new()).unwrap();
        let port = server_ep.local_addr().unwrap().port();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);

        let server_fp = server_id.fingerprint();
        let (rogue, server_side) = tokio::join!(
            connect(&rogue_id, addr, &server_fp),
            accept_once(&server_ep)
        );
        assert_rejected_by_verifier("rogue", server_side);
        if let Ok(conn) = rogue {
            await_peer_rejection("rogue", &conn).await;
        }

        server_ep.close(0u32.into(), b"done");
    }

    #[tokio::test]
    async fn tofu_rejects_unknown_client_while_others_are_allowed() {
        // The production allow-list is built from the device registry, so it is
        // normally non-empty: an unlisted peer must still be refused while a
        // listed peer is connected on the same endpoint.
        let server_id = Identity::generate().unwrap();
        let allowed_id = Identity::generate().unwrap();
        let stranger_id = Identity::generate().unwrap();

        let mut allowed = HashSet::new();
        allowed.insert(allowed_id.fingerprint());
        let server_ep = server(&server_id, 0, allowed).unwrap();
        let port = server_ep.local_addr().unwrap().port();
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        let server_fp = server_id.fingerprint();

        // A listed peer gets in. `accept_once` must run *concurrently* with the
        // client connect: the server cannot produce its first flight until the
        // `Incoming` is awaited, and awaiting it only after the `join!` resolves
        // would deadlock both ends until quinn's idle timeout.
        let (allowed_conn, allowed_side) = tokio::join!(
            connect(&allowed_id, addr, &server_fp),
            accept_once(&server_ep)
        );
        let allowed_conn = allowed_conn.expect("listed peer connects");
        let allowed_side = allowed_side.expect("listed peer handshake");
        assert_eq!(
            peer_fingerprint(&allowed_side).as_deref(),
            Some(allowed_id.fingerprint().as_str())
        );

        // An unlisted peer on the same endpoint must not. The server has to take
        // this attempt off `Endpoint::accept()` too, otherwise `PinnedClients`
        // never runs and the stranger would merely time out — an `Err` that would
        // satisfy the assertion without proving the allow-list rejected it.
        let (stranger_conn, stranger_side) = tokio::join!(
            connect(&stranger_id, addr, &server_fp),
            accept_once(&server_ep)
        );
        // The refusal is observable on the server side, where the allow-list
        // lives: `PinnedClients::verify_client_cert` returned `UnknownIssuer`.
        assert_rejected_by_verifier("stranger", stranger_side);
        await_peer_rejection("stranger", &stranger_conn.expect("stranger handshake")).await;

        allowed_conn.close(0u32.into(), b"done");
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
