//! Live mTLS QUIC mesh: allow-list from the device registry (TOFU), peer link
//! bookkeeping, and the accept loop for [`QUIC_PORT`](crate::QUIC_PORT).
//!
//! ## Why this exists
//!
//! The coordinator already minted a persistent `Identity` (mDNS TXT, pairing
//! URI, `GET /api/node`) but never bound a socket, so the advertised
//! `quic_port=8443` was a lie. [`spawn_mesh_server`] binds it for real and
//! accepts connections from **paired** devices only.
//!
//! ## Allow-list = registry ∩ paired ∩ has-cert (TOFU, ADR-028)
//!
//! [`MeshState::refresh_allow_list`] rebuilds the rustls client-cert
//! allow-list from `store.list_devices()`. A row qualifies when
//! `status == "paired"` **and** `cert_fp` is non-empty. That second condition
//! is what separates a real mesh peer from an HTTP-only heartbeat: a device
//! that has only ever talked to the REST API has no certificate fingerprint
//! yet and is rejected during the TLS handshake.
//!
//! Fingerprints are stored lowercased (they are `sha256(cert_der)` hex, see
//! `dllm_net::transport::Identity::fingerprint`). A value that is not a real
//! sha256 hex digest simply can never match a presented certificate, so
//! junk rows fail closed instead of widening the allow-list.
//!
//! Rejection happens inside rustls, so a rejected peer's fingerprint is never
//! handed to us — what we *can* log (and do) is which registry rows were
//! excluded and why.
//!
//! ## Honest telemetry
//!
//! [`PeerLink::rtt_us`] is `None` until the peer has acknowledged at least one
//! of our packets ([`PeerLink::sample_rtt`] gates on
//! `stats().frame_rx.acks`), because `quinn::Connection::rtt()` otherwise
//! reports its built-in initial estimate rather than a measurement.
//! [`crate::QUIC_PORT`]-level `bottleneck`/`latency_ms` stay `null` for the
//! same reason: stage execution is not wired yet (see ADR-028).

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dllm_net::transport::{Identity, peer_fingerprint, server as quic_server};
use dllm_store::Store;
use serde::Serialize;

use crate::unix_ms_now;

/// RTT samples kept per link. A short rolling window keeps `rtt_us` stable
/// against a single retransmit spike without hiding a real regression.
pub const MESH_RTT_WINDOW: usize = 8;

/// Cadence of the mesh monitor loop: one RTT sample per link plus a
/// `last_seen` refresh, so a QUIC-only peer still counts as active in
/// `GET /v1/devices` without sending HTTP heartbeats.
pub const MESH_TICK_SECS: u64 = 5;

/// One live mTLS QUIC link to a paired peer.
///
/// The `quinn::Connection` handle is kept here (not just a flag) so future
/// stage execution can open streams straight off the live link; `GET /v1/mesh`
/// only reads the metadata fields.
pub struct PeerLink {
    /// Registry `device_id` resolved from the peer certificate fingerprint.
    pub device_id: String,
    /// Lowercase hex `sha256(cert_der)` of the peer certificate.
    pub fingerprint: String,
    /// Peer address (port is ephemeral; the fingerprint is the identity).
    pub remote: SocketAddr,
    /// Unix ms when the handshake completed.
    pub connected_since_ms: i64,
    /// Unix ms of the most recent monitor tick that saw this link alive.
    pub last_seen_ms: i64,
    /// Rolling window of measured RTT samples in microseconds.
    rtt_samples: VecDeque<u64>,
    conn: quinn::Connection,
}

impl PeerLink {
    /// Build a link for an accepted connection. `last_seen_ms` starts at
    /// `connected_since_ms` — the handshake itself is the first liveness
    /// proof we have.
    pub fn new(device_id: String, fingerprint: String, conn: quinn::Connection) -> Self {
        let now_ms = unix_ms_now();
        Self {
            device_id,
            fingerprint,
            remote: conn.remote_address(),
            connected_since_ms: now_ms,
            last_seen_ms: now_ms,
            rtt_samples: VecDeque::with_capacity(MESH_RTT_WINDOW),
            conn,
        }
    }

    /// Rolling median RTT in microseconds, or `None` when no real sample
    /// exists yet. Never synthesizes a value from the connection's built-in
    /// initial estimate.
    pub fn rtt_us(&self) -> Option<u64> {
        if self.rtt_samples.is_empty() {
            return None;
        }
        let mut v: Vec<u64> = self.rtt_samples.iter().copied().collect();
        v.sort_unstable();
        Some(v[v.len() / 2])
    }

    /// Take one RTT sample if the peer has acknowledged real traffic.
    ///
    /// `frame_rx.acks > 0` means the peer ACKed at least one of our packets,
    /// so `rtt()` is derived from a measured round trip. Before that the
    /// sample is skipped (window stays empty → `rtt_us` stays `null`).
    pub fn sample_rtt(&mut self) -> Option<u64> {
        if self.conn.stats().frame_rx.acks == 0 {
            return None;
        }
        let us = self.conn.rtt().as_micros().min(u128::from(u64::MAX)) as u64;
        if self.rtt_samples.len() == MESH_RTT_WINDOW {
            self.rtt_samples.pop_front();
        }
        self.rtt_samples.push_back(us);
        Some(us)
    }

    /// Refresh `last_seen_ms` to now (called once per monitor tick).
    pub fn touch(&mut self) {
        self.last_seen_ms = unix_ms_now();
    }

    /// True while quinn reports no close reason.
    pub fn is_connected(&self) -> bool {
        self.conn.close_reason().is_none()
    }

    /// The live connection (future stage execution; not used by the API yet).
    pub fn conn(&self) -> &quinn::Connection {
        &self.conn
    }

    /// Close the link with an application error code.
    pub fn close(&self, reason: &[u8]) {
        self.conn.close(0u32.into(), reason);
    }
}

/// One entry of `GET /v1/mesh`.
#[derive(Debug, Clone, Serialize)]
pub struct MeshPeerView {
    pub device_id: String,
    /// Friendly OS hostname; `None` = unknown (server never invents one).
    pub device_name: Option<String>,
    pub fingerprint: String,
    /// Rolling median RTT; `null` until a real sample exists.
    pub rtt_us: Option<u64>,
    /// False once quinn reports a close reason (the row is kept for forensics).
    pub connected: bool,
    pub connected_since_ms: i64,
    pub last_seen_ms: i64,
}

/// Mesh summary surfaced as `GET /api/stats` → `mesh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct MeshSummary {
    /// Links with a live QUIC connection.
    pub peers_connected: usize,
    /// Registry rows currently eligible for the QUIC allow-list.
    pub allowed_peers: usize,
}

#[derive(Default)]
struct MeshInner {
    /// Live links keyed by peer certificate fingerprint (the TOFU identity;
    /// a peer that reconnects replaces its own entry instead of duplicating).
    links: HashMap<String, PeerLink>,
    /// allow-list: lowercase fingerprint -> `device_id`.
    allow: HashMap<String, String>,
}

/// Shared live mesh state: peer links + the allow-list they are admitted
/// from. Cheap to clone via `Arc`; every critical section is a `Mutex` held
/// across pure in-memory work only (never across an `await`).
pub struct MeshState {
    inner: Mutex<MeshInner>,
    /// Actual bound UDP port; `0` = the mesh is not listening.
    bound_port: AtomicU16,
}

impl Default for MeshState {
    fn default() -> Self {
        Self::new_inner()
    }
}

impl MeshState {
    /// Fresh, unbound mesh state (no links, empty allow-list, port `0`).
    pub fn new() -> Arc<Self> {
        Arc::new(Self::new_inner())
    }

    fn new_inner() -> Self {
        Self {
            inner: Mutex::new(MeshInner::default()),
            bound_port: AtomicU16::new(0),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, MeshInner> {
        // A poisoned mesh mutex would mean a handler panicked mid-update;
        // recovering keeps `GET /v1/mesh` answering instead of 500-ing.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// UDP port the mesh actually bound; `0` when it is not listening.
    pub fn bound_port(&self) -> u16 {
        self.bound_port.load(Ordering::Relaxed)
    }

    /// Record the bound port so `GET /v1/mesh` reports the real socket
    /// (and `0` stays "not listening" when the bind failed).
    pub fn set_bound_port(&self, port: u16) {
        self.bound_port.store(port, Ordering::Relaxed);
    }

    /// `endpoint` string for the API: `0.0.0.0:<port>`, or `"unbound"`.
    pub fn endpoint(&self) -> String {
        match self.bound_port() {
            0 => "unbound".to_string(),
            p => format!("0.0.0.0:{p}"),
        }
    }

    /// Size of the current QUIC allow-list (paired rows with a `cert_fp`).
    pub fn allowed_peer_count(&self) -> usize {
        self.lock().allow.len()
    }

    /// Resolve an accepted connection's fingerprint to its registry
    /// `device_id`. `None` = not allowed on QUIC (unpaired, revoked, or an
    /// HTTP-only heartbeat with no certificate yet).
    pub fn resolve_device_id(&self, fingerprint: &str) -> Option<String> {
        let fp = normalize_fingerprint(fingerprint);
        self.lock().allow.get(&fp).cloned()
    }

    /// Rebuild the allow-list from the device registry (TOFU).
    ///
    /// Also drops live links whose fingerprint just left the allow-list, so a
    /// revoke takes effect on the next tick instead of leaving a usable
    /// socket behind. Returns `(allowed, excluded)` counts; every excluded
    /// row is logged with the reason, since a rejected handshake is invisible
    /// to us (rustls refuses it before `accept()` resolves).
    pub fn refresh_allow_list(&self, store: &Store) -> (usize, usize) {
        let rows = store.list_devices().unwrap_or_default();
        let mut allow: HashMap<String, String> = HashMap::new();
        let mut excluded: Vec<(String, String, String)> = Vec::new();
        for r in rows {
            let fp = normalize_fingerprint(&r.cert_fp);
            if r.status != "paired" {
                excluded.push((
                    r.device_id.clone(),
                    r.cert_fp.clone(),
                    format!("status={} (not paired)", r.status),
                ));
            } else if fp.is_empty() {
                excluded.push((
                    r.device_id.clone(),
                    r.cert_fp.clone(),
                    "no cert_fp yet (HTTP heartbeat only) — must pair first".to_string(),
                ));
            } else {
                allow.insert(fp, r.device_id.clone());
            }
        }
        for (device_id, cert_fp, reason) in &excluded {
            tracing::debug!(
                %device_id,
                cert_fp = %cert_fp,
                %reason,
                "QUIC allow-list: device excluded from the mesh"
            );
        }
        let mut inner = self.lock();
        let allowed = allow.len();
        inner.allow = allow;
        // Revocation must tear down the live link, not just the allow-list.
        let revoked: Vec<String> = inner
            .links
            .keys()
            .filter(|fp| !inner.allow.contains_key(*fp))
            .cloned()
            .collect();
        for fp in &revoked {
            if let Some(link) = inner.links.remove(fp) {
                tracing::info!(
                    device_id = %link.device_id,
                    fingerprint = %fp,
                    "closing mesh link: fingerprint no longer in the allow-list"
                );
                link.close(b"dllm: allow-list changed");
            }
        }
        drop(inner);
        tracing::info!(
            allowed,
            excluded = excluded.len(),
            linked = self.lock().links.len(),
            "QUIC allow-list rebuilt from the device registry (TOFU)"
        );
        (allowed, excluded.len())
    }

    /// Record a new link, replacing any previous link for the same
    /// fingerprint (a reconnect must not leave a stale duplicate row).
    fn insert(&self, link: PeerLink) {
        let fp = link.fingerprint.clone();
        let previous = self.lock().links.insert(fp, link);
        if let Some(prev) = previous {
            tracing::info!(
                device_id = %prev.device_id,
                "replaced an older mesh link for the same fingerprint"
            );
            prev.close(b"dllm: superseded by a newer link");
        }
    }

    /// One monitor tick: sample RTT, refresh `last_seen`, drop dead links and
    /// re-read the registry. Returns the `device_id`s whose `last_seen` was
    /// refreshed (the caller writes them to SQLite).
    pub fn tick(&self, store: &Store) -> Vec<String> {
        self.refresh_allow_list(store);
        let mut touched = Vec::new();
        // (fingerprint, device_id) of links the peer took down.
        let mut dead: Vec<(String, String)> = Vec::new();
        {
            let mut inner = self.lock();
            for link in inner.links.values_mut() {
                if !link.is_connected() {
                    dead.push((link.fingerprint.clone(), link.device_id.clone()));
                    continue;
                }
                link.sample_rtt();
                link.touch();
                touched.push(link.device_id.clone());
            }
            for (fp, _) in &dead {
                inner.links.remove(fp);
            }
        }
        for (_, device_id) in dead {
            tracing::info!(device_id, "mesh link closed by the peer");
        }
        touched
    }

    /// Drop one link by fingerprint (used when the handshake is rejected
    /// after a provisional insert, and by tests).
    pub fn remove(&self, fingerprint: &str) -> Option<PeerLink> {
        let fp = normalize_fingerprint(fingerprint);
        self.lock().links.remove(&fp)
    }

    /// Snapshot of every known link for `GET /v1/mesh`, sorted by `device_id`
    /// so the response is stable between polls.
    pub fn peers(&self, store: &Store) -> Vec<MeshPeerView> {
        let names: HashMap<String, Option<String>> = store
            .list_devices()
            .unwrap_or_default()
            .into_iter()
            .map(|r| {
                (
                    r.device_id.clone(),
                    r.device_name.filter(|s| !s.trim().is_empty()),
                )
            })
            .collect();
        let mut out: Vec<MeshPeerView> = self
            .lock()
            .links
            .values()
            .map(|l| MeshPeerView {
                device_id: l.device_id.clone(),
                device_name: names.get(&l.device_id).cloned().flatten(),
                fingerprint: l.fingerprint.clone(),
                rtt_us: l.rtt_us(),
                connected: l.is_connected(),
                connected_since_ms: l.connected_since_ms,
                last_seen_ms: l.last_seen_ms,
            })
            .collect();
        out.sort_by(|a, b| a.device_id.cmp(&b.device_id));
        out
    }

    /// IP address a paired peer's QUIC connection was accepted from, or `None`
    /// when that peer has no live link.
    ///
    /// Why the coordinator needs this: the ggml-rpc protocol is a *separate*
    /// plaintext TCP server, and a worker on another machine may advertise only
    /// its RPC **port** (it knows the port it was told to listen on) without
    /// knowing which of its interfaces the coordinator will reach. The address
    /// the peer connected *from* is the one the coordinator already proved is
    /// routable, so it is the honest host for `host:port`.
    ///
    /// The port half comes from the registry (`rpc_endpoint`); this only supplies
    /// the host. Returns `None` rather than a loopback guess — dialing
    /// `127.0.0.1:<port>` for a remote peer would silently hit the *coordinator's*
    /// own CPU, which is the single most confusing failure this feature could
    /// have.
    pub fn remote_ip_for(&self, device_id: &str) -> Option<std::net::IpAddr> {
        self.lock()
            .links
            .values()
            .find(|l| l.device_id == device_id)
            .map(|l| l.remote.ip())
    }

    /// `GET /api/stats` → `mesh` summary.
    pub fn summary(&self) -> MeshSummary {
        let inner = self.lock();
        MeshSummary {
            peers_connected: inner
                .links
                .values()
                .filter(|l| l.is_connected())
                .count(),
            allowed_peers: inner.allow.len(),
        }
    }
}

/// Lowercase + trim a stored/presented fingerprint so registry rows written
/// by any client match the hex produced by `Identity::fingerprint`.
fn normalize_fingerprint(fp: &str) -> String {
    fp.trim().to_ascii_lowercase()
}

/// Bind the mTLS QUIC mesh server and accept paired peers until the process
/// ends (or the task is aborted).
///
/// Synchronous bind + allow-list build, so the caller can log the real port
/// and peer count immediately and can tell "bound" from "failed" by checking
/// [`MeshState::bound_port`]. **A bind failure never stops the HTTP server**:
/// single-device chat does not need the mesh, so the error is logged and the
/// returned handle is already finished.
pub fn spawn_mesh_server(
    identity: Identity,
    port: u16,
    store: Arc<Store>,
    mesh: Arc<MeshState>,
) -> tokio::task::JoinHandle<()> {
    let (allowed, excluded) = mesh.refresh_allow_list(&store);
    let endpoint = match quic_server(&identity, port, mesh.lock().allow.keys().cloned().collect()) {
        Ok(ep) => ep,
        Err(e) => {
            tracing::error!(
                error = %e,
                "QUIC mesh bind failed; continuing with HTTP only (single-device chat does not need the mesh)"
            );
            return tokio::spawn(async {});
        }
    };
    let bound = endpoint.local_addr().map(|a| a.port()).unwrap_or(port);
    mesh.set_bound_port(bound);
    tracing::info!(
        port = bound,
        allowed_peers = allowed,
        excluded_devices = excluded,
        fingerprint = %identity.fingerprint(),
        "QUIC mesh server listening (mTLS TOFU, paired peers only)"
    );
    tokio::spawn(async move {
        run_mesh_server(endpoint, store, mesh).await;
    })
}

/// Accept loop + monitor tick.
///
/// Handshake failures are logged without the fingerprint on purpose: rustls
/// rejects an unlisted client *before* the certificate is ours to read, and
/// inventing an identity for it would be exactly the kind of lie this
/// endpoint is meant to remove.
async fn run_mesh_server(endpoint: quinn::Endpoint, store: Arc<Store>, mesh: Arc<MeshState>) {
    let mut ticker = tokio::time::interval(Duration::from_secs(MESH_TICK_SECS));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick fires immediately: publish a `last_seen` for links that
    // connected before the loop started.
    loop {
        tokio::select! {
            incoming = endpoint.accept() => match incoming {
                Some(incoming) => match incoming.await {
                    Ok(conn) => register_peer(conn, &store, &mesh),
                    Err(e) => tracing::warn!(
                        error = %e,
                        "QUIC handshake rejected (peer not paired, revoked, or bad certificate)"
                    ),
                },
                None => {
                    tracing::info!("QUIC endpoint closed; mesh server exiting");
                    return;
                }
            },
            _ = ticker.tick() => {
                for device_id in mesh.tick(&store) {
                    // Phase 0 inline write; production: spawn_blocking.
                    if let Err(e) = store.touch_last_seen(&device_id) {
                        tracing::warn!(device_id, error = %e, "failed to refresh mesh peer last_seen");
                    }
                }
            }
        }
    }
}

/// Admit one accepted connection: resolve its certificate fingerprint to a
/// registry row, record the link, and stamp `last_seen`.
fn register_peer(conn: quinn::Connection, store: &Store, mesh: &Arc<MeshState>) {
    let fingerprint = match peer_fingerprint(&conn) {
        Some(fp) => fp,
        None => {
            tracing::error!(
                remote = %conn.remote_address(),
                "accepted peer exposed no certificate; dropping (cannot resolve a device)"
            );
            conn.close(0u32.into(), b"dllm: no peer certificate");
            return;
        }
    };
    let device_id = match mesh.resolve_device_id(&fingerprint) {
        Some(id) => id,
        None => {
            // Unreachable while the allow-list is fresh, but fail closed if a
            // row was revoked between the bind and this handshake.
            tracing::warn!(
                fingerprint = %fingerprint,
                remote = %conn.remote_address(),
                "rejecting QUIC peer: fingerprint is not in the paired allow-list"
            );
            conn.close(0u32.into(), b"dllm: peer not paired");
            return;
        }
    };
    tracing::info!(
        device_id,
        fingerprint = %fingerprint,
        remote = %conn.remote_address(),
        "mesh peer connected"
    );
    mesh.insert(PeerLink::new(device_id.clone(), fingerprint, conn));
    if let Err(e) = store.touch_last_seen(&device_id) {
        tracing::warn!(device_id, error = %e, "failed to stamp mesh peer last_seen");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn temp_store(tag: &str) -> (std::path::PathBuf, Arc<Store>) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "dllm-mesh-{}-{tag}-{n}.db",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let store = Arc::new(Store::open(&path).expect("open store"));
        (path, store)
    }

    fn cleanup(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(path.with_extension("db-wal"));
        let _ = std::fs::remove_file(path.with_extension("db-shm"));
    }

    const FP_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const FP_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    /// Only paired rows that carry a certificate fingerprint may join the
    /// QUIC mesh: an HTTP-heartbeat-only device has `cert_fp = ""` and must
    /// pair (approve + fingerprint) first.
    #[test]
    fn allow_list_is_paired_rows_with_a_cert_fingerprint() {
        let (path, store) = temp_store("allow");
        // Paired with an uppercase fingerprint -> lowercased on the way in.
        store
            .upsert_device("paired-1", "worker", "[]", &FP_A.to_ascii_uppercase(), "self")
            .unwrap();
        store.upsert_device("paired-2", "worker", "[]", FP_B, "self").unwrap();
        // Heartbeat-only device: row exists and is paired, but no cert_fp.
        store.upsert_device("http-only", "worker", "[]", "", "self").unwrap();
        // Revoked peer that does carry a fingerprint.
        store.upsert_device("revoked-1", "worker", "[]", FP_B, "self").unwrap();
        store.set_status("revoked-1", "revoked").unwrap();

        let mesh = MeshState::new();
        let (allowed, excluded) = mesh.refresh_allow_list(&store);

        assert_eq!(allowed, 2);
        // http-only + revoked-1 are excluded, with the reason logged.
        assert_eq!(excluded, 2);
        assert_eq!(mesh.allowed_peer_count(), 2);
        assert_eq!(mesh.resolve_device_id(FP_A).as_deref(), Some("paired-1"));
        // Lookup is case-insensitive in both directions.
        assert_eq!(
            mesh.resolve_device_id(&FP_A.to_ascii_uppercase()).as_deref(),
            Some("paired-1")
        );
        assert_eq!(mesh.resolve_device_id(FP_B).as_deref(), Some("paired-2"));
        // Excluded devices resolve to nothing -> a handshake would be refused.
        assert!(mesh.resolve_device_id("").is_none());
        assert_eq!(
            mesh.resolve_device_id(&"b".repeat(64)).as_deref(),
            Some("paired-2"),
            "a revoked row's fingerprint must not be admitted under another id"
        );

        cleanup(&path);
    }

    /// A fresh mesh is not listening: the API reports port `0` / `"unbound"`.
    #[test]
    fn fresh_state_is_unbound_with_no_peers() {
        let mesh = MeshState::new();
        assert_eq!(mesh.bound_port(), 0);
        assert_eq!(mesh.endpoint(), "unbound");
        assert_eq!(mesh.allowed_peer_count(), 0);
        let s = mesh.summary();
        assert_eq!(s.peers_connected, 0);
        assert_eq!(s.allowed_peers, 0);
    }

    /// Revoking a device closes its live link, not just the allow-list entry.
    #[tokio::test]
    async fn revoking_a_device_tears_down_its_live_link() {
        let (path, store) = temp_store("revoke");
        let mesh = MeshState::new();

        // Real loopback pair so the PeerLink holds a live quinn::Connection,
        // registered under the fingerprint the peer will actually present.
        let server_id = Identity::generate().unwrap();
        let client_id = Identity::generate().unwrap();
        store
            .upsert_device("peer-1", "worker", "[]", &client_id.fingerprint(), "self")
            .unwrap();
        mesh.refresh_allow_list(&store);
        assert_eq!(mesh.allowed_peer_count(), 1);

        let endpoint = quic_server(&server_id, 0, HashSet::from([client_id.fingerprint()])).unwrap();
        let addr = SocketAddr::from(([127, 0, 0, 1], endpoint.local_addr().unwrap().port()));
        let server_fp = server_id.fingerprint();
        let (client_conn, server_conn) = tokio::join!(
            dllm_net::transport::connect(&client_id, addr, &server_fp),
            async { endpoint.accept().await.unwrap().await.unwrap() }
        );
        let client_conn = client_conn.unwrap();
        mesh.insert(PeerLink::new(
            "peer-1".to_string(),
            client_id.fingerprint(),
            server_conn,
        ));
        // A tick while the device is still paired keeps the link and stamps it.
        assert_eq!(mesh.tick(&store), vec!["peer-1".to_string()]);
        assert_eq!(mesh.summary().peers_connected, 1);
        assert_eq!(mesh.summary().allowed_peers, 1);

        store.set_status("peer-1", "revoked").unwrap();
        mesh.refresh_allow_list(&store);

        assert_eq!(mesh.allowed_peer_count(), 0);
        assert_eq!(mesh.summary().peers_connected, 0);
        assert!(mesh.remove(&client_id.fingerprint()).is_none());

        client_conn.close(0u32.into(), b"done");
        endpoint.close(0u32.into(), b"done");
        cleanup(&path);
    }

    /// RTT stays `null` until the peer has acknowledged real traffic, then
    /// reports a measured median from the rolling window.
    #[tokio::test]
    async fn rtt_is_null_until_a_real_sample_exists() {
        let (path, store) = temp_store("rtt");
        let mesh = MeshState::new();
        let server_id = Identity::generate().unwrap();
        let client_id = Identity::generate().unwrap();
        store
            .upsert_device("peer-1", "worker", "[]", &client_id.fingerprint(), "self")
            .unwrap();
        mesh.refresh_allow_list(&store);

        let endpoint = quic_server(&server_id, 0, HashSet::from([client_id.fingerprint()])).unwrap();
        let addr = SocketAddr::from(([127, 0, 0, 1], endpoint.local_addr().unwrap().port()));
        let server_fp = server_id.fingerprint();
        let (client_conn, server_conn) = tokio::join!(
            dllm_net::transport::connect(&client_id, addr, &server_fp),
            async { endpoint.accept().await.unwrap().await.unwrap() }
        );
        let client_conn = client_conn.unwrap();
        mesh.insert(PeerLink::new(
            "peer-1".to_string(),
            client_id.fingerprint(),
            server_conn,
        ));

        // Before any sample the API must report `rtt_us: null`, never a guess.
        let peers = mesh.peers(&store);
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].device_id, "peer-1");
        assert!(peers[0].connected);
        assert!(peers[0].rtt_us.is_none());
        assert_eq!(peers[0].device_name, None);
        assert_eq!(mesh.summary().peers_connected, 1);
        assert_eq!(mesh.summary().allowed_peers, 1);

        // The completed handshake means the peer already ACKed our packets,
        // so one monitor tick produces a real measurement and stamps last_seen.
        assert_eq!(mesh.tick(&store), vec!["peer-1".to_string()]);
        assert!(mesh.peers(&store)[0].rtt_us.expect("a real RTT sample exists") > 0);

        // The window is bounded: extra samples never grow it past the cap.
        for _ in 0..(MESH_RTT_WINDOW * 3) {
            let mut inner = mesh.lock();
            inner
                .links
                .get_mut(&client_id.fingerprint())
                .expect("link")
                .sample_rtt();
        }
        let (len, rtt) = {
            let inner = mesh.lock();
            let link = inner.links.get(&client_id.fingerprint()).expect("link");
            (link.rtt_samples.len(), link.rtt_us())
        };
        assert_eq!(len, MESH_RTT_WINDOW);
        assert!(rtt.expect("median of the window") > 0);

        client_conn.close(0u32.into(), b"done");
        endpoint.close(0u32.into(), b"done");
        cleanup(&path);
    }

    /// End-to-end: `spawn_mesh_server` binds a real socket, admits a paired
    /// peer, and resolves its certificate fingerprint to the registry row.
    /// An HTTP-heartbeat-only peer (no `cert_fp`) is refused by the handshake.
    #[tokio::test]
    async fn mesh_server_binds_and_admits_only_paired_peers() {
        let (path, store) = temp_store("e2e");
        let server_id = Identity::generate().unwrap();
        let peer_id = Identity::generate().unwrap();
        let stranger_id = Identity::generate().unwrap();
        // Paired with a certificate; the heartbeat-only row stays in the store
        // (so the registry is realistic) but must not be mesh-eligible.
        store
            .upsert_device("paired-1", "worker", "[]", &peer_id.fingerprint(), "self")
            .unwrap();
        store.upsert_device("http-only", "worker", "[]", "", "self").unwrap();

        let mesh = MeshState::new();
        // Port 0 -> the OS assigns one; bound_port must report the real socket.
        let handle = spawn_mesh_server(server_id.clone(), 0, store.clone(), mesh.clone());
        let port = mesh.bound_port();
        assert_ne!(port, 0, "the mesh must bind a real UDP port");
        assert_eq!(mesh.endpoint(), format!("0.0.0.0:{port}"));
        assert_eq!(mesh.allowed_peer_count(), 1);

        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        let server_fp = server_id.fingerprint();
        let paired = dllm_net::transport::connect(&peer_id, addr, &server_fp).await;
        assert!(paired.is_ok(), "a paired peer must complete the handshake");
        let paired = paired.unwrap();

        // The accept loop runs on its own task; wait for the link to land.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while mesh.summary().peers_connected != 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "mesh accept loop never registered the paired peer"
            );
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let peers = mesh.peers(&store);
        assert_eq!(peers[0].device_id, "paired-1");
        assert_eq!(peers[0].fingerprint, peer_id.fingerprint());
        assert!(peers[0].connected);

        // An unpaired peer (no registry row at all) must never be admitted.
        //
        // Note we must NOT assert `connect(..).is_err()`: TLS 1.3 lets the
        // client finish its handshake before the server has validated the
        // client certificate, so a rejected client still observes `Ok(..)` and
        // only learns of the refusal via the server's CONNECTION_CLOSE. The
        // security property that actually matters is that the peer is never
        // *admitted* — no link, no registry row, no place in the API output.
        let stranger = dllm_net::transport::connect(&stranger_id, addr, &server_fp).await;
        assert!(
            stranger.is_ok(),
            "TLS 1.3 completes the client side before the server rejects; \
             admission is what must be refused, not connect()"
        );
        // Give the accept loop several chances to (wrongly) admit it.
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(25)).await;
            assert_eq!(
                mesh.summary().peers_connected,
                1,
                "an unpaired fingerprint was admitted to the mesh"
            );
        }
        let peers = mesh.peers(&store);
        assert_eq!(peers.len(), 1, "stranger must not appear in the peer list");
        assert_eq!(peers[0].device_id, "paired-1");
        assert_eq!(mesh.allowed_peer_count(), 1);

        // Dropping the peer is noticed on the next monitor tick.
        paired.close(0u32.into(), b"done");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while mesh.summary().peers_connected != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "closed mesh link was never reaped"
            );
            mesh.tick(&store);
            tokio::time::sleep(Duration::from_millis(25)).await;
        }

        handle.abort();
        cleanup(&path);
    }

    /// A failed bind must not take the process down: the error is logged and
    /// the returned handle finishes while the state stays "unbound".
    #[tokio::test]
    async fn mesh_bind_failure_leaves_the_state_unbound() {
        let (path, store) = temp_store("bindfail");
        // Occupy a UDP port, then ask the mesh to bind the same one.
        let blocker = quic_server(&Identity::generate().unwrap(), 0, HashSet::new()).unwrap();
        let taken = blocker.local_addr().unwrap().port();
        let mesh = MeshState::new();
        let handle = spawn_mesh_server(
            Identity::generate().unwrap(),
            taken,
            store.clone(),
            mesh.clone(),
        );
        // The bind is attempted synchronously inside `spawn_mesh_server`, so by the
        // time it returns the failure is already known. Awaiting the handle
        // (rather than polling `is_finished()`, which races the scheduler)
        // proves the stub task terminates without panicking.
        handle.await.expect("mesh task must not panic on bind failure");
        assert_eq!(mesh.bound_port(), 0);
        assert_eq!(mesh.endpoint(), "unbound");
        // The allow-list is still readable, so the API keeps answering.
        assert_eq!(mesh.allowed_peer_count(), 0);

        blocker.close(0u32.into(), b"done");
        cleanup(&path);
    }
}