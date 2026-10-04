//! RPC worker endpoints and the de-duplication the shim refuses to do for us.
//!
//! `dllm_shim_add_rpc_server` appends to a process-global ggml device registry
//! on every call and — per the header — "the caller owns de-duplication". It
//! also has **no removal**: once an endpoint is registered its devices are in
//! the global list for the life of the process, and every later `tensor_split`
//! is indexed against that grown list. So this module is not a convenience, it
//! is the only thing standing between a re-registration and a silently
//! mis-indexed split.
//!
//! [`RpcRegistry`] is therefore process-global and idempotent: the first
//! [`register`](RpcRegistry::register) of an endpoint performs the FFI call,
//! every later one is a no-op that still reports how many devices the endpoint
//! contributed, so the caller can rebuild its `tensor_split` layout without
//! re-dialling.

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

/// Default ggml-rpc port. Mirrors `tools/rpc/rpc-server.cpp`'s default so a
/// worker started without `--rpc-port` is still dialable.
pub const DEFAULT_RPC_PORT: u16 = 50052;

/// A `host:port` pair for a ggml-rpc worker.
///
/// ggml's endpoint parser (`ggml_backend_rpc_add_server`) accepts IPv4 and
/// hostnames; it does *not* accept bracketed IPv6 literals reliably, so a
/// literal `::1` is rejected here with a message that says so instead of
/// failing inside the DLL where the reason is invisible.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RpcEndpoint {
    /// Hostname or IPv4 literal of the worker, as the coordinator can reach it.
    /// Never a bracketed IPv6 literal — ggml's endpoint parser rejects those.
    pub host: String,
    /// ggml-rpc TCP port.
    pub port: u16,
}

impl RpcEndpoint {
    /// Build an endpoint from parts that are already known-good (this skips the
    /// validation [`FromStr`] performs).
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        Self {
            host: host.into(),
            port,
        }
    }

    /// `host:port`, the exact string handed to `dllm_shim_add_rpc_server`.
    pub fn as_endpoint_string(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

impl fmt::Display for RpcEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.host, self.port)
    }
}

/// Why an endpoint string was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndpointError {
    /// The string was empty or only whitespace.
    Empty,
    /// No `host:port` separator at all.
    NoPort,
    /// The port is not a number in `1..=65535`.
    BadPort { raw: String },
    /// A bracketed IPv6 literal: ggml's endpoint parser does not accept it.
    Ipv6NotSupported { raw: String },
    /// A host containing whitespace or a stray separator, which would let
    /// `"a:1 b:2"` masquerade as two endpoints.
    BadHost { raw: String },
}

impl fmt::Display for EndpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "endpoint is empty"),
            Self::NoPort => write!(f, "endpoint has no ':port' suffix"),
            Self::BadPort { raw } => write!(f, "endpoint port {raw:?} is not in 1..=65535"),
            Self::Ipv6NotSupported { raw } => write!(
                f,
                "endpoint {raw:?} looks like an IPv6 literal; ggml-rpc endpoints must be IPv4 or a \
                 hostname, so a coordinator must publish its IPv4 address"
            ),
            Self::BadHost { raw } => write!(f, "endpoint host in {raw:?} is not a bare hostname/IP"),
        }
    }
}

impl std::error::Error for EndpointError {}

impl FromStr for RpcEndpoint {
    type Err = EndpointError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let raw = s.trim();
        if raw.is_empty() {
            return Err(EndpointError::Empty);
        }
        if raw.starts_with('[') {
            return Err(EndpointError::Ipv6NotSupported {
                raw: raw.to_string(),
            });
        }
        let (host, port) = raw
            .rsplit_once(':')
            .ok_or_else(|| EndpointError::NoPort)?;
        let port: u16 = port
            .parse()
            .map_err(|_| EndpointError::BadPort {
                raw: port.to_string(),
            })?;
        if port == 0 {
            return Err(EndpointError::BadPort {
                raw: port.to_string(),
            });
        }
        if host.is_empty()
            || host.split_whitespace().count() != 1
            || host.contains(':')
            || host.contains('/')
        {
            return Err(EndpointError::BadHost {
                raw: raw.to_string(),
            });
        }
        Ok(Self::new(host, port))
    }
}

/// De-duplicate endpoints while preserving **first-seen order**.
///
/// Order is the point, not a side effect: `tensor_split` is indexed by ggml
/// device slot and slots follow registration order, so a `HashSet`-style
/// "sorted unique" pass would silently reshuffle the pipeline. Later
/// duplicates are dropped and reported so a caller can log "worker X was
/// announced twice by two registries" instead of quietly running with the wrong
/// number of devices.
pub fn dedup_endpoints<I>(endpoints: I) -> (Vec<RpcEndpoint>, Vec<RpcEndpoint>)
where
    I: IntoIterator<Item = RpcEndpoint>,
{
    let mut seen: HashSet<RpcEndpoint> = HashSet::new();
    let mut unique = Vec::new();
    let mut dropped = Vec::new();
    for ep in endpoints {
        if seen.insert(ep.clone()) {
            unique.push(ep);
        } else {
            dropped.push(ep);
        }
    }
    (unique, dropped)
}

/// One successful `dllm_shim_add_rpc_server`: how many ggml devices the
/// endpoint contributed to the global list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Registration {
    /// The endpoint that was registered.
    pub endpoint: RpcEndpoint,
    /// How many ggml devices it contributed to the global device list.
    pub devices: usize,
}

/// Process-wide record of which RPC endpoints the loaded shim already knows.
///
/// Why it has to be global: the ggml device list the shim mutates is global, so
/// "have I already registered this?" is a property of the process, not of one
/// session. Two coordinators' worth of state in one process would be the bug.
#[derive(Debug, Default)]
pub struct RpcRegistry {
    /// Endpoint -> devices it contributed. Insertion order is registration
    /// order, which is ggml slot order for everything after the local devices.
    registered: Vec<(RpcEndpoint, usize)>,
}

impl RpcRegistry {
    /// Empty registry; nothing has been handed to the shim yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Endpoints already registered, in registration (== ggml slot) order.
    pub fn registered(&self) -> Vec<RpcEndpoint> {
        self.registered.iter().map(|(e, _)| e.clone()).collect()
    }

    /// Devices contributed by the endpoint, or `None` if never registered.
    pub fn devices_for(&self, endpoint: &RpcEndpoint) -> Option<usize> {
        self.registered
            .iter()
            .find(|(e, _)| e == endpoint)
            .map(|(_, n)| *n)
    }

    /// True when `endpoint` has already been handed to
    /// `dllm_shim_add_rpc_server` in this process.
    pub fn contains(&self, endpoint: &RpcEndpoint) -> bool {
        self.devices_for(endpoint).is_some()
    }

    /// Record an endpoint as registered with `devices` ggml devices.
    ///
    /// Pure bookkeeping: the caller has *already* made the FFI call. Split out
    /// from [`Self::register`] so the ordering logic is testable without the
    /// DLL, and so a test can build the exact state a live shim would have.
    pub fn record(&mut self, endpoint: RpcEndpoint, devices: usize) {
        match self.registered.iter_mut().find(|(e, _)| *e == endpoint) {
            Some(slot) => slot.1 = devices,
            None => self.registered.push((endpoint, devices)),
        }
    }

    /// Forget an endpoint. **Does not** unregister it from ggml — the ABI has
    /// no removal — so this only affects future bookkeeping. Only correct when
    /// the process is about to reload the model, never as a way to drop a
    /// dead worker's slots.
    pub fn forget(&mut self, endpoint: &RpcEndpoint) {
        self.registered.retain(|(e, _)| e != endpoint);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_round_trips_through_its_wire_form() {
        let ep: RpcEndpoint = "192.168.1.42:50052".parse().unwrap();
        assert_eq!(ep.host, "192.168.1.42");
        assert_eq!(ep.port, 50052);
        assert_eq!(ep.as_endpoint_string(), "192.168.1.42:50052");
        assert_eq!(ep.to_string(), "192.168.1.42:50052");
        // Hostnames are legal too (mDNS/NetBIOS names on a LAN).
        let named: RpcEndpoint = "pixel-8.lan:50052".parse().unwrap();
        assert_eq!(named.host, "pixel-8.lan");
    }

    #[test]
    fn bad_endpoints_are_named_errors_not_silent_defaults() {
        assert_eq!("".parse::<RpcEndpoint>(), Err(EndpointError::Empty));
        assert_eq!("   ".parse::<RpcEndpoint>(), Err(EndpointError::Empty));
        assert_eq!("host".parse::<RpcEndpoint>(), Err(EndpointError::NoPort));
        assert!(matches!(
            "host:0".parse::<RpcEndpoint>(),
            Err(EndpointError::BadPort { .. })
        ));
        assert!(matches!(
            "host:70000".parse::<RpcEndpoint>(),
            Err(EndpointError::BadPort { .. })
        ));
        assert!(matches!(
            "host:abc".parse::<RpcEndpoint>(),
            Err(EndpointError::BadPort { .. })
        ));
        assert!(matches!(
            ":50052".parse::<RpcEndpoint>(),
            Err(EndpointError::BadHost { .. })
        ));
        // Two endpoints crammed into one string must not parse as a host.
        assert!(matches!(
            "a:1 b:2".parse::<RpcEndpoint>(),
            Err(EndpointError::BadHost { .. })
        ));
        assert!(matches!(
            "[::1]:50052".parse::<RpcEndpoint>(),
            Err(EndpointError::Ipv6NotSupported { .. })
        ));
    }

    #[test]
    fn dedup_preserves_first_seen_order_and_reports_the_dropped_dupes() {
        let input = vec![
            "c:50052".parse().unwrap(),
            "a:50052".parse().unwrap(),
            "c:50052".parse().unwrap(),
            "b:1".parse().unwrap(),
            "a:50052".parse().unwrap(),
        ];
        let (unique, dropped) = dedup_endpoints(input);
        // Order, not sorted order: the pipeline stage order depends on it.
        assert_eq!(
            unique.iter().map(|e| e.to_string()).collect::<Vec<_>>(),
            vec!["c:50052", "a:50052", "b:1"]
        );
        assert_eq!(
            dropped.iter().map(|e| e.to_string()).collect::<Vec<_>>(),
            vec!["c:50052", "a:50052"]
        );
    }

    #[test]
    fn dedup_is_case_sensitive_on_host_but_that_is_stated_not_accidental() {
        // ggml resolves DNS names case-insensitively, but the endpoint string
        // is also used as a registry key here, so `Worker` and `worker` are two
        // keys. Documented rather than silently folded: folding would make the
        // registry key disagree with what the shim was handed.
        let input = vec![
            "worker:50052".parse::<RpcEndpoint>().unwrap(),
            "Worker:50052".parse().unwrap(),
        ];
        let (unique, dropped) = dedup_endpoints(input);
        assert_eq!(unique.len(), 2);
        assert!(dropped.is_empty());
    }

    /// The shim deliberately does NOT dedup, so a re-add double-counts. This
    /// is the registry behaviour that prevents it.
    #[test]
    fn registry_remembers_registration_so_a_double_add_never_happens() {
        let mut reg = RpcRegistry::new();
        let w1: RpcEndpoint = "w1:50052".parse().unwrap();
        let w2: RpcEndpoint = "w2:50052".parse().unwrap();

        assert!(!reg.contains(&w1));
        reg.record(w1.clone(), 1);
        reg.record(w2.clone(), 2);
        assert!(reg.contains(&w1));
        assert_eq!(reg.devices_for(&w1), Some(1));
        assert_eq!(reg.devices_for(&w2), Some(2));
        // Registration order is ggml slot order.
        assert_eq!(
            reg.registered()
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>(),
            vec!["w1:50052", "w2:50052"]
        );

        // The caller asks again: it gets the remembered answer, so it must not
        // make the FFI call. Total device slots stay 3, not 4.
        let (unique, _) = dedup_endpoints(vec![w1.clone(), w1.clone(), w2.clone()]);
        let fresh: Vec<&RpcEndpoint> = unique.iter().filter(|e| !reg.contains(e)).collect();
        assert!(fresh.is_empty(), "both endpoints already registered");
        let total_slots: usize = unique
            .iter()
            .map(|e| reg.devices_for(e).unwrap_or(0))
            .sum();
        assert_eq!(total_slots, 3);
    }

    #[test]
    fn recording_the_same_endpoint_twice_does_not_duplicate_the_slot_block() {
        let mut reg = RpcRegistry::new();
        let w: RpcEndpoint = "w:50052".parse().unwrap();
        reg.record(w.clone(), 1);
        reg.record(w.clone(), 1);
        assert_eq!(reg.registered().len(), 1);
        assert_eq!(reg.devices_for(&w), Some(1));
        // Re-recording with a different device count updates in place rather
        // than adding a second block, which would shift every later slot.
        reg.record(w.clone(), 3);
        assert_eq!(reg.registered().len(), 1);
        assert_eq!(reg.devices_for(&w), Some(3));
    }
}
