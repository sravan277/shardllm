//! Headless 2-node pipeline drill on 127.0.0.1 (transport + planner + commit).
//!
//! Proves integration without a 2nd device:
//! worker `server()` on 8443 with coordinator fp allowed,
//! coordinator `connect()` with strict fp check, plan over Control,
//! 8 activation frames over Activation (`frame_channel` +
//! `spawn_frame_recv_loop` on the worker), `KvTentative` per frame,
//! piggyback commit at pos 7, truncate drill at pos 6 (resend 6,7 as new
//! tokens), re-commit to `committed_pos == Some(7)`.

use std::{
    collections::HashSet,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use dllm_core::{CommitTracker, DeviceSpec, plan_layers};
use dllm_net::{
    Ack, ActivationFrame,
    tensor_format,
    transport::{
        Identity, StreamKind, accept_stream, connect, frame_channel, open_stream,
        recv_ack, recv_frame, send_ack, send_frame, server, spawn_frame_recv_loop,
    },
};

const PORT: u16 = 8443;
const SESSION_ID: u128 = 0x70_6970_655f_7061_6972_0001;
const TIMEOUT: Duration = Duration::from_secs(15);

async fn timeout_of<T>(
    fut: impl std::future::Future<Output = Result<T, dllm_net::transport::TransportError>>,
    what: &str,
) -> T {
    tokio::time::timeout(TIMEOUT, fut)
        .await
        .unwrap_or_else(|_| panic!("timeout waiting for {what}"))
        .unwrap_or_else(|e| panic!("{what} failed: {e:?}"))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Simulated pairing: mint 2 identities, swap fingerprints.
    let coord_id = Identity::generate().expect("coordinator Identity::generate failed");
    let worker_id = Identity::generate().expect("worker Identity::generate failed");
    let coord_fp = coord_id.fingerprint();
    let worker_fp = worker_id.fingerprint();
    assert_ne!(coord_fp, worker_fp, "fingerprints must differ");

    // 2. Worker server on 8443 with coordinator fp pinned; coordinator connects strict.
    // Worker runs inline in a spawned task so no `quinn::Endpoint` type annotation
    // is needed here (keeps this example at zero manifest edits).
    let mut allowed = HashSet::new();
    allowed.insert(coord_fp.clone());
    let worker_ep = server(&worker_id, PORT, allowed)
        .unwrap_or_else(|e| panic!("worker server() on port {PORT} failed: {e:?}"));
    assert_eq!(
        worker_ep.local_addr().expect("worker local_addr").port(),
        PORT,
        "worker must bind port {PORT}"
    );
    let worker_handle = tokio::spawn(async move {
        let incoming = tokio::time::timeout(TIMEOUT, worker_ep.accept())
            .await
            .expect("worker: timeout waiting for incoming connection")
            .ok_or_else(|| anyhow::anyhow!("worker: endpoint closed, no incoming"))?;
        let conn = tokio::time::timeout(TIMEOUT, incoming)
            .await
            .expect("worker: timeout during handshake")
            .map_err(|e| anyhow::anyhow!("worker handshake failed: {e:?}"))?;

        // --- Control: plan summary frame -> Received ack ---
        let (kind, mut c_send, mut c_recv) =
            timeout_of(accept_stream(&conn), "worker accept Control").await;
        assert_eq!(kind, StreamKind::Control, "worker: first stream must be Control");
        let plan_frame = timeout_of(recv_frame(&mut c_recv), "worker recv plan frame").await;
        let plan: dllm_core::PipelinePlan =
            serde_json::from_slice(&plan_frame.payload)
                .expect("worker: plan payload must decode as PipelinePlan JSON");
        assert_eq!(plan.stages.len(), 2, "worker: plan must have 2 stages, got {:?}", plan.stages);
        assert_eq!(plan.num_layers(), 28, "worker: plan must cover 28 layers");
        send_ack(&mut c_send, &Ack::Received { pos: 0, token: plan.plan_id as u32 })
            .await
            .expect("worker: send plan ack failed");

        // --- Activation: 8 frames via bounded queue, KvTentative per frame ---
        let (kind2, mut a_send, a_recv) =
            timeout_of(accept_stream(&conn), "worker accept Activation").await;
        assert_eq!(kind2, StreamKind::Activation, "worker: second stream must be Activation");
        let (tx, mut rx) = frame_channel();
        let recv_handle = spawn_frame_recv_loop(a_recv, tx);

        let mut got: u32 = 0;
        for expected in 0..8u32 {
            let frame = tokio::time::timeout(TIMEOUT, rx.recv())
                .await
                .expect("worker: timeout waiting for activation frame")
                .expect("worker: activation channel closed early");
            assert_eq!(
                frame.token_position, expected,
                "worker: activation pos mismatch (got {}, want {expected})",
                frame.token_position
            );
            send_ack(
                &mut a_send,
                &Ack::KvTentative { pos: frame.token_position, token: frame.token_position * 10 + 7 },
            )
            .await
            .expect("worker: send KvTentative failed");
            got += 1;
        }

        // --- Ack-kind stream: Truncate(pos 6) from coordinator ---
        let (kind3, mut t_send, mut t_recv) =
            timeout_of(accept_stream(&conn), "worker accept Ack(truncate)").await;
        assert_eq!(kind3, StreamKind::Ack, "worker: third stream must be Ack, got {kind3:?}");
        let trunc = timeout_of(recv_ack(&mut t_recv), "worker recv truncate").await;
        assert_eq!(
            trunc,
            Ack::Truncate { pos: 6, token: 0 },
            "worker: expected Truncate{{pos:6}}, got {trunc:?}"
        );
        let saw_truncate = true;
        send_ack(&mut t_send, &Ack::Received { pos: 6, token: 0 })
            .await
            .expect("worker: send truncate receipt failed");

        // --- Resends 6,7 as new tokens on the still-open Activation stream ---
        for expected in [6u32, 7u32] {
            let frame = tokio::time::timeout(TIMEOUT, rx.recv())
                .await
                .expect("worker: timeout waiting for resend frame")
                .expect("worker: activation channel closed before resend");
            assert_eq!(
                frame.token_position, expected,
                "worker: resend pos mismatch (got {}, want {expected})",
                frame.token_position
            );
            send_ack(
                &mut a_send,
                &Ack::KvTentative { pos: frame.token_position, token: frame.token_position * 10 + 99 },
            )
            .await
            .expect("worker: send resend KvTentative failed");
            got += 1;
        }

        assert_eq!(got, 10, "worker: must see 8 + 2 resend frames, saw {got}");
        recv_handle.abort();
        // Graceful shutdown: hold conn+endpoint until the coordinator closes,
        // so in-flight resend ACKs flush instead of racing endpoint drop.
        let _ = tokio::time::timeout(Duration::from_secs(10), conn.closed()).await;
        anyhow::Ok((got, saw_truncate))
    });

    let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), PORT);
    let coord_conn = connect(&coord_id, addr, &worker_fp)
        .await
        .unwrap_or_else(|e| panic!("coordinator connect() with strict fp failed: {e:?}"));

    // 3. Planner: 28 layers over 2 devices incl. local 16.6 tok/s reference.
    let devices = vec![
        DeviceSpec::new("coord-local", 16.6, 1000.0, 1024),
        DeviceSpec::new("worker-remote", 16.6, 1000.0, 1024),
    ];
    let plan = plan_layers(28, &devices);
    assert_eq!(plan.num_layers(), 28, "plan must cover 28 layers, got {:?}", plan.stages);
    assert_eq!(plan.stages.len(), 2, "2-device plan must have 2 stages, got {:?}", plan.stages);
    assert_eq!(plan.stages[0].start, 0, "plan must start at layer 0");
    assert_eq!(plan.stages[1].end, 27, "plan must end at layer 27");
    assert_eq!(
        plan.stages[0].end + 1,
        plan.stages[1].start,
        "plan stages must be contiguous: {:?}",
        plan.stages
    );

    // --- Control stream: plan summary -> worker Ack ---
    let (mut c_send, mut c_recv) =
        timeout_of(open_stream(&coord_conn, StreamKind::Control), "coord open Control").await;
    let plan_bytes = serde_json::to_vec(&plan).expect("plan must serialize to JSON");
    let plan_frame = ActivationFrame {
        session_id: SESSION_ID,
        plan_id: plan.plan_id,
        token_position: 0,
        source_stage: 0,
        target_stage: 1,
        tensor_format: tensor_format::RAW,
        payload: plan_bytes,
    };
    timeout_of(send_frame(&mut c_send, &plan_frame), "coord send plan frame").await;
    let plan_ack = timeout_of(recv_ack(&mut c_recv), "coord recv plan ack").await;
    assert_eq!(
        plan_ack,
        Ack::Received { pos: 0, token: plan.plan_id as u32 },
        "coordinator: plan ack mismatch, got {plan_ack:?}"
    );

    // --- Activation stream: 8 tiny synthetic frames, increasing pos ---
    let (mut a_send, mut a_recv) =
        timeout_of(open_stream(&coord_conn, StreamKind::Activation), "coord open Activation").await;
    for pos in 0..8u32 {
        let frame = ActivationFrame {
            session_id: SESSION_ID,
            plan_id: plan.plan_id,
            token_position: pos,
            source_stage: 0,
            target_stage: 1,
            tensor_format: tensor_format::F16,
            payload: vec![pos as u8; 16],
        };
        timeout_of(send_frame(&mut a_send, &frame), "coord send activation").await;
    }
    let mut tracker = CommitTracker::new();
    let mut first_tokens = Vec::new();
    for expected in 0..8u32 {
        let ack = timeout_of(recv_ack(&mut a_recv), "coord recv KvTentative").await;
        match ack {
            Ack::KvTentative { pos, token } => {
                assert_eq!(pos, expected, "coordinator: KvTentative pos mismatch (got {pos}, want {expected})");
                first_tokens.push((pos, token));
                let outcome = tracker.on_ack(ack);
                assert_eq!(outcome, dllm_core::CommitOutcome::Buffered { pos }, "on_ack must buffer pos {pos}");
            }
            other => panic!("coordinator: expected KvTentative, got {other:?}"),
        }
    }
    assert_eq!(tracker.pending_count(), 8, "tracker must hold 8 tentative rows");

    // Piggyback demo: committing pos 7 on a clone implicitly commits 0..=7.
    {
        let mut demo = tracker.clone();
        let newly = demo.on_commit(7);
        assert_eq!(newly, vec![0, 1, 2, 3, 4, 5, 6, 7], "piggyback on_commit(7) must commit full prefix, got {newly:?}");
        assert_eq!(demo.committed_pos, Some(7), "demo piggyback must end at Some(7)");
    }

    // Truncate drill needs an uncommitted suffix: commit prefix 0..=5 first,
    // then truncate at 6, resend 6,7 as new tokens, re-commit 7.
    let committed_prefix = tracker.on_commit(5);
    assert_eq!(committed_prefix, vec![0, 1, 2, 3, 4, 5], "prefix commit mismatch: {committed_prefix:?}");

    // Send Truncate(pos 6) over a dedicated Ack-priority stream.
    let (mut t_send, mut t_recv) =
        timeout_of(open_stream(&coord_conn, StreamKind::Ack), "coord open Ack(truncate)").await;
    timeout_of(send_ack(&mut t_send, &Ack::Truncate { pos: 6, token: 0 }), "coord send truncate").await;
    let trunc_receipt = timeout_of(recv_ack(&mut t_recv), "coord recv truncate receipt").await;
    assert_eq!(
        trunc_receipt,
        Ack::Received { pos: 6, token: 0 },
        "worker must confirm truncate receipt, got {trunc_receipt:?}"
    );
    let dropped = tracker.on_truncate(6);
    assert_eq!(dropped, vec![6, 7], "on_truncate(6) must drop [6,7], got {dropped:?}");

    // Resend 6,7 as new tokens (new payload marker +100).
    for pos in [6u32, 7u32] {
        let frame = ActivationFrame {
            session_id: SESSION_ID,
            plan_id: plan.plan_id,
            token_position: pos,
            source_stage: 0,
            target_stage: 1,
            tensor_format: tensor_format::F16,
            payload: vec![pos.wrapping_add(100) as u8; 16],
        };
        timeout_of(send_frame(&mut a_send, &frame), "coord resend activation").await;
    }
    for expected in [6u32, 7u32] {
        let ack = timeout_of(recv_ack(&mut a_recv), "coord recv resend KvTentative").await;
        match ack {
            Ack::KvTentative { pos, token } => {
                assert_eq!(pos, expected, "resend ack pos mismatch (got {pos}, want {expected})");
                let old = first_tokens.iter().find(|(p, _)| *p == pos).map(|(_, t)| *t);
                if let Some(old_tok) = old {
                    assert_ne!(token, old_tok, "resend token for pos {pos} must differ from first pass");
                }
                tracker.on_ack(ack);
            }
            other => panic!("coordinator: expected resend KvTentative, got {other:?}"),
        }
    }

    // Re-commit piggyback at pos 7.
    let recommitted = tracker.on_commit(7);
    assert_eq!(recommitted, vec![6, 7], "re-commit(7) must commit resends [6,7], got {recommitted:?}");
    assert_eq!(
        tracker.committed_pos,
        Some(7),
        "CommitTracker must end with committed_pos==Some(7), got {:?}",
        tracker.committed_pos
    );

    coord_conn.close(0u32.into(), b"pipe_pair done");

    let (worker_frames, worker_saw_truncate) = tokio::time::timeout(TIMEOUT, worker_handle)
        .await
        .expect("timeout waiting for worker task")
        .expect("worker task panicked")
        .expect("worker task failed");
    assert!(worker_saw_truncate, "worker must have seen truncate");
    assert_eq!(worker_frames, 10, "worker must have seen 10 frames, saw {worker_frames}");

    println!(
        "PIPE_PAIR PASS stages={} layers=28 frames={} resends=2 committed_pos=7 truncate_seen=true plan_id={}",
        plan.stages.len(),
        worker_frames,
        plan.plan_id
    );
    Ok(())
}
