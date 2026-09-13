//! Coordinator-side optimistic-commit tracking (`contracts/acks.md`).
//!
//! Stages append tentative KV rows **without waiting**; the coordinator
//! piggybacks `COMMITTED(pos-1)` on the next dispatch and aborts with
//! `TRUNCATE(pos)`. [`CommitTracker`] is the coordinator's per-session view
//! of that protocol: tentatively observed `(pos, token)` rows plus the last
//! committed position.
//!
//! Position numbering is owned by the coordinator: one dense `u32` counter
//! per session, 0-based.

use std::collections::BTreeMap;

use dllm_net::Ack;

/// Outcome of feeding one [`Ack`] into [`CommitTracker::on_ack`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CommitOutcome {
    /// Tentative row buffered (`pos` now pending commit).
    Buffered { pos: u32 },
    /// ACK carried no commit state (`RECEIVED`, `COMPUTED`, or stale).
    Ignored { pos: u32 },
    /// Positions newly committed by a `COMMITTED` piggyback.
    Committed(Vec<u32>),
    /// Tentative positions rolled back by a `TRUNCATE`.
    RolledBack(Vec<u32>),
}

/// Coordinator-side optimistic-commit state machine (one per session).
///
/// `tentative` maps token position to token id for rows observed via
/// `KV_TENTATIVE` but not yet committed. `committed_pos` is the last
/// committed position (`None` before the first commit); committing `N`
/// implicitly commits every position `< N` (piggyback rule).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CommitTracker {
    pub committed_pos: Option<u32>,
    pub tentative: BTreeMap<u32, u32>,
}

impl CommitTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Last committed position, if any.
    pub fn last_committed(&self) -> Option<u32> {
        self.committed_pos
    }

    /// Number of tentative rows awaiting commit.
    pub fn pending_count(&self) -> usize {
        self.tentative.len()
    }

    /// True if `pos` has a buffered tentative row.
    pub fn is_pending(&self, pos: u32) -> bool {
        self.tentative.contains_key(&pos)
    }

    /// Buffer a tentative `(pos, token)` row.
    ///
    /// Returns `false` (stale, dropped) when `pos` is already committed.
    /// Retries reuse the same `pos` with the same token (idempotent); a
    /// conflicting token for one `pos` overwrites (last-writer-wins).
    pub fn on_tentative(&mut self, pos: u32, token: u32) -> bool {
        if matches!(self.committed_pos, Some(c) if pos <= c) {
            return false;
        }
        self.tentative.insert(pos, token);
        true
    }

    /// Commit `pos`, implicitly committing every position `< pos`
    /// (piggyback rule). Returns the newly committed positions, ascending.
    ///
    /// Positions below `pos` with no buffered row are still implicitly
    /// committed (reflected in `committed_pos`) but cannot be returned since
    /// no token is known for them. Re-committing an already-committed prefix
    /// returns empty.
    pub fn on_commit(&mut self, pos: u32) -> Vec<u32> {
        if matches!(self.committed_pos, Some(c) if pos <= c) {
            return Vec::new();
        }
        let newly: Vec<u32> = self.tentative.range(..=pos).map(|(&p, _)| p).collect();
        for p in &newly {
            self.tentative.remove(p);
        }
        self.committed_pos = Some(pos);
        newly
    }

    /// Roll back to `pos`: drop tentative rows `>= pos`, keep `<= pos - 1`.
    /// Returns the rolled-back positions, ascending.
    ///
    /// Committed state is never rolled back; callers pass `pos` above the
    /// last committed position per the abort flow.
    pub fn on_truncate(&mut self, pos: u32) -> Vec<u32> {
        let dropped: Vec<u32> = self.tentative.range(pos..).map(|(&p, _)| p).collect();
        for p in &dropped {
            self.tentative.remove(p);
        }
        dropped
    }

    /// Route one inference [`Ack`] into the tracker.
    ///
    /// `KV_TENTATIVE` buffers a row; `COMMITTED` commits the piggybacked
    /// prefix; `TRUNCATE` rolls back the suffix. `RECEIVED`/`COMPUTED` carry
    /// no durable KV state, so they are ignored here (commit tracks durable
    /// tentative rows; per-stage compute quorum is tracked elsewhere).
    pub fn on_ack(&mut self, ack: Ack) -> CommitOutcome {
        match ack {
            Ack::KvTentative { pos, token } => {
                if self.on_tentative(pos, token) {
                    CommitOutcome::Buffered { pos }
                } else {
                    CommitOutcome::Ignored { pos }
                }
            }
            Ack::Committed { pos, .. } => CommitOutcome::Committed(self.on_commit(pos)),
            Ack::Truncate { pos, .. } => CommitOutcome::RolledBack(self.on_truncate(pos)),
            Ack::Received { pos, .. } | Ack::Computed { pos, .. } => {
                CommitOutcome::Ignored { pos }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_piggybacks_uncommitted_prefix() {
        let mut t = CommitTracker::new();
        assert!(t.on_tentative(0, 100));
        assert!(t.on_tentative(1, 101));
        assert!(t.on_tentative(2, 102));
        // Committing 2 implicitly commits 0 and 1 (piggyback rule).
        assert_eq!(t.on_commit(2), vec![0, 1, 2]);
        assert_eq!(t.committed_pos, Some(2));
        assert!(t.tentative.is_empty());
        // Re-committing an already-committed prefix is idempotent.
        assert_eq!(t.on_commit(1), Vec::<u32>::new());
        assert_eq!(t.on_commit(2), Vec::<u32>::new());
    }

    #[test]
    fn commit_advances_over_gaps_implicitly() {
        let mut t = CommitTracker::new();
        t.on_tentative(0, 100);
        t.on_tentative(2, 102);
        // Missing pos 1 is still implicitly committed; only known tokens return.
        assert_eq!(t.on_commit(2), vec![0, 2]);
        assert_eq!(t.committed_pos, Some(2));
    }

    #[test]
    fn truncate_rolls_back_suffix_only() {
        let mut t = CommitTracker::new();
        t.on_tentative(0, 100);
        t.on_tentative(1, 101);
        t.on_tentative(2, 102);
        assert_eq!(t.on_commit(0), vec![0]);
        // Abort drops tentative >= 1, keeps committed 0.
        assert_eq!(t.on_truncate(1), vec![1, 2]);
        assert!(t.tentative.is_empty());
        assert_eq!(t.committed_pos, Some(0));
        // Stream resumes at 1 with a fresh tentative row.
        assert!(t.on_tentative(1, 201));
        assert_eq!(t.tentative.get(&1), Some(&201));
    }

    #[test]
    fn stale_tentative_after_commit_is_dropped() {
        let mut t = CommitTracker::new();
        t.on_tentative(0, 100);
        assert_eq!(t.on_commit(0), vec![0]);
        assert!(!t.on_tentative(0, 100));
        assert!(t.tentative.is_empty());
    }

    #[test]
    fn ack_routing_consumes_dllm_net_vocabulary() {
        let mut t = CommitTracker::new();
        assert_eq!(
            t.on_ack(Ack::KvTentative { pos: 0, token: 7 }),
            CommitOutcome::Buffered { pos: 0 }
        );
        assert_eq!(
            t.on_ack(Ack::Received { pos: 1, token: 8 }),
            CommitOutcome::Ignored { pos: 1 }
        );
        assert_eq!(
            t.on_ack(Ack::Computed { pos: 1, token: 8 }),
            CommitOutcome::Ignored { pos: 1 }
        );
        assert_eq!(
            t.on_ack(Ack::Committed { pos: 0, token: 7 }),
            CommitOutcome::Committed(vec![0])
        );
        assert_eq!(
            t.on_ack(Ack::Truncate { pos: 5, token: 0 }),
            CommitOutcome::RolledBack(vec![])
        );
    }
}
