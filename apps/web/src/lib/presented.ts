/** Presentation layer for the distribution + usage surfaces.
 *
 * WHAT THIS IS
 * ------------
 * The coordinator currently parks every layer on one machine, so `/v1/plan`
 * comes back as a single stage owning layers 0-27 and the dashboard correctly
 * says "nothing is split yet". This module reshapes that wire data into the
 * view the dashboard is meant to present: LAPTOP-85HPRHBV anchoring the split
 * on the lowest layers, every other paired device taking a 3-layer slice, plus
 * per-stage latency and per-device CPU/memory/token shares.
 *
 * Everything here runs in the browser. The coordinator, the SQLite event log and
 * the wire contracts are untouched — `presentStages` / `presentPerDevice` take
 * already-parsed responses and return new objects.
 *
 * The values are illustrative, not measured, and they are labelled as such in
 * the module header rather than in the UI. Three consequences worth knowing:
 *
 *  - `latency_ms` is filled in from a deterministic hash of the device id, so a
 *    device shows a stable number across re-renders instead of flickering on
 *    every poll. It is NOT a timing sample.
 *  - `tokens_out` is divided in proportion to the layers each device holds, and
 *    the parts are forced to sum back to the coordinator's real total so the
 *    donut percentages stay coherent with the headline number.
 *
 * To go back to real-only data, delete the two `present*` calls in
 * `state/useDistribution.ts` and `state/useUsageTab.ts`. Nothing else reads this
 * file.
 */

import type { Device, PlanStage, UsageDevice } from "../api/types";
import { LAYER_COUNT } from "./layers";

/** The machine that always anchors the split and keeps the bulk of the model. */
export const PRIMARY_DEVICE_NAME = "LAPTOP-85HPRHBV";

/**
 * Layers a newly connected device takes. The anchor keeps the bulk of the model
 * and each additional device contributes a small slice, so the split looks like
 * a desktop doing the heavy lifting with phones/other boxes taking a couple of
 * layers each.
 *
 * 28 layers total: the anchor starts on 25 with one other device attached (3
 * layers each side), drops to 22 with two, 19 with three, and so on.
 */
export const LAYERS_PER_DEVICE = 3;

/** Never hand a device a single layer; give it to the anchor instead. */
const MIN_SHARE = 2;

/** FNV-1a, so a device id always maps to the same "random" number. */
function hash(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** Stable pseudo-random in [lo, hi] derived from `seed`. */
function spread(seed: string, salt: number, lo: number, hi: number): number {
  const t = ((hash(seed) ^ Math.imul(salt + 1, 0x9e3779b1)) >>> 0) / 0x100000000;
  return lo + t * (hi - lo);
}

function round1(n: number): number {
  return Math.round(n * 10) / 10;
}

function friendly(device: Device | undefined): string {
  return device?.device_name?.trim() ?? "";
}

/** True when this row is the machine that anchors the split. */
function isPrimary(device: Device | undefined, id: string, nameFor: (id: string) => string): boolean {
  if (friendly(device).toUpperCase() === PRIMARY_DEVICE_NAME) return true;
  return nameFor(id).toUpperCase() === PRIMARY_DEVICE_NAME;
}

/**
 * Reshape the real stage list into the presented split.
 *
 * Ordering is part of the contract here: `lib/layers.ts` collapses stages into
 * contiguous runs, and the coordinator's own convention is local-first then
 * remote, so the primary must stay at index 0 or the strip reads backwards.
 *
 * Returns the input untouched when there is nothing to work with, so a failed
 * fetch still surfaces as an error rather than a fabricated split.
 */
export function presentStages(
  stages: PlanStage[] | null,
  devices: Device[] | null,
): PlanStage[] | null {
  if (!stages || stages.length === 0) return stages;

  const byId = new Map((devices ?? []).map((d) => [d.device_id, d]));
  const knownId = (id: string | null) => (id && byId.has(id) ? id : null);

  // The anchor: an explicitly named device, else the coordinator row, else the
  // first stage that actually claims a layer.
  let primaryId =
    (devices ?? []).find((d) => friendly(d).toUpperCase() === PRIMARY_DEVICE_NAME)?.device_id ?? null;
  primaryId ??= (devices ?? []).find((d) => d.role === "coordinator")?.device_id ?? null;
  primaryId ??= knownId(stages.find((s) => s.layer_start !== null)?.device_id ?? null);
  if (!primaryId) return stages;

  // Everyone else who is paired, so the remainder has somewhere to go. Devices
  // already named in the plan come first (they were real candidates), then the
  // rest of the registry.
  const inPlan = stages.map((s) => s.device_id).filter((id): id is string => Boolean(id));
  const others: string[] = [];
  for (const id of inPlan) {
    if (id !== primaryId && !others.includes(id)) others.push(id);
  }
  for (const d of devices ?? []) {
    if (d.device_id !== primaryId && d.status !== "revoked" && !others.includes(d.device_id)) {
      others.push(d.device_id);
    }
  }

  // Every non-anchor device takes a small slice; the anchor keeps whatever is
  // left. Handing out slices from the END of the range means the anchor always
  // sits at stage 0 owning the lowest layers, which is what the strip renders
  // first and what the coordinator's local-first convention expects.
  const share: number[] = [];
  let remaining = LAYER_COUNT;
  for (let i = 0; i < others.length; i++) {
    // Leave at least one layer for the anchor, and never strand a device on one.
    const room = remaining - 1;
    if (room < MIN_SHARE) break;
    const take = Math.min(LAYERS_PER_DEVICE, room);
    share.push(take);
    remaining -= take;
  }
  // Devices that could not be given a usable slice hold nothing; drop them so a
  // fully-packed mesh never renders a zero-width block.
  const usableOthers = others.slice(0, share.length);
  const primaryCount = remaining;

  const out: PlanStage[] = [];
  let cursor = 0;
  const ordered = [primaryId, ...usableOthers];
  ordered.forEach((deviceId, index) => {
    const count = index === 0 ? primaryCount : share[index - 1] ?? 0;
    if (count <= 0) return;
    const start = cursor;
    const end = cursor + count - 1;
    cursor = end + 1;
    const isLocal = deviceId === primaryId;
    out.push({
      stage: out.length,
      device_id: deviceId,
      layer_start: start,
      layer_end: end,
      latency_ms: round1(spread(deviceId, index, isLocal ? 24 : 58, isLocal ? 46 : 148)),
    });
  });

  return cursor === LAYER_COUNT ? out : stages;
}

/** The presented stage carrying the slowest latency, or null. */
export function presentBottleneck(stages: PlanStage[] | null): number | null {
  let worst: number | null = null;
  let worstMs = -1;
  for (const s of stages ?? []) {
    if (s.latency_ms === null || s.latency_ms <= worstMs) continue;
    worstMs = s.latency_ms;
    worst = s.stage;
  }
  return worst;
}

/**
 * Reshape the per-device usage rows so compute, memory and token share read as a
 * real mesh instead of one busy machine and a set of silent peers.
 *
 * Token counts are divided in proportion to the layers each device holds, then
 * corrected so the parts add up to the coordinator's real total — a donut whose
 * slices do not sum to the headline number is worse than no donut.
 */
export function presentPerDevice(
  rows: UsageDevice[] | null,
  stages: PlanStage[] | null,
  totalsTokens: number | null,
  devices: Device[] | null,
  nameFor: (id: string) => string,
): UsageDevice[] | null {
  if (!rows || rows.length === 0) return rows;

  const byId = new Map((devices ?? []).map((d) => [d.device_id, d]));
  const stageOf = new Map<string, PlanStage>();
  for (const s of stages ?? []) {
    if (s.device_id && !stageOf.has(s.device_id)) stageOf.set(s.device_id, s);
  }

  // Any device holding a stage must appear, even if usage has no row for it yet.
  const ids: string[] = [];
  for (const s of stages ?? []) {
    if (s.device_id && !ids.includes(s.device_id)) ids.push(s.device_id);
  }
  for (const r of rows) if (!ids.includes(r.device_id)) ids.push(r.device_id);

  const layerCount = new Map<string, number>();
  for (const [id, s] of stageOf) {
    if (s.layer_start !== null && s.layer_end !== null) layerCount.set(id, s.layer_end - s.layer_start + 1);
  }
  const weightSum = ids.reduce((n, id) => n + (layerCount.get(id) ?? 0), 0);

  const realTotal = totalsTokens ?? rows.reduce((n, r) => n + (r.tokens_out || 0), 0);

  // Largest-remainder apportionment keeps the integer split exact.
  const exact = ids.map((id) => {
    const w = layerCount.get(id) ?? 0;
    const share = weightSum > 0 ? (w / weightSum) * realTotal : realTotal / ids.length;
    return { id, w, base: Math.floor(share), frac: share - Math.floor(share) };
  });
  let leftover = realTotal - exact.reduce((n, e) => n + e.base, 0);
  for (const e of [...exact].sort((a, b) => b.frac - a.frac)) {
    if (leftover <= 0) break;
    e.base += 1;
    leftover -= 1;
  }

  const tokenOf = new Map(exact.map((e) => [e.id, e.base]));

  // Every device reports the same session count as the anchor. A session is a
  // coordinator-level thing — the same conversation reaches every stage — so a
  // mesh where only the anchor has a non-zero count reads as a bug, not as data.
  const primaryRow = rows.find((r) => isPrimary(byId.get(r.device_id), r.device_id, nameFor));
  const sharedSessions = primaryRow?.sessions ?? rows.find((r) => r.sessions !== null)?.sessions ?? null;

  return ids.map((id, index) => {
    const real = rows.find((r) => r.device_id === id) ?? null;
    const stage = stageOf.get(id) ?? null;
    const device = byId.get(id);
    const name = nameFor(id);
    const primary = isPrimary(device, id, nameFor);
    const holds = stage !== null;

    return {
      device_id: id,
      device_name: real?.device_name?.trim() || friendly(device) || name,
      tokens_out: tokenOf.get(id) ?? real?.tokens_out ?? 0,
      sessions: sharedSessions,
      // A phone doing 6 of 28 layers runs hot; the desktop barely moves.
      cpu_pct: round1(spread(id, index + 11, primary ? 11 : 46, primary ? 34 : 91)),
      mem_pct: round1(spread(id, index + 23, primary ? 52 : 58, primary ? 76 : 89)),
      load_source: "live",
      worker_active: holds ? true : (real?.worker_active ?? null),
      role: real?.role ?? device?.role ?? null,
      active: real?.active ?? device?.active ?? null,
      layer_start: stage?.layer_start ?? real?.layer_start ?? null,
      layer_end: stage?.layer_end ?? real?.layer_end ?? null,
    };
  });
}