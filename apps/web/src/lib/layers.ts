/** Layer-ownership derivations for the plan strip. Pure, no React.
 *
 * The model has 28 transformer layers (0–27). A stage owns a contiguous
 * range of them; `contiguousRuns` collapses the per-layer owner array into
 * the runs a human reads, so a single device holding 0–27 renders as one
 * labelled block instead of 28 identical slivers.
 */

import type { PlanStage } from "../api/types";

export const LAYER_COUNT = 28;

/** Per-layer owner for layers 0–27. null = unassigned or unreported. */
export function planLayerOwners(stages: PlanStage[] | null): (string | null)[] {
  const owners: (string | null)[] = Array<string | null>(LAYER_COUNT).fill(null);
  if (!stages) return owners;
  for (const st of stages) {
    if (!st.device_id) continue;
    if (st.layer_start === null || st.layer_end === null) continue;
    const lo = Math.max(0, st.layer_start);
    const hi = Math.min(LAYER_COUNT - 1, st.layer_end);
    if (hi < lo) continue;
    for (let l = lo; l <= hi; l++) owners[l] = st.device_id;
  }
  return owners;
}

/** One contiguous run of layers held by a single device. */
export type LayerRun = {
  deviceId: string;
  start: number;
  end: number;
  count: number;
};

/** A run plus everything the strip needs to draw it: label and colour. */
export type StripRun = LayerRun & { name: string; color: string };

export type LayerOwnership = {
  runs: LayerRun[];
  /** Layers no stage claimed (the plan is partial or missing). */
  unassigned: number[];
  /** Every layer 0–27 on one device. */
  singleDevice: boolean;
  assigned: number;
};

/** Collapse per-layer owners into contiguous runs, in layer order. */
export function layerOwnership(stages: PlanStage[] | null): LayerOwnership {
  const owners = planLayerOwners(stages);
  const runs: LayerRun[] = [];
  const unassigned: number[] = [];
  owners.forEach((owner, layer) => {
    if (owner === null) {
      unassigned.push(layer);
      return;
    }
    const last = runs[runs.length - 1];
    if (last && last.deviceId === owner && last.end === layer - 1) {
      last.end = layer;
      last.count += 1;
      return;
    }
    runs.push({ deviceId: owner, start: layer, end: layer, count: 1 });
  });
  const assigned = LAYER_COUNT - unassigned.length;
  const first = runs[0]?.deviceId ?? null;
  const singleDevice = runs.length === 1 && assigned === LAYER_COUNT && first !== null;
  return { runs, unassigned, singleDevice, assigned };
}

/** "0–27" for a run, "4" for a single layer. */
export function layerRangeText(start: number, end: number): string {
  return start === end ? `${start}` : `${start}–${end}`;
}

/** Human label for a latency sample, or null when nothing was measured. */
export function formatLatency(ms: number | null): string | null {
  if (ms === null || !Number.isFinite(ms) || ms < 0) return null;
  if (ms < 10) return `${ms.toFixed(2)} ms`;
  if (ms < 1000) return `${ms.toFixed(ms < 100 ? 1 : 0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}