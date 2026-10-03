/** Join the plan with display data: layer runs get a device name and a colour
 *  from the shared palette, so the same device looks the same everywhere. */

import type { PlanStage } from "../api/types";
import { layerOwnership, type StripRun } from "./layers";
import { colorOrder, usageColorFor } from "./palette";

export type Strip = {
  runs: StripRun[];
  unassigned: number[];
  singleDevice: boolean;
  assigned: number;
};

/** `stages` from `/v1/plan` or `/v1/usage`. null = nothing reported. */
export function stripRuns(stages: PlanStage[] | null, nameFor: (id: string) => string): Strip {
  const ownership = layerOwnership(stages);
  const order = colorOrder(ownership.runs.map((r) => r.deviceId));
  return {
    ...ownership,
    runs: ownership.runs.map((r) => ({ ...r, name: nameFor(r.deviceId), color: usageColorFor(r.deviceId, order) })),
  };
}