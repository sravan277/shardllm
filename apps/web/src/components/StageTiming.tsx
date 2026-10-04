/** Per-stage timing table.
 *
 * `latency_ms` is null on every stage until the coordinator measures one, and
 * renders as an explicit "not measured yet" instead of a zero or an invented
 * estimate. `bottleneck` is accepted and honoured when the coordinator names
 * one; pass null to leave the slowest stage unmarked.
 */

import { memo } from "react";
import type { PlanStage } from "../api/types";
import { formatLatency } from "../lib/layers";

export type NamedStage = PlanStage & { name: string };

export const StageTiming = memo(function StageTiming({
  stages,
  bottleneck,
}: {
  stages: NamedStage[];
  bottleneck: number | null;
}) {
  if (stages.length === 0) {
    return <p className="hint">No stages reported — the coordinator returned a plan without layer ranges.</p>;
  }
  const measured = stages.filter((s) => s.latency_ms !== null).length;
  const slowest = bottleneck === null ? null : stages.find((s) => s.stage === bottleneck) ?? null;

  return (
    <div className="stage-table-wrap">
      <table className="stage-table">
        <caption className="visually-hidden">Pipeline stages with their layer range and measured latency</caption>
        <thead>
          <tr>
            <th scope="col">Stage</th>
            <th scope="col">Device</th>
            <th scope="col">Layers</th>
            <th scope="col">Latency</th>
          </tr>
        </thead>
        <tbody>
          {stages.map((s, i) => {
            const isBottleneck = bottleneck !== null && s.stage === bottleneck;
            const latency = formatLatency(s.latency_ms);
            const range =
              s.layer_start === null || s.layer_end === null
                ? "not reported"
                : `${s.layer_start}–${s.layer_end} (${s.layer_end - s.layer_start + 1})`;
            return (
              <tr key={`${s.device_id ?? "none"}:${i}`} className={isBottleneck ? "bottleneck" : undefined}>
                <th scope="row">
                  {s.stage + 1}
                  {isBottleneck ? <span className="pill warn">bottleneck</span> : null}
                </th>
                <td>{s.name}</td>
                <td className="mono">{range}</td>
                <td className="mono">{latency ?? "not measured yet"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="hint">
        {measured === 0
          ? "Stage timing not measured yet — send a message to generate one, then refresh."
          : `${measured} of ${stages.length} stages report a measured latency.`}
        {slowest
          ? ` Stage ${slowest.stage + 1} on ${slowest.name} is the bottleneck.`
          : bottleneck === null
            ? " No bottleneck identified yet."
            : ""}
      </p>
    </div>
  );
});