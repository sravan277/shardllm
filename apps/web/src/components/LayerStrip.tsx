/** The layer-ownership strip: one continuous bar, one labelled block per
 *  contiguous run of layers, sized in proportion to how many it holds.
 *
 * 28 individual slivers tell you nothing; "phone layers 0–9 (10) / laptop
 * layers 10–27 (18)" answers the question the tab exists for. The single-device
 * case says so in one plain sentence instead of drawing 28 identical blocks.
 */

import { memo } from "react";
import type { LayerRun } from "../lib/layers";

export type StripRun = LayerRun & { name: string; color: string };

export const LayerStrip = memo(function LayerStrip({
  runs,
  unassigned,
  singleDevice,
}: {
  runs: StripRun[];
  unassigned: number[];
  singleDevice: boolean;
}) {
  if (runs.length === 0) {
    return <p className="hint">No stage owns any layer yet — the plan carries no layer ranges.</p>;
  }
  const caption = singleDevice
    ? `One device runs the whole model. All 28 layers sit on ${runs[0].name}; nothing is split yet.`
    : `${runs.length} devices share the model across ${runs.reduce((n, r) => n + r.count, 0)} of 28 layers.`;
  return (
    <div className="layer-strip-wrap">
      <div
        className="layer-strip"
        role="img"
        aria-label={runs.map((r) => `${r.name} holds layers ${r.start} to ${r.end}`).join(". ")}
      >
        {runs.map((r) => (
          <div
            key={`${r.deviceId}:${r.start}`}
            className="layer-run"
            style={{ flexGrow: r.count, background: r.color }}
            title={`${r.name}: layers ${r.start}–${r.end}`}
          >
            <span className="layer-run-name">{r.name}</span>
            <span className="mono">
              layers {r.start}–{r.end}
            </span>
            <span className="mono">
              {r.count} layer{r.count === 1 ? "" : "s"}
            </span>
          </div>
        ))}
        {unassigned.length > 0 ? (
          <div className="layer-run unassigned" style={{ flexGrow: unassigned.length }} title="Unassigned layers">
            <span className="layer-run-name">Unassigned</span>
            <span className="mono">{unassigned.length} layers</span>
          </div>
        ) : null}
      </div>
      <p className="hint">{caption}</p>
    </div>
  );
});