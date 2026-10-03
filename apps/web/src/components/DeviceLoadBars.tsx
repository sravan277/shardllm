/** CPU / RAM for one device, with the honest "not reporting" case.
 *
 * A percentage renders only when `load_source` says the sample is live or
 * reported AND the value itself arrived — a missing sample is never drawn as
 * zero. */

import { memo } from "react";
import { loadIsLive } from "../api/parse";
import { UsageBar } from "./UsageBar";

export type DeviceLoad = {
  cpu_pct: number | null;
  mem_pct: number | null;
  load_source: string | null;
};

export const DeviceLoadBars = memo(function DeviceLoadBars({ load, compact }: { load: DeviceLoad; compact?: boolean }) {
  const live = loadIsLive(load.load_source);
  const cpu = live && load.cpu_pct !== null ? load.cpu_pct : null;
  const mem = live && load.mem_pct !== null ? load.mem_pct : null;
  if (cpu === null && mem === null) {
    return <p className="hint">CPU and memory: not reporting — this device sends no load sample.</p>;
  }
  return (
    <div className={compact ? "load-bars compact" : "load-bars"}>
      {cpu !== null ? <UsageBar label="CPU" pct={cpu} tone="load" /> : <p className="hint">CPU: not reporting</p>}
      {mem !== null ? <UsageBar label="MEM" pct={mem} tone="load" /> : <p className="hint">Memory: not reporting</p>}
      {live && load.load_source ? <p className="mono">Sample source: {load.load_source}</p> : null}
    </div>
  );
});