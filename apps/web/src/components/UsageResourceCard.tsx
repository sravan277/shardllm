/** Per-device card for the Usage tab (admin breakdown only). */

import { memo } from "react";
import type { UsageDevice } from "../api/types";
import { shortId } from "../lib/format";
import { DeviceLoadBars } from "./DeviceLoadBars";

export const UsageResourceCard = memo(function UsageResourceCard({ d }: { d: UsageDevice }) {
  const label = d.device_name?.trim() || shortId(d.device_id);
  return (
    <div className="card">
      <h2>{label}</h2>
      {label !== d.device_id ? <p className="mono">{d.device_id}</p> : null}
      <p className="pill-row">
        {d.worker_active === true ? <span className="pill live">compute worker active</span> : null}
        <span className={d.active === true ? "pill live" : "pill"}>{d.active === true ? "active" : "idle"}</span>
        {d.role ? <span className="pill">{d.role}</span> : null}
      </p>
      <DeviceLoadBars load={d} />
      <p className="mono meta-chips">
        {d.sessions !== null ? <span>{`${d.sessions} session${d.sessions === 1 ? "" : "s"}`}</span> : null}
        <span>{`${d.tokens_out} tokens out`}</span>
      </p>
    </div>
  );
});