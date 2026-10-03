/** One registry row on the Devices tab.
 *
 * Approve and Revoke are mutually exclusive and driven by the row's real
 * `status`, never both shown. Revoked rows are rendered by the caller in their
 * own section precisely so a revoked device can still be approved again.
 */

import { memo } from "react";
import type { Device } from "../api/types";
import { relativeTime, shortId } from "../lib/format";
import { DeviceLoadBars } from "./DeviceLoadBars";

export const DeviceCard = memo(function DeviceCard({
  device,
  isSelf,
  busy,
  busyAction,
  now,
  onAct,
}: {
  device: Device;
  isSelf: boolean;
  busy: boolean;
  busyAction: "approve" | "revoke" | "delete" | null;
  now: number;
  onAct: (device: Device, action: "approve" | "revoke" | "delete") => void;
}) {
  const label = device.device_name?.trim() || shortId(device.device_id);
  const seen = relativeTime(device.last_seen, now) ?? "never seen";
  const paired = device.status === "paired";
  return (
    <article className="card device-card">
      <h3>{label}</h3>
      {label !== device.device_id ? <p className="mono">{device.device_id}</p> : null}
      <p className="pill-row">
        <span className={paired ? "pill live" : "pill warn"}>{paired ? "paired" : device.status}</span>
        {isSelf ? <span className="pill live">this coordinator</span> : null}
        {device.active ? <span className="pill live">active</span> : <span className="pill">idle</span>}
        {device.worker_active === true ? <span className="pill live">compute worker</span> : null}
        {device.role ? <span className="pill">{device.role}</span> : null}
      </p>
      <p className="hint">
        Last seen {seen}. Paired {relativeTime(device.paired_at, now) ?? "at an unknown time"}.
      </p>
      <DeviceLoadBars load={device} compact />
      <div className="row dev-actions">
        {paired ? (
          <button className="danger" disabled={busy} onClick={() => onAct(device, "revoke")}>
            {busy && busyAction === "revoke" ? "Revoking…" : "Revoke"}
          </button>
        ) : (
          <button disabled={busy} onClick={() => onAct(device, "approve")}>
            {busy && busyAction === "approve" ? "Approving…" : "Approve"}
          </button>
        )}
        <button className="danger" disabled={busy} onClick={() => onAct(device, "delete")}>
          {busy && busyAction === "delete" ? "Deleting…" : "Delete"}
        </button>
      </div>
      {isSelf ? (
        <p className="hint">This is the coordinator itself. Delete is refused by the mesh; use Revoke to lock it out.</p>
      ) : null}
    </article>
  );
});