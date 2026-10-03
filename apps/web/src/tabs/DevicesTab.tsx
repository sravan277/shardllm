/** Devices tab: who is in this mesh, what each one is doing, and how a new
 *  device gets in.
 *
 * Paired and revoked devices live in two separate sections on purpose. An
 * earlier version collapsed revoked rows into a "3 revoked (hidden)" note,
 * which made re-approval impossible; here every revoked row is a full card
 * with its own Approve button.
 */

import { useEffect, useMemo, useState } from "react";
import type { Devices } from "../state/useDevices";
import type { Device } from "../api/types";
import { DeviceCard } from "../components/DeviceCard";
import { EmptyState } from "../components/EmptyState";
import { PairingCard } from "../components/PairingCard";
import { Section } from "../components/Section";

/** Re-render once a minute so "last seen 12s ago" stays honest. */
const NOW_TICK_MS = 30000;

export function DevicesTab({ devices, base }: { devices: Devices; base: string }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), NOW_TICK_MS);
    return () => window.clearInterval(timer);
  }, []);

  const { paired, revoked, other } = useMemo(() => {
    const p: Device[] = [];
    const r: Device[] = [];
    const o: Device[] = [];
    for (const d of devices.devices ?? []) {
      if (d.status === "revoked") r.push(d);
      else if (d.status === "paired") p.push(d);
      else o.push(d);
    }
    // Active first, then this coordinator, then alphabetical by id.
    const rank = (d: Device) => (d.active ? 0 : 2) + (d.device_id === devices.selfId ? -1 : 0);
    p.sort((a, b) => rank(a) - rank(b) || a.device_id.localeCompare(b.device_id));
    r.sort((a, b) => a.device_id.localeCompare(b.device_id));
    return { paired: p, revoked: r, other: o };
  }, [devices.devices, devices.selfId]);

  const refreshButton = (
    <button onClick={devices.refresh} disabled={devices.loading}>
      {devices.loading ? "Refreshing…" : "Refresh"}
    </button>
  );

  return (
    <main className="fleet">
      <Section
        title="Devices"
        hint={
          devices.loading && devices.devices === null
            ? "Reading the device registry…"
            : devices.devices === null
              ? (devices.error ?? "The registry could not be read.")
              : `${paired.length} paired, ${revoked.length} revoked. The list refreshes every 15 seconds while this tab is open.`
        }
        action={refreshButton}
      >
        {devices.actionNote ? <p className="notice inline">{devices.actionNote}</p> : null}
      </Section>

      <Section title="Pair a new device" hint="One scan or one paste pairs a phone or laptop onto this coordinator.">
        <PairingCard pairing={devices.pairing} base={base} />
      </Section>

      {devices.devices === null ? (
        <Section title="Paired devices">
          <EmptyState
            title="No registry to read"
            body={
              devices.error ??
              "The coordinator did not answer GET /v1/devices. Start dllm serve and refresh, or upgrade the coordinator."
            }
          />
        </Section>
      ) : (
        <>
          <Section title={`Paired (${paired.length})`} hint="Devices allowed to serve pipeline stages and read the event log.">
            {paired.length === 0 ? (
              <EmptyState title="No devices paired yet" body="Pair one with the code above; it shows up here and can take a share of the model." />
            ) : (
              <div className="usage-grid">
                {paired.map((d) => (
                  <DeviceCard
                    key={d.device_id}
                    device={d}
                    isSelf={d.device_id === devices.selfId}
                    busy={devices.busyId === d.device_id}
                    busyAction={devices.busyAction}
                    now={now}
                    onAct={(dev, action) => void devices.act(dev, action)}
                  />
                ))}
              </div>
            )}
          </Section>

          <Section
            title={`Revoked (${revoked.length})`}
            hint="Kept in the registry so access can be restored without re-pairing. Approve puts a device back in the mesh."
          >
            {revoked.length === 0 ? (
              <EmptyState title="Nothing revoked" body="Every paired device is allowed in. Revoke one above if it should lose access." />
            ) : (
              <div className="usage-grid">
                {revoked.map((d) => (
                  <DeviceCard
                    key={d.device_id}
                    device={d}
                    isSelf={false}
                    busy={devices.busyId === d.device_id}
                    busyAction={devices.busyAction}
                    now={now}
                    onAct={(dev, action) => void devices.act(dev, action)}
                  />
                ))}
              </div>
            )}
          </Section>

          {other.length > 0 ? (
            <Section title={`Other status (${other.length})`} hint="Statuses this build does not recognise. Nothing is assumed about them.">
              <div className="usage-grid">
                {other.map((d) => (
                  <DeviceCard
                    key={d.device_id}
                    device={d}
                    isSelf={d.device_id === devices.selfId}
                    busy={devices.busyId === d.device_id}
                    busyAction={devices.busyAction}
                    now={now}
                    onAct={(dev, action) => void devices.act(dev, action)}
                  />
                ))}
              </div>
            </Section>
          ) : null}
        </>
      )}
    </main>
  );
}