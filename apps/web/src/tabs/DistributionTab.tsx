/** Distribution tab: how the model is split and what each stage costs.
 *
 * This is the headline answer to "how is the distribution actually done?":
 *
 *  1. one continuous layer strip, one labelled block per device run;
 *  2. a per-stage table with measured latency, and the bottleneck called out;
 *  3. per-device CPU, memory and token share, only where reported;
 *  4. link + mesh numbers, with an explicit empty state where nothing measures
 *     them yet.
 *
 * Nothing here is estimated. `latency_ms`, `bottleneck` and `bandwidth` are all
 * null server-side today, and each one renders as "not measured yet" instead of
 * a plausible-looking number.
 */

import { useMemo } from "react";
import type { Distribution } from "../state/useDistribution";
import { stripRuns } from "../lib/strip";
import { formatDuration } from "../lib/format";
import { LayerStrip } from "../components/LayerStrip";
import { StageTiming, type NamedStage } from "../components/StageTiming";
import { DeviceLoadBars } from "../components/DeviceLoadBars";
import { EmptyState, NotMeasured } from "../components/EmptyState";
import { Section } from "../components/Section";
import type { UsageDevice } from "../api/types";

export function DistributionTab({ data }: { data: Distribution }) {
  const nameFor = data.nameFor;
  const stages = data.plan?.stages ?? data.usage?.stages ?? null;
  const strip = useMemo(() => stripRuns(stages, nameFor), [nameFor, stages]);
  const measuredStages = useMemo(() => (stages ?? []).filter((s) => s.latency_ms !== null).length, [stages]);

  const namedStages: NamedStage[] = useMemo(
    () =>
      (stages ?? []).map((s) => ({
        ...s,
        name: s.device_id ? nameFor(s.device_id) : "no device assigned",
      })),
    [nameFor, stages],
  );

  // Devices that hold a stage come first; then anything else with usage data.
  const compute = useMemo<UsageDevice[]>(() => {
    const per = data.usage?.perDevice;
    if (!per || per.length === 0) return [];
    const stageIds = new Set(strip.runs.map((r) => r.deviceId));
    return [...per].sort((a, b) => {
      const sa = stageIds.has(a.device_id) ? 0 : 1;
      const sb = stageIds.has(b.device_id) ? 0 : 1;
      return sa - sb || a.device_id.localeCompare(b.device_id);
    });
  }, [data.usage, strip.runs]);

  const totals = data.usage?.totals ?? null;
  const bandwidth = data.usage?.bandwidth ?? null;
  const mesh = data.stats?.mesh ?? null;
  const tps = data.stats?.tokens_per_second ?? null;

  return (
    <main className="fleet">
      <Section
        title="Distribution"
        hint={
          data.loading
            ? "Reading the plan and usage…"
            : stages === null
              ? (data.error ?? "The coordinator reported no plan. Refresh once it is running.")
              : `${namedStages.length} stage${namedStages.length === 1 ? "" : "s"} across ${strip.runs.length} device${strip.runs.length === 1 ? "" : "s"}, ${strip.assigned} of 28 layers assigned.`
        }
        action={
          <button onClick={data.refresh} disabled={data.loading}>
            {data.loading ? "Refreshing…" : "Refresh"}
          </button>
        }
      >
        {data.usage?.forbidden ? (
          <p className="hint warn">
            The coordinator returned 403 for usage, so only totals are shown. Per-device compute is admin only.
          </p>
        ) : null}
      </Section>

      <Section
        title="Layer ownership"
        hint="Each block is one device, sized by how many of the 28 layers it holds."
      >
        <LayerStrip runs={strip.runs} unassigned={strip.unassigned} singleDevice={strip.singleDevice} />
      </Section>

      <Section
        title="Stage timing"
        hint={
          measuredStages === 0
            ? "No stage has reported a latency sample yet."
            : `${measuredStages} of ${(stages ?? []).length} stages report a measured latency.`
        }
      >
        <StageTiming stages={namedStages} bottleneck={null} />
      </Section>

      <Section
        title="Compute per device"
        hint="CPU and memory render only where the device actually reports a load sample."
      >
        {compute.length === 0 ? (
          <EmptyState
            title="No per-device compute"
            body={
              data.usage?.forbidden
                ? "The coordinator withholds the per-device breakdown from non-admin callers. Totals are below."
                : "No device has reported usage yet. Pair a device and send a message to fill this in."
            }
          />
        ) : (
          <div className="usage-grid">
            {compute.map((d) => (
              <article className="card" key={d.device_id}>
                <h2>{d.device_name?.trim() || data.nameFor(d.device_id)}</h2>
                <p className="pill-row">
                  {d.worker_active === true ? <span className="pill live">compute worker</span> : null}
                  <span className={d.active === true ? "pill live" : "pill"}>{d.active === true ? "active" : "idle"}</span>
                  {d.role ? <span className="pill">{d.role}</span> : null}
                  {layerOf(strip.runs, d.device_id) ? (
                    <span className="pill">holds {layerOf(strip.runs, d.device_id)}</span>
                  ) : null}
                </p>
                <DeviceLoadBars load={d} />
                <p className="mono meta-chips">
                  <span>{`${d.tokens_out} tokens out`}</span>
                  {d.sessions !== null ? <span>{`${d.sessions} session${d.sessions === 1 ? "" : "s"}`}</span> : null}
                </p>
              </article>
            ))}
          </div>
        )}
        <p className="hint">
          {totals?.tokens_out_total !== null && totals?.tokens_out_total !== undefined
            ? `${totals.tokens_out_total} tokens out across the mesh.`
            : "Token totals: not reported."}
          {totals?.sessions_total !== null && totals?.sessions_total !== undefined
            ? ` ${totals.sessions_total} session${totals.sessions_total === 1 ? "" : "s"} recorded.`
            : ""}
        </p>
      </Section>

      <Section title="Link and throughput" hint="Measured link cost of moving activations between devices.">
        <div className="kv-grid">
          <div className="kv-item">
            <span className="kv-key">Link bandwidth</span>
            {bandwidth && bandwidth.mbps !== null ? (
              <span className="kv-val">{`${bandwidth.mbps.toFixed(1)} Mbit/s`}</span>
            ) : (
              <NotMeasured what="Link bandwidth" />
            )}
          </div>
          <div className="kv-item">
            <span className="kv-key">Mesh peers</span>
            {mesh?.peers_connected !== undefined ? (
              <span className="kv-val">
                {`${mesh.peers_connected} connected of ${mesh.allowed_peers ?? "?"} allowed`}
              </span>
            ) : (
              <NotMeasured what="Mesh peer count" />
            )}
          </div>
          <div className="kv-item">
            <span className="kv-key">Decode throughput</span>
            {tps !== null ? <span className="kv-val">{`${tps.toFixed(2)} tokens/s`}</span> : <NotMeasured what="Decode throughput" />}
          </div>
          <div className="kv-item">
            <span className="kv-key">Coordinator uptime</span>
            <span className="kv-val">{formatDuration(data.stats?.uptime_s) ?? "not reported"}</span>
          </div>
        </div>
        <p className="hint">
          Activations cross the LAN as 2 KiB per token. Per-hop throughput and round-trip time arrive with the QUIC
          pipeline; until then this panel says so rather than guessing.
        </p>
      </Section>

      {data.plan?.note ? <p className="hint">Plan note: {data.plan.note}</p> : null}
    </main>
  );
}

/** "layers 0–9" for the chip on a device card, or null when it holds nothing. */
function layerOf(runs: { deviceId: string; start: number; end: number }[], deviceId: string): string | null {
  const r = runs.find((x) => x.deviceId === deviceId);
  if (!r) return null;
  return r.start === r.end ? `layer ${r.start}` : `layers ${r.start}–${r.end}`;
}