/** Usage tab: what the mesh has produced, and who produced it.
 *
 * The per-device breakdown is admin only, so a 403 (or `is_admin:false`) shows
 * the totals plus a note rather than an empty chart. The layer split that used
 * to live here now lives on the Distribution tab.
 */

import { useMemo } from "react";
import type { UsageTab } from "../state/useUsageTab";
import { colorOrder, usageColorFor } from "../lib/palette";
import { shortId } from "../lib/format";
import { UsageDonut, type DonutSlice } from "../components/UsageDonut";
import { UsageResourceCard } from "../components/UsageResourceCard";
import { DeviceLoadBars } from "../components/DeviceLoadBars";
import { EmptyState } from "../components/EmptyState";
import { Section } from "../components/Section";

export function UsageTab({ data }: { data: UsageTab }) {
  const usage = data.usage;
  const totals = usage?.totals ?? null;
  const perDevice = usage?.perDevice ?? null;

  const slices: DonutSlice[] = useMemo(
    () => (perDevice ?? []).map((e) => ({ key: e.device_id, label: e.device_name?.trim() || shortId(e.device_id), tokens: e.tokens_out })),
    [perDevice],
  );
  const total = useMemo(() => slices.reduce((n, s) => n + s.tokens, 0), [slices]);
  const order = useMemo(() => colorOrder(slices.map((s) => s.key)), [slices]);
  const names = useMemo(
    () => new Map((perDevice ?? []).map((e) => [e.device_id, e.device_name?.trim() || shortId(e.device_id)])),
    [perDevice],
  );

  const worker = useMemo(() => {
    const entry = (perDevice ?? []).find((e) => e.worker_active === true);
    if (entry) {
      return {
        active: true as const,
        load: { cpu_pct: entry.cpu_pct, mem_pct: entry.mem_pct, load_source: entry.load_source },
        who: names.get(entry.device_id) ?? null,
      };
    }
    if (totals?.worker_active === true) {
      return {
        active: true as const,
        load: { cpu_pct: totals.cpu_pct, mem_pct: totals.mem_pct, load_source: totals.load_source },
        who: null,
      };
    }
    return { active: false as const, load: null, who: null };
  }, [names, perDevice, totals]);

  return (
    <main className="fleet">
      <Section
        title="Usage"
        hint={
          data.loading
            ? "Reading usage…"
            : usage === null
              ? (data.error ?? "Usage could not be read.")
              : "Live totals from GET /v1/usage."
        }
        action={
          <button onClick={data.refresh} disabled={data.loading}>
            {data.loading ? "Refreshing…" : "Refresh"}
          </button>
        }
      />

      <Section title="Totals">
        <p className="mono meta-chips">
          <span>
            {totals?.tokens_out_total !== null && totals?.tokens_out_total !== undefined
              ? `${totals.tokens_out_total} tokens out`
              : "Tokens out: not reported"}
          </span>
<span>
              {totals?.sessions_total !== null && totals?.sessions_total !== undefined
                ? `${totals.sessions_total} session${totals.sessions_total === 1 ? "" : "s"}`
                : "Sessions: not reported"}
            </span>
        </p>
        {usage?.forbidden ? (
          <p className="hint warn">
            This coordinator returned 403 for usage, so the per-device breakdown is not available to you. Totals above are real.
          </p>
        ) : null}
      </Section>

      <Section title="Compute worker" hint="The device currently holding pipeline stages.">
        {worker.active ? (
          <>
            <p className="pill-row">
              <span className="pill live">compute worker active</span>
              {worker.who ? <span className="pill">{worker.who}</span> : null}
            </p>
            <DeviceLoadBars load={worker.load} />
          </>
        ) : (
          <p className="hint">No compute worker is reporting. One device runs every layer until a peer pairs as a worker.</p>
        )}
      </Section>

      <Section title="Tokens out per device" hint="Share of everything the mesh has generated.">
        {usage?.forbidden ? (
          <EmptyState title="Breakdown not available" body="Per-device token shares are admin only on this coordinator. The totals above are the honest part." />
        ) : slices.length === 0 || total === 0 ? (
          <EmptyState title="No tokens yet" body="Send a message in chat and the share per device appears here." />
        ) : (
          <div className="donut-wrap">
            <UsageDonut slices={slices} order={order} total={total} />
            <ul className="legend">
              {slices.map((s) => {
                const pct = total > 0 ? (s.tokens / total) * 100 : 0;
                return (
                  <li key={s.key} className="mono">
                    <span className="sw" style={{ background: usageColorFor(s.key, order) }} aria-hidden />
                    {`${s.label} — ${s.tokens} tokens (${pct.toFixed(1)}%)`}
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </Section>

      {perDevice && perDevice.length > 0 ? (
        <Section title="Resources per device" hint="Load bars appear only where a device reports a sample; the rest say so.">
          <div className="usage-grid">
            {perDevice.map((d) => (
              <UsageResourceCard key={d.device_id} d={d} />
            ))}
          </div>
        </Section>
      ) : null}
    </main>
  );
}