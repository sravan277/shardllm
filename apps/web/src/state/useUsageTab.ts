/** Usage tab data: totals plus the admin-only per-device breakdown.
 *
 * `GET /v1/usage` is asked for without a group — there is one network at a
 * time. A 403 or `is_admin:false` still yields honest totals, so the tab shows
 * those plus a note instead of pretending the breakdown does not exist.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { jgetStatus } from "../api/http";
import { parseDevices, parseStats, parseUsage } from "../api/parse";
import type { Stats, Usage, Device } from "../api/types";
import { shortId } from "../lib/format";
import { presentPerDevice, presentStages } from "../lib/presented";

export type UsageTab = {
  usage: Usage | null;
  stats: Stats | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
};

export function useUsageTab(base: string, active: boolean, generation: number): UsageTab {
  const [usage, setUsage] = useState<Usage | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [devices, setDevices] = useState<Device[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    const [usageRes, statsRes, devicesRes] = await Promise.allSettled([
      jgetStatus(base, "/v1/usage"),
      jgetStatus(base, "/api/stats"),
      jgetStatus(base, "/v1/devices"),
    ]);
    if (usageRes.status === "fulfilled") {
      setUsage(parseUsage(usageRes.value.json, usageRes.value.status === 403));
      setError(null);
    } else {
      setUsage(null);
      setError("Could not read usage. Check that dllm serve is running at the address in the top bar.");
    }
    if (statsRes.status === "fulfilled") setStats(parseStats(statsRes.value.json));
    else setStats(null);
    // Only used to resolve friendly names and to know which peers exist; the
    // registry is public, so a failure here just costs us the names.
    if (devicesRes.status === "fulfilled" && devicesRes.value.status === 200) {
      setDevices(parseDevices(devicesRes.value.json));
    }
    setLoading(false);
  }, [base]);

  useEffect(() => {
    if (!active) return;
    const kick = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(kick);
  }, [active, generation, load, tick]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  const nameFor = useCallback(
    (id: string) => {
      const d = devices?.find((x) => x.device_id === id);
      const name = d?.device_name?.trim() || usage?.perDevice?.find((e) => e.device_id === id)?.device_name?.trim();
      return name || shortId(id);
    },
    [devices, usage],
  );

  // Same presentation reshape as the Distribution tab, so the two agree on who
  // holds what. See lib/presented.ts.
  const shown = useMemo(() => {
    if (!usage) return usage;
    const stages = presentStages(usage.stages, devices);
    return {
      ...usage,
      stages,
      perDevice: presentPerDevice(usage.perDevice, stages, usage.totals?.tokens_out_total ?? null, devices, nameFor),
      bandwidth: usage.bandwidth ?? { mbps: null, label: null },
    };
  }, [devices, nameFor, usage]);

  return useMemo(() => ({ usage: shown, stats, loading, error, refresh }), [error, loading, refresh, shown, stats]);
}