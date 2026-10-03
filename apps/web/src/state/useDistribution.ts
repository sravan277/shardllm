/** Distribution tab data: the live pipeline plan merged with real load.
 *
 * Sources, in order of authority:
 * - `GET /v1/plan`   -> stages, layer ranges, `latency_ms`, `bottleneck`
 * - `GET /v1/usage`  -> per-device tokens/load and `bandwidth`
 * - `GET /v1/devices`-> friendly names (the plan only carries ids)
 * - `GET /api/stats` -> mesh link counts and throughput, if ever reported
 *
 * `latency_ms` and `bottleneck` may be absent OR explicitly null (the
 * coordinator emits `null` until it measures a stage), so both are treated as
 * "not measured yet" and nothing is substituted for them.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { jgetStatus, jget } from "../api/http";
import { parseDevices, parsePlan, parseStats, parseUsage } from "../api/parse";
import type { Device, Plan, Stats, Usage } from "../api/types";
import { shortId } from "../lib/format";

export type Distribution = {
  plan: Plan | null;
  usage: Usage | null;
  stats: Stats | null;
  devices: Device[] | null;
  /** null until the first load resolves. */
  loading: boolean;
  error: string | null;
  refresh: () => void;
  /** id -> friendly name, merged from devices and usage. */
  nameFor: (id: string) => string;
};

export function useDistribution(base: string, active: boolean, generation: number): Distribution {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [devices, setDevices] = useState<Device[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    const [planRes, usageRes, statsRes, devicesRes] = await Promise.allSettled([
      jgetStatus(base, "/v1/plan"),
      jgetStatus(base, "/v1/usage"),
      jget<unknown>(base, "/api/stats").then((v) => ({ status: 200, json: v })),
      jgetStatus(base, "/v1/devices"),
    ]);

    if (planRes.status === "fulfilled" && planRes.value.status === 200) {
      setPlan(parsePlan(planRes.value.json));
      setError(null);
    } else {
      setPlan(null);
      const status = planRes.status === "fulfilled" ? planRes.value.status : 0;
      setError(
        status === 404
          ? "This coordinator has no plan endpoint yet — upgrade it to see the layer split."
          : "Could not read the plan. Check that dllm serve is running at the address in the top bar.",
      );
    }

    if (usageRes.status === "fulfilled") {
      // A 403 body still carries honest totals, so parse it either way.
      setUsage(parseUsage(usageRes.value.json, usageRes.value.status === 403));
    } else {
      setUsage(null);
    }
    if (statsRes.status === "fulfilled") setStats(parseStats(statsRes.value.json));
    else setStats(null);
    if (devicesRes.status === "fulfilled") setDevices(parseDevices(devicesRes.value.json));
    else setDevices(null);

    setLoading(false);
  }, [base]);

  useEffect(() => {
    if (!active) return;
    // Timer kick: the first paint shows the current data, then the reload.
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

  return useMemo(
    () => ({ plan, usage, stats, devices, loading, error, refresh, nameFor }),
    [devices, error, loading, nameFor, plan, refresh, stats, usage],
  );
}