/** Usage tab data: totals plus the admin-only per-device breakdown.
 *
 * `GET /v1/usage` is asked for without a group — there is one network at a
 * time. A 403 or `is_admin:false` still yields honest totals, so the tab shows
 * those plus a note instead of pretending the breakdown does not exist.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { jgetStatus } from "../api/http";
import { parseStats, parseUsage } from "../api/parse";
import type { Stats, Usage } from "../api/types";

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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    const [usageRes, statsRes] = await Promise.allSettled([
      jgetStatus(base, "/v1/usage"),
      jgetStatus(base, "/api/stats"),
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
    setLoading(false);
  }, [base]);

  useEffect(() => {
    if (!active) return;
    const kick = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(kick);
  }, [active, generation, load, tick]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  return useMemo(() => ({ usage, stats, loading, error, refresh }), [error, loading, refresh, stats, usage]);
}