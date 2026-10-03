/** Devices tab data: the registry, its per-row actions, and the pairing code.
 *
 * `GET /v1/devices` is polled every 15 s while the tab is open. Quiet polls
 * never touch the loading flag, so the list does not flash a spinner; only the
 * first load and an explicit refresh do. `/v1/usage` rides along because that
 * is where CPU/mem actually live — the registry rows carry no load at all.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, jget, jwrite } from "../api/http";
import { mergeDeviceLoad, parseDevices, parsePairingUri, parseUsage } from "../api/parse";
import type { Device, PairingUri } from "../api/types";

/** Quiet-poll interval for the registry. */
export const DEVICE_POLL_MS = 15000;

export type DeviceAction = "approve" | "revoke" | "delete";

export type Devices = {
  /** null = the endpoint failed; [] = the registry is genuinely empty. */
  devices: Device[] | null;
  pairing: PairingUri | null;
  error: string | null;
  loading: boolean;
  refresh: () => void;
  busyId: string | null;
  busyAction: DeviceAction | null;
  /** Last action result / failure, as cause + fix. */
  actionNote: string | null;
  act: (device: Device, action: DeviceAction) => Promise<void>;
  /** Device id of this coordinator (GET /api/node), for the "this device" row. */
  selfId: string | null;
};

export function useDevices(base: string, active: boolean, generation: number): Devices {
  const [devices, setDevices] = useState<Device[] | null>(null);
  const [pairing, setPairing] = useState<PairingUri | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<DeviceAction | null>(null);
  const [actionNote, setActionNote] = useState<string | null>(null);
  const [selfId, setSelfId] = useState<string | null>(null);
  const loadedRef = useRef(false);

  const load = useCallback(
    async (quiet: boolean) => {
      if (!quiet) setLoading(true);
      try {
        const [registry, pair, node] = await Promise.allSettled([
          jget<unknown>(base, "/v1/devices"),
          jget<unknown>(base, "/api/pairing-uri"),
          jget<unknown>(base, "/api/node"),
        ]);
        let list = registry.status === "fulfilled" ? parseDevices(registry.value) : null;
        if (list !== null) {
          // CPU/mem/worker flags live on /v1/usage, so pull them alongside.
          try {
            const usage = parseUsage(await jget<unknown>(base, "/v1/usage"));
            list = mergeDeviceLoad(list, usage.perDevice);
          } catch {
            /* usage is optional here: rows just show "not reporting" */
          }
          setDevices(list);
          setError(null);
        } else {
          setDevices(null);
          setError(fixable(registry.status === "rejected" ? registry.reason : new Error("no registry"), "GET /v1/devices"));
        }
        setPairing(pair.status === "fulfilled" ? parsePairingUri(pair.value) : null);
        if (node.status === "fulfilled") {
          const id = (node.value as { node_id?: unknown } | null)?.node_id;
          setSelfId(typeof id === "string" ? id : null);
        }
      } finally {
        if (!quiet) setLoading(false);
        loadedRef.current = true;
      }
    },
    [base],
  );

  // First open (and every reconnect) loads loudly; the poll stays quiet.
  useEffect(() => {
    if (!active) return;
    void load(loadedRef.current);
  }, [active, generation, load]);

  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => void load(true), DEVICE_POLL_MS);
    return () => window.clearInterval(timer);
  }, [active, load]);

  const refresh = useCallback(() => {
    void load(false);
  }, [load]);

  const act = useCallback(
    async (device: Device, action: DeviceAction) => {
      const id = device.device_id;
      const label = device.device_name?.trim() || "this device";
      setBusyId(id);
      setBusyAction(action);
      setActionNote(null);
      const path = `/v1/devices/${encodeURIComponent(id)}${action === "delete" ? "" : `/${action}`}`;
      try {
        await jwrite(base, path, action === "delete" ? "DELETE" : "POST");
        const done = action === "approve" ? "Approved" : action === "revoke" ? "Revoked" : "Deleted";
        setActionNote(`${done} ${label}.`);
        await load(true);
      } catch (err) {
        // The coordinator's own refusal (e.g. 400 "cannot delete self
        // coordinator") is shown verbatim plus the way out.
        const cause = err instanceof ApiError ? err.message : String(err);
        const fix =
          action === "delete" && device.device_id === selfId
            ? "Use Revoke instead — deleting the coordinator would orphan the mesh."
            : "Refresh to see the current state, then try again.";
        setActionNote(`${cause}. ${fix}`);
      } finally {
        setBusyId(null);
        setBusyAction(null);
      }
    },
    [base, load, selfId],
  );

  return useMemo(
    () => ({ devices, pairing, error, loading, refresh, busyId, busyAction, actionNote, act, selfId }),
    [act, actionNote, busyAction, busyId, devices, error, loading, pairing, refresh, selfId],
  );
}

/** Turn a fetch failure into a cause + fix sentence. */
function fixable(err: unknown, what: string): string {
  if (err instanceof ApiError) {
    if (err.status === 404) return `${what} is not on this coordinator — upgrade it to see paired devices.`;
    return `${what} answered ${err.status}. Check that dllm serve is running at the address in the top bar.`;
  }
  return `Could not reach ${what}. Check that dllm serve is running at the address in the top bar.`;
}