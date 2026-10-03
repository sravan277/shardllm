/** Defensive parsers for every coordinator payload the UI reads.
 *
 * Rules that hold across all of them:
 * - a missing key, an explicit `null`, and a wrong type all collapse to the
 *   honest "unknown" value (`null` / omitted) rather than a zero or a guess;
 * - several key spellings are accepted (`device_id` / `deviceId` / `id`)
 *   because coordinators in the field have shipped more than one shape;
 * - nothing here touches the network, so the whole file is pure.
 */

import type {
  Bandwidth,
  Device,
  DeviceStatus,
  NodeInfo,
  PairingUri,
  Plan,
  PlanStage,
  Stats,
  Usage,
  UsageDevice,
  UsageTotals,
} from "./types";

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.trim() ? v.trim() : null;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function nonNegInt(v: unknown): number | null {
  const n = num(v);
  return n === null || n < 0 ? null : Math.floor(n);
}

/** Percentages: finite, and never outside 0–100 (a bigger number is a unit bug). */
function pct(v: unknown): number | null {
  const n = num(v);
  return n === null || n < 0 || n > 100 ? null : n;
}

function bool(v: unknown): boolean | null {
  return typeof v === "boolean" ? v : null;
}

function strArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];
}

function firstStr(o: Record<string, unknown>, ...keys: string[]): string | null {
  for (const k of keys) {
    const s = str(o[k]);
    if (s) return s;
  }
  return null;
}

function firstBool(o: Record<string, unknown>, ...keys: string[]): boolean | null {
  for (const k of keys) {
    const b = bool(o[k]);
    if (b !== null) return b;
  }
  return null;
}

function parseStatus(v: unknown): DeviceStatus {
  const s = str(v)?.toLowerCase();
  if (s === "paired" || s === "revoked") return s;
  return "unknown";
}

/** `load_source` gates every load bar: only live/reported samples render. */
export function loadIsLive(source: string | null): boolean {
  return source === "live" || source === "reported";
}

// ---------------------------------------------------------------- devices

/** `GET /v1/devices` -> `{devices:[...]}`, bare array, or a single row. */
export function parseDevices(raw: unknown): Device[] | null {
  if (!isRecord(raw) && !Array.isArray(raw)) return null;
  let rows: unknown[];
  if (Array.isArray(raw)) rows = raw;
  else if (isRecord(raw) && Array.isArray(raw["devices"])) rows = raw["devices"];
  else if (isRecord(raw) && str(raw["device_id"])) rows = [raw];
  else return null;
  const out: Device[] = [];
  for (const r of rows) {
    const d = parseDevice(r);
    if (d) out.push(d);
  }
  return out;
}

/** One device row. Load fields are absent on `/v1/devices` today and parsed
 *  anyway, so the row lights up the moment the coordinator starts sending
 *  them. */
export function parseDevice(raw: unknown): Device | null {
  if (!isRecord(raw)) return null;
  const id = firstStr(raw, "device_id", "deviceId", "id");
  if (!id) return null;
  return {
    device_id: id,
    device_name: firstStr(raw, "device_name", "deviceName", "name"),
    role: firstStr(raw, "role"),
    status: parseStatus(raw["status"]),
    active: bool(raw["active"]) ?? false,
    last_seen: firstStr(raw, "last_seen", "lastSeen"),
    paired_at: firstStr(raw, "paired_at", "pairedAt"),
    permissions: strArray(raw["permissions"]),
    worker_active: firstBool(raw, "worker_active", "workerActive"),
    cpu_pct: pct(raw["cpu_pct"] ?? raw["cpuPct"]),
    mem_pct: pct(raw["mem_pct"] ?? raw["memPct"]),
    load_source: firstStr(raw, "load_source", "loadSource"),
  };
}

/** Overlay `/v1/usage` load + worker fields onto a device row. Registry-only
 *  fields (status, pairing, last_seen) always win — they are the authority. */
export function mergeDeviceLoad(devices: Device[], perDevice: UsageDevice[] | null): Device[] {
  if (!perDevice) return devices;
  const byId = new Map(perDevice.map((e) => [e.device_id, e]));
  return devices.map((d) => {
    const e = byId.get(d.device_id);
    if (!e) return d;
    return {
      ...d,
      device_name: d.device_name ?? e.device_name,
      role: d.role ?? e.role,
      cpu_pct: d.cpu_pct ?? e.cpu_pct,
      mem_pct: d.mem_pct ?? e.mem_pct,
      load_source: d.load_source ?? e.load_source,
      worker_active: d.worker_active ?? e.worker_active,
    };
  });
}

// ------------------------------------------------------------------- plan

/** `GET /v1/plan` (or any `{stages}` / `{plan:{stages}}` payload). */
export function parsePlan(raw: unknown): Plan {
  if (!isRecord(raw)) return { plan_id: null, stages: null, bottleneck: null, note: null };
  const stages = parseStages(raw["stages"] ?? (isRecord(raw["plan"]) ? (raw["plan"] as Record<string, unknown>)["stages"] : null));
  return {
    plan_id: firstStr(raw, "plan_id", "planId"),
    stages,
    bottleneck: nonNegInt(raw["bottleneck"]),
    note: firstStr(raw, "note"),
  };
}

/** A stage list is only usable when at least one row has finite bounds. */
function parseStages(v: unknown): PlanStage[] | null {
  if (!Array.isArray(v)) return null;
  const out: PlanStage[] = [];
  v.forEach((e, i) => {
    if (!isRecord(e)) return;
    out.push({
      stage: nonNegInt(e["stage"]) ?? i,
      device_id: firstStr(e, "device_id", "deviceId"),
      layer_start: nonNegInt(e["layer_start"] ?? e["layerStart"]),
      layer_end: nonNegInt(e["layer_end"] ?? e["layerEnd"]),
      latency_ms: num(e["latency_ms"] ?? e["latencyMs"]),
    });
  });
  return out.length > 0 ? out : null;
}

// ------------------------------------------------------------------ usage

/** `GET /v1/usage`. `forbidden` is set from a 403 body or `is_admin:false`. */
export function parseUsage(raw: unknown, forbidden = false): Usage {
  if (!isRecord(raw)) {
    return { totals: null, perDevice: null, stages: null, bandwidth: null, forbidden };
  }
  const totals = parseTotals(raw);
  const perDevice = parsePerDevice(raw);
  const admin = totals?.is_admin;
  const withhold = forbidden || admin === false;
  return {
    totals,
    perDevice: withhold ? null : perDevice,
    stages: parseStages(raw["plan"] !== undefined ? (isRecord(raw["plan"]) ? (raw["plan"] as Record<string, unknown>)["stages"] : null) : raw["stages"]),
    bandwidth: parseBandwidth(raw["bandwidth"]),
    forbidden: withhold,
  };
}

function parseTotals(raw: Record<string, unknown>): UsageTotals | null {
  const hasAny =
    raw["tokens_out_total"] !== undefined ||
    raw["sessions_total"] !== undefined ||
    raw["is_admin"] !== undefined ||
    raw["worker_active"] !== undefined;
  if (!hasAny) return null;
  return {
    tokens_out_total: nonNegInt(raw["tokens_out_total"] ?? raw["tokensOutTotal"]),
    sessions_total: nonNegInt(raw["sessions_total"] ?? raw["sessionsTotal"]),
    is_admin: bool(raw["is_admin"] ?? raw["isAdmin"]),
    worker_active: firstBool(raw, "worker_active", "workerActive"),
    cpu_pct: pct(raw["cpu_pct"] ?? raw["cpuPct"]),
    mem_pct: pct(raw["mem_pct"] ?? raw["memPct"]),
    load_source: firstStr(raw, "load_source", "loadSource"),
  };
}

/** `per_device` entries. Returns [] for a present-but-empty list, null when
 *  the key is missing entirely (the coordinator withheld the breakdown). */
function parsePerDevice(raw: Record<string, unknown>): UsageDevice[] | null {
  const list = raw["per_device"] ?? raw["perDevice"] ?? raw["devices"];
  if (!Array.isArray(list)) return null;
  const out: UsageDevice[] = [];
  for (const e of list) {
    if (!isRecord(e)) continue;
    const id = firstStr(e, "device_id", "deviceId", "id");
    if (!id) continue;
    out.push({
      device_id: id,
      device_name: firstStr(e, "device_name", "deviceName", "name"),
      tokens_out: nonNegInt(e["tokens_out"] ?? e["tokensOut"]) ?? 0,
      sessions: nonNegInt(e["sessions"] ?? e["session_count"]),
      cpu_pct: pct(e["cpu_pct"] ?? e["cpuPct"]),
      mem_pct: pct(e["mem_pct"] ?? e["memPct"]),
      load_source: firstStr(e, "load_source", "loadSource"),
      worker_active: firstBool(e, "worker_active", "workerActive"),
      role: firstStr(e, "role"),
      active: bool(e["active"]),
      layer_start: nonNegInt(e["layer_start"]),
      layer_end: nonNegInt(e["layer_end"]),
    });
  }
  return out;
}

/** `bandwidth` is a hard null server-side today; shape kept for the day it
 *  stops being null. Anything unrecognised stays null (never rounded up). */
function parseBandwidth(v: unknown): Bandwidth | null {
  if (!isRecord(v)) return null;
  return { mbps: num(v["mbps"] ?? v["mbit_s"] ?? v["mbps_total"]), label: firstStr(v, "label") };
}

// ------------------------------------------------------- node/stats/pairing

export function parseNode(raw: unknown): NodeInfo | null {
  if (!isRecord(raw)) return null;
  const id = firstStr(raw, "node_id", "nodeId", "id");
  if (!id) return null;
  return {
    node_id: id,
    fingerprint: firstStr(raw, "fingerprint") ?? "",
    quic_port: num(raw["quic_port"] ?? raw["quicPort"]) ?? 0,
    version: firstStr(raw, "version") ?? "",
  };
}

export function parseStats(raw: unknown): Stats | null {
  if (!isRecord(raw)) return null;
  const meshRaw = raw["mesh"];
  const mesh = isRecord(meshRaw)
    ? { peers_connected: nonNegInt(meshRaw["peers_connected"]) ?? undefined, allowed_peers: nonNegInt(meshRaw["allowed_peers"]) ?? undefined }
    : null;
  return {
    uptime_s: num(raw["uptime_s"]) ?? undefined,
    engine: firstStr(raw, "engine") ?? undefined,
    sessions: nonNegInt(raw["sessions"]) ?? undefined,
    events: nonNegInt(raw["events"]) ?? undefined,
    node_id: firstStr(raw, "node_id", "nodeId") ?? undefined,
    tokens_per_second: num(raw["tokens_per_second"] ?? raw["tok_s"] ?? raw["tps"]),
    mesh,
  };
}

export function parsePairingUri(raw: unknown): PairingUri | null {
  if (!isRecord(raw)) return null;
  return {
    uri: firstStr(raw, "uri", "pairing_uri"),
    candidates: strArray(raw["candidates"]),
    warning: firstStr(raw, "warning"),
  };
}

// ------------------------------------------------------------- SSE tokens

export type TokenParse = { action: "append"; text: string } | { action: "ignore" } | { action: "commit" };

/**
 * Contract token payload is `{"pos":N,"text":"..."}` (extra token/margin keys
 * ignored). Legacy servers wrap everything as `event:token` with
 * `{kind,payload}`: ignore session_created/user_message, append payload.text
 * chunks, and treat commit-ish kinds (or a token with `done:true` and empty
 * text) as completion — so chat works against old AND new coordinators.
 * Legacy payloads are often a JSON STRING, so nested parsing is required;
 * without it the raw JSON leaks into bubbles and typing never clears.
 */
export function parseTokenData(raw: string): TokenParse {
  let d: unknown;
  try {
    d = JSON.parse(raw);
  } catch {
    return { action: "append", text: raw };
  }
  if (typeof d !== "object" || d === null) {
    return typeof d === "string" && d ? { action: "append", text: d } : { action: "ignore" };
  }
  const o = d as Record<string, unknown>;
  const doneTop = o["done"] === true;
  if (typeof o["kind"] === "string") {
    const kind = o["kind"].toLowerCase();
    if (kind === "session_created" || kind === "user_message") return { action: "ignore" };
    if (kind === "commit" || kind === "done" || kind === "complete" || kind === "completed") return { action: "commit" };
    const p = o["payload"] ?? o["data"];
    const fromObj = (obj: Record<string, unknown>): TokenParse | null => {
      const t = obj["text"];
      const done = obj["done"] === true || doneTop;
      if (typeof t === "string" && t) return { action: "append", text: t };
      if (done) return { action: "commit" };
      return null;
    };
    if (typeof p === "string") {
      if (!p) return { action: "ignore" };
      try {
        const inner = JSON.parse(p) as unknown;
        if (typeof inner === "object" && inner !== null) {
          return fromObj(inner as Record<string, unknown>) ?? { action: "ignore" };
        }
        return isTextKind(kind) ? { action: "append", text: p } : { action: "ignore" };
      } catch {
        return isTextKind(kind) ? { action: "append", text: p } : { action: "ignore" };
      }
    }
    if (typeof p === "object" && p !== null) {
      return fromObj(p as Record<string, unknown>) ?? { action: "ignore" };
    }
    if (typeof o["text"] === "string" && o["text"]) return { action: "append", text: o["text"] as string };
    if (doneTop) return { action: "commit" };
    return { action: "ignore" };
  }
  if (typeof o["text"] === "string") {
    if (o["text"]) return { action: "append", text: o["text"] as string };
    if (doneTop) return { action: "commit" };
    return { action: "ignore" };
  }
  if (doneTop) return { action: "commit" };
  return { action: "ignore" };
}

function isTextKind(kind: string): boolean {
  return kind === "token" || kind === "chunk" || kind === "text";
}