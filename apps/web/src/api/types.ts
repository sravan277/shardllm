/** Wire types for the coordinator LAN API.
 *
 * Everything the server may omit, or send as an explicit `null`, is typed
 * that way here: `?` or `| null`, never a defaulted number. Parsers in
 * ./parse.ts turn unknown JSON into these shapes and drop anything they
 * cannot vouch for, so no screen has to guess.
 */

/** One entry of `GET /api/models` (`stages[].range` is display-only). */
export type Model = {
  id: string;
  quant?: string;
  params?: string;
  size_mb?: number;
  stages?: { range: string }[];
};

/** `GET /api/node`. */
export type NodeInfo = {
  node_id: string;
  fingerprint: string;
  quic_port: number;
  version: string;
};

/** `GET /api/stats`. Every field is optional: old coordinators send less. */
export type Stats = {
  uptime_s?: number;
  engine?: string;
  sessions?: number;
  events?: number;
  node_id?: string;
  /** Not measured by any coordinator yet; rendered only when actually sent. */
  tokens_per_second?: number | null;
  mesh?: { peers_connected?: number; allowed_peers?: number } | null;
};

/** Registry status. `unknown` covers statuses this build does not know. */
export type DeviceStatus = "paired" | "revoked" | "unknown";

/** One row of `GET /v1/devices`, merged with `/v1/usage` load when present. */
export type Device = {
  device_id: string;
  /** null = never reported a hostname; callers fall back to a short id. */
  device_name: string | null;
  role: string | null;
  status: DeviceStatus;
  /** Server-derived inside a 90 s window; never recomputed here. */
  active: boolean;
  last_seen: string | null;
  paired_at: string | null;
  permissions: string[];
  worker_active: boolean | null;
  cpu_pct: number | null;
  mem_pct: number | null;
  load_source: string | null;
};

/** One pipeline stage of `GET /v1/plan`. `/v1/usage` omits the index, in which
 *  case the array position is the stage number. */
export type PlanStage = {
  stage: number;
  device_id: string | null;
  layer_start: number | null;
  layer_end: number | null;
  /** Explicitly null until a stage timing is measured; never invented. */
  latency_ms: number | null;
};

/** `GET /v1/plan`. `stages: null` means "endpoint said nothing usable". */
export type Plan = {
  plan_id: string | null;
  stages: PlanStage[] | null;
  /** null = no bottleneck identified (or not measured yet). */
  bottleneck: number | null;
  note: string | null;
};

/** One entry of `GET /v1/usage` `per_device`. */
export type UsageDevice = {
  device_id: string;
  device_name: string | null;
  tokens_out: number;
  sessions: number | null;
  cpu_pct: number | null;
  mem_pct: number | null;
  load_source: string | null;
  worker_active: boolean | null;
  role: string | null;
  active: boolean | null;
  layer_start: number | null;
  layer_end: number | null;
};

export type UsageTotals = {
  tokens_out_total: number | null;
  sessions_total: number | null;
  is_admin: boolean | null;
  worker_active: boolean | null;
  cpu_pct: number | null;
  mem_pct: number | null;
  load_source: string | null;
};

/** `GET /v1/usage`. */
export type Usage = {
  totals: UsageTotals | null;
  /** null = breakdown withheld (admin only); [] = breakdown is genuinely empty. */
  perDevice: UsageDevice[] | null;
  stages: PlanStage[] | null;
  /** Always null server-side today; rendered as an explicit empty state. */
  bandwidth: Bandwidth | null;
  /** 403 / `is_admin:false`: totals are honest, the breakdown is not available. */
  forbidden: boolean;
};

export type UsageTotalsSource = {
  tokens_out_total: number | null;
  sessions_total: number | null;
  is_admin: boolean | null;
  worker_active: boolean | null;
  cpu_pct: number | null;
  mem_pct: number | null;
  load_source: string | null;
};

/** Link measurements. Both fields null while nothing measures the link. */
export type Bandwidth = { mbps: number | null; label: string | null };

/** `GET /api/pairing-uri`. */
export type PairingUri = {
  uri: string | null;
  candidates: string[];
  warning: string | null;
};

/** Local-only chat row: created by this browser, mirrored on the server. */
export type LocalChat = {
  id: string;
  title?: string;
  model?: string;
  created_at?: string;
  updated_at?: string;
  last_token_at?: string;
  tokens_out?: number;
};

/** Live/dead/reconnecting, split out so a dead EventSource is a visible state. */
export type StreamPhase = "idle" | "live" | "reconnecting";

export type Health = "unknown" | "live" | "down";