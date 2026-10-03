import { useCallback, useEffect, useRef, useState } from "react";
import * as QRCode from "qrcode";
import { appendToken, displayTitle, finalizeStream, type StreamMsg } from "./chatStream";

type Msg = StreamMsg;
type Model = { id: string; quant?: string; params?: string; size_mb?: number; stages?: { range: string }[] };
type NodeInfo = { node_id: string; fingerprint: string; quic_port: number; version: string };
type Stats = { uptime_s?: number; engine?: string; sessions?: number; events?: number; node_id?: string };
/** Local-only chat row: created by this browser, mirrored on the server. */
type LocalChat = {
  id: string;
  title?: string;
  model?: string;
  created_at?: string;
  updated_at?: string;
  last_token_at?: string;
  tokens_out?: number;
};
type PlanStage = { stage?: number; device_id?: string; layer_start?: number; layer_end?: number };
type UsageDeviceEntry = {
  device_id: string;
  device_name?: string;
  tokens_out: number;
  sessions?: number;
  cpu_pct?: number | null;
  mem_pct?: number | null;
  load_source?: string | null;
  worker_active?: boolean | null;
  role?: string | null;
  active?: boolean | null;
};
type UsageTotals = {
  tokens_out_total: number | null;
  sessions_total: number | null;
  is_admin: boolean | null;
  worker_active: boolean | null;
  cpu_pct: number | null;
  mem_pct: number | null;
  load_source: string | null;
};
type NetworkSummary = {
  id: string;
  name?: string;
  open_join?: boolean;
  has_password?: boolean;
  member_count?: number;
  active_count?: number;
  is_admin?: boolean;
};
type NetworkMember = {
  device_id?: string;
  device_name?: string;
  name?: string;
  active?: boolean;
  role?: string;
  last_seen?: string;
};

async function jget<T>(base: string, path: string): Promise<T> {
  const r = await fetch(`${base}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return (await r.json()) as T;
}

function shortId(id: string): string {
  return id.length > 20 ? `${id.slice(0, 9)}…${id.slice(-4)}` : id;
}

function memberLabel(m: NetworkMember): string {
  const name = (m.device_name ?? m.name ?? "").trim();
  if (name) return name;
  if (m.device_id) return shortId(m.device_id);
  return "unknown device";
}

/** localStorage keys: per coordinator origin, local to this browser. */
function chatsKey(base: string): string {
  return `dllm:chats:${base}`;
}
function threadsKey(base: string): string {
  return `dllm:threads:${base}`;
}

function loadLocalChats(base: string): LocalChat[] {
  try {
    const raw = window.localStorage.getItem(chatsKey(base));
    if (!raw) return [];
    const arr = JSON.parse(raw) as unknown;
    if (!Array.isArray(arr)) return [];
    return (arr as Record<string, unknown>[])
      .filter((e) => typeof e === "object" && e !== null && typeof (e as { id?: unknown }).id === "string" && (e as { id: string }).id)
      .map((e) => {
        const o = e as Record<string, unknown>;
        const pick = (k: string): string | undefined => (typeof o[k] === "string" && (o[k] as string) ? (o[k] as string) : undefined);
        const num = (k: string): number | undefined =>
          typeof o[k] === "number" && Number.isFinite(o[k] as number) && (o[k] as number) >= 0 ? Math.floor(o[k] as number) : undefined;
        return {
          id: o["id"] as string,
          title: pick("title"),
          model: pick("model"),
          created_at: pick("created_at"),
          updated_at: pick("updated_at"),
          last_token_at: pick("last_token_at"),
          tokens_out: num("tokens_out"),
        } satisfies LocalChat;
      });
  } catch {
    return [];
  }
}

function loadLocalThreads(base: string): Record<string, Msg[]> {
  try {
    const raw = window.localStorage.getItem(threadsKey(base));
    if (!raw) return {};
    const o = JSON.parse(raw) as unknown;
    if (typeof o !== "object" || o === null) return {};
    const out: Record<string, Msg[]> = {};
    for (const [k, v] of Object.entries(o as Record<string, unknown>)) {
      if (!Array.isArray(v)) continue;
      const msgs: Msg[] = [];
      for (const m of v as unknown[]) {
        if (typeof m !== "object" || m === null) continue;
        const r = m as Record<string, unknown>;
        if ((r["role"] === "you" || r["role"] === "mesh") && typeof r["text"] === "string") {
          msgs.push({ role: r["role"] as "you" | "mesh", text: r["text"] as string });
        }
      }
      out[k] = msgs;
    }
    return out;
  } catch {
    return {};
  }
}

/** Newest-first by last activity; untimestamped chats sink. */
function sortChatsNewestFirst(list: LocalChat[]): LocalChat[] {
  const key = (s: LocalChat): string => s.last_token_at ?? s.updated_at ?? s.created_at ?? "";
  return [...list].sort((a, b) => (key(b) < key(a) ? -1 : key(b) > key(a) ? 1 : 0));
}

/** Normalize GET /v1/networks: bare array or {networks:[...]}. */
function normalizeNetworks(raw: unknown): NetworkSummary[] {
  const arr = Array.isArray(raw) ? raw : (raw as { networks?: unknown })?.networks;
  if (!Array.isArray(arr)) return [];
  const out: NetworkSummary[] = [];
  for (const e of arr as Record<string, unknown>[]) {
    if (typeof e !== "object" || e === null) continue;
    const id = e["id"];
    if (typeof id !== "string" || !id) continue;
    const opt = (v: unknown): boolean | undefined => (typeof v === "boolean" ? v : undefined);
    const cnt = (v: unknown): number | undefined =>
      typeof v === "number" && Number.isFinite(v) && v >= 0 ? Math.floor(v) : undefined;
    out.push({
      id,
      name: typeof e["name"] === "string" && e["name"] ? e["name"] : undefined,
      open_join: opt(e["open_join"]),
      has_password: opt(e["has_password"]),
      member_count: cnt(e["member_count"]),
      active_count: cnt(e["active_count"]),
      is_admin: opt(e["is_admin"]),
    });
  }
  return out;
}

/** Parse GET /v1/networks/{id}/devices: {paired,active} or {devices} or bare array. */
function parseNetworkMembers(raw: unknown): { paired: NetworkMember[]; active: NetworkMember[] } {
  const asMember = (e: unknown): NetworkMember | null => {
    if (typeof e !== "object" || e === null) return null;
    const o = e as Record<string, unknown>;
    const str = (v: unknown): string | undefined => (typeof v === "string" && v ? v : undefined);
    return {
      device_id: str(o["device_id"] ?? o["id"]),
      device_name: str(o["device_name"] ?? o["name"]),
      name: str(o["name"]),
      active: typeof o["active"] === "boolean" ? o["active"] : undefined,
      role: str(o["role"]),
      last_seen: str(o["last_seen"]),
    };
  };
  if (Array.isArray(raw)) {
    const all = raw.map(asMember).filter((m): m is NetworkMember => m !== null);
    return { paired: all.filter((m) => m.active !== true), active: all.filter((m) => m.active === true) };
  }
  if (typeof raw === "object" && raw !== null) {
    const o = raw as Record<string, unknown>;
    if (Array.isArray(o["paired"]) || Array.isArray(o["active"])) {
      const paired = (Array.isArray(o["paired"]) ? o["paired"] : []).map(asMember).filter((m): m is NetworkMember => m !== null);
      const active = (Array.isArray(o["active"]) ? o["active"] : []).map(asMember).filter((m): m is NetworkMember => m !== null);
      return { paired, active };
    }
    const list = o["devices"] ?? o["members"];
    if (Array.isArray(list)) {
      const all = (list as unknown[]).map(asMember).filter((m): m is NetworkMember => m !== null);
      return { paired: all.filter((m) => m.active !== true), active: all.filter((m) => m.active === true) };
    }
  }
  return { paired: [], active: [] };
}

/** QR payload for a network: dllm://net?id=…&name=…&host=…&port=… */
function buildNetworkQrPayload(baseUrl: string, net: NetworkSummary): string | null {
  try {
    const u = new URL(baseUrl);
    const q = new URLSearchParams({
      id: net.id,
      name: net.name ?? net.id,
      host: u.hostname || "127.0.0.1",
      port: u.port || "8080",
    });
    return `dllm://net?${q.toString()}`;
  } catch {
    return `dllm://net?id=${encodeURIComponent(net.id)}`;
  }
}

/** Parse a pasted dllm://net?... code. Null = not a network code. */
function parseNetworkQr(text: string): { id: string; host?: string; port?: string; name?: string } | null {
  const t = text.trim();
  if (!t.toLowerCase().startsWith("dllm://net")) return null;
  try {
    // URL parser needs a // host; rewrite dllm://net?... as https://x/... query.
    const qIndex = t.indexOf("?");
    const query = qIndex >= 0 ? t.slice(qIndex + 1) : "";
    const q = new URLSearchParams(query);
    const id = (q.get("id") ?? "").trim();
    if (!id) return null;
    return {
      id,
      host: q.get("host") ?? undefined,
      port: q.get("port") ?? undefined,
      name: q.get("name") ?? undefined,
    };
  } catch {
    return null;
  }
}

/** Fixed palette for Usage donut + layer strip (no chart lib). */
const USAGE_PALETTE = ["#2dd4bf", "#f5b544", "#60a5fa", "#f472b6", "#a78bfa", "#34d399", "#fb7185", "#facc15"];
const UNATTRIBUTED_KEY = "__unattributed__";
const UNATTRIBUTED_COLOR = "#5b6b7c";

function usageColorFor(key: string, order: string[]): string {
  if (key === UNATTRIBUTED_KEY) return UNATTRIBUTED_COLOR;
  const i = order.indexOf(key);
  return USAGE_PALETTE[(i < 0 ? 0 : i) % USAGE_PALETTE.length];
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
}
function intOrNull(v: unknown): number | null {
  const n = numOrNull(v);
  return n === null ? null : Math.floor(n);
}

/** Opportunistic GET /v1/usage parsing: accepts several shapes, null = unusable. */
function parseUsagePerDevice(raw: unknown): UsageDeviceEntry[] | null {
  if (typeof raw !== "object" || raw === null) return null;
  const o = raw as Record<string, unknown>;
  const cand = o["per_device"] ?? o["perDevice"] ?? o["devices"] ?? o["usage"] ?? null;
  if (!Array.isArray(cand)) return null;
  const out: UsageDeviceEntry[] = [];
  for (const e of cand as Record<string, unknown>[]) {
    if (typeof e !== "object" || e === null) continue;
    const id = (e["device_id"] ?? e["deviceId"] ?? e["id"]) as unknown;
    if (typeof id !== "string" || !id) continue;
    const tok = (e["tokens_out"] ?? e["tokensOut"] ?? e["tokens"]) as unknown;
    const nm = (e["device_name"] ?? e["deviceName"] ?? e["name"]) as unknown;
    const sess = (e["sessions"] ?? e["session_count"] ?? e["count"]) as unknown;
    const cpu = e["cpu_pct"] ?? e["cpuPct"] ?? e["cpu"];
    const mem = e["mem_pct"] ?? e["memPct"] ?? e["mem"];
    const ls = e["load_source"] ?? e["loadSource"] ?? e["source"];
    const wa = e["worker_active"] ?? e["workerActive"];
    const role = e["role"];
    const active = e["active"];
    out.push({
      device_id: id,
      device_name: typeof nm === "string" && nm ? nm : undefined,
      tokens_out: typeof tok === "number" && Number.isFinite(tok) && tok >= 0 ? Math.floor(tok) : 0,
      sessions: typeof sess === "number" && Number.isFinite(sess) && sess >= 0 ? Math.floor(sess) : undefined,
      cpu_pct: numOrNull(cpu),
      mem_pct: numOrNull(mem),
      load_source: typeof ls === "string" ? ls : null,
      worker_active: typeof wa === "boolean" ? wa : null,
      role: typeof role === "string" ? role : null,
      active: typeof active === "boolean" ? active : null,
    });
  }
  return out;
}

function parseUsageTotals(raw: unknown): UsageTotals | null {
  if (typeof raw !== "object" || raw === null) return null;
  const o = raw as Record<string, unknown>;
  const hasAny =
    o["tokens_out_total"] !== undefined ||
    o["tokensOutTotal"] !== undefined ||
    o["total_tokens"] !== undefined ||
    o["sessions_total"] !== undefined ||
    o["sessionsTotal"] !== undefined ||
    o["total_sessions"] !== undefined ||
    o["is_admin"] !== undefined ||
    o["worker_active"] !== undefined;
  if (!hasAny) return null;
  const isAdmin = o["is_admin"] ?? o["isAdmin"];
  const workerActive = o["worker_active"] ?? o["workerActive"];
  const cpu = o["cpu_pct"] ?? o["cpuPct"];
  const mem = o["mem_pct"] ?? o["memPct"];
  const ls = o["load_source"] ?? o["loadSource"];
  return {
    tokens_out_total: intOrNull(o["tokens_out_total"] ?? o["tokensOutTotal"] ?? o["total_tokens"]),
    sessions_total: intOrNull(o["sessions_total"] ?? o["sessionsTotal"] ?? o["total_sessions"]),
    is_admin: typeof isAdmin === "boolean" ? isAdmin : null,
    worker_active: typeof workerActive === "boolean" ? workerActive : null,
    cpu_pct: numOrNull(cpu),
    mem_pct: numOrNull(mem),
    load_source: typeof ls === "string" ? ls : null,
  };
}

/** Opportunistic plan parsing from /v1/usage (.plan / .stages) or /v1/plan. */
function parseUsagePlan(raw: unknown): PlanStage[] | null {
  if (typeof raw !== "object" || raw === null) return null;
  const o = raw as Record<string, unknown>;
  const direct = o["stages"];
  if (Array.isArray(direct)) return direct as PlanStage[];
  const plan = o["plan"];
  if (typeof plan === "object" && plan !== null && Array.isArray((plan as Record<string, unknown>)["stages"])) {
    return (plan as Record<string, unknown>)["stages"] as PlanStage[];
  }
  return null;
}

/** Layer owner lookup: 28 entries (layers 0–27), null = unassigned/unknown. */
function planLayerOwners(stages: PlanStage[] | null): (string | null)[] {
  const owners: (string | null)[] = Array<string | null>(28).fill(null);
  if (!stages) return owners;
  for (const st of stages) {
    const id = typeof st.device_id === "string" && st.device_id ? st.device_id : null;
    if (id === null) continue;
    const lo = typeof st.layer_start === "number" ? st.layer_start : NaN;
    const hi = typeof st.layer_end === "number" ? st.layer_end : NaN;
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) continue;
    for (let l = Math.max(0, Math.floor(lo)); l <= Math.min(27, Math.floor(hi)); l++) owners[l] = id;
  }
  return owners;
}

/** Compress a sorted layer list into "0–27" style ranges + count. */
function layerRanges(layers: number[]): { text: string; count: number } {
  if (layers.length === 0) return { text: "none", count: 0 };
  const sorted = [...layers].sort((a, b) => a - b);
  const ranges: string[] = [];
  let s = sorted[0];
  let p = sorted[0];
  for (let i = 1; i <= sorted.length; i++) {
    const c = sorted[i];
    if (c === p + 1) {
      p = c;
      continue;
    }
    ranges.push(s === p ? `${s}` : `${s}–${p}`);
    s = c;
    p = c;
  }
  return { text: ranges.join(", "), count: sorted.length };
}

type TokenParse = { action: "append"; text: string } | { action: "ignore" } | { action: "commit" };

/**
 * Contract token payload is {"pos":N,"text":"..."} (extra token/margin keys
 * ignored). Legacy servers wrap everything as event:token with
 * {kind,payload}: ignore session_created/user_message, append payload.text
 * chunks, treat commit-ish kinds (or token with done:true and empty text) as
 * completion — so chat works old AND new. Legacy payloads are often a JSON
 * STRING, so nested parsing is required; without it the raw JSON would leak
 * into bubbles and typing would never clear (no commit event exists yet).
 */
function parseTokenData(raw: string): TokenParse {
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
    const kind = (o["kind"] as string).toLowerCase();
    if (kind === "session_created" || kind === "user_message") return { action: "ignore" };
    if (kind === "commit" || kind === "done" || kind === "complete" || kind === "completed") {
      return { action: "commit" };
    }
    const p = (o["payload"] ?? o["data"]) as unknown;
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
        // Plain-string token payload (non-JSON): only append for token kinds.
        return kind === "token" || kind === "chunk" || kind === "text"
          ? { action: "append", text: p }
          : { action: "ignore" };
      } catch {
        return kind === "token" || kind === "chunk" || kind === "text"
          ? { action: "append", text: p }
          : { action: "ignore" };
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

export default function App() {
  // Default coordinator = the host that served this page (works on phones,
  // other laptops, any LAN device). Only during `npm run dev` (:5173, page
  // NOT served by dllm) do we fall back to loopback.
  const [base, setBase] = useState(() => {
    try {
      if (window.location.protocol.startsWith("http") && window.location.port !== "5173") {
        return window.location.origin;
      }
    } catch {
      /* fall through to loopback */
    }
    return "http://127.0.0.1:8080";
  });
  const [health, setHealth] = useState<"unknown" | "live" | "down">("unknown");
  const [tab, setTab] = useState<"chat" | "models" | "networks" | "usage">(() => {
    try {
      const t = new URLSearchParams(window.location.search).get("tab");
      return t === "models" || t === "networks" || t === "usage" ? t : "chat";
    } catch {
      return "chat";
    }
  });
  // Local-only chats: created by this browser, keyed per coordinator origin.
  // Server SSE resume still works per session, but the server list is never
  // merged in — other devices never appear here.
  const [localSessions, setLocalSessions] = useState<LocalChat[]>(() => {
    try {
      if (window.location.protocol.startsWith("http") && window.location.port !== "5173") {
        return sortChatsNewestFirst(loadLocalChats(window.location.origin));
      }
    } catch {
      /* fall through */
    }
    return sortChatsNewestFirst(loadLocalChats("http://127.0.0.1:8080"));
  });
  // Per-session threads so switching conversations keeps history.
  const [msgsBySession, setMsgsBySession] = useState<Record<string, Msg[]>>(() => {
    try {
      if (window.location.protocol.startsWith("http") && window.location.port !== "5173") {
        return loadLocalThreads(window.location.origin);
      }
    } catch {
      /* fall through */
    }
    return loadLocalThreads("http://127.0.0.1:8080");
  });
  const [chatDraft, setChatDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [session, setSession] = useState<string | null>(null);
  const [lastEvent, setLastEvent] = useState<string>("");
  const [models, setModels] = useState<Model[]>([]);
  const [notice, setNotice] = useState("");
  const esRef = useRef<EventSource | null>(null);
  const busyRef = useRef(false);
  const stopControllerRef = useRef<AbortController | null>(null);
  // In-flight assistant response key (session-scoped). Tokens accumulate into
  // ONE bubble per key via appendToken; bare `commit` events never clear it.
  const streamKeyRef = useRef<string | null>(null);
  const idleTimerRef = useRef<number | null>(null);
  const streamSeq = useRef(0);
  // Resume cursors + seen event ids per session (ids are global rowids, so a
  // cursor from one session must never skip another session's replay).
  const cursorBySession = useRef<Record<string, string>>({});
  const seenIdBySession = useRef<Record<string, number>>({});
  const threadRef = useRef<HTMLDivElement | null>(null);
  // Right chat drawer + row menus.
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [usagePopId, setUsagePopId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  // Model-split layer panel inside the chat tab.
  const [layerOpen, setLayerOpen] = useState(false);
  const sessionRef = useRef<string | null>(null);
  const lastEventRef = useRef<string>("");
  const [node, setNode] = useState<NodeInfo | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);

  // Networks tab state.
  const [networks, setNetworks] = useState<NetworkSummary[] | null>(null);
  const [networksLoading, setNetworksLoading] = useState(false);
  const [selectedNetworkId, setSelectedNetworkId] = useState<string | null>(null);
  const [netMembers, setNetMembers] = useState<Record<string, { paired: NetworkMember[]; active: NetworkMember[] }>>({});
  const [netDetailLoading, setNetDetailLoading] = useState<string | null>(null);
  const [netQr, setNetQr] = useState<Record<string, string | null>>({});
  const [createName, setCreateName] = useState("");
  const [createPassword, setCreatePassword] = useState("");
  const [createOpenJoin, setCreateOpenJoin] = useState(true);
  const [createBusy, setCreateBusy] = useState(false);
  const [joinPasswords, setJoinPasswords] = useState<Record<string, string>>({});
  const [joinBusyId, setJoinBusyId] = useState<string | null>(null);
  const [joinCode, setJoinCode] = useState("");

  // Usage tab: GET /v1/usage?group=selectedGroup. Admin-only breakdown;
  // non-admin / 403 falls back to totals-only with an honest note.
  const [usageTotals, setUsageTotals] = useState<UsageTotals | null>(null);
  const [usagePerDevice, setUsagePerDevice] = useState<UsageDeviceEntry[] | null>(null);
  const [planStages, setPlanStages] = useState<PlanStage[] | null>(null);
  const [usageSource, setUsageSource] = useState<"usage" | "plan-only" | "unknown">("unknown");
  const [usageLoading, setUsageLoading] = useState(false);
  const [usageForbidden, setUsageForbidden] = useState(false);

  const setBusyBoth = (v: boolean) => {
    busyRef.current = v;
    setBusy(v);
  };

  const setLastEventBoth = (v: string) => {
    lastEventRef.current = v;
    setLastEvent(v);
  };

  // Persist local chats + threads per coordinator origin.
  useEffect(() => {
    try {
      window.localStorage.setItem(chatsKey(base), JSON.stringify(localSessions));
    } catch {
      /* storage full or blocked — chats stay in memory */
    }
  }, [base, localSessions]);
  useEffect(() => {
    try {
      const pruned: Record<string, Msg[]> = {};
      for (const s of localSessions) {
        if (msgsBySession[s.id]) pruned[s.id] = msgsBySession[s.id].map((m) => ({ role: m.role, text: m.text }));
      }
      window.localStorage.setItem(threadsKey(base), JSON.stringify(pruned));
    } catch {
      /* storage full or blocked — threads stay in memory */
    }
  }, [base, localSessions, msgsBySession]);
  // Switching coordinator loads that origin's local chats (never the server's).
  useEffect(() => {
    setLocalSessions(sortChatsNewestFirst(loadLocalChats(base)));
    setMsgsBySession(loadLocalThreads(base));
    setSession(null);
    sessionRef.current = null;
    esRef.current?.close();
    setBusyBoth(false);
    cursorBySession.current = {};
    seenIdBySession.current = {};
    streamKeyRef.current = null;
    setLastEventBoth("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [base]);

  const ping = useCallback(async () => {
    try {
      await jget(base, "/api/health");
      setHealth("live");
    } catch {
      setHealth("down");
    }
  }, [base]);

  useEffect(() => {
    ping();
    const t = setInterval(ping, 5000);
    return () => clearInterval(t);
  }, [ping]);

  // Keep ?tab=chat deep-link working.
  useEffect(() => {
    try {
      const u = new URL(window.location.href);
      u.searchParams.set("tab", tab);
      window.history.replaceState(null, "", u.toString());
    } catch {
      /* non-URL context — skip */
    }
  }, [tab]);

  const clearIdle = useCallback(() => {
    if (idleTimerRef.current !== null) {
      window.clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
  }, []);

  /** End the in-flight response: drop its key (later tokens open a new
   * bubble) and clear the typing indicator. Never deletes text. */
  const finalizeCurrent = useCallback(
    (id?: string | null) => {
      clearIdle();
      const key = streamKeyRef.current;
      streamKeyRef.current = null;
      setBusyBoth(false);
      if (key && id) {
        setMsgsBySession((prev) => {
          const cur = prev[id];
          if (!cur) return prev;
          const next = finalizeStream(cur, key);
          return next === cur ? prev : { ...prev, [id]: next };
        });
      }
    },
    [clearIdle],
  );

  const appendChunk = useCallback(
    (id: string, chunk: string) => {
      if (!chunk) return;
      let key = streamKeyRef.current;
      if (!key) {
        streamSeq.current += 1;
        key = `${id}#${Date.now()}#${streamSeq.current}`;
        streamKeyRef.current = key;
      }
      const k = key;
      setBusyBoth(true);
      setMsgsBySession((prev) => ({ ...prev, [id]: appendToken(prev[id] ?? [], k, chunk) }));
      setLocalSessions((prev) =>
        prev.map((s) => (s.id === id ? { ...s, updated_at: new Date().toISOString(), last_token_at: new Date().toISOString() } : s)),
      );
      // The server sends no explicit end-of-response (commit is per-token
      // durability, done is stripped from SSE shapes), so an idle gap means
      // the response finished. Any new token cancels this and keeps appending.
      clearIdle();
      idleTimerRef.current = window.setTimeout(() => {
        if (streamKeyRef.current === k) finalizeCurrent(id);
      }, 2500);
    },
    [clearIdle, finalizeCurrent],
  );

  const attach = useCallback(
    (id: string, resume: string) => {
      esRef.current?.close();
      clearIdle();
      sessionRef.current = id;
      const url = `${base}/v1/sessions/${id}/events${resume ? `?last_event=${encodeURIComponent(resume)}` : ""}`;
      const es = new EventSource(url);
      // Record the cursor, skipping already-seen ids so re-attaches never
      // duplicate replayed turns. Returns false when the event is a replay
      // duplicate and must be ignored.
      const noteEvent = (e: Event): boolean => {
        const ev = e as MessageEvent;
        if (!ev.lastEventId) return true;
        const n = Number.parseInt(ev.lastEventId, 10);
        if (Number.isFinite(n)) {
          if (n <= (seenIdBySession.current[id] ?? -1)) return false;
          seenIdBySession.current[id] = n;
        }
        cursorBySession.current[id] = ev.lastEventId;
        if (sessionRef.current === id) setLastEventBoth(ev.lastEventId);
        return true;
      };
      es.addEventListener("token", (e) => {
        const ev = e as MessageEvent;
        if (!noteEvent(e)) return;
        const parsed = parseTokenData(ev.data);
        if (parsed.action === "ignore") return;
        // Token payloads with done:true (or legacy commit-ish kinds) end the
        // response; anything else appends to this stream's single bubble.
        if (parsed.action === "commit") {
          finalizeCurrent(id);
          return;
        }
        appendChunk(id, parsed.text);
      });
      es.addEventListener("commit", (e) => {
        // Durability mark per token, NOT end-of-response: record the cursor
        // but keep the pending bubble open so the next token still appends.
        noteEvent(e);
      });
      es.addEventListener("status", (e) => {
        const ev = e as MessageEvent;
        if (!noteEvent(e)) return;
        // Replayed user turns arrive as status rows — rebuild them as "you"
        // bubbles so history reload shows full turns. Anything else is a note.
        try {
          const d = JSON.parse(ev.data) as { role?: unknown; text?: unknown };
          if (d.role === "user" && typeof d.text === "string" && d.text) {
            const text = d.text;
            streamKeyRef.current = null; // turn boundary: next token opens a fresh bubble
            setMsgsBySession((prev) => {
              const cur = prev[id] ?? [];
              const last = cur[cur.length - 1];
              if (last?.role === "you" && last.text === text) return prev;
              return { ...prev, [id]: [...cur, { role: "you", text }] };
            });
            return;
          }
        } catch {
          /* non-JSON status falls through to notice */
        }
        setNotice(String(ev.data).slice(0, 300));
      });
      // Legacy servers terminate with done/complete events (or just close the
      // stream) instead of contract silence — treat any of them as done.
      const finish = (e: Event) => {
        noteEvent(e);
        finalizeCurrent(id);
      };
      es.addEventListener("done", finish);
      es.addEventListener("complete", finish);
      es.addEventListener("completed", finish);
      es.onerror = () => {
        if (busyRef.current) {
          setNotice("Stream interrupted — will resume from last event on next send.");
          finalizeCurrent(id);
        }
      };
      esRef.current = es;
    },
    [appendChunk, base, clearIdle, finalizeCurrent],
  );

  useEffect(
    () => () => {
      esRef.current?.close();
      stopControllerRef.current?.abort();
      if (idleTimerRef.current !== null) window.clearTimeout(idleTimerRef.current);
    },
    [],
  );

  const preferredModel = useCallback((): string => {
    const cur = localSessions.find((s) => s.id === session)?.model;
    if (cur) return cur;
    if (models.length > 0) return models[0].id;
    return "";
  }, [models, session, localSessions]);

  const send = useCallback(
    async (textArg?: string) => {
      const text = (textArg ?? chatDraft).trim();
      if (!text || busyRef.current) return;
      setChatDraft("");
      setBusyBoth(true);
      setNotice("");
      // Seal any previous in-flight bubble before this turn starts.
      if (streamKeyRef.current) finalizeCurrent(sessionRef.current);
      clearIdle();
      const ctrl = new AbortController();
      stopControllerRef.current = ctrl;
      try {
        let id = sessionRef.current;
        if (!id) {
          const body = preferredModel() ? { model: preferredModel() } : {};
          const r = await fetch(`${base}/v1/sessions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: ctrl.signal,
          });
          if (!r.ok) throw new Error(`HTTP ${r.status} POST /v1/sessions`);
          const s = (await r.json()) as { id?: string; session_id?: string; sessionId?: string; model?: string; title?: string };
          id = s.id ?? s.session_id ?? s.sessionId ?? "";
          if (!id) throw new Error("server returned no session id");
          const now = new Date().toISOString();
          const row: LocalChat = {
            id,
            model: typeof s.model === "string" ? s.model : preferredModel() || undefined,
            title: typeof s.title === "string" ? s.title : undefined,
            created_at: now,
            updated_at: now,
          };
          setLocalSessions((prev) => (prev.some((p) => p.id === id) ? prev : sortChatsNewestFirst([row, ...prev])));
          setSession(id);
          sessionRef.current = id;
          setMsgsBySession((prev) => (prev[id!] ? prev : { ...prev, [id!]: [] }));
        }
        const target = id;
        setLocalSessions((prev) =>
          prev.map((s) => (s.id === target ? { ...s, updated_at: new Date().toISOString() } : s)),
        );
        setMsgsBySession((prev) => {
          const cur = prev[target] ?? [];
          const last = cur[cur.length - 1];
          if (last?.role === "you" && last.text === text) return prev;
          return { ...prev, [target]: [...cur, { role: "you", text }] };
        });
        streamKeyRef.current = null; // fresh bubble for this response
        attach(target, cursorBySession.current[target] ?? "");
        streamSeq.current += 1;
        streamKeyRef.current = `${target}#${Date.now()}#${streamSeq.current}`;
        const r = await fetch(`${base}/v1/sessions/${target}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
          signal: ctrl.signal,
        });
        if (!r.ok) throw new Error(`HTTP ${r.status} POST messages`);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          setNotice("Stopped — partial reply kept.");
        } else {
          setNotice(`Send failed: ${err}. Is dllm serve running at ${base}?`);
        }
        finalizeCurrent(sessionRef.current);
      } finally {
        if (stopControllerRef.current === ctrl) stopControllerRef.current = null;
      }
    },
    [attach, base, chatDraft, clearIdle, finalizeCurrent, preferredModel],
  );

  /** Stop generation: abort the in-flight POST, close SSE, tell the server,
   * finalize the bubble, clear busy. Never deletes text. */
  const stopGeneration = useCallback(async () => {
    const id = sessionRef.current;
    stopControllerRef.current?.abort();
    stopControllerRef.current = null;
    esRef.current?.close();
    esRef.current = null;
    if (id) {
      try {
        await fetch(`${base}/v1/sessions/${encodeURIComponent(id)}/stop`, { method: "POST" });
      } catch {
        /* stopping is best-effort — local state still finalizes */
      }
      finalizeCurrent(id);
    } else {
      finalizeCurrent(null);
    }
    setBusyBoth(false);
    setNotice("Stopped — partial reply kept.");
  }, [base, finalizeCurrent]);

  const newChat = useCallback(async () => {
    if (busyRef.current) return;
    finalizeCurrent(sessionRef.current);
    clearIdle();
    try {
      const body = preferredModel() ? { model: preferredModel() } : {};
      const r = await fetch(`${base}/v1/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const s = (await r.json()) as { id?: string; session_id?: string; sessionId?: string; model?: string; title?: string };
      const id = s.id ?? s.session_id ?? s.sessionId ?? "";
      if (!id) throw new Error("server returned no session id");
      const now = new Date().toISOString();
      const row: LocalChat = {
        id,
        model: typeof s.model === "string" ? s.model : preferredModel() || undefined,
        title: typeof s.title === "string" ? s.title : undefined,
        created_at: now,
        updated_at: now,
      };
      setLocalSessions((prev) => (prev.some((p) => p.id === id) ? prev : sortChatsNewestFirst([row, ...prev])));
      setSession(id);
      sessionRef.current = id;
      setMsgsBySession((prev) => ({ ...prev, [id]: [] }));
      streamKeyRef.current = null;
      setNotice("");
      attach(id, "");
    } catch (err) {
      setNotice(`New chat failed: ${err}. Is dllm serve running at ${base}?`);
    }
  }, [attach, base, clearIdle, finalizeCurrent, preferredModel]);

  const selectSession = useCallback(
    (id: string) => {
      if (streamKeyRef.current) finalizeCurrent(sessionRef.current);
      clearIdle();
      setSession(id);
      sessionRef.current = id;
      setMsgsBySession((prev) => (prev[id] ? prev : { ...prev, [id]: [] }));
      streamKeyRef.current = null;
      setBusyBoth(false);
      setMenuOpenId(null);
      setUsagePopId(null);
      setDeleteConfirmId(null);
      setRenamingId(null);
      // Empty local thread + unknown cursor replays the full log, and the
      // status/token handlers rebuild one message per turn.
      attach(id, cursorBySession.current[id] ?? "");
    },
    [attach, clearIdle, finalizeCurrent],
  );

  // Rename updates the local row first, then tries the server. A 404/405
  // means a stale coordinator — the local name is still kept.
  const renameSession = useCallback(
    async (id: string, title: string) => {
      const t = title.trim();
      if (!t) return;
      setLocalSessions((prev) => sortChatsNewestFirst(prev.map((s) => (s.id === id ? { ...s, title: t } : s))));
      setRenamingId(null);
      setMenuOpenId(null);
      try {
        const r = await fetch(`${base}/v1/sessions/${encodeURIComponent(id)}/rename`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: t }),
        });
        if (r.status === 404 || r.status === 405) {
          setNotice("Renamed here — coordinator needs upgrade so other devices keep the old name.");
          return;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
      } catch (err) {
        setNotice(`Server rename failed: ${err}. Local name kept.`);
      }
    },
    [base],
  );

  // Delete removes the local row + thread, then tries the server so the
  // coordinator drops the event log too. Unknown sessions still clear locally.
  const deleteSession = useCallback(
    async (id: string) => {
      try {
        const r = await fetch(`${base}/v1/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
        if (r.status === 404 || r.status === 405) {
          setNotice("Deleted here — session was already gone on the coordinator.");
        } else if (!r.ok) {
          throw new Error(`HTTP ${r.status}`);
        }
      } catch (err) {
        setNotice(`Server delete failed: ${err}. Removing local copy anyway.`);
      }
      setLocalSessions((prev) => prev.filter((s) => s.id !== id));
      setMsgsBySession((prev) => {
        if (!(id in prev)) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
      delete cursorBySession.current[id];
      delete seenIdBySession.current[id];
      if (sessionRef.current === id) {
        esRef.current?.close();
        stopControllerRef.current?.abort();
        stopControllerRef.current = null;
        clearIdle();
        sessionRef.current = null;
        setSession(null);
        streamKeyRef.current = null;
        setBusyBoth(false);
      }
      setDeleteConfirmId(null);
      setMenuOpenId(null);
      setUsagePopId(null);
    },
    [base, clearIdle],
  );

  const loadModels = useCallback(async () => {
    try {
      const d = await jget<{ models: Model[] }>(base, "/api/models");
      setModels(d.models ?? []);
    } catch {
      setNotice("Model catalog unreachable — start dllm serve first.");
    }
  }, [base]);

  // Coordinator identity for the header line. Runs whenever the base URL
  // changes so the header is honest on every tab.
  const loadIdentity = useCallback(async () => {
    try {
      setNode(await jget<NodeInfo>(base, "/api/node"));
    } catch {
      setNode(null);
    }
    try {
      setStats(await jget<Stats>(base, "/api/stats"));
    } catch {
      setStats(null);
    }
  }, [base]);

  useEffect(() => {
    loadIdentity();
  }, [loadIdentity]);

  // Networks list. New endpoint; failure leaves an honest empty state.
  const loadNetworks = useCallback(async (quiet?: boolean) => {
    if (!quiet) setNetworksLoading(true);
    try {
      const raw = await jget<unknown>(base, "/v1/networks");
      const list = normalizeNetworks(raw);
      setNetworks(list);
      setSelectedNetworkId((prev) => {
        if (prev && list.some((n) => n.id === prev)) return prev;
        return list.length > 0 ? list[0].id : null;
      });
    } catch {
      setNetworks(null);
    }
    if (!quiet) setNetworksLoading(false);
  }, [base]);

  const loadNetworkDetail = useCallback(
    async (id: string) => {
      setNetDetailLoading(id);
      try {
        const raw = await jget<unknown>(base, `/v1/networks/${encodeURIComponent(id)}/devices`);
        const parsed = parseNetworkMembers(raw);
        setNetMembers((prev) => ({ ...prev, [id]: parsed }));
      } catch {
        setNetMembers((prev) => (prev[id] ? prev : { ...prev, [id]: { paired: [], active: [] } }));
        setNotice("Network detail unavailable — coordinator may need upgrade. State unchanged.");
      }
      setNetDetailLoading(null);
    },
    [base],
  );

  const createNetwork = useCallback(async () => {
    const name = createName.trim();
    if (!name || createBusy) return;
    setCreateBusy(true);
    try {
      const body: Record<string, unknown> = { name, open_join: createOpenJoin };
      if (createPassword) body["password"] = createPassword;
      const r = await fetch(`${base}/v1/networks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setNotice(`Network “${name}” created — visible to all LAN devices on this coordinator.`);
      setCreateName("");
      setCreatePassword("");
      await loadNetworks(true);
    } catch (err) {
      setNotice(`Create failed: ${err}. State unchanged.`);
    }
    setCreateBusy(false);
  }, [base, createName, createPassword, createOpenJoin, createBusy, loadNetworks]);

  const joinNetwork = useCallback(
    async (id: string, password?: string) => {
      setJoinBusyId(id);
      try {
        const body = password ? { password } : {};
        const r = await fetch(`${base}/v1/networks/${encodeURIComponent(id)}/join`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (r.status === 401 || r.status === 403) {
          setNotice("Join refused — wrong or missing password. State unchanged.");
          return;
        }
        if (r.status === 404) {
          setNotice("Network not found on this coordinator — state unchanged.");
          return;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setNotice("Joined network.");
        await loadNetworkDetail(id);
        await loadNetworks(true);
      } catch (err) {
        setNotice(`Join failed: ${err}. State unchanged.`);
      }
      setJoinBusyId(null);
    },
    [base, loadNetworkDetail, loadNetworks],
  );

  const joinFromCode = useCallback(async () => {
    const parsed = parseNetworkQr(joinCode);
    if (!parsed) {
      setNotice("That code is not a network invite — paste a dllm://net?... code.");
      return;
    }
    try {
      const u = new URL(base);
      if (parsed.host && parsed.host !== u.hostname) {
        setNotice(`Invite points at ${parsed.host} — switch the coordinator box there first, then join.`);
        return;
      }
    } catch {
      /* base is loopback — attempt join anyway */
    }
    await joinNetwork(parsed.id);
    setJoinCode("");
  }, [joinCode, base, joinNetwork]);

  // Render each network's QR as a bundled data URL (no CDN: LAN-offline rule).
  useEffect(() => {
    if (!networks) return;
    let live = true;
    for (const n of networks) {
      if (netQr[n.id] !== undefined) continue;
      const payload = buildNetworkQrPayload(base, n);
      if (!payload) {
        setNetQr((prev) => (prev[n.id] !== undefined ? prev : { ...prev, [n.id]: null }));
        continue;
      }
      QRCode.toDataURL(payload, { width: 200, margin: 1 })
        .then((url) => {
          if (live) setNetQr((prev) => (prev[n.id] !== undefined ? prev : { ...prev, [n.id]: url }));
        })
        .catch(() => {
          if (live) setNetQr((prev) => (prev[n.id] !== undefined ? prev : { ...prev, [n.id]: null }));
        });
    }
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [networks, base]);

  const copyNetworkPayload = useCallback(
    async (id: string) => {
      const n = networks?.find((x) => x.id === id);
      if (!n) return;
      const payload = buildNetworkQrPayload(base, n);
      if (!payload) return;
      try {
        await navigator.clipboard.writeText(payload);
        setNotice("Network invite copied.");
      } catch {
        setNotice("Copy blocked by the browser — select the invite text manually.");
      }
    },
    [base, networks],
  );

  // Usage loader: GET /v1/usage?group=selectedGroup. 403 or is_admin:false
  // means totals-only + admin note. Falls back to /v1/plan for the strip.
  const loadUsage = useCallback(async () => {
    setUsageLoading(true);
    setUsageForbidden(false);
    const group = selectedNetworkId ? `?group=${encodeURIComponent(selectedNetworkId)}` : "";
    try {
      const r = await fetch(`${base}/v1/usage${group}`, { cache: "no-store" });
      if (r.status === 403) {
        let totals: UsageTotals | null = null;
        let plan: PlanStage[] | null = null;
        try {
          const raw: unknown = await r.json();
          totals = parseUsageTotals(raw);
          plan = parseUsagePlan(raw);
        } catch {
          /* 403 with no body — totals stay null */
        }
        setUsageTotals(totals);
        setUsagePerDevice(null);
        if (plan !== null) setPlanStages(plan);
        setUsageSource(plan !== null ? "usage" : "unknown");
        setUsageForbidden(true);
        setUsageLoading(false);
        return;
      }
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const raw: unknown = await r.json();
      const totals = parseUsageTotals(raw);
      const pd = parseUsagePerDevice(raw);
      const pl = parseUsagePlan(raw);
      setUsageTotals(totals);
      // Non-admin coordinators serve totals without a breakdown.
      if (totals?.is_admin === false) {
        setUsagePerDevice(null);
        setUsageForbidden(true);
      } else {
        setUsagePerDevice(pd);
        setUsageForbidden(false);
      }
      if (pl !== null) {
        setPlanStages(pl);
        setUsageSource("usage");
      } else {
        try {
          const d = await jget<unknown>(base, "/v1/plan");
          setPlanStages(parseUsagePlan(d));
        } catch {
          setPlanStages(null);
        }
        setUsageSource("usage");
      }
      setUsageLoading(false);
      return;
    } catch {
      /* fall through to plan-only fallback */
    }
    try {
      const d = await jget<unknown>(base, "/v1/plan");
      setPlanStages(parseUsagePlan(d));
    } catch {
      setPlanStages(null);
    }
    setUsageTotals(null);
    setUsagePerDevice(null);
    setUsageSource("plan-only");
    setUsageLoading(false);
  }, [base, selectedNetworkId]);

  // Model-split panel loads the honest usage sources lazily on expand.
  useEffect(() => {
    if (tab === "chat" && layerOpen) {
      loadUsage();
    }
  }, [tab, layerOpen, loadUsage]);

  // Escape closes drawer menus/popovers/confirm flows.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMenuOpenId(null);
        setUsagePopId(null);
        setDeleteConfirmId(null);
        setRenamingId(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Keep the newest bubble in view.
  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight });
  }, [msgsBySession, session, busy]);

  useEffect(() => {
    if (tab === "models" || tab === "chat") loadModels();
    if (tab === "networks") {
      loadIdentity();
      loadNetworks();
    }
    if (tab === "usage") {
      loadIdentity();
      loadNetworks(true);
      loadUsage();
    }
  }, [tab, loadModels, loadIdentity, loadNetworks, loadUsage]);

  // Header identity line: only parts the server actually returned.
  let hostHint = "";
  try {
    const h = new URL(base).hostname;
    hostHint = h === "127.0.0.1" || h === "localhost" || h === "::1" ? "on this machine" : `remote host ${h}`;
  } catch {
    hostHint = "";
  }
  const identityBits = [
    node ? `node ${(node.node_id ?? "").slice(0, 12)}` : "",
    stats?.engine ? stats.engine : "",
    node?.version ? `v${node.version}` : "",
    hostHint,
  ].filter(Boolean);

  // Chat derivations (local-only).
  const msgs: Msg[] = session ? (msgsBySession[session] ?? []) : [];
  const headerModel =
    localSessions.find((s) => s.id === session)?.model ?? (models.length > 0 ? models[0].id : "");
  const currentTitle = session
    ? displayTitle(session, {
        title: localSessions.find((s) => s.id === session)?.title,
        model: headerModel,
      })
    : "New conversation";
  const convEntries: { sid: string; s: LocalChat }[] = sortChatsNewestFirst(localSessions).map((s) => ({ sid: s.id, s }));

  // ---- Usage tab derivations (read-only, never invented) ----
  const usageNameById = new Map((usagePerDevice ?? []).map((e) => [e.device_id, e.device_name?.trim() || shortId(e.device_id)]));
  const usageDonutSlices: { key: string; label: string; tokens: number }[] = (usagePerDevice ?? []).map((e) => ({
    key: e.device_id,
    label: e.device_name?.trim() || shortId(e.device_id),
    tokens: e.tokens_out,
  }));
  const usageTotal = usageDonutSlices.reduce((n, s) => n + s.tokens, 0);
  const layerOwners = planLayerOwners(planStages);
  const layerOwnerKeys: string[] = [...new Set(layerOwners.filter((o): o is string => o !== null))];
  const usageColorOrder: string[] = [...new Set([
    ...usageDonutSlices.filter((s) => s.key !== UNATTRIBUTED_KEY).map((s) => s.key),
    ...layerOwnerKeys,
  ])];
  const singleDeviceRun = layerOwners.length === 28 && layerOwners.every((o) => o !== null && o === layerOwners[0]);
  const layersByDevice = new Map<string, number[]>();
  layerOwners.forEach((owner, layer) => {
    if (owner === null) return;
    const arr = layersByDevice.get(owner) ?? [];
    arr.push(layer);
    layersByDevice.set(owner, arr);
  });
  const layerName = (id: string): string => usageNameById.get(id) ?? shortId(id);
  const workerActive = usageTotals?.worker_active === true || (usagePerDevice ?? []).some((e) => e.worker_active === true);
  const workerEntry = (usagePerDevice ?? []).find((e) => e.worker_active === true) ?? null;
  const workerCpu = workerEntry?.cpu_pct ?? usageTotals?.cpu_pct ?? null;
  const workerMem = workerEntry?.mem_pct ?? usageTotals?.mem_pct ?? null;
  const workerSource = workerEntry?.load_source ?? usageTotals?.load_source ?? null;
  const workerLive = workerSource === "live" || workerSource === "reported";
  const showTotalsOnly = usageForbidden || usagePerDevice === null;

  const selectedNetwork = networks?.find((n) => n.id === selectedNetworkId) ?? null;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden />
          <div>
            <h1>DLLM Mesh</h1>
            <p>Local LAN inference · no cloud in the loop</p>
            {identityBits.length > 0 && <p className="mono">{identityBits.join(" · ")}</p>}
          </div>
        </div>
        <nav className="nav" aria-label="Sections">
          {(["chat", "models", "networks", "usage"] as const).map((t) => (
            <button
              key={t}
              className={tab === t ? "on" : ""}
              onClick={() => setTab(t)}
              aria-current={tab === t ? "page" : undefined}
            >
              {t[0].toUpperCase() + t.slice(1)}
            </button>
          ))}
        </nav>
        <div className="conn">
          <input value={base} onChange={(e) => setBase(e.target.value)} spellCheck={false} aria-label="Coordinator URL" />
          <span className={`pill ${health}`}>{health === "live" ? "coordinator live" : health === "down" ? "no coordinator" : "checking…"}</span>
          <button
            className="drawer-toggle"
            onClick={() => {
              if (tab !== "chat") {
                setTab("chat");
                setDrawerOpen(true);
              } else {
                setDrawerOpen((v) => !v);
              }
            }}
            aria-expanded={tab === "chat" ? drawerOpen : undefined}
            aria-label="Toggle chat list"
            title="Chat list"
          >
            Chats
          </button>
        </div>
      </header>

      {notice && <div className="notice">{notice}</div>}

      {tab === "chat" && (
        <main className="chatgpt">
          <section className="thread-col" aria-label="Conversation">
            <div className="thread-head">
              <div className="thread-id">
                <strong>{currentTitle}</strong>
                <span className="mono">
                  {headerModel || "no model catalog yet"}
                  {session ? ` · ${session.slice(0, 12)}` : ""}
                  {lastEvent ? ` · resume from ${lastEvent}` : ""}
                </span>
              </div>
              <button
                className={layerOpen ? "ghost on" : "ghost"}
                onClick={() => setLayerOpen((v) => !v)}
                aria-expanded={layerOpen}
              >
                Model split
              </button>
            </div>
            {layerOpen && (
              <div className="layer-panel">
                <h2>Model split — 28 segments (layers 0–27)</h2>
                {planStages === null ? (
                  <p className="hint">Plan not reported — open this panel after a refresh; needs GET /v1/plan or /v1/usage plan.</p>
                ) : (
                  <>
                    <div className="strip" role="img" aria-label="Layer ownership, layers 0 to 27">
                      {layerOwners.map((owner, layer) => (
                        <span
                          key={layer}
                          className="seg"
                          title={owner === null ? `layer ${layer}: unassigned` : `layer ${layer}: ${layerName(owner)}`}
                          style={{ background: owner === null ? "transparent" : usageColorFor(owner, usageColorOrder) }}
                        />
                      ))}
                    </div>
                    <p className="hint">
                      {singleDeviceRun
                        ? "Single-device run: layers 0–27 on this coordinator — split view activates when pipeline sharding lands."
                        : "Live layer ownership from /v1/plan (or /v1/usage plan)."}
                    </p>
                    {layersByDevice.size === 0 ? (
                      <p className="hint">No stage assigned — plan carries no layer ranges.</p>
                    ) : (
                      [...layersByDevice.entries()].map(([id, layers]) => {
                        const r = layerRanges(layers);
                        return (
                          <p key={id} className="mono">
                            {layerName(id)}: layers {r.text} ({r.count})
                          </p>
                        );
                      })
                    )}
                  </>
                )}
              </div>
            )}
            <div className="thread" ref={threadRef} role="log" aria-live="polite" aria-label="Messages">
              {msgs.length === 0 && !busy && (
                <div className="empty">
                  <h2>Start a conversation</h2>
                  <p>Talk to your own mesh — start dllm serve, then send a message. Tokens stream here over local SSE.</p>
                </div>
              )}
              {msgs.map((m, i) => (
                <div key={i} className={`bubble ${m.role}`}>
                  {m.text}
                </div>
              ))}
              {busy && <div className="typing">meshing…</div>}
            </div>
            <form
              className="composer"
              onSubmit={(e) => {
                e.preventDefault();
                void send();
              }}
            >
              <input
                value={chatDraft}
                onChange={(e) => setChatDraft(e.target.value)}
                placeholder="Message the mesh…"
                aria-label="Message the mesh"
              />
              {busy ? (
                <button type="button" onClick={() => void stopGeneration()}>
                  Stop
                </button>
              ) : (
                <button type="submit" disabled={!chatDraft.trim()}>
                  Send
                </button>
              )}
            </form>
            <p className="hint thread-meta">
              {`${localSessions.length} local chat${localSessions.length === 1 ? "" : "s"} on this browser`}
            </p>
          </section>
          {drawerOpen && <div className="drawer-backdrop" onClick={() => setDrawerOpen(false)} aria-hidden />}
          <aside className={`chat-drawer${drawerOpen ? " open" : ""}`} aria-label="Chat list" aria-hidden={!drawerOpen}>
            <div className="drawer-head">
              <span>Chats ({localSessions.length})</span>
              <button onClick={() => void newChat()} disabled={busy}>
                New chat
              </button>
            </div>
            <p className="hint">Local only — chats live in this browser per coordinator, never synced across devices.</p>
            {convEntries.length === 0 && (
              <p className="hint">No chats yet — start a new chat.</p>
            )}
            {convEntries.map(({ sid, s }) => {
              const title = displayTitle(sid, { title: s.title, model: s.model });
              const menuOpen = menuOpenId === sid;
              return (
                <div key={sid} className={`drawer-row${sid === session ? " active" : ""}`}>
                  <button className="row-main" onClick={() => selectSession(sid)} title={sid}>
                    <span className="row-title">{title}</span>
                    <span className="mono row-sub">
                      {`${s.model ?? headerModel ?? "no model"}${s.tokens_out != null ? ` · ${s.tokens_out} out` : ""}${s.last_token_at ? ` · ${s.last_token_at}` : ""}`}
                    </span>
                  </button>
                  <button
                    className="dots"
                    aria-label={`Options for ${title}`}
                    aria-expanded={menuOpen}
                    onClick={() => {
                      setMenuOpenId(menuOpen ? null : sid);
                      setUsagePopId(null);
                      setDeleteConfirmId(null);
                      setRenamingId(null);
                    }}
                  >
                    ⋯
                  </button>
                  {menuOpen && (
                    <div className="menu" role="menu">
                      <button
                        onClick={() => {
                          setRenamingId(sid);
                          setRenameDraft(typeof s.title === "string" ? s.title : "");
                          setMenuOpenId(null);
                          setDeleteConfirmId(null);
                          setUsagePopId(null);
                        }}
                      >
                        Rename
                      </button>
                      <button
                        onClick={() => {
                          setUsagePopId(usagePopId === sid ? null : sid);
                          setRenamingId(null);
                          setDeleteConfirmId(null);
                        }}
                      >
                        Usage
                      </button>
                      <button
                        className="danger"
                        onClick={() => {
                          setDeleteConfirmId(sid);
                          setRenamingId(null);
                          setUsagePopId(null);
                          setMenuOpenId(null);
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  )}
                  {renamingId === sid && (
                    <form
                      className="rename"
                      onSubmit={(e) => {
                        e.preventDefault();
                        void renameSession(sid, renameDraft);
                      }}
                    >
                      <input
                        value={renameDraft}
                        onChange={(e) => setRenameDraft(e.target.value)}
                        aria-label={`New name for ${title}`}
                        placeholder="Chat name"
                      />
                      <button type="submit">Save</button>
                      <button type="button" onClick={() => setRenamingId(null)}>
                        Cancel
                      </button>
                    </form>
                  )}
                  {deleteConfirmId === sid && (
                    <div className="confirm">
                      <span>Delete this chat here and on the coordinator?</span>
                      <button className="danger" onClick={() => void deleteSession(sid)}>
                        Confirm delete
                      </button>
                      <button onClick={() => setDeleteConfirmId(null)}>Cancel</button>
                    </div>
                  )}
                  {usagePopId === sid && (
                    <div className="usage-pop">
                      <strong>Usage</strong>
                      <p className="mono">Model: {s.model ?? "not reported"}</p>
                      <p className="mono">
                        Tokens out: {typeof s.tokens_out === "number" ? s.tokens_out : "not reported"}
                      </p>
                      <p className="mono">
                        Messages: {(msgsBySession[sid] ?? []).length}
                      </p>
                      <p className="mono">
                        Last activity: {(s.last_token_at ?? s.updated_at ?? s.created_at) ?? "not reported"}
                      </p>
                    </div>
                  )}
                </div>
              );
            })}
          </aside>
        </main>
      )}

      {tab === "models" && (
        <main className="cards">
          {models.length === 0 && <p className="hint">No catalog yet — start the coordinator, or pull a model with dllm pull.</p>}
          {models.map((m) => (
            <div key={m.id} className="card">
              <h2>{m.id}</h2>
              <p className="hint">
                {[m.quant, m.params, m.size_mb ? `${m.size_mb} MB` : ""].filter(Boolean).join(" · ") || "curated GGUF"}
              </p>
              <div className="shards">{(m.stages ?? []).map((s) => <span key={s.range}>{s.range}</span>)}</div>
              <button
                onClick={() => {
                  setTab("chat");
                  setChatDraft(`Run ${m.id}: hello mesh`);
                }}
              >
                Run in chat
              </button>
            </div>
          ))}
        </main>
      )}

      {tab === "networks" && (
        <main className="fleet">
          <div className="card fleet-head">
            <div>
              <h2>{networks ? `${networks.length} network${networks.length === 1 ? "" : "s"}` : "Networks"}</h2>
              <p className="hint">
                {networks
                  ? "Live networks from GET /v1/networks — nothing here is guessed."
                  : "No data — start/upgrade coordinator (GET /v1/networks unavailable)."}
              </p>
            </div>
            <button onClick={() => { loadIdentity(); loadNetworks(); }} disabled={networksLoading}>
              {networksLoading ? "Refreshing…" : "Refresh"}
            </button>
          </div>

          <div className="card">
            <h2>Create a network</h2>
            <p className="hint">Visible to all LAN devices on this coordinator.</p>
            <div className="row">
              <input
                value={createName}
                onChange={(e) => setCreateName(e.target.value)}
                placeholder="Network name"
                aria-label="Network name"
              />
              <input
                value={createPassword}
                onChange={(e) => setCreatePassword(e.target.value)}
                placeholder="Password (optional)"
                aria-label="Network password"
                type="password"
              />
            </div>
            <p className="hint">
              <label>
                <input
                  type="checkbox"
                  checked={createOpenJoin}
                  onChange={(e) => setCreateOpenJoin(e.target.checked)}
                />{" "}
                Open join (no approval needed)
              </label>
            </p>
            <div className="row">
              <button onClick={() => void createNetwork()} disabled={createBusy || !createName.trim()}>
                {createBusy ? "Creating…" : "Create"}
              </button>
            </div>
          </div>

          <div className="card">
            <h2>Join from invite code</h2>
            <p className="hint">Paste a dllm://net?... code from another device, then join.</p>
            <div className="row">
              <input
                value={joinCode}
                onChange={(e) => setJoinCode(e.target.value)}
                placeholder="dllm://net?id=…"
                aria-label="Network invite code"
                spellCheck={false}
              />
              <button onClick={() => void joinFromCode()} disabled={!joinCode.trim()}>
                Join
              </button>
            </div>
          </div>

          {networks === null ? (
            <div className="card">
              <h2>Networks</h2>
              <p className="hint">No data — start/upgrade coordinator. This panel renders only what the server returns.</p>
            </div>
          ) : networks.length === 0 ? (
            <div className="card">
              <h2>Networks</h2>
              <p className="hint">No networks yet — create one above.</p>
            </div>
          ) : (
            <div className="cards">
              {networks.map((n) => {
                const detail = netMembers[n.id];
                const loading = netDetailLoading === n.id;
                const selected = selectedNetworkId === n.id;
                const payload = buildNetworkQrPayload(base, n);
                const qr = netQr[n.id];
                const pw = joinPasswords[n.id] ?? "";
                return (
                  <div key={n.id} className="card">
                    <h2>{n.name?.trim() || shortId(n.id)}</h2>
                    {n.name?.trim() && n.name.trim() !== n.id && <p className="mono">{n.id}</p>}
                    <p>
                      {n.open_join ? <span className="pill live">open join</span> : <span className="pill">approval</span>}{" "}
                      {n.has_password ? <span className="pill">password</span> : <span className="pill">no password</span>}{" "}
                      {n.is_admin ? <span className="pill live">admin</span> : null}
                    </p>
                    <p className="mono">
                      {typeof n.active_count === "number" ? `${n.active_count} active` : "active: not reported"}
                      {" · "}
                      {typeof n.member_count === "number" ? `${n.member_count} members` : "members: not reported"}
                    </p>
                    {payload && (
                      <>
                        {qr ? (
                          <img className="qr" src={qr} alt={`Invite QR for ${n.name ?? n.id}`} width={200} height={200} />
                        ) : (
                          <p className="hint">QR unavailable — use the invite text below.</p>
                        )}
                        <p className="mono pair-uri">{payload}</p>
                        <div className="row">
                          <button onClick={() => void copyNetworkPayload(n.id)}>Copy invite</button>
                        </div>
                      </>
                    )}
                    <div className="row dev-actions">
                      <button
                        onClick={() => {
                          setSelectedNetworkId(n.id);
                          if (!detail) void loadNetworkDetail(n.id);
                        }}
                      >
                        {selected ? "Selected" : "Select"}
                      </button>
                      <button onClick={() => void loadNetworkDetail(n.id)} disabled={loading}>
                        {loading ? "Loading…" : detail ? "Reload members" : "Members"}
                      </button>
                    </div>
                    {n.has_password ? (
                      <div className="row">
                        <input
                          value={pw}
                          onChange={(e) => setJoinPasswords((prev) => ({ ...prev, [n.id]: e.target.value }))}
                          placeholder="Password"
                          aria-label={`Password for ${n.name ?? n.id}`}
                          type="password"
                        />
                        <button onClick={() => void joinNetwork(n.id, pw || undefined)} disabled={joinBusyId === n.id}>
                          {joinBusyId === n.id ? "Joining…" : "Join"}
                        </button>
                      </div>
                    ) : (
                      <div className="row">
                        <button onClick={() => void joinNetwork(n.id)} disabled={joinBusyId === n.id}>
                          {joinBusyId === n.id ? "Joining…" : "Join"}
                        </button>
                      </div>
                    )}
                    {detail && (
                      <div>
                        <h3 className="fleet-h">Active now ({detail.active.length})</h3>
                        {detail.active.length === 0 ? (
                          <p className="hint">None active right now.</p>
                        ) : (
                          detail.active.map((m, i) => (
                            <p key={m.device_id ?? `${n.id}-a-${i}`} className="mono">
                              {memberLabel(m)}{m.role ? ` · ${m.role}` : ""}{m.last_seen ? ` · seen ${m.last_seen}` : ""}
                            </p>
                          ))
                        )}
                        <h3 className="fleet-h">Paired ({detail.paired.length})</h3>
                        {detail.paired.length === 0 ? (
                          <p className="hint">No idle paired devices.</p>
                        ) : (
                          detail.paired.map((m, i) => (
                            <p key={m.device_id ?? `${n.id}-p-${i}`} className="mono">
                              {memberLabel(m)}{m.role ? ` · ${m.role}` : ""}{m.last_seen ? ` · seen ${m.last_seen}` : ""}
                            </p>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {selectedNetwork && (
            <p className="hint">Usage group: {selectedNetwork.name?.trim() || shortId(selectedNetwork.id)} — the usage tab reports this network.</p>
          )}
        </main>
      )}

      {tab === "usage" && (
        <main className="fleet usage">
          <div className="card fleet-head">
            <div>
              <h2>Usage{selectedNetwork ? ` — ${selectedNetwork.name?.trim() || shortId(selectedNetwork.id)}` : ""}</h2>
              <p className="hint">
                {usageLoading
                  ? "Loading usage…"
                  : usageSource === "usage"
                    ? `Live shares from GET /v1/usage${selectedNetworkId ? `?group=${selectedNetworkId}` : ""}.`
                    : usageSource === "plan-only"
                      ? "GET /v1/usage unavailable — layer strip from /v1/plan only."
                      : "Usage source unknown — refresh to probe the coordinator."}
              </p>
            </div>
            <div className="row">
              {networks && networks.length > 0 && (
                <select
                  value={selectedNetworkId ?? ""}
                  onChange={(e) => setSelectedNetworkId(e.target.value || null)}
                  aria-label="Usage group"
                >
                  <option value="">All groups</option>
                  {networks.map((n) => (
                    <option key={n.id} value={n.id}>
                      {n.name?.trim() || shortId(n.id)}
                    </option>
                  ))}
                </select>
              )}
              <button onClick={() => { loadIdentity(); loadNetworks(true); loadUsage(); }} disabled={usageLoading}>
                {usageLoading ? "Refreshing…" : "Refresh"}
              </button>
            </div>
          </div>

          {usageForbidden && (
            <div className="card">
              <h2>Totals only</h2>
              <p className="hint">Per-device breakdown is admin only — showing totals.</p>
              <p className="mono">
                {usageTotals?.tokens_out_total != null ? `${usageTotals.tokens_out_total} tokens out` : "tokens out: not reported"}
                {" · "}
                {usageTotals?.sessions_total != null ? `${usageTotals.sessions_total} sessions` : "sessions: not reported"}
              </p>
            </div>
          )}

          {!usageForbidden && usageTotals && (usageTotals.tokens_out_total != null || usageTotals.sessions_total != null) && (
            <div className="card">
              <h2>Totals</h2>
              <p className="mono">
                {usageTotals.tokens_out_total != null ? `${usageTotals.tokens_out_total} tokens out` : "tokens out: not reported"}
                {" · "}
                {usageTotals.sessions_total != null ? `${usageTotals.sessions_total} sessions` : "sessions: not reported"}
              </p>
            </div>
          )}

          <div className="card">
            <h2>Compute worker</h2>
            {workerActive ? (
              <>
                <p>
                  <span className="pill live">compute worker active</span>
                </p>
                {workerLive && workerCpu !== null ? (
                  <UsageBar label="CPU" pct={workerCpu} />
                ) : (
                  <p className="hint">CPU: not reporting</p>
                )}
                {workerLive && workerMem !== null ? (
                  <UsageBar label="MEM" pct={workerMem} />
                ) : (
                  <p className="hint">MEM: not reporting</p>
                )}
                {workerLive && workerSource && <p className="mono">measured [{workerSource}]</p>}
              </>
            ) : (
              <p className="hint">Compute worker not reporting.</p>
            )}
          </div>

          <div className="usage-top">
            <div className="card">
              <h2>Compute donut — tokens_out per device</h2>
              {showTotalsOnly ? (
                <p className="hint">Per-device breakdown is admin only — totals shown above.</p>
              ) : usageDonutSlices.length === 0 || usageTotal === 0 ? (
                <p className="hint">No tokens_out reported yet — send a chat to generate usage.</p>
              ) : (
                <div className="donut-wrap">
                  <UsageDonut slices={usageDonutSlices} order={usageColorOrder} total={usageTotal} />
                  <ul className="legend">
                    {usageDonutSlices.map((s) => {
                      const pct = usageTotal > 0 ? (s.tokens / usageTotal) * 100 : 0;
                      return (
                        <li key={s.key} className="mono">
                          <span className="sw" style={{ background: usageColorFor(s.key, usageColorOrder) }} aria-hidden />
                          {s.label} — {s.tokens} tokens ({pct.toFixed(1)}%)
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
              {!showTotalsOnly && (
                <p className="hint">Shares from /v1/usage per_device.</p>
              )}
            </div>

            <div className="card">
              <h2>Layer split — 28 segments (layers 0–27)</h2>
              {planStages === null ? (
                <p className="hint">Plan not reported — GET /v1/plan (or /v1/usage plan) returned nothing usable.</p>
              ) : (
                <>
                  <div className="strip" role="img" aria-label="Layer ownership, layers 0 to 27">
                    {layerOwners.map((owner, layer) => (
                      <span
                        key={layer}
                        className="seg"
                        title={owner === null ? `layer ${layer}: unassigned` : `layer ${layer}: ${layerName(owner)}`}
                        style={{ background: owner === null ? "transparent" : usageColorFor(owner, usageColorOrder) }}
                      />
                    ))}
                  </div>
                  <p className="hint">
                    {singleDeviceRun
                      ? "Single-device run: layers 0–27 on this coordinator — split view activates when pipeline sharding lands."
                      : "Live layer ownership from /v1/plan (or /v1/usage plan)."}
                  </p>
                  {layersByDevice.size === 0 ? (
                    <p className="hint">No stage assigned — plan carries no layer ranges.</p>
                  ) : (
                    [...layersByDevice.entries()].map(([id, layers]) => {
                      const r = layerRanges(layers);
                      return (
                        <p key={id} className="mono">
                          {layerName(id)}: layers {r.text} ({r.count})
                        </p>
                      );
                    })
                  )}
                </>
              )}
            </div>
          </div>

          {!showTotalsOnly && usagePerDevice !== null && usagePerDevice.length > 0 && (
            <div className="card">
              <h2>Resources per device</h2>
              <div className="usage-grid">
                {usagePerDevice.map((d) => (
                  <UsageResourceCard key={d.device_id} d={d} />
                ))}
              </div>
            </div>
          )}

          <div className="card">
            <h2>Link bandwidth</h2>
            <p className="hint">Link bandwidth not yet measured — activation frames are 2 KiB/token (WIRE_DIM 1024 fp16); per-hop throughput arrives with the QUIC pipeline.</p>
          </div>
        </main>
      )}

      <footer>proto dllm/1 · KV block 16 · checkpoints every 64–128 tokens · cloud never sees chat</footer>
    </div>
  );
}

/** Pure-SVG donut (no chart lib): tokens_out share per device. */
function UsageDonut({
  slices,
  order,
  total,
}: {
  slices: { key: string; label: string; tokens: number }[];
  order: string[];
  total: number;
}) {
  const R = 70;
  const C = 2 * Math.PI * R;
  let acc = 0;
  return (
    <svg className="donut" width="180" height="180" viewBox="0 0 180 180" role="img" aria-label={`tokens_out share, ${total} total`}>
      <circle cx="90" cy="90" r={R} fill="none" stroke="#1e2530" strokeWidth="28" />
      {slices.map((s) => {
        const frac = total > 0 ? s.tokens / total : 0;
        const len = frac * C;
        const offset = acc;
        acc += len;
        if (len <= 0) return null;
        const gap = slices.length > 1 ? 2 : 0;
        return (
          <circle
            key={s.key}
            cx="90"
            cy="90"
            r={R}
            fill="none"
            stroke={usageColorFor(s.key, order)}
            strokeWidth="28"
            strokeDasharray={`${Math.max(0, len - gap)} ${C - Math.max(0, len - gap)}`}
            strokeDashoffset={-offset}
            transform="rotate(-90 90 90)"
          >
            <title>{`${s.label}: ${s.tokens} tokens`}</title>
          </circle>
        );
      })}
      <text x="90" y="86" textAnchor="middle" className="donut-total">{total}</text>
      <text x="90" y="104" textAnchor="middle" className="donut-sub">tokens out</text>
    </svg>
  );
}

/** Labeled resource bar (0–100 only; never rendered without a real sample). */
function UsageBar({ label, pct }: { label: string; pct: number }) {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div className="meter">
      <div className="mono">
        {label} {clamped.toFixed(0)}%
      </div>
      <div className="bar" aria-label={`${label} ${clamped.toFixed(0)} percent`}>
        <i style={{ width: `${clamped}%` }} />
      </div>
    </div>
  );
}

/** Per-device resource card for the Usage tab (admin breakdown only). */
function UsageResourceCard({ d }: { d: UsageDeviceEntry }) {
  const label = d.device_name?.trim() || shortId(d.device_id);
  const live = d.load_source === "live" || d.load_source === "reported";
  const cpu = live && typeof d.cpu_pct === "number" ? d.cpu_pct : null;
  const mem = live && typeof d.mem_pct === "number" ? d.mem_pct : null;
  return (
    <div className="card">
      <h2>{label}</h2>
      {label !== d.device_id && <p className="mono">{d.device_id}</p>}
      <p>
        {d.worker_active === true ? (
          <span className="pill live">compute worker active</span>
        ) : null}{" "}
        {d.active === true ? <span className="pill live">active</span> : <span className="pill">idle</span>}{" "}
        {d.role && <span className="pill">{d.role}</span>}
      </p>
      {cpu !== null ? (
        <UsageBar label="CPU" pct={cpu} />
      ) : (
        <p className="hint">CPU: not reporting</p>
      )}
      {mem !== null ? (
        <UsageBar label="MEM" pct={mem} />
      ) : (
        <p className="hint">MEM: not reporting</p>
      )}
      {d.load_source && live && <p className="mono">measured [{d.load_source}]</p>}
      <p className="mono">
        {d.sessions != null ? `${d.sessions} sessions · ` : ""}{d.tokens_out} tokens_out
      </p>
    </div>
  );
}
