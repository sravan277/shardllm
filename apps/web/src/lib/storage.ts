/** Browser-local chat + thread storage, namespaced per coordinator.
 *
 * These rows are created by this browser only — the coordinator's session list
 * is never merged in, so another device's chats never appear here. Writes are
 * debounced by the caller (see `useChat`), not here, so a token burst does not
 * re-serialize the whole thread on every event.
 */

import type { StreamMsg } from "../chatStream";
import type { LocalChat } from "../api/types";

export function chatsKey(base: string): string {
  return `dllm:chats:${base}`;
}

export function threadsKey(base: string): string {
  return `dllm:threads:${base}`;
}

/** Newest-first by last activity; untimestamped chats sink to the bottom. */
export function sortChatsNewestFirst(list: LocalChat[]): LocalChat[] {
  const key = (s: LocalChat): string => s.last_token_at ?? s.updated_at ?? s.created_at ?? "";
  return [...list].sort((a, b) => (key(b) < key(a) ? -1 : key(b) > key(a) ? 1 : 0));
}

export function loadLocalChats(base: string): LocalChat[] {
  try {
    const raw = window.localStorage.getItem(chatsKey(base));
    if (!raw) return [];
    const arr = JSON.parse(raw) as unknown;
    if (!Array.isArray(arr)) return [];
    return (arr as unknown[])
      .filter((e): e is Record<string, unknown> => typeof e === "object" && e !== null)
      .map((o) => {
        const pick = (k: string): string | undefined => (typeof o[k] === "string" && o[k] ? (o[k] as string) : undefined);
        const num = (k: string): number | undefined => {
          const v = o[k];
          return typeof v === "number" && Number.isFinite(v) && v >= 0 ? Math.floor(v) : undefined;
        };
        return {
          id: o["id"] as string,
          title: pick("title"),
          model: pick("model"),
          created_at: pick("created_at"),
          updated_at: pick("updated_at"),
          last_token_at: pick("last_token_at"),
          tokens_out: num("tokens_out"),
        } satisfies LocalChat;
      })
      .filter((c) => typeof c.id === "string" && c.id.length > 0);
  } catch {
    return [];
  }
}

export function loadLocalThreads(base: string): Record<string, StreamMsg[]> {
  try {
    const raw = window.localStorage.getItem(threadsKey(base));
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed !== "object" || parsed === null) return {};
    const out: Record<string, StreamMsg[]> = {};
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (!Array.isArray(v)) continue;
      const msgs: StreamMsg[] = [];
      for (const m of v) {
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

/** One debounced write of both keys. Failures keep everything in memory. */
export function saveLocalState(base: string, chats: LocalChat[], threads: Record<string, StreamMsg[]>): void {
  try {
    window.localStorage.setItem(chatsKey(base), JSON.stringify(chats));
  } catch {
    /* storage full or blocked — chats stay in memory */
  }
  try {
    const pruned: Record<string, StreamMsg[]> = {};
    for (const c of chats) {
      const t = threads[c.id];
      if (t) pruned[c.id] = t.map((m) => ({ role: m.role, text: m.text }));
    }
    window.localStorage.setItem(threadsKey(base), JSON.stringify(pruned));
  } catch {
    /* storage full or blocked — threads stay in memory */
  }
}