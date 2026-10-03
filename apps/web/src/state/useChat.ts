/** Chat state: local conversations, the SSE stream, and the send lifecycle.
 *
 * Two things here are deliberate performance work, not style:
 *
 * 1. **Token buffering.** A coordinator streams one `token` event per token
 *    (~17/s here). `appendChunk` only appends to a pending buffer and
 *    schedules one `requestAnimationFrame`; that frame does the single
 *    `setMsgsBySession` pair. A 20 tok/s stream costs ~10 renders per second
 *    instead of 20 full-tree renders.
 * 2. **Debounced persistence.** `saveLocalState` runs on a timer, so a token
 *    burst costs one `JSON.stringify` per key instead of two per token.
 *
 * Reconnection is explicit: `es.onerror` flips `streamPhase` to
 * `reconnecting` (a visible state, not a notice string), and once the health
 * ping says the coordinator is back, the retry effect re-attaches the
 * EventSource from the stored `lastEventId` cursor — no chat reselect needed.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { appendToken, displayTitle, finalizeStream } from "../chatStream";
import { applyIds, stampedThreads, type ChatMsg } from "../lib/messages";
import { parseTokenData } from "../api/parse";
import type { LocalChat, Model, StreamPhase } from "../api/types";
import { loadLocalChats, loadLocalThreads, saveLocalState, sortChatsNewestFirst } from "../lib/storage";

/** Minimum gap between two activity-stamp writes (drawer ordering only). */
const TOUCH_INTERVAL_MS = 700;
const RETRY_BASE_MS = 400;
const RETRY_MAX_MS = 15000;
/** Quiet gap after the last token that means the response is over. */
const IDLE_FINALIZE_MS = 2500;
/** Debounce window for localStorage writes. */
const SAVE_DEBOUNCE_MS = 400;

export type Chat = {
  /** Coordinator URL these messages belong to (tabs need it for extra reads). */
  base: string;
  messages: ChatMsg[];
  sessions: LocalChat[];
  sessionId: string | null;
  busy: boolean;
  notice: string;
  setNotice: (v: string) => void;
  lastEvent: string;
  streamPhase: StreamPhase;
  title: string;
  model: string;
  /** Send one turn. The composer owns the draft so a tab switch never loses it. */
  send: (text: string) => Promise<void>;
  stop: () => Promise<void>;
  newChat: () => Promise<void>;
  select: (id: string) => void;
  rename: (id: string, title: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  /** Reset all per-coordinator state after the coordinator URL changes. */
  forgetCoordinator: (nextBase: string) => void;
};

export function useChat(base: string, generation: number, reachable: boolean): Chat {
  const [sessions, setSessions] = useState<LocalChat[]>(() => sortChatsNewestFirst(loadLocalChats(base)));
  const [msgsBySession, setMsgsBySession] = useState<Record<string, ChatMsg[]>>(() => stampedThreads(loadLocalThreads(base)));
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [lastEvent, setLastEvent] = useState("");
  const [streamPhase, setStreamPhase] = useState<StreamPhase>("idle");
  const [retryTick, setRetryTick] = useState(0);
  const [models, setModels] = useState<Model[]>([]);

  const esRef = useRef<EventSource | null>(null);
  const busyRef = useRef(false);
  const sessionRef = useRef<string | null>(null);
  const lastEventRef = useRef("");
  const stopControllerRef = useRef<AbortController | null>(null);
  const streamKeyRef = useRef<string | null>(null);
  const streamSeq = useRef(0);
  const idleTimerRef = useRef<number | null>(null);
  const cursorBySession = useRef<Record<string, string>>({});
  const seenIdBySession = useRef<Record<string, number>>({});
  const msgIdsRef = useRef<Record<string, ChatMsg[]>>({});
  const pendingRef = useRef<{ id: string; key: string; text: string } | null>(null);
  const frameRef = useRef<number | null>(null);
  const touchedRef = useRef(0);
  const attemptRef = useRef(0);

  const setBusyBoth = useCallback((v: boolean) => {
    busyRef.current = v;
    setBusy(v);
  }, []);

  const setLastEventBoth = useCallback((v: string) => {
    lastEventRef.current = v;
    setLastEvent(v);
  }, []);

  /** Stamp a session's activity times, rate-limited during a token burst so
   *  the drawer keeps ordering without a state write per token. */
  const touch = useCallback((id: string, force: boolean) => {
    const now = Date.now();
    if (!force && now - touchedRef.current < TOUCH_INTERVAL_MS) return;
    touchedRef.current = now;
    const iso = new Date(now).toISOString();
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, updated_at: iso, last_token_at: iso } : s)));
  }, []);

  /** Replace one session's messages, keeping ids stable. */
  const writeThread = useCallback((id: string, build: (prev: ChatMsg[]) => ChatMsg[] | null) => {
    setMsgsBySession((prev) => {
      const next = build(prev[id] ?? []);
      return next === null ? prev : { ...prev, [id]: next };
    });
  }, []);

  const clearIdle = useCallback(() => {
    if (idleTimerRef.current !== null) {
      window.clearTimeout(idleTimerRef.current);
      idleTimerRef.current = null;
    }
  }, []);

  const flushTokens = useCallback(() => {
    frameRef.current = null;
    const pending = pendingRef.current;
    pendingRef.current = null;
    if (!pending || !pending.text) return;
    const { id, key, text } = pending;
    writeThread(id, (prev) => applyIds(appendToken(prev, key, text), msgIdsRef.current[id] ?? []));
    touch(id, false);
  }, [touch, writeThread]);

  /** End the in-flight response: drop its key (later tokens open a new bubble)
   *  and clear the typing indicator. Never deletes text. */
  const finalizeCurrent = useCallback(
    (id?: string | null) => {
      clearIdle();
      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
      const key = streamKeyRef.current;
      pendingRef.current = null;
      streamKeyRef.current = null;
      setBusyBoth(false);
      if (!key || !id) return;
      writeThread(id, (prev) => {
        if (prev.length === 0) return null;
        return applyIds(finalizeStream(prev, key), msgIdsRef.current[id] ?? []);
      });
      touch(id, true);
    },
    [clearIdle, setBusyBoth, touch, writeThread],
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
      if (!busyRef.current) setBusyBoth(true);
      const pending = pendingRef.current;
      if (pending && pending.id === id && pending.key === key) pending.text += chunk;
      else pendingRef.current = { id, key, text: chunk };
      if (frameRef.current === null) frameRef.current = window.requestAnimationFrame(flushTokens);
      // The server sends no explicit end-of-response (commit is per-token
      // durability, done is stripped from SSE shapes), so an idle gap means
      // the response finished. Any new token cancels this and keeps appending.
      clearIdle();
      idleTimerRef.current = window.setTimeout(() => {
        if (streamKeyRef.current === key) finalizeCurrent(id);
      }, IDLE_FINALIZE_MS);
    },
    [clearIdle, finalizeCurrent, flushTokens, setBusyBoth],
  );

  const attach = useCallback(
    (id: string, resume: string) => {
      esRef.current?.close();
      clearIdle();
      sessionRef.current = id;
      const cursor = resume ? `?last_event=${encodeURIComponent(resume)}` : "";
      const es = new EventSource(`${base}/v1/sessions/${encodeURIComponent(id)}/events${cursor}`);
      es.onopen = () => {
        attemptRef.current = 0;
        setStreamPhase("live");
      };
      es.onerror = () => {
        // A dead stream is a visible state, not a silent notice: the pill
        // shows "reconnecting" and the retry effect resumes from the cursor.
        setStreamPhase("reconnecting");
        setRetryTick((t) => t + 1);
        if (busyRef.current) finalizeCurrent(id);
      };
      // Record the cursor, skipping already-seen ids so re-attaches never
      // duplicate replayed turns. Returns false for a replay duplicate.
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
        if (!noteEvent(e)) return;
        const parsed = parseTokenData((e as MessageEvent).data);
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
            writeThread(id, (prev) => {
              const last = prev[prev.length - 1];
              if (last?.role === "you" && last.text === text) return null;
              return applyIds([...prev, { role: "you", text }], msgIdsRef.current[id] ?? []);
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
      esRef.current = es;
      setStreamPhase("reconnecting");
    },
    [appendChunk, base, clearIdle, finalizeCurrent, setLastEventBoth, writeThread],
  );

  // Retry a dead stream once the coordinator answers again, with backoff.
  useEffect(() => {
    const id = sessionRef.current;
    if (!id || streamPhase !== "reconnecting" || !reachable) return;
    const attempt = Math.min(attemptRef.current, 5);
    attemptRef.current += 1;
    const delay = Math.min(RETRY_MAX_MS, RETRY_BASE_MS * 2 ** attempt);
    const timer = window.setTimeout(() => {
      if (sessionRef.current === id) attach(id, cursorBySession.current[id] ?? "");
    }, delay);
    return () => window.clearTimeout(timer);
  }, [attach, reachable, retryTick, streamPhase]);

  // A manual reconnect (the connection pill) re-attaches from the same cursor.
  useEffect(() => {
    if (generation === 0) return;
    const id = sessionRef.current;
    if (id) attach(id, cursorBySession.current[id] ?? "");
  }, [attach, generation]);

  /** Reset every piece of per-coordinator state and load `nextBase`'s local
   *  chats (never the server's). Called by App right after the URL changes:
   *  an event-driven reset rather than an effect, so the switch never paints
   *  the old coordinator's threads. `nextBase` is passed in because this
   *  closure still points at the previous URL at the moment it is called. */
  const forgetCoordinator = useCallback(
    (nextBase: string) => {
      const stamped = stampedThreads(loadLocalThreads(nextBase));
      setSessions(sortChatsNewestFirst(loadLocalChats(nextBase)));
      setMsgsBySession(stamped);
      msgIdsRef.current = stamped;
      setSessionId(null);
      sessionRef.current = null;
      esRef.current?.close();
      esRef.current = null;
      setBusyBoth(false);
      cursorBySession.current = {};
      seenIdBySession.current = {};
      streamKeyRef.current = null;
      pendingRef.current = null;
      attemptRef.current = 0;
      setStreamPhase("idle");
      setLastEventBoth("");
      setNotice("");
    },
    [setBusyBoth, setLastEventBoth],
  );

  // Debounced persistence: one write per key per burst, not per token.
  useEffect(() => {
    const timer = window.setTimeout(() => saveLocalState(base, sessions, msgsBySession), SAVE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [base, msgsBySession, sessions]);

  useEffect(
    () => () => {
      esRef.current?.close();
      stopControllerRef.current?.abort();
      if (idleTimerRef.current !== null) window.clearTimeout(idleTimerRef.current);
      if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
    },
    [],
  );

  const preferredModel = useCallback((): string => sessions.find((s) => s.id === sessionId)?.model ?? models[0]?.id ?? "", [
    models,
    sessionId,
    sessions,
  ]);

  const loadModels = useCallback(async () => {
    try {
      const res = await fetch(`${base}/api/models`, { cache: "no-store" });
      if (!res.ok) return;
      const d = (await res.json()) as { models?: Model[] };
      setModels(Array.isArray(d.models) ? d.models : []);
    } catch {
      /* catalog unreachable: the composer still sends without a model */
    }
  }, [base]);

  useEffect(() => {
    const kick = window.setTimeout(() => void loadModels(), 0);
    return () => window.clearTimeout(kick);
  }, [loadModels]);

  /** Create a session, adopt it, and return its id. null = notice is set. */
  const createSession = useCallback(
    async (signal?: AbortSignal): Promise<string | null> => {
      const model = preferredModel();
      try {
        const r = await fetch(`${base}/v1/sessions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(model ? { model } : {}),
          signal,
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const s = (await r.json()) as { id?: string; session_id?: string; model?: string; title?: string };
        const id = s.id ?? s.session_id ?? "";
        if (!id) throw new Error("the coordinator returned no session id");
        const now = new Date().toISOString();
        const row: LocalChat = {
          id,
          model: typeof s.model === "string" ? s.model : model || undefined,
          title: typeof s.title === "string" ? s.title : undefined,
          created_at: now,
          updated_at: now,
        };
        setSessions((prev) => (prev.some((p) => p.id === id) ? prev : sortChatsNewestFirst([row, ...prev])));
        setSessionId(id);
        sessionRef.current = id;
        setMsgsBySession((prev) => (prev[id] ? prev : { ...prev, [id]: [] }));
        return id;
      } catch (err) {
        setNotice(`Could not open a chat: ${describe(err)}. Check that dllm serve is running at ${base}.`);
        return null;
      }
    },
    [base, preferredModel],
  );

  const send = useCallback(
    async (textArg: string) => {
      const text = textArg.trim();
      if (!text || busyRef.current) return;
      setBusyBoth(true);
      setNotice("");
      // Seal any previous in-flight bubble before this turn starts.
      if (streamKeyRef.current) finalizeCurrent(sessionRef.current);
      clearIdle();
      const ctrl = new AbortController();
      stopControllerRef.current = ctrl;
      try {
        const target = sessionRef.current ?? (await createSession(ctrl.signal));
        if (!target) {
          setBusyBoth(false);
          return;
        }
        writeThread(target, (prev) => {
          const last = prev[prev.length - 1];
          if (last?.role === "you" && last.text === text) return null;
          return applyIds([...prev, { role: "you", text }], msgIdsRef.current[target] ?? []);
        });
        touch(target, true);
        streamKeyRef.current = null; // fresh bubble for this response
        attach(target, cursorBySession.current[target] ?? "");
        streamSeq.current += 1;
        streamKeyRef.current = `${target}#${Date.now()}#${streamSeq.current}`;
        const r = await fetch(`${base}/v1/sessions/${encodeURIComponent(target)}/messages`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
          signal: ctrl.signal,
        });
        if (!r.ok) throw new Error(`HTTP ${r.status} on POST messages`);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          setNotice("Stopped — partial reply kept.");
        } else {
          setNotice(`Send failed: ${describe(err)}. Check that dllm serve is running at ${base}.`);
        }
        finalizeCurrent(sessionRef.current);
      } finally {
        if (stopControllerRef.current === ctrl) stopControllerRef.current = null;
      }
    },
    [attach, base, clearIdle, createSession, finalizeCurrent, setBusyBoth, touch, writeThread],
  );

  /** Stop generation: abort the in-flight POST, close SSE, tell the server,
   *  finalize the bubble, clear busy. Never deletes text. */
  const stop = useCallback(async () => {
    const id = sessionRef.current;
    stopControllerRef.current?.abort();
    stopControllerRef.current = null;
    esRef.current?.close();
    esRef.current = null;
    setStreamPhase("idle");
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
  }, [base, finalizeCurrent, setBusyBoth]);

  const newChat = useCallback(async () => {
    if (busyRef.current) return;
    finalizeCurrent(sessionRef.current);
    clearIdle();
    streamKeyRef.current = null;
    const id = await createSession();
    if (!id) return;
    setNotice("");
    attach(id, "");
  }, [attach, clearIdle, createSession, finalizeCurrent]);

  const select = useCallback(
    (id: string) => {
      if (streamKeyRef.current) finalizeCurrent(sessionRef.current);
      clearIdle();
      setSessionId(id);
      sessionRef.current = id;
      setMsgsBySession((prev) => (prev[id] ? prev : { ...prev, [id]: [] }));
      streamKeyRef.current = null;
      setBusyBoth(false);
      // Empty local thread + unknown cursor replays the full log, and the
      // status/token handlers rebuild one message per turn.
      attach(id, cursorBySession.current[id] ?? "");
    },
    [attach, clearIdle, finalizeCurrent, setBusyBoth],
  );

  // Rename updates the local row first, then tries the server. A 404/405
  // means a stale coordinator — the local name is still kept.
  const rename = useCallback(
    async (id: string, title: string) => {
      const t = title.trim();
      if (!t) return;
      setSessions((prev) => sortChatsNewestFirst(prev.map((s) => (s.id === id ? { ...s, title: t } : s))));
      try {
        const r = await fetch(`${base}/v1/sessions/${encodeURIComponent(id)}/rename`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: t }),
        });
        if (r.status === 404 || r.status === 405) {
          setNotice("Renamed here — the coordinator needs an upgrade so other devices keep the old name.");
          return;
        }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
      } catch (err) {
        setNotice(`The coordinator refused the rename: ${describe(err)}. The name is kept in this browser.`);
      }
    },
    [base],
  );

  // Delete removes the local row + thread, then tries the server so the
  // coordinator drops the event log too. Unknown sessions still clear locally.
  const remove = useCallback(
    async (id: string) => {
      try {
        const r = await fetch(`${base}/v1/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
        if (r.status === 404 || r.status === 405) {
          setNotice("Deleted here — the chat was already gone on the coordinator.");
        } else if (!r.ok) {
          throw new Error(`HTTP ${r.status}`);
        }
      } catch (err) {
        setNotice(`The coordinator refused the delete: ${describe(err)}. The copy in this browser is removed anyway.`);
      }
      setSessions((prev) => prev.filter((s) => s.id !== id));
      setMsgsBySession((prev) => {
        if (!(id in prev)) return prev;
        const next = { ...prev };
        delete next[id];
        return next;
      });
      delete cursorBySession.current[id];
      delete seenIdBySession.current[id];
      delete msgIdsRef.current[id];
      if (sessionRef.current === id) {
        esRef.current?.close();
        esRef.current = null;
        stopControllerRef.current?.abort();
        stopControllerRef.current = null;
        clearIdle();
        sessionRef.current = null;
        setSessionId(null);
        streamKeyRef.current = null;
        setStreamPhase("idle");
        setBusyBoth(false);
      }
    },
    [base, clearIdle, setBusyBoth],
  );

  const messages = useMemo(() => (sessionId ? (msgsBySession[sessionId] ?? []) : []), [msgsBySession, sessionId]);
  const model = sessions.find((s) => s.id === sessionId)?.model ?? models[0]?.id ?? "";

  return {
    base,
    messages,
    sessions,
    sessionId,
    busy,
    notice,
    setNotice,
    lastEvent,
    streamPhase,
    title: sessionId ? displayTitle(sessionId, { title: sessions.find((s) => s.id === sessionId)?.title, model }) : "New conversation",
    model,
    send,
    stop,
    newChat,
    select,
    rename,
    remove,
    forgetCoordinator,
  };
}

function describe(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}