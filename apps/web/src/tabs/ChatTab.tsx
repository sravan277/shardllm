/** Chat tab: the thread, the composer, and an inline peek at the layer split.
 *
 * The bubble list is keyed by stable message ids and each bubble is memoized,
 * so a token update re-renders one bubble instead of the whole tree. Autoscroll
 * is throttled to one jump per 120 ms and only while the reader is already at
 * the bottom, so streaming never yanks the view away from someone reading up.
 */

import { memo, useEffect, useRef, useState } from "react";
import type { Chat } from "../state/useChat";
import type { ChatMsg } from "../lib/messages";
import { useDistribution } from "../state/useDistribution";
import { stripRuns } from "../lib/strip";
import { LayerStrip } from "../components/LayerStrip";
import { ChatDrawer } from "./ChatDrawer";

const SCROLL_THROTTLE_MS = 120;
const BOTTOM_SLACK_PX = 80;

export function ChatTab({ chat, draft, onDraft }: { chat: Chat; draft: string; onDraft: (v: string) => void }) {
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [splitOpen, setSplitOpen] = useState(false);
  const threadRef = useRef<HTMLDivElement | null>(null);
  const stickToBottomRef = useRef(true);
  const scrollTimerRef = useRef<number | null>(null);

  const split = useDistribution(chat.base, splitOpen, 0);
  const runs = stripRuns(split.plan?.stages ?? null, split.nameFor);

  const onThreadScroll = () => {
    const el = threadRef.current;
    if (!el) return;
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < BOTTOM_SLACK_PX;
  };

  // Throttle, do not debounce: the timer is never restarted, so a continuous
  // stream still scrolls once per window instead of starving until it stops.
  useEffect(() => {
    if (scrollTimerRef.current !== null) return;
    scrollTimerRef.current = window.setTimeout(() => {
      scrollTimerRef.current = null;
      const el = threadRef.current;
      if (el && stickToBottomRef.current) el.scrollTo({ top: el.scrollHeight });
    }, SCROLL_THROTTLE_MS);
  }, [chat.messages, chat.busy, chat.sessionId]);

  useEffect(
    () => () => {
      if (scrollTimerRef.current !== null) window.clearTimeout(scrollTimerRef.current);
    },
    [],
  );

  return (
    <main className="chatgpt">
      <section className="thread-col" aria-label="Conversation">
        <div className="thread-head">
          <div className="thread-id">
            <strong>{chat.title}</strong>
            <span className="mono meta-chips">
              <span>{chat.model || "no model catalog yet"}</span>
              {chat.sessionId ? <span>{`session ${chat.sessionId.slice(0, 12)}`}</span> : null}
              {chat.lastEvent ? <span>{`resuming from event ${chat.lastEvent}`}</span> : null}
            </span>
          </div>
          <div className="row head-actions">
            <button
              className={splitOpen ? "ghost on" : "ghost"}
              onClick={() => setSplitOpen((v) => !v)}
              aria-expanded={splitOpen}
            >
              Model split
            </button>
            <button className="ghost" onClick={() => setDrawerOpen((v) => !v)} aria-expanded={drawerOpen} aria-label="Toggle chat list">
              Chats
            </button>
          </div>
        </div>

        {splitOpen ? (
          <div className="layer-panel">
            <h2>How this model is split</h2>
            {runs.runs.length > 0 ? (
              <LayerStrip runs={runs.runs} unassigned={runs.unassigned} singleDevice={runs.singleDevice} />
            ) : (
              <p className="hint">
                {split.error ?? "No plan reported yet — the coordinator needs GET /v1/plan. The Distribution tab has the full split."}
              </p>
            )}
          </div>
        ) : null}

        <div
          className="thread"
          ref={threadRef}
          onScroll={onThreadScroll}
          role="log"
          aria-live="polite"
          aria-label="Messages"
        >
          {chat.messages.length === 0 && !chat.busy ? (
            <div className="empty">
              <h2>Start a conversation</h2>
              <p>Talk to your own mesh. Start dllm serve, then send a message; tokens stream in here over local SSE.</p>
            </div>
          ) : null}
          {chat.messages.map((m) => (
            <MessageBubble key={m.id} msg={m} />
          ))}
          {chat.busy ? <div className="typing">meshing</div> : null}
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            if (draft.trim()) {
              void chat.send(draft);
              onDraft("");
            }
          }}
        >
          <input
            value={draft}
            onChange={(e) => onDraft(e.target.value)}
            placeholder="Message the mesh"
            aria-label="Message the mesh"
          />
          {chat.busy ? (
            <button type="button" onClick={() => void chat.stop()}>
              Stop
            </button>
          ) : (
            <button type="submit" disabled={!draft.trim()}>
              Send
            </button>
          )}
        </form>
        <p className="hint thread-meta">
          {`${chat.sessions.length} chat${chat.sessions.length === 1 ? "" : "s"} stored in this browser`}
        </p>
      </section>

      {drawerOpen ? <div className="drawer-backdrop" onClick={() => setDrawerOpen(false)} aria-hidden /> : null}
      <ChatDrawer chat={chat} open={drawerOpen} onClose={() => setDrawerOpen(false)} />
    </main>
  );
}

const MessageBubble = memo(function MessageBubble({ msg }: { msg: ChatMsg }) {
  return <div className={`bubble ${msg.role}`}>{msg.text}</div>;
});