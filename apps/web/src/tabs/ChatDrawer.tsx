/** Chat list drawer: this browser's local conversations only.
 *
 * All per-row UI state (open menu, rename box, delete confirmation) lives here
 * so the chat hook stays about data, and Escape closes whichever one is open. */

import { useEffect, useState } from "react";
import { displayTitle } from "../chatStream";
import type { LocalChat } from "../api/types";
import type { Chat } from "../state/useChat";
import { relativeTime } from "../lib/format";
import { sortChatsNewestFirst } from "../lib/storage";

export function ChatDrawer({ chat, open, onClose }: { chat: Chat; open: boolean; onClose: () => void }) {
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [usagePopId, setUsagePopId] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setMenuOpenId(null);
      setUsagePopId(null);
      setDeleteConfirmId(null);
      setRenamingId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const closeAll = () => {
    setMenuOpenId(null);
    setUsagePopId(null);
    setDeleteConfirmId(null);
    setRenamingId(null);
  };

  const pick = (id: string) => {
    chat.select(id);
    closeAll();
    onClose();
  };

  const entries: { sid: string; s: LocalChat }[] = sortChatsNewestFirst(chat.sessions).map((s) => ({ sid: s.id, s }));

  return (
    <aside className={`chat-drawer${open ? " open" : ""}`} aria-label="Chat list" aria-hidden={!open}>
      <div className="drawer-head">
        <span>{`Chats (${chat.sessions.length})`}</span>
        <button
          onClick={() => {
            void chat.newChat();
            onClose();
          }}
          disabled={chat.busy}
        >
          New chat
        </button>
      </div>
      <p className="hint">Stored in this browser only, per coordinator. Other devices never see these chats.</p>
      {entries.length === 0 ? (
        <div className="empty">
          <h3>No chats yet</h3>
          <p>Start one with New chat, or send a message and it will appear here.</p>
        </div>
      ) : null}
      {entries.map(({ sid, s }) => {
        const title = displayTitle(sid, { title: s.title, model: s.model ?? chat.model });
        const menuOpen = menuOpenId === sid;
        const seen = relativeTime(s.last_token_at ?? null);
        return (
          <div key={sid} className={`drawer-row${sid === chat.sessionId ? " active" : ""}`}>
            <button className="row-main" onClick={() => pick(sid)} title={sid}>
              <span className="row-title">{title}</span>
              <span className="mono row-sub meta-chips">
                <span>{s.model ?? chat.model ?? "no model"}</span>
                {s.tokens_out !== undefined ? <span>{`${s.tokens_out} tokens out`}</span> : null}
                {seen ? <span>{seen}</span> : null}
              </span>
            </button>
            <button
              className="dots"
              aria-label={`Options for ${title}`}
              aria-expanded={menuOpen}
              onClick={() => {
                closeAll();
                setMenuOpenId(menuOpen ? null : sid);
              }}
            >
              ⋯
            </button>
            {menuOpen ? (
              <div className="menu" role="menu">
                <button
                  onClick={() => {
                    setRenamingId(sid);
                    setRenameDraft(typeof s.title === "string" ? s.title : "");
                    setMenuOpenId(null);
                  }}
                >
                  Rename
                </button>
                <button
                  onClick={() => {
                    setUsagePopId(usagePopId === sid ? null : sid);
                    setMenuOpenId(null);
                  }}
                >
                  Usage
                </button>
                <button
                  className="danger"
                  onClick={() => {
                    setDeleteConfirmId(sid);
                    setMenuOpenId(null);
                  }}
                >
                  Delete
                </button>
              </div>
            ) : null}
            {renamingId === sid ? (
              <form
                className="rename"
                onSubmit={(e) => {
                  e.preventDefault();
                  void chat.rename(sid, renameDraft).then(closeAll);
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
            ) : null}
            {deleteConfirmId === sid ? (
              <div className="confirm">
                <span>Delete this chat here and on the coordinator?</span>
                <button
                  className="danger"
                  onClick={() => {
                    void chat.remove(sid).then(closeAll);
                  }}
                >
                  Confirm delete
                </button>
                <button onClick={() => setDeleteConfirmId(null)}>Cancel</button>
              </div>
            ) : null}
            {usagePopId === sid ? (
              <div className="usage-pop">
                <strong>Usage</strong>
                <p className="mono">Model: {s.model ?? "not reported"}</p>
                <p className="mono">Tokens out: {s.tokens_out ?? "not reported"}</p>
                <p className="mono">
                  Last activity: {s.last_token_at ?? s.updated_at ?? s.created_at ?? "not reported"}
                </p>
              </div>
            ) : null}
          </div>
        );
      })}
    </aside>
  );
}