/** Pure streaming helpers for the chat thread (no React, no DOM).
 *
 * Backend contract (`GET /v1/sessions/{id}/events`): the server emits one
 * `token {"pos","text"}` event plus one `commit {"pos"}` durability mark PER
 * token — `commit` is not end-of-response, and the `done` flag is stripped
 * from SSE shapes. So the thread must keep one pending assistant bubble open
 * per in-flight response (`streamKey`) and only finalize on an explicit end
 * signal (done/complete event, token with done:true, idle timeout, or the
 * next user turn) — never on a bare `commit`.
 */

export type StreamMsg = { role: "you" | "mesh"; text: string; streamKey?: string };

/** Append `text` to this stream's single pending bubble, or open one.
 *
 * Rules:
 * - empty text is a no-op (same array reference back);
 * - a newer "you" message seals older streams: tokens after it always open a
 *   fresh bubble, so replayed history renders one assistant message per turn;
 * - interleaved streams merge into their own bubble (no data loss), at the
 *   position of that stream's latest bubble.
 */
export function appendToken(messages: StreamMsg[], streamKey: string, text: string): StreamMsg[] {
  if (!text) return messages;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === "you") break;
    if (m.role === "mesh" && m.streamKey === streamKey) {
      const next = messages.slice();
      next[i] = { role: "mesh", text: m.text + text, streamKey };
      return next;
    }
  }
  return [...messages, { role: "mesh", text, streamKey }];
}

/** Mark a stream finished: drop its key so later tokens open a new bubble.
 * Streams with no tokens leave the thread untouched (same reference back —
 * a commit with no tokens never creates an empty bubble).
 */
export function finalizeStream(messages: StreamMsg[], streamKey: string): StreamMsg[] {
  let changed = false;
  const next = messages.map((m) => {
    if (m.role === "mesh" && m.streamKey === streamKey) {
      changed = true;
      return { role: m.role, text: m.text };
    }
    return m;
  });
  return changed ? next : messages;
}

/** Drawer title: server `title` when present, else id prefix + model. */
export function displayTitle(id: string, s: { title?: unknown; model?: unknown }): string {
  const t = typeof s.title === "string" ? s.title.trim() : "";
  if (t) return t;
  const m = typeof s.model === "string" && s.model.trim() ? s.model.trim() : "";
  const short = id.length > 12 ? id.slice(0, 12) : id;
  return m ? `${short} · ${m}` : short;
}
