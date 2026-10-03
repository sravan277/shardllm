/** Stable identity for chat bubbles. Pure, no React.
 *
 * The token stream rewrites one bubble dozens of times per second, so message
 * identity has to survive `appendToken`/`finalizeStream` array surgery. React
 * only needs the key to be *stable*, not unique forever: a slot keeps its id
 * when the role matches and the slot is either the same stream or has just been
 * finalized. Without this, every token would remount the whole bubble list and
 * blow away the native selection and scroll position inside it.
 */

import type { StreamMsg } from "../chatStream";

export type ChatMsg = StreamMsg & { id: string };

let seq = 0;

/** Re-attach ids to `msgs`, reusing the id each slot already had in `prev`. */
export function applyIds(msgs: StreamMsg[], prev: ChatMsg[]): ChatMsg[] {
  return msgs.map((m, i) => {
    const old = prev[i];
    if (old && old.role === m.role && old.text === m.text && old.streamKey === m.streamKey) return old;
    const sameSlot = !!old && old.role === m.role && (old.streamKey === m.streamKey || m.streamKey === undefined);
    return { role: m.role, text: m.text, streamKey: m.streamKey, id: sameSlot ? old.id : newId() };
  });
}

/** Browser-stored threads with fresh ids attached. */
export function stampedThreads(stored: Record<string, StreamMsg[]>): Record<string, ChatMsg[]> {
  const out: Record<string, ChatMsg[]> = {};
  for (const [k, v] of Object.entries(stored)) out[k] = applyIds(v, []);
  return out;
}

function newId(): string {
  seq += 1;
  return `m${seq}`;
}