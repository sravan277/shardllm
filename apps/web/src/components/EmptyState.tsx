/** Empty states that invite one concrete next action. Never a bare "no data". */

import type { ReactNode } from "react";

export function EmptyState({ title, body, children }: { title: string; body: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      <p>{body}</p>
      {children ? <div className="row empty-actions">{children}</div> : null}
    </div>
  );
}

/** "Not measured yet" — the honest replacement for a number we do not have. */
export function NotMeasured({ what }: { what: string }) {
  return <p className="hint">{what} not measured yet — the coordinator has not reported a sample.</p>;
}