/** Panel wrapper: a titled card with an optional action slot. Keeps every tab
 *  on the same surface rhythm without repeating the markup. */

import type { ReactNode } from "react";

export function Section({
  title,
  hint,
  action,
  children,
  className,
}: {
  title: string;
  hint?: ReactNode;
  action?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`card${className ? ` ${className}` : ""}`}>
      <div className="fleet-head">
        <div>
          <h2>{title}</h2>
          {hint ? <p className="hint">{hint}</p> : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}