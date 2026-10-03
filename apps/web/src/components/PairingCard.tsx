/** Pairing card: the QR, the plain-text URI, and every candidate address.
 *
 * The QR is encoded locally with the `qrcode` package into a data URL — no CDN,
 * so this works on a LAN with no internet. The URI stays visible as text so a
 * device without a camera, or one whose camera will not focus, still has a way in.
 */

import { memo, useEffect, useState } from "react";
import * as QRCode from "qrcode";
import type { PairingUri } from "../api/types";
import { NotMeasured } from "./EmptyState";

export const PairingCard = memo(function PairingCard({ pairing, base }: { pairing: PairingUri | null; base: string }) {
  // Keyed by uri so the rendered image is derived from state, never cleared by
  // an effect: a changed uri simply no longer matches and shows nothing until
  // its own encode resolves.
  const [encoded, setEncoded] = useState<{ uri: string; url: string } | null>(null);
  const uri = pairing?.uri ?? null;

  useEffect(() => {
    if (!uri) return;
    let live = true;
    QRCode.toDataURL(uri, { width: 220, margin: 1 })
      .then((url) => {
        if (live) setEncoded({ uri, url });
      })
      .catch(() => {
        /* the plain-text URI below stays the fallback */
      });
    return () => {
      live = false;
    };
  }, [uri]);

  const qr = encoded && encoded.uri === uri ? encoded.url : null;

  if (!pairing) {
    return <p className="hint">No pairing code available — this coordinator does not answer /api/pairing-uri.</p>;
  }
  if (!uri) {
    return <p className="hint">The coordinator returned no pairing address. Check its LAN interface, then refresh.</p>;
  }

  return (
    <div className="pairing">
      <div className="pairing-code">
        {qr ? (
          <img className="qr" src={qr} alt={`Pairing QR code: ${uri}`} width={220} height={220} />
        ) : (
          <NotMeasured what="The QR image" />
        )}
      </div>
      <div className="pairing-detail">
        <h3>Pair a device</h3>
        <p className="hint">Open this in the other device's app to join this mesh.</p>
        <p className="mono pair-uri">{uri}</p>
        <button
          className="secondary"
          onClick={() => {
            void navigator.clipboard?.writeText(uri).catch(() => {
              /* clipboard blocked: the text above stays selectable */
            });
          }}
        >
          Copy address
        </button>
        {pairing.candidates.length > 0 ? (
          <>
            <h3 className="sub">Every address this machine offers</h3>
            <ul className="mono candidate-list">
              {pairing.candidates.map((c) => (
                <li key={c}>{`${c} (the code above uses ${pairing.uri?.includes(c) ? "this one" : "a fallback"})`}</li>
              ))}
            </ul>
          </>
        ) : null}
        {pairing.warning ? <p className="hint warn">{pairing.warning}</p> : null}
        <p className="mono">Serving at {base}</p>
      </div>
    </div>
  );
});