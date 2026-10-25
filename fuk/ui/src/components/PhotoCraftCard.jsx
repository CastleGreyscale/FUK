/**
 * PhotoCraft card for the Image tab: connection status plus the document-level
 * actions (depth → layer for slap comps, @-layers → EliGen).
 *
 * Polls the bridge every 5 s, only while the tab is active and the page
 * visible. Per-input pulls live on PhotoCraftButton next to each input.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  depthToPhotoCraftLayer,
  getPhotoCraftStatus,
  launchPhotoCraft,
  pullEligenFromPhotoCraft,
} from '../utils/photocraftApi';

const POLL_MS = 5000;

export default function PhotoCraftCard({ active = true, disabled = false, onEligen }) {
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(null);       // label of the running action
  const [message, setMessage] = useState(null); // {kind: 'ok'|'warn'|'error', text}
  const [openAs, setOpenAs] = useState('none');

  const refresh = useCallback(async () => {
    try {
      setStatus(await getPhotoCraftStatus());
    } catch (err) {
      setStatus({ reachable: false, error: err.message });
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    refresh();
    const id = setInterval(() => {
      if (document.visibilityState === 'visible') refresh();
    }, POLL_MS);
    return () => clearInterval(id);
  }, [active, refresh]);

  const run = async (label, fn) => {
    setBusy(label);
    setMessage(null);
    try {
      const res = await fn();
      const warnings = res?.warnings || [];
      setMessage(warnings.length
        ? { kind: 'warn', text: warnings.join(' · ') }
        : { kind: 'ok', text: res?.okText || 'Done' });
      return res;
    } catch (err) {
      setMessage({ kind: 'error', text: err.message });
      return null;
    } finally {
      setBusy(null);
      refresh();
    }
  };

  const doc = status?.documents?.find(d => d.active);
  const entities = (status?.layers || []).filter(l => l.is_entity);
  const reachable = !!status?.reachable;

  const handleLaunch = () => run('Launching…', async () => {
    const res = await launchPhotoCraft();
    return { ...res, okText: res.already_running ? 'Already running' : 'Starting PhotoCraft…' };
  });

  const handleDepth = () => run('Estimating depth…', async () => {
    const res = await depthToPhotoCraftLayer({ open_as: openAs });
    return { ...res, okText: `Added layer “${res.name}”` };
  });

  const handleEligen = () => run('Pulling entities…', async () => {
    const res = await pullEligenFromPhotoCraft({});
    onEligen?.(res);
    return { ...res, okText: `${res.entities.length} entities → EliGen: ${res.entities.map(e => e.prompt).join(', ')}` };
  });

  return (
    <div className="fuk-card fuk-photocraft-card">
      <h3 className="fuk-card-title fuk-mb-3">
        <span className={`fuk-photocraft-dot ${reachable ? 'on' : 'off'}`} />
        PhotoCraft
      </h3>

      {reachable ? (
        <p className="fuk-help-text">
          {doc
            ? <>{doc.name} · {doc.width}×{doc.height} · {status.depth || 8}-bit{status.has_selection ? ' · selection' : ''}</>
            : 'No document open'}
        </p>
      ) : (
        <>
          <p className="fuk-help-text">
            {status?.busy ? 'Busy…' : (status?.error_kind === 'timeout'
              ? 'PhotoCraft is not drawing; bring its window into view.'
              : 'Not connected.')}
          </p>
          {!status?.busy && (
            <button type="button" className="fuk-btn fuk-btn-secondary fuk-btn-full"
                    onClick={handleLaunch} disabled={!!busy}>
              {busy || 'Launch PhotoCraft'}
            </button>
          )}
        </>
      )}

      {reachable && doc && (
        <>
          <div className="fuk-input-inline fuk-mt-2">
            <button type="button" className="fuk-btn fuk-btn-secondary fuk-input--flex-2"
                    onClick={handleDepth} disabled={disabled || !!busy}
                    title="Estimate depth of the composite (Depth Anything 3, near = white) and add it as a layer">
              {busy === 'Estimating depth…' ? busy : 'Depth → layer'}
            </button>
            <select className="fuk-input" value={openAs} onChange={e => setOpenAs(e.target.value)}
                    title="Also open the full-precision depth map as its own document">
              <option value="none">layer only</option>
              <option value="png16">+ 16-bit doc</option>
              <option value="exr">+ 32-bit EXR doc</option>
            </select>
          </div>

          <button type="button" className="fuk-btn fuk-btn-secondary fuk-btn-full fuk-mt-2"
                  onClick={handleEligen} disabled={disabled || !!busy || entities.length === 0}
                  title="Layers named @<prompt> become EliGen entities, top of the Layers panel first">
            {busy === 'Pulling entities…' ? busy : `@ layers → EliGen (${entities.length})`}
          </button>
        </>
      )}

      {message && (
        <p className={`fuk-help-text fuk-photocraft-msg ${message.kind}`}>{message.text}</p>
      )}
    </div>
  );
}
