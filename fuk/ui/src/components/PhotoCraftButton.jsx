/**
 * "From PhotoCraft" button for Image tab inputs.
 *
 * mode:
 *   image   – menu: Composite / Active layer / any layer; onPulled(url)
 *   inpaint – composite + mask (selection, else `_mask`); onPulled({image, mask})
 *   mask    – same pull, for the mask slot only; onPulled({image, mask})
 *
 * The layer list is fetched when the menu opens, so nothing polls PhotoCraft
 * while the button sits idle.
 */

import { useEffect, useRef, useState } from 'react';
import {
  getPhotoCraftStatus,
  pullFromPhotoCraft,
  pullInpaintFromPhotoCraft,
} from '../utils/photocraftApi';

export default function PhotoCraftButton({ mode = 'image', onPulled, disabled = false, label }) {
  const [busy, setBusy] = useState(false);
  const [menu, setMenu] = useState(null);     // null = closed; {layers} or {error}
  const [error, setError] = useState(null);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!menu) return;
    const close = (e) => { if (!rootRef.current?.contains(e.target)) setMenu(null); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [menu]);

  const run = async (fn) => {
    setBusy(true);
    setError(null);
    setMenu(null);
    try {
      await fn();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const pullImage = (opts) => run(async () => {
    const res = await pullFromPhotoCraft(opts);
    onPulled?.(res.url, res);
  });

  const handleClick = async () => {
    if (mode !== 'image') {
      run(async () => {
        const res = await pullInpaintFromPhotoCraft({});
        if (mode === 'mask' && !res.mask) {
          throw new Error('No selection and no _mask layer in PhotoCraft');
        }
        onPulled?.(res);
      });
      return;
    }
    if (menu) { setMenu(null); return; }
    setError(null);
    try {
      const status = await getPhotoCraftStatus();
      setMenu(status.reachable
        ? { layers: (status.layers || []).filter(l => !l.is_entity && !l.is_mask), doc: status.documents?.find(d => d.active) }
        : { error: status.error || 'PhotoCraft is not running' });
    } catch (err) {
      setMenu({ error: err.message });
    }
  };

  const text = label || (mode === 'mask' ? 'Mask from PhotoCraft' : 'From PhotoCraft');

  return (
    <div className="fuk-photocraft-button" ref={rootRef}>
      <button
        type="button"
        className="fuk-btn fuk-btn-secondary fuk-btn-full"
        onClick={handleClick}
        disabled={disabled || busy}
        title={mode === 'image'
          ? 'Pull the composite or a layer of the active PhotoCraft document'
          : 'Pull the composite and a mask from the PhotoCraft selection (or a _mask layer)'}
      >
        {busy ? 'Pulling…' : text}
      </button>

      {menu && (
        <div className="fuk-photocraft-menu">
          {menu.error ? (
            <div className="fuk-photocraft-menu-error">{menu.error}</div>
          ) : (
            <>
              {menu.doc && <div className="fuk-photocraft-menu-title">{menu.doc.name}</div>}
              <button type="button" onClick={() => pullImage({ source: 'composite' })}>Composite</button>
              <button type="button" onClick={() => pullImage({ source: 'active' })}>Active layer</button>
              {menu.layers.length > 0 && <div className="fuk-photocraft-menu-sep" />}
              {menu.layers.map(layer => (
                <button
                  key={layer.id}
                  type="button"
                  className={layer.visible ? '' : 'hidden-layer'}
                  onClick={() => pullImage({ source: 'layer', layer: layer.name })}
                >
                  {layer.name}
                </button>
              ))}
            </>
          )}
        </div>
      )}

      {error && <p className="fuk-help-text fuk-photocraft-error">{error}</p>}
    </div>
  );
}
