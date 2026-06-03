/**
 * ConformHDButton
 *
 * Two-pass video conform: re-renders an existing i2v proxy generation through
 * Wan VACE at the original source-still resolution, using the proxy video as
 * the motion control and the source still as the appearance anchor.
 *
 * Mounted in GenerationHistory action bars. Renders null when the generation
 * is not an eligible i2v proxy (so it can be dropped in unconditionally).
 *
 * See: docs/HD_PROXY_VIDEO_SYSTEM.md
 */

import { useState, useEffect, useLayoutEffect, useRef } from 'react';
import { ArrowUp, X } from './Icons';
import { API_URL } from '../utils/constants';

function isConformable(generation) {
  if (!generation || generation.type !== 'video') return false;
  if (generation.subtype === 'hd_conform') return false;
  // The /project/generations endpoint surfaces `model` (the task name) at the
  // top level of the entry — metadata is not included for video gens.
  const model = (generation.model || '').toString().toLowerCase();
  if (!model) return false;
  if (model.includes('vace')) return false;     // already a conform
  if (model.includes('t2v')) return false;      // no source still to anchor on
  return model.includes('i2v') || model.includes('flf2v');
}

export default function ConformHDButton({ generation, variant = 'compact' }) {
  const [open, setOpen] = useState(false);
  const [strength, setStrength] = useState(0.30);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [popStyle, setPopStyle] = useState({ position: 'fixed', left: -9999, top: -9999, visibility: 'hidden' });
  const btnRef = useRef(null);
  const popRef = useRef(null);

  // Position the popover relative to the button each time it opens (and on
  // window resize/scroll while open). Fixed positioning escapes the parent
  // `overflow: hidden` on .gen-history-item and any stacking context in the
  // fullscreen gallery.
  useLayoutEffect(() => {
    if (!open) return;
    const reposition = () => {
      if (!btnRef.current || !popRef.current) return;
      const b = btnRef.current.getBoundingClientRect();
      const p = popRef.current.getBoundingClientRect();
      const margin = 8;
      const vw = window.innerWidth;
      const vh = window.innerHeight;

      // Prefer above-and-right-aligned to the button (matches the visual feel
      // of a per-button menu). Flip below if there isn't room above.
      let top = b.top - p.height - margin;
      if (top < margin) top = b.bottom + margin;
      let left = b.right - p.width;
      left = Math.max(margin, Math.min(left, vw - p.width - margin));
      top = Math.max(margin, Math.min(top, vh - p.height - margin));
      setPopStyle({ position: 'fixed', left, top, visibility: 'visible' });
    };
    // Two-pass: first render places popover offscreen, then we measure and reposition.
    reposition();
    const ro = new ResizeObserver(reposition);
    if (popRef.current) ro.observe(popRef.current);
    window.addEventListener('resize', reposition);
    window.addEventListener('scroll', reposition, true);
    return () => {
      ro.disconnect();
      window.removeEventListener('resize', reposition);
      window.removeEventListener('scroll', reposition, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e) => {
      if (popRef.current && popRef.current.contains(e.target)) return;
      if (btnRef.current && btnRef.current.contains(e.target)) return;
      setOpen(false);
    };
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  // Reset offscreen each time we close so the next open re-measures cleanly.
  useEffect(() => {
    if (!open) setPopStyle({ position: 'fixed', left: -9999, top: -9999, visibility: 'hidden' });
  }, [open]);

  if (!isConformable(generation)) return null;

  const runConform = async (e) => {
    e.stopPropagation();
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/video/hd_conform`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_id: generation.id,
          denoising_strength: strength,
        }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setOpen(false);
      // History panel listens for this event to refresh; the queued gen will
      // appear once it completes and writes its metadata.json.
      window.dispatchEvent(new CustomEvent('fuk-generation-complete', {
        detail: { source: 'hd_conform', generation_id: data.generation_id },
      }));
    } catch (err) {
      setError(err.message || 'Conform failed');
    } finally {
      setBusy(false);
    }
  };

  const isLarge = variant === 'large';
  const btnClass = isLarge
    ? 'gen-history-conform-hd large'
    : 'gen-history-conform-hd';

  return (
    <div className="gen-history-conform-hd-wrap">
      <button
        ref={btnRef}
        className={btnClass}
        onClick={(e) => { e.stopPropagation(); setOpen(o => !o); }}
        title="Conform HD — re-render this proxy at source resolution via VACE"
        disabled={busy}
      >
        <ArrowUp />
        {isLarge && <span>Conform HD</span>}
      </button>
      {open && (
        <div
          ref={popRef}
          className="gen-history-conform-hd-popover"
          style={popStyle}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="conform-hd-header">
            <span>Conform HD</span>
            <button className="conform-hd-close" onClick={() => setOpen(false)} aria-label="Close">
              <X />
            </button>
          </div>
          <div className="conform-hd-row">
            <label htmlFor="conform-hd-strength">Detail strength</label>
            <span className="conform-hd-value">{strength.toFixed(2)}</span>
          </div>
          <input
            id="conform-hd-strength"
            type="range"
            min="0.15"
            max="0.50"
            step="0.01"
            value={strength}
            onChange={(e) => setStrength(parseFloat(e.target.value))}
            disabled={busy}
          />
          <div className="conform-hd-hint">
            0.20 texture-only · 0.30 default · 0.40+ more freedom
          </div>
          {error && <div className="conform-hd-error">{error}</div>}
          <button
            className="conform-hd-run"
            onClick={runConform}
            disabled={busy}
          >
            {busy ? 'Queueing…' : 'Run conform'}
          </button>
        </div>
      )}
    </div>
  );
}
