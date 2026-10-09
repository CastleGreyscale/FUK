/**
 * MaskPainter
 * Modal for painting an inpaint mask over an image.
 *
 * The mask is painted at the image's own resolution on a canvas stacked over
 * it, then saved to the project cache as a white-on-black PNG
 * (white = repaint, black = keep). Everything else in the UI passes paths
 * around, so the saved mask comes back as a path like any other input.
 *
 * Strokes are kept as a list of operations and replayed on undo, which costs
 * nothing in memory however large the image is.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { X } from './Icons';
import { buildImageUrl, API_URL } from '../utils/constants';

// Colour the mask is shown in. It is painted opaque and faded with CSS, so
// overlapping strokes do not build up darker patches.
const TINT = '#ff3b3b';
const MIN_BRUSH = 2;

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new window.Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not load ${url}`));
    img.src = url;
  });
}

// Turn a saved white-on-black mask back into a tinted layer whose alpha is the
// mask, which is the form the paint canvas works in.
function maskToLayer(maskImg, width, height) {
  const layer = document.createElement('canvas');
  layer.width = width;
  layer.height = height;
  const ctx = layer.getContext('2d');
  ctx.drawImage(maskImg, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height);
  const d = pixels.data;
  const r = parseInt(TINT.slice(1, 3), 16);
  const g = parseInt(TINT.slice(3, 5), 16);
  const b = parseInt(TINT.slice(5, 7), 16);
  for (let i = 0; i < d.length; i += 4) {
    d[i + 3] = Math.max(d[i], d[i + 1], d[i + 2]);
    d[i] = r;
    d[i + 1] = g;
    d[i + 2] = b;
  }
  ctx.putImageData(pixels, 0, 0);
  return layer;
}

function strokeSegment(ctx, op, from, to) {
  ctx.globalCompositeOperation = op.erase ? 'destination-out' : 'source-over';
  ctx.strokeStyle = TINT;
  ctx.fillStyle = TINT;
  ctx.lineWidth = op.size;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  if (from.x === to.x && from.y === to.y) {
    ctx.beginPath();
    ctx.arc(from.x, from.y, op.size / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

function applyOp(ctx, op, width, height) {
  if (op.type === 'stroke') {
    const pts = op.points;
    strokeSegment(ctx, op, pts[0], pts[0]);
    for (let i = 1; i < pts.length; i++) strokeSegment(ctx, op, pts[i - 1], pts[i]);
  } else if (op.type === 'clear') {
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, width, height);
  } else if (op.type === 'fill') {
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = TINT;
    ctx.fillRect(0, 0, width, height);
  } else if (op.type === 'invert') {
    // xor against a full opaque fill swaps painted and unpainted.
    ctx.globalCompositeOperation = 'xor';
    ctx.fillStyle = TINT;
    ctx.fillRect(0, 0, width, height);
  }
  ctx.globalCompositeOperation = 'source-over';
}

export default function MaskPainter({ imagePath, maskPath = null, onSave, onClose }) {
  const canvasRef = useRef(null);
  const baseLayerRef = useRef(null);   // mask loaded on open, under all ops
  const opsRef = useRef([]);
  const strokeRef = useRef(null);      // stroke in progress

  const [size, setSize] = useState(null);       // image dimensions once loaded
  const [brush, setBrush] = useState(64);
  const [erase, setErase] = useState(false);
  const [opCount, setOpCount] = useState(0);
  const [cursor, setCursor] = useState(null);   // {x, y, d} in CSS px within the frame
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const imageUrl = buildImageUrl(imagePath);
  const maxBrush = size ? Math.round(Math.max(size.width, size.height) / 2) : 512;

  // Load the image for its dimensions, and the existing mask if there is one.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const img = await loadImage(imageUrl);
        if (cancelled) return;
        const width = img.naturalWidth;
        const height = img.naturalHeight;
        if (maskPath) {
          try {
            const maskImg = await loadImage(buildImageUrl(maskPath));
            if (cancelled) return;
            baseLayerRef.current = maskToLayer(maskImg, width, height);
          } catch (err) {
            console.warn('[MaskPainter] Existing mask could not be loaded:', err);
          }
        }
        setBrush(Math.max(MIN_BRUSH, Math.round(Math.max(width, height) / 20)));
        setSize({ width, height });
      } catch (err) {
        if (!cancelled) setError('Could not load the image to paint on.');
      }
    })();
    return () => { cancelled = true; };
  }, [imageUrl, maskPath]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (baseLayerRef.current) ctx.drawImage(baseLayerRef.current, 0, 0);
    for (const op of opsRef.current) applyOp(ctx, op, canvas.width, canvas.height);
  }, []);

  // The canvas is sized from `size`, which also wipes it — draw once it exists.
  useEffect(() => {
    if (size) redraw();
  }, [size, redraw]);

  const pushOp = useCallback((op) => {
    opsRef.current.push(op);
    setOpCount(opsRef.current.length);
  }, []);

  const runOp = useCallback((type) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const op = { type };
    applyOp(canvas.getContext('2d'), op, canvas.width, canvas.height);
    pushOp(op);
  }, [pushOp]);

  const undo = useCallback(() => {
    if (strokeRef.current || !opsRef.current.length) return;
    opsRef.current.pop();
    setOpCount(opsRef.current.length);
    redraw();
  }, [redraw]);

  // Pointer position in image pixels, plus what the cursor ring needs.
  const locate = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scale = canvas.width / rect.width;
    return {
      point: { x: (e.clientX - rect.left) * scale, y: (e.clientY - rect.top) * scale },
      css: { x: e.clientX - rect.left, y: e.clientY - rect.top, scale },
    };
  };

  const handlePointerDown = (e) => {
    if (e.button !== 0 || saving) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    canvas.setPointerCapture(e.pointerId);
    const { point } = locate(e);
    const op = { type: 'stroke', erase, size: brush, points: [point] };
    strokeRef.current = op;
    strokeSegment(canvas.getContext('2d'), op, point, point);
  };

  const handlePointerMove = (e) => {
    const { point, css } = locate(e);
    setCursor({ x: css.x, y: css.y, d: brush / css.scale });
    const op = strokeRef.current;
    if (!op) return;
    const last = op.points[op.points.length - 1];
    strokeSegment(canvasRef.current.getContext('2d'), op, last, point);
    op.points.push(point);
  };

  const endStroke = () => {
    const op = strokeRef.current;
    if (!op) return;
    strokeRef.current = null;
    canvasRef.current.getContext('2d').globalCompositeOperation = 'source-over';
    pushOp(op);
  };

  const requestClose = useCallback(() => {
    if (saving) return;
    if (opsRef.current.length && !window.confirm('Discard the changes to this mask?')) return;
    onClose();
  }, [saving, onClose]);

  const handleSave = async () => {
    const canvas = canvasRef.current;
    if (!canvas || saving) return;
    setSaving(true);
    setError(null);
    try {
      // Flatten the tinted layer to white, then put black behind it.
      const out = document.createElement('canvas');
      out.width = canvas.width;
      out.height = canvas.height;
      const ctx = out.getContext('2d');
      ctx.drawImage(canvas, 0, 0);
      ctx.globalCompositeOperation = 'source-in';
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, out.width, out.height);
      ctx.globalCompositeOperation = 'destination-over';
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, out.width, out.height);

      const res = await fetch(`${API_URL}/image/mask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: out.toDataURL('image/png') }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !data.success) {
        throw new Error(data.detail || `Save failed (${res.status})`);
      }
      onSave(data.output_url);
    } catch (err) {
      console.error('[MaskPainter] Save failed:', err);
      setError(err.message || 'Could not save the mask.');
      setSaving(false);
    }
  };

  // Keys are taken in the capture phase and stopped there, so the app's own
  // shortcuts (tab switching, generate, cancel) stay quiet while this is open.
  useEffect(() => {
    const onKey = (e) => {
      e.stopPropagation();
      if (e.target?.tagName === 'INPUT' && e.target.type === 'number') return;
      const ctrl = e.ctrlKey || e.metaKey;
      if (e.key === 'Escape') { requestClose(); return; }
      if (ctrl && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
      if (ctrl) return;
      if (e.key === 'b') setErase(false);
      else if (e.key === 'e') setErase(true);
      else if (e.key === 'x') setErase(v => !v);
      else if (e.key === '[') setBrush(b => Math.max(MIN_BRUSH, Math.round(b / 1.25)));
      else if (e.key === ']') setBrush(b => Math.min(maxBrush, Math.max(b + 1, Math.round(b * 1.25))));
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [requestClose, undo, maxBrush]);

  return (
    <div className="mask-painter-overlay">
      <div className="mask-painter">
        <div className="mask-painter-header">
          <span className="mask-painter-title">Paint Mask</span>
          <button className="mask-painter-close" onClick={requestClose} title="Close (Esc)">
            <X style={{ width: '1.25rem', height: '1.25rem' }} />
          </button>
        </div>

        <div className="mask-painter-toolbar">
          <div className="fuk-btn-group">
            <button
              type="button"
              className={`fuk-btn fuk-btn-sm ${erase ? 'fuk-btn-secondary' : 'fuk-btn-primary'}`}
              onClick={() => setErase(false)}
              title="Brush (B)"
            >Brush</button>
            <button
              type="button"
              className={`fuk-btn fuk-btn-sm ${erase ? 'fuk-btn-primary' : 'fuk-btn-secondary'}`}
              onClick={() => setErase(true)}
              title="Eraser (E) — X swaps between the two"
            >Eraser</button>
          </div>

          <label className="mask-painter-size" title="Brush size in image pixels ( [ and ] )">
            <span className="fuk-label">Size</span>
            <input
              type="range"
              className="fuk-slider"
              value={brush}
              min={MIN_BRUSH}
              max={maxBrush}
              step={1}
              onChange={(e) => setBrush(parseInt(e.target.value))}
            />
            <input
              type="number"
              className="fuk-input fuk-input--w-80"
              value={brush}
              min={MIN_BRUSH}
              max={maxBrush}
              onChange={(e) => {
                const v = parseInt(e.target.value);
                if (!Number.isNaN(v)) setBrush(Math.min(maxBrush, Math.max(MIN_BRUSH, v)));
              }}
            />
          </label>

          <div className="fuk-btn-group">
            <button type="button" className="fuk-btn fuk-btn-secondary fuk-btn-sm"
              onClick={undo} disabled={!opCount} title="Undo (Ctrl+Z)">Undo</button>
            <button type="button" className="fuk-btn fuk-btn-secondary fuk-btn-sm"
              onClick={() => runOp('invert')} disabled={!size}
              title="Swap the painted and unpainted areas">Invert</button>
            <button type="button" className="fuk-btn fuk-btn-secondary fuk-btn-sm"
              onClick={() => runOp('fill')} disabled={!size}
              title="Mask the whole image">Fill</button>
            <button type="button" className="fuk-btn fuk-btn-secondary fuk-btn-sm"
              onClick={() => runOp('clear')} disabled={!size}
              title="Remove the whole mask">Clear</button>
          </div>
        </div>

        <div className="mask-painter-stage">
          {size ? (
            <div className="mask-painter-frame">
              <img src={imageUrl} alt="" className="mask-painter-image" draggable={false} />
              <canvas
                ref={canvasRef}
                width={size.width}
                height={size.height}
                className="mask-painter-canvas"
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={endStroke}
                onPointerCancel={endStroke}
                onPointerLeave={() => setCursor(null)}
              />
              {cursor && (
                <div
                  className={`mask-painter-cursor ${erase ? 'mask-painter-cursor--erase' : ''}`}
                  style={{ left: cursor.x, top: cursor.y, width: cursor.d, height: cursor.d }}
                />
              )}
            </div>
          ) : (
            <p className="fuk-help-text">{error ? '' : 'Loading image…'}</p>
          )}
        </div>

        <div className="mask-painter-footer">
          <span className="mask-painter-hint">
            {error
              ? <span className="mask-painter-error">{error}</span>
              : <>Red is repainted, the rest is kept.{size && ` ${size.width}×${size.height}`}</>}
          </span>
          <div className="fuk-btn-group">
            <button type="button" className="fuk-btn fuk-btn-secondary" onClick={requestClose} disabled={saving}>
              Cancel
            </button>
            <button type="button" className="fuk-btn fuk-btn-primary" onClick={handleSave} disabled={!size || saving}>
              {saving ? 'Saving…' : 'Save Mask'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
