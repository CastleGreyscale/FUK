/**
 * CompareImage Component
 *
 * A/B "before / after" comparison viewer.
 * - Two images are layered in the same box. Image B (the "after") is revealed
 *   on the left of a draggable vertical bar; image A (the "before") shows on the
 *   right. Drag the bar across to wipe between them.
 * - Zoom mirrors ZoomableImage's mechanic: scroll wheel zooms, moving the mouse
 *   pans the view. Both images share ONE transform so they stay pixel-aligned
 *   while you zoom/pan — the only thing that differs between them is the wipe clip.
 *
 * The wipe clip lives on a non-transformed wrapper so the divider stays a fixed
 * vertical line on screen while the images zoom/pan underneath it.
 */

import { useState, useRef, useCallback } from 'react';

export default function CompareImage({
  srcA,
  srcB,
  labelA = 'A',
  labelB = 'B',
  minZoom = 1,
  maxZoom = 6,
}) {
  const [zoom, setZoom] = useState(1);
  const [origin, setOrigin] = useState({ x: 50, y: 50 });
  const [dividePct, setDividePct] = useState(50);
  const containerRef = useRef(null);
  const draggingDivider = useRef(false);

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // Moving the mouse pans the zoomed view (unless we're dragging the divider,
  // in which case the window listeners below own the mouse).
  const handleMouseMove = useCallback((e) => {
    if (draggingDivider.current) return;
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setOrigin({ x: clamp(x, 0, 100), y: clamp(y, 0, 100) });
  }, []);

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    const delta = e.deltaY > 0 ? -0.5 : 0.5;
    setZoom(prev => clamp(prev + delta, minZoom, maxZoom));
  }, [minZoom, maxZoom]);

  // Grab the divider handle and wipe across. Uses window listeners so the drag
  // keeps tracking even if the cursor leaves the image box.
  const handleDividerDown = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    draggingDivider.current = true;
    const onMove = (ev) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const pct = ((ev.clientX - rect.left) / rect.width) * 100;
      setDividePct(clamp(pct, 0, 100));
    };
    const onUp = () => {
      draggingDivider.current = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, []);

  // Shared transform — identical for both images so they stay aligned.
  const imgStyle = {
    transform: `scale(${zoom})`,
    transformOrigin: `${origin.x}% ${origin.y}%`,
  };

  return (
    <div
      ref={containerRef}
      className="compare-container"
      onMouseMove={handleMouseMove}
      onWheel={handleWheel}
    >
      {/* before (right side) */}
      <img className="compare-layer" src={srcA} alt={labelA} style={imgStyle} draggable={false} />

      {/* after (left side) — clipped by the divider position */}
      <div className="compare-clip" style={{ clipPath: `inset(0 ${100 - dividePct}% 0 0)` }}>
        <img className="compare-layer" src={srcB} alt={labelB} style={imgStyle} draggable={false} />
      </div>

      {/* divider bar + grab handle */}
      <div
        className="compare-divider"
        style={{ left: `${dividePct}%` }}
        onMouseDown={handleDividerDown}
        title="Drag to wipe between A and B"
      >
        <div className="compare-divider-handle">
          <span>‹</span>
          <span>›</span>
        </div>
      </div>

      {/* corner labels */}
      <div className="compare-label compare-label--b" style={{ opacity: dividePct > 6 ? 1 : 0.15 }}>{labelB}</div>
      <div className="compare-label compare-label--a" style={{ opacity: dividePct < 94 ? 1 : 0.15 }}>{labelA}</div>

      <div className="zoomable-indicator">{zoom.toFixed(1)}x</div>
    </div>
  );
}
