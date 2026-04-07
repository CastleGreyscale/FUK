/**
 * Utilities Tab
 * Houses miscellaneous tools that don't belong in the main pipeline.
 * Sub-tabs: Spec Tool, (more coming)
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import Footer from '../components/Footer';
import LoraDatasetBuilder from './LoraDatasetBuilder';

// ============================================================================
// Helpers
// ============================================================================

const snapTo16  = (v) => Math.ceil(v / 16) * 16;
const snapTo4m1 = (v) => Math.ceil((Math.max(1, v) - 1) / 4) * 4 + 1;

const hexToRgba = (hex, alpha) => {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
};

// ============================================================================
// Data
// ============================================================================

const ASPECT_RATIOS = [
  { label: '1:1   — Square', w: 1, h: 1 },
  { label: '4:3   — Academy', w: 4, h: 3 },
  { label: '16:9  — Widescreen', w: 16, h: 9 },
  { label: '1.85:1 — Cinema flat', w: 185,  h: 100 },
  { label: '2.35:1 — Widescreen cinema', w: 235, h: 100 }, 
  { label: '2.39:1 — Anamorphic cinema', w: 239, h: 100 },
  { label: '21:9  — Ultra-wide', w: 21, h: 9 },
  { label: 'Custom', w: null, h: null },
];

const PRESET_RESOLUTIONS = [
  { label: '512',    width: 512  },
  { label: '1024',   width: 1024 },
  { label: '720p',   width: 1280 },
  { label: '1080p',  width: 1920 },
  { label: '2K DCI', width: 2048 },
  { label: '1440p',  width: 2560 },
  { label: '4K UHD', width: 3840 },
  { label: '4K',     width: 4096 },
];

const COMPOSITION_GUIDES = [
  { key: 'thirds',     label: 'Rule of Thirds' },
  { key: 'golden',     label: 'Golden Spiral' },
  { key: 'vp',         label: 'Vanishing Points (RoT)' },
  { key: 'actionSafe', label: 'Action Safe (90%)' },
  { key: 'titleSafe',  label: 'Title Safe (80%)' },
  { key: 'center',     label: 'Center Cross' },
];

// ============================================================================
// Golden spiral helper
// ============================================================================

function drawGoldenSpiral(ctx, W, H, gc) {
  const phi = 1.618034;

  // Inscribe a phi:1 golden rectangle in the canvas, centered
  let gw, gh, gx, gy;
  if (W / H >= phi) {
    gh = H; gw = H * phi; gx = (W - gw) / 2; gy = 0;
  } else {
    gw = W; gh = W / phi; gx = 0; gy = (H - gh) / 2;
  }

  ctx.strokeStyle = gc;
  ctx.setLineDash([]);

  const ITERS = 8;

  // Shared subdivision walker
  const walk = (fn) => {
    let x1=gx, y1=gy, x2=gx+gw, y2=gy+gh;
    for (let i = 0; i < ITERS; i++) {
      const sq = Math.min(x2-x1, y2-y1);
      if (sq < 1.5) break;
      fn(i, x1, y1, x2, y2, sq);
      switch (i % 4) {
        case 0: x1 += sq; break;
        case 1: y1 += sq; break;
        case 2: x2 -= sq; break;
        case 3: y2 -= sq; break;
      }
    }
  };

  // Subdivision lines
  walk((i, x1, y1, x2, y2, sq) => {
    ctx.beginPath();
    switch (i % 4) {
      case 0: ctx.moveTo(x1+sq, y1); ctx.lineTo(x1+sq, y2); break;
      case 1: ctx.moveTo(x1, y1+sq); ctx.lineTo(x2, y1+sq); break;
      case 2: ctx.moveTo(x2-sq, y1); ctx.lineTo(x2-sq, y2); break;
      case 3: ctx.moveTo(x1, y2-sq); ctx.lineTo(x2, y2-sq); break;
    }
    ctx.stroke();
  });

  // Spiral arcs — each is a quarter-circle inscribed in its square
  // Arc centers and angles for each cut direction:
  //   case 0 (cut left):   center = bottom-right of left square  → from -PI/2 to PI  (ccw)
  //   case 1 (cut top):    center = bottom-left  of top  square  → from 0     to -PI/2 (ccw)
  //   case 2 (cut right):  center = top-left     of right square → from PI/2  to 0    (ccw)
  //   case 3 (cut bottom): center = top-right    of bot  square  → from PI    to PI/2  (ccw)
  walk((i, x1, y1, x2, y2, sq) => {
    let cx, cy, sa, ea;
    switch (i % 4) {
      case 0: cx=x1+sq; cy=y2;    sa=-Math.PI/2;    ea=Math.PI;       break;
      case 1: cx=x1;    cy=y1+sq; sa=0;              ea=-Math.PI/2;    break;
      case 2: cx=x2-sq; cy=y1;    sa=Math.PI/2;      ea=0;             break;
      case 3: cx=x2;    cy=y2-sq; sa=Math.PI;        ea=Math.PI/2;     break;
    }
    ctx.beginPath();
    ctx.moveTo(cx + sq * Math.cos(sa), cy + sq * Math.sin(sa));
    ctx.arc(cx, cy, sq, sa, ea, true); // counterclockwise
    ctx.stroke();
  });
}

// ============================================================================
// Canvas drawing
// ============================================================================

function drawGuides(canvas, width, height, { guides, guideColor, bgColor, bgAlpha, frameCount }) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = hexToRgba(bgColor, bgAlpha / 100);
  ctx.fillRect(0, 0, width, height);

  const gc      = hexToRgba(guideColor, 0.7);
  const gcFaint = hexToRgba(guideColor, 0.35);
  ctx.lineWidth = Math.max(1, Math.round(width / 800));
  ctx.setLineDash([]);

  if (guides.thirds) {
    ctx.strokeStyle = gc;
    for (let i = 1; i <= 2; i++) {
      const x = Math.round((width  / 3) * i);
      const y = Math.round((height / 3) * i);
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y);  ctx.stroke();
    }
  }

  if (guides.golden) {
    drawGoldenSpiral(ctx, width, height, gc);
    ctx.lineWidth = Math.max(1, Math.round(width / 800)); // restore after spiral resets it
  }

  // Vanishing points from rule-of-thirds intersections → lines to frame corners
  if (guides.vp) {
    ctx.strokeStyle = hexToRgba(guideColor, 0.28);
    ctx.setLineDash([]);
    const corners = [[0,0],[width,0],[0,height],[width,height]];
    const pts = [
      [width/3, height/3],   [2*width/3, height/3],
      [width/3, 2*height/3], [2*width/3, 2*height/3],
    ];
    for (const [px,py] of pts) {
      for (const [cx,cy] of corners) {
        ctx.beginPath(); ctx.moveTo(px,py); ctx.lineTo(cx,cy); ctx.stroke();
      }
    }
  }

  if (guides.actionSafe) {
    ctx.strokeStyle = gc;
    ctx.setLineDash([Math.round(width / 80), Math.round(width / 80)]);
    const mx = width * 0.05, my = height * 0.05;
    ctx.strokeRect(mx, my, width - mx * 2, height - my * 2);
    ctx.setLineDash([]);
  }

  if (guides.titleSafe) {
    ctx.strokeStyle = gcFaint;
    ctx.setLineDash([Math.round(width / 120), Math.round(width / 120)]);
    const mx = width * 0.1, my = height * 0.1;
    ctx.strokeRect(mx, my, width - mx * 2, height - my * 2);
    ctx.setLineDash([]);
  }

  if (guides.center) {
    ctx.strokeStyle = gc;
    const cx = width / 2, cy = height / 2;
    const cs = Math.min(width, height) * 0.03;
    ctx.beginPath(); ctx.moveTo(cx - cs, cy); ctx.lineTo(cx + cs, cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx, cy - cs); ctx.lineTo(cx, cy + cs); ctx.stroke();
  }

  ctx.strokeStyle = hexToRgba(guideColor, 0.2);
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(0, 0, width, height);

  if (frameCount != null) {
    const fontSize = Math.max(10, Math.round(width / 60));
    const pad      = Math.round(fontSize * 0.6);
    const label    = `${frameCount}f  (4m+1)`;
    ctx.font       = `${fontSize}px monospace`;
    const tw       = ctx.measureText(label).width;
    const bx       = width  - tw - pad * 2 - 2;
    const by       = height - fontSize - pad * 2 - 2;
    const bw       = tw + pad * 2;
    const bh       = fontSize + pad * 2;
    ctx.fillStyle  = 'rgba(0,0,0,0.55)';
    ctx.fillRect(bx, by, bw, bh);
    ctx.fillStyle  = hexToRgba(guideColor, 0.9);
    ctx.fillText(label, bx + pad, by + bh - pad - 1);
  }
}

// ============================================================================
// Spec Tool sub-tab
// ============================================================================

const CHIP = { padding: '0.15rem 0.4rem', fontSize: '0.72rem' };

function SpecTool({ config }) {
  const d = config?.defaults?.spec_tool ?? {};

  const [ratioIdx,    setRatioIdx]    = useState(d.ratio_idx        ?? 0);
  const [customW,     setCustomW]     = useState(16);
  const [customH,     setCustomH]     = useState(9);
  const [rawW,        setRawW]        = useState(d.resolution_width  ?? 1920);
  // Frame count stored as string so the field can be empty while typing
  const [frameStr,    setFrameStr]    = useState('');
  const [guides,      setGuides]      = useState(d.guides ?? { thirds: true, golden: false, vp: false, actionSafe: false, titleSafe: false, center: false });
  const [guideColor,  setGuideColor]  = useState(d.guide_color      ?? '#ffffff');
  const [bgColor,     setBgColor]     = useState(d.bg_color         ?? '#1a1a1a');
  const [bgAlpha,     setBgAlpha]     = useState(d.bg_alpha         ?? 100);
  const [burnIn,      setBurnIn]      = useState(d.burn_in          ?? true);
  const [blendDir,    setBlendDir]    = useState('');
  const [blendName,   setBlendName]   = useState(d.blend_filename   ?? 'template');
  const [blendFps,    setBlendFps]    = useState(d.blend_fps        ?? 24);
  const [blendSaving, setBlendSaving] = useState(false);
  const [blendResult, setBlendResult] = useState(null); // { ok, msg }

  const canvasRef = useRef(null);
  const PREVIEW_W = 420;

  const ratio  = ASPECT_RATIOS[ratioIdx];
  const ratioW = ratio.w ?? customW;
  const ratioH = ratio.h ?? customH;

  const snapW      = snapTo16(rawW);
  const snapH      = snapTo16((snapW / ratioW) * ratioH);
  const parsedF    = parseInt(frameStr);
  const snapFrames = !isNaN(parsedF) && parsedF > 0 ? snapTo4m1(parsedF) : null;
  const previewH   = Math.round((PREVIEW_W / ratioW) * ratioH);

  const drawParams = { guides, guideColor, bgColor, bgAlpha, frameCount: burnIn ? snapFrames : null };

  useEffect(() => {
    drawGuides(canvasRef.current, PREVIEW_W, previewH, drawParams);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ratioIdx, customW, customH, guides, guideColor, bgColor, bgAlpha, burnIn, snapFrames, previewH]);

  const toggleGuide = (key) => setGuides(prev => ({ ...prev, [key]: !prev[key] }));

  const handleFrameBlur = () => {
    if (snapFrames !== null) setFrameStr(String(snapFrames));
  };

  const handleBrowseBlendDir = async () => {
    try {
      const res = await fetch('/api/browser/directory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'Choose save folder for .blend template' }),
      });
      const data = await res.json();
      if (data.success && data.directory) setBlendDir(data.directory);
    } catch (err) { console.error('Browse dir failed:', err); }
  };

  const handleSaveBlend = useCallback(async () => {
    if (!blendDir) return;
    setBlendSaving(true);
    setBlendResult(null);
    try {
      const res = await fetch('/api/utilities/blend-template', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          width:       snapW,
          height:      snapH,
          frame_count: snapFrames ?? 25,
          fps:         blendFps,
          save_dir:    blendDir,
          filename:    blendName || 'template',
        }),
      });
      const data = await res.json();
      if (data.success) {
        setBlendResult({ ok: true, msg: data.path });
      } else {
        setBlendResult({ ok: false, msg: data.detail || 'Unknown error' });
      }
    } catch (err) {
      setBlendResult({ ok: false, msg: err.message });
    } finally {
      setBlendSaving(false);
    }
  }, [snapW, snapH, snapFrames, blendFps, blendDir, blendName]);

  const handleDownloadPNG = useCallback(() => {
    const offscreen = document.createElement('canvas');
    offscreen.width  = snapW;
    offscreen.height = snapH;
    drawGuides(offscreen, snapW, snapH, drawParams);
    const link = document.createElement('a');
    const fSuffix = snapFrames != null ? `_${snapFrames}f` : '';
    link.download = `template_${snapW}x${snapH}${fSuffix}.png`;
    link.href = offscreen.toDataURL('image/png');
    link.click();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapW, snapH, snapFrames, guides, guideColor, bgColor, bgAlpha, burnIn]);

  const widthAdjusted  = rawW !== snapW;
  const framesAdjusted = snapFrames !== null && parsedF !== snapFrames;

  // ── Shared small-label style ──
  const lbl = { fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: '0.2rem', display: 'block' };

  return (
    <div style={{ display: 'flex', gap: '1.5rem', padding: '1rem 1.25rem', height: '100%', boxSizing: 'border-box', overflow: 'hidden' }}>

      {/* ── Left panel: two-column grid of controls ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gridAutoRows: 'min-content',
        gap: '0.6rem',
        width: '520px',
        flexShrink: 0,
        alignContent: 'start',
      }}>

        {/* Aspect Ratio — full width */}
        <div className="fuk-card" style={{ gridColumn: '1 / -1', padding: '0.6rem 0.75rem' }}>
          <span style={lbl}>Aspect Ratio</span>
          <select className="fuk-select" value={ratioIdx}
            onChange={e => setRatioIdx(Number(e.target.value))}
            style={{ width: '100%' }}>
            {ASPECT_RATIOS.map((r, i) => <option key={i} value={i}>{r.label}</option>)}
          </select>
          {ratio.w === null && (
            <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.4rem', alignItems: 'center' }}>
              <input type="number" min="1" max="999" className="fuk-input" value={customW}
                onChange={e => setCustomW(Math.max(1, parseInt(e.target.value) || 1))}
                style={{ width: '4rem', textAlign: 'center' }} />
              <span style={{ color: 'var(--text-muted)' }}>:</span>
              <input type="number" min="1" max="999" className="fuk-input" value={customH}
                onChange={e => setCustomH(Math.max(1, parseInt(e.target.value) || 1))}
                style={{ width: '4rem', textAlign: 'center' }} />
            </div>
          )}
        </div>

        {/* Resolution — full width */}
        <div className="fuk-card" style={{ gridColumn: '1 / -1', padding: '0.6rem 0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.4rem' }}>
            <span style={{ ...lbl, marginBottom: 0 }}>Resolution</span>
            {/* ×16 snapped badge inline */}
            <span style={{
              fontFamily: 'monospace', fontSize: '0.85rem',
              color: widthAdjusted ? 'var(--accent-color, #a855f7)' : 'var(--text-secondary)',
            }}>
              {snapW} × {snapH}
              {widthAdjusted && <span style={{ fontSize: '0.65rem', marginLeft: '0.3rem', opacity: 0.8 }}>↑×16</span>}
            </span>
          </div>
          {/* Preset chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', marginBottom: '0.5rem' }}>
            {PRESET_RESOLUTIONS.map(p => (
              <button key={p.label} className="fuk-btn fuk-btn-secondary" style={CHIP} onClick={() => setRawW(p.width)}>
                {p.label}
              </button>
            ))}
          </div>
          {/* W / H inputs */}
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.4rem' }}>
            <div style={{ flex: 1 }}>
              <span style={lbl}>Width</span>
              <input type="number" min="16" max="16384" className="fuk-input"
                value={rawW}
                onChange={e => setRawW(Math.max(16, parseInt(e.target.value) || 16))}
                onBlur={() => setRawW(snapW)}
                style={{ width: '100%', textAlign: 'center' }} />
            </div>
            <span style={{ color: 'var(--text-muted)', paddingBottom: '0.35rem' }}>×</span>
            <div style={{ flex: 1 }}>
              <span style={lbl}>Height (derived)</span>
              <div className="fuk-input"
                style={{ width: '100%', textAlign: 'center', background: 'var(--bg-secondary)', color: 'var(--text-muted)', userSelect: 'none' }}>
                {Math.round((rawW / ratioW) * ratioH)}
              </div>
            </div>
          </div>
        </div>

        {/* Frame Count */}
        <div className="fuk-card" style={{ padding: '0.6rem 0.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.4rem' }}>
            <span style={{ ...lbl, marginBottom: 0 }}>Frame Count</span>
            {snapFrames !== null && (
              <span style={{
                fontFamily: 'monospace', fontSize: '0.85rem',
                color: framesAdjusted ? 'var(--accent-color, #a855f7)' : 'var(--text-secondary)',
              }}>
                {snapFrames}f
                {framesAdjusted && <span style={{ fontSize: '0.65rem', marginLeft: '0.3rem', opacity: 0.8 }}>↑4m+1</span>}
              </span>
            )}
          </div>
          <input
            type="number" min="1" max="99999"
            className="fuk-input"
            placeholder="e.g. 25"
            value={frameStr}
            onChange={e => setFrameStr(e.target.value)}
            onBlur={handleFrameBlur}
            style={{ width: '100%', textAlign: 'center' }}
          />
        </div>

        {/* Burn-in */}
        <div className="fuk-card" style={{ padding: '0.6rem 0.75rem', display: 'flex', alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.82rem', width: '100%' }}>
            <input type="checkbox" checked={burnIn} onChange={e => setBurnIn(e.target.checked)} />
            Burn-in frame count
          </label>
        </div>

        {/* Composition Guides */}
        <div className="fuk-card" style={{ padding: '0.6rem 0.75rem' }}>
          <span style={lbl}>Composition Guides</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', marginBottom: '0.55rem' }}>
            {COMPOSITION_GUIDES.map(g => (
              <label key={g.key} style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', cursor: 'pointer', fontSize: '0.8rem' }}>
                <input type="checkbox" checked={guides[g.key]} onChange={() => toggleGuide(g.key)} />
                {g.label}
              </label>
            ))}
          </div>
          <span style={lbl}>Line Color</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <input type="color" value={guideColor} onChange={e => setGuideColor(e.target.value)}
              style={{ width: '1.8rem', height: '1.8rem', border: 'none', background: 'none', cursor: 'pointer', padding: 0 }} />
            <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-muted)' }}>{guideColor}</span>
            <button className="fuk-btn fuk-btn-secondary" style={CHIP} onClick={() => setGuideColor('#ffffff')}>White</button>
            <button className="fuk-btn fuk-btn-secondary" style={CHIP} onClick={() => setGuideColor('#ff0000')}>Red</button>
          </div>
        </div>

        {/* Background */}
        <div className="fuk-card" style={{ padding: '0.6rem 0.75rem' }}>
          <span style={lbl}>Background</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
            <input type="color" value={bgColor} onChange={e => setBgColor(e.target.value)}
              style={{ width: '1.8rem', height: '1.8rem', border: 'none', background: 'none', cursor: 'pointer', padding: 0 }} />
            <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-muted)' }}>{bgColor}</span>
            <button className="fuk-btn fuk-btn-secondary" style={CHIP} onClick={() => setBgColor('#000000')}>Black</button>
            <button className="fuk-btn fuk-btn-secondary" style={CHIP} onClick={() => setBgColor('#808080')}>Gray</button>
          </div>
          <span style={lbl}>Opacity: {bgAlpha}%{bgAlpha < 100 && <span style={{ marginLeft: '0.4rem', color: 'var(--accent-color, #a855f7)' }}>transparent PNG</span>}</span>
          <input type="range" min="0" max="100" value={bgAlpha}
            onChange={e => setBgAlpha(Number(e.target.value))}
            style={{ width: '100%' }} />
        </div>

        {/* Download PNG — full width */}
        <div style={{ gridColumn: '1 / -1' }}>
          <button className="fuk-btn fuk-btn-primary" onClick={handleDownloadPNG} style={{ width: '100%' }}>
            Download Template PNG
            <span style={{ opacity: 0.65, fontSize: '0.8em', marginLeft: '0.5rem' }}>
              {snapW}×{snapH}{snapFrames != null ? ` · ${snapFrames}f` : ''}
            </span>
          </button>
        </div>

        {/* Save .blend Template — full width */}
        <div className="fuk-card" style={{ gridColumn: '1 / -1', padding: '0.6rem 0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <span style={lbl}>Save Blender Template (.blend)</span>

          {/* Row 1: filename + fps */}
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-end' }}>
            <div style={{ flex: 1 }}>
              <span style={lbl}>Filename</span>
              <input
                className="fuk-input"
                value={blendName}
                onChange={e => setBlendName(e.target.value)}
                placeholder="template"
                style={{ width: '100%' }}
              />
            </div>
            <div style={{ width: '5.5rem' }}>
              <span style={lbl}>FPS</span>
              <select className="fuk-select" value={blendFps} onChange={e => setBlendFps(Number(e.target.value))} style={{ width: '100%' }}>
                {[23.976, 24, 25, 29.97, 30, 48, 60].map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Row 2: directory picker */}
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <div
              className="fuk-input"
              style={{ flex: 1, color: blendDir ? 'var(--text-primary)' : 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', userSelect: 'none', fontSize: '0.8rem' }}
            >
              {blendDir || 'No folder selected'}
            </div>
            <button className="fuk-btn fuk-btn-secondary" style={CHIP} onClick={handleBrowseBlendDir}>
              Browse
            </button>
          </div>

          {/* Row 3: save button + result */}
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button
              className="fuk-btn fuk-btn-secondary"
              onClick={handleSaveBlend}
              disabled={!blendDir || blendSaving}
              style={{ flexShrink: 0 }}
            >
              {blendSaving ? 'Saving…' : 'Save .blend'}
            </button>
            {blendResult && (
              <span style={{
                fontSize: '0.72rem',
                fontFamily: 'monospace',
                color: blendResult.ok ? 'var(--success-color, #22c55e)' : 'var(--error-color, #ef4444)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {blendResult.ok ? `✓ ${blendResult.msg}` : `✗ ${blendResult.msg}`}
              </span>
            )}
          </div>
        </div>

      </div>

      {/* ── Right panel — preview ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem', minWidth: 0, overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem', flexWrap: 'wrap', flexShrink: 0 }}>
          <span className="fuk-label">Preview</span>
          <span style={{ fontFamily: 'monospace', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {snapW} × {snapH}{snapFrames != null ? ` · ${snapFrames}f` : ''} &nbsp;|&nbsp; {ratioW}:{ratioH}
          </span>
        </div>

        {/* Checkerboard to show alpha */}
        <div style={{
          display: 'inline-flex',
          backgroundImage: 'repeating-conic-gradient(#555 0% 25%, #333 0% 50%)',
          backgroundSize: '14px 14px',
          borderRadius: '4px',
          border: '1px solid var(--border-color)',
          overflow: 'hidden',
          alignSelf: 'flex-start',
          maxWidth: '100%',
          maxHeight: 'calc(100% - 2rem)',
        }}>
          <canvas
            ref={canvasRef}
            width={PREVIEW_W}
            height={previewH}
            style={{ display: 'block', maxWidth: '100%', maxHeight: 'calc(100vh - 260px)' }}
          />
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Main Utilities Tab
// ============================================================================

const SUB_TABS = [
  { key: 'spec', label: 'Spec Tool' },
  { key: 'lora', label: 'LoRA Dataset Builder' },
];

export default function UtilitiesTab({ activeTab, setActiveTab, config }) {
  const [subTab, setSubTab] = useState('spec');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Inner sub-tab bar */}
      <div style={{
        display: 'flex',
        gap: '0.25rem',
        padding: '0.5rem 1rem 0',
        borderBottom: '1px solid var(--border-color)',
        background: 'var(--bg-secondary)',
        flexShrink: 0,
      }}>
        {SUB_TABS.map(t => (
          <button
            key={t.key}
            disabled={t.disabled}
            onClick={() => !t.disabled && setSubTab(t.key)}
            className={`fuk-tab ${subTab === t.key ? 'active' : ''}`}
            style={{ opacity: t.disabled ? 0.4 : 1, cursor: t.disabled ? 'not-allowed' : 'pointer', fontSize: '0.8rem' }}
          >
            {t.label}
            {t.disabled && <span style={{ marginLeft: '0.35rem', fontSize: '0.65rem', color: 'var(--text-muted)' }}>soon</span>}
          </button>
        ))}
      </div>

      {/* Sub-tab content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {subTab === 'spec' && <SpecTool config={config} />}
        {subTab === 'lora' && <LoraDatasetBuilder config={config} />}
      </div>

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={false}
        canGenerate={false}
        onGenerate={() => {}}
      />
    </div>
  );
}
