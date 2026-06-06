/**
 * Utilities Tab
 * Houses miscellaneous tools that don't belong in the main pipeline.
 * Sub-tabs: Spec Tool, (more coming)
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import Footer from '../components/Footer';
import LoraDatasetBuilder from './LoraDatasetBuilder';
import ImageDescribeTool from './ImageDescribeTool';

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

const CHIP_CLASS = 'spec-tool-chip';

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

  return (
    <div className="spec-tool">

      {/* ── Left panel: two-column grid of controls ── */}
      <div className="spec-tool-controls">

        {/* Aspect Ratio — full width */}
        <div className="fuk-card spec-tool-card spec-tool-card-full">
          <span className="spec-tool-label">Aspect Ratio</span>
          <select className="fuk-select" value={ratioIdx} onChange={e => setRatioIdx(Number(e.target.value))}>
            {ASPECT_RATIOS.map((r, i) => <option key={i} value={i}>{r.label}</option>)}
          </select>
          {ratio.w === null && (
            <div className="spec-tool-custom-ratio">
              <input type="number" min="1" max="999" className="fuk-input" value={customW}
                onChange={e => setCustomW(Math.max(1, parseInt(e.target.value) || 1))} />
              <span className="spec-tool-separator">:</span>
              <input type="number" min="1" max="999" className="fuk-input" value={customH}
                onChange={e => setCustomH(Math.max(1, parseInt(e.target.value) || 1))} />
            </div>
          )}
        </div>

        {/* Resolution — full width */}
        <div className="fuk-card spec-tool-card spec-tool-card-full">
          <div className="spec-tool-resolution-header">
            <span className="spec-tool-label">Resolution</span>
            <span className={`spec-tool-snapped ${widthAdjusted ? 'spec-tool-snapped--adjusted' : ''}`}>
              {snapW} × {snapH}
              {widthAdjusted && <span className="spec-tool-snapped-badge">↑×16</span>}
            </span>
          </div>
          <div className="spec-tool-preset-chips">
            {PRESET_RESOLUTIONS.map(p => (
              <button key={p.label} className={`fuk-btn fuk-btn-secondary ${CHIP_CLASS}`} onClick={() => setRawW(p.width)}>
                {p.label}
              </button>
            ))}
          </div>
          <div className="spec-tool-row">
            <div className="spec-tool-col">
              <span className="spec-tool-label">Width</span>
              <input type="number" min="16" max="16384" className="fuk-input"
                value={rawW}
                onChange={e => setRawW(Math.max(16, parseInt(e.target.value) || 16))}
                onBlur={() => setRawW(snapW)} />
            </div>
            <span className="spec-tool-separator-bottom">×</span>
            <div className="spec-tool-col">
              <span className="spec-tool-label">Height (derived)</span>
              <div className="fuk-input spec-tool-derived-display">
                {Math.round((rawW / ratioW) * ratioH)}
              </div>
            </div>
          </div>
        </div>

        {/* Frame Count */}
        <div className="fuk-card spec-tool-card">
          <div className="spec-tool-resolution-header">
            <span className="spec-tool-label">Frame Count</span>
            {snapFrames !== null && (
              <span className={`spec-tool-snapped ${framesAdjusted ? 'spec-tool-snapped--adjusted' : ''}`}>
                {snapFrames}f
                {framesAdjusted && <span className="spec-tool-snapped-badge">↑4m+1</span>}
              </span>
            )}
          </div>
          <input
            type="number" min="1" max="99999"
            className="fuk-input spec-tool-input-center"
            placeholder="e.g. 25"
            value={frameStr}
            onChange={e => setFrameStr(e.target.value)}
            onBlur={handleFrameBlur}
          />
        </div>

        {/* Burn-in */}
        <div className="fuk-card spec-tool-card spec-tool-card--row">
          <label className="spec-tool-burn-in-label">
            <input type="checkbox" className="fuk-checkbox" checked={burnIn} onChange={e => setBurnIn(e.target.checked)} />
            Burn-in frame count
          </label>
        </div>

        {/* Composition Guides */}
        <div className="fuk-card spec-tool-card">
          <span className="spec-tool-label">Composition Guides</span>
          <div className="spec-tool-checklist">
            {COMPOSITION_GUIDES.map(g => (
              <label key={g.key}>
                <input type="checkbox" className="fuk-checkbox" checked={guides[g.key]} onChange={() => toggleGuide(g.key)} />
                {g.label}
              </label>
            ))}
          </div>
          <span className="spec-tool-label">Line Color</span>
          <div className="spec-tool-color-row">
            <input type="color" value={guideColor} onChange={e => setGuideColor(e.target.value)} className="spec-tool-color-input" />
            <span className="spec-tool-color-hex">{guideColor}</span>
            <button className={`fuk-btn fuk-btn-secondary ${CHIP_CLASS}`} onClick={() => setGuideColor('#ffffff')}>White</button>
            <button className={`fuk-btn fuk-btn-secondary ${CHIP_CLASS}`} onClick={() => setGuideColor('#ff0000')}>Red</button>
          </div>
        </div>

        {/* Background */}
        <div className="fuk-card spec-tool-card">
          <span className="spec-tool-label">Background</span>
          <div className="spec-tool-bg-row">
            <input type="color" value={bgColor} onChange={e => setBgColor(e.target.value)} className="spec-tool-color-input" />
            <span className="spec-tool-color-hex">{bgColor}</span>
            <button className={`fuk-btn fuk-btn-secondary ${CHIP_CLASS}`} onClick={() => setBgColor('#000000')}>Black</button>
            <button className={`fuk-btn fuk-btn-secondary ${CHIP_CLASS}`} onClick={() => setBgColor('#808080')}>Gray</button>
          </div>
          <span className="spec-tool-label">
            Opacity: {bgAlpha}%
            {bgAlpha < 100 && <span className="spec-tool-opacity-accent">transparent PNG</span>}
          </span>
          <input type="range" className="fuk-slider" min="0" max="100" value={bgAlpha} onChange={e => setBgAlpha(Number(e.target.value))} />
        </div>

        {/* Download PNG — full width */}
        <div className="spec-tool-card-full">
          <button className="fuk-btn fuk-btn-primary fuk-btn-full" onClick={handleDownloadPNG}>
            Download Template PNG
            <span className="spec-tool-download-info">
              {snapW}×{snapH}{snapFrames != null ? ` · ${snapFrames}f` : ''}
            </span>
          </button>
        </div>

        {/* Save .blend Template — full width */}
        <div className="fuk-card spec-tool-card spec-tool-card-full--col">
          <span className="spec-tool-label">Save Blender Template (.blend)</span>

          <div className="spec-tool-blend-row">
            <div className="spec-tool-col">
              <span className="spec-tool-label">Filename</span>
              <input className="fuk-input" value={blendName} onChange={e => setBlendName(e.target.value)} placeholder="template" />
            </div>
            <div className="spec-tool-blend-fps">
              <span className="spec-tool-label">FPS</span>
              <select className="fuk-select" value={blendFps} onChange={e => setBlendFps(Number(e.target.value))}>
                {[23.976, 24, 25, 29.97, 30, 48, 60].map(f => (
                  <option key={f} value={f}>{f}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="spec-tool-blend-dir-row">
            <div className={`fuk-input spec-tool-blend-dir ${blendDir ? 'spec-tool-blend-dir--set' : 'spec-tool-blend-dir--empty'}`}>
              {blendDir || 'No folder selected'}
            </div>
            <button className={`fuk-btn fuk-btn-secondary ${CHIP_CLASS}`} onClick={handleBrowseBlendDir}>
              Browse
            </button>
          </div>

          <div className="spec-tool-blend-result-row">
            <button
              className="fuk-btn fuk-btn-secondary"
              onClick={handleSaveBlend}
              disabled={!blendDir || blendSaving}
            >
              {blendSaving ? 'Saving…' : 'Save .blend'}
            </button>
            {blendResult && (
              <span className={`spec-tool-blend-result ${blendResult.ok ? 'spec-tool-blend-result--ok' : 'spec-tool-blend-result--error'}`}>
                {blendResult.ok ? `✓ ${blendResult.msg}` : `✗ ${blendResult.msg}`}
              </span>
            )}
          </div>
        </div>

      </div>

      {/* ── Right panel — preview ── */}
      <div className="spec-tool-preview-panel">
        <div className="spec-tool-preview-header">
          <span className="fuk-label">Preview</span>
          <span className="spec-tool-preview-meta">
            {snapW} × {snapH}{snapFrames != null ? ` · ${snapFrames}f` : ''} &nbsp;|&nbsp; {ratioW}:{ratioH}
          </span>
        </div>

        <div className="spec-tool-checkerboard">
          <canvas
            ref={canvasRef}
            width={PREVIEW_W}
            height={previewH}
            className="spec-tool-preview-canvas"
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
  { key: 'describe', label: 'Describe & Tag' },
];

export default function UtilitiesTab({ activeTab, setActiveTab, config }) {
  const [subTab, setSubTab] = useState('spec');

  return (
    <div className="utilities-tab">

      {/* Inner sub-tab bar */}
      <div className="utilities-subtab-bar">
        {SUB_TABS.map(t => (
          <button
            key={t.key}
            disabled={t.disabled}
            onClick={() => !t.disabled && setSubTab(t.key)}
            className={`fuk-tab ${subTab === t.key ? 'active' : ''}`}
          >
            {t.label}
            {t.disabled && <span className="utilities-subtab-soon">soon</span>}
          </button>
        ))}
      </div>

      {/* Sub-tab content */}
      <div className="utilities-subtab-content">
        {subTab === 'spec' && <SpecTool config={config} />}
        {subTab === 'lora' && <LoraDatasetBuilder config={config} />}
        {subTab === 'describe' && <ImageDescribeTool />}
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
