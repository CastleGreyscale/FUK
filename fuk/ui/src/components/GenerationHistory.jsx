/**
 * Generation History Component
 * Shows past generations with drag-and-drop, pagination, and pinning
 */

import { useState, useEffect, useLayoutEffect, useCallback, useRef, useMemo } from 'react';
import { Film, Camera, Clock, Trash2, RefreshCw, ChevronDown, ChevronRight, Enhance, Zap, ArrowUp, Layers, Download, PinIcon, ImportIcon, SequenceIcon, ThumbsUp, ThumbsDown, Maximize2, Columns, X } from './Icons';
import { buildImageUrl, API_URL } from '../utils/constants';
import { useVideoPlayback } from '../hooks/useVideoPlayback';
import ZoomableImage from './ZoomableImage';
import CompareImage from './CompareImage';
import VideoSyncController from './VideoSyncController';
import ConformHDButton from './ConformHDButton';
import { setPanelPreview, upsertPanel } from '../utils/storyboardApi';
import { saveGenerationToLocation } from '../utils/historyApi';

// Pin-to-storyboard glyph: stacked thumbnails with an arrow.
const StoryboardPinIcon = ({ className, style }) => (
  <svg className={className} style={style} width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="5" width="8" height="6" rx="1" />
    <rect x="13" y="5" width="8" height="6" rx="1" />
    <rect x="3" y="14" width="8" height="6" rx="1" />
    <rect x="13" y="14" width="8" height="6" rx="1" />
  </svg>
);



// Save-a-copy button. Opens a native "Save As" dialog (server-side) and copies
// the generation's png/mp4 to the chosen location. `compact` = icon only for
// action bars; `large` = labeled button for the gallery detail panel.
function SaveButton({ generation, variant = 'compact' }) {
  const [busy, setBusy] = useState(false);

  const handleSave = async (e) => {
    e.stopPropagation();
    if (busy) return;
    setBusy(true);
    try {
      const res = await saveGenerationToLocation(generation);
      if (res && res.success === false && !res.cancelled && res.error) {
        alert(`Save failed: ${res.error}`);
      }
    } finally {
      setBusy(false);
    }
  };

  const isLarge = variant === 'large';
  return (
    <button
      className={`gen-history-save${isLarge ? ' large' : ''}`}
      onClick={handleSave}
      disabled={busy}
      title="Save a copy to another location"
    >
      <Download />
      {isLarge && <span>{busy ? 'Saving…' : 'Save'}</span>}
    </button>
  );
}

// Hover preview popup component
function HoverPreview({ generation, position, videoRef }) {
  const popupRef = useRef(null);
  const [style, setStyle] = useState({ position: 'fixed', left: -9999, top: -9999, zIndex: 9999, visibility: 'hidden' });

  const isVideo = generation.type === 'video' || generation.type === 'interpolate' ||
                  (generation.type === 'preprocess' && generation.subtype === 'video') ||
                  (generation.type === 'upscale' && generation.subtype === 'video') ||
                  (generation.type === 'layers' && generation.subtype === 'video');
  const isSequence = generation.isSequence;
  const previewUrl = buildImageUrl(generation.preview);
  const displayName = generation.name || generation.id || 'Unknown';

  // Full date + time combined
  const dateTime = [generation.date, generation.timestamp].filter(Boolean).join(' · ');

  useLayoutEffect(() => {
    if (!popupRef.current) return;
    const { offsetWidth: w, offsetHeight: h } = popupRef.current;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const margin = 8;

    let left = position.x + 16;
    let top = position.y - Math.round(h / 2);

    // Flip to left if it overflows right edge
    if (left + w > vw - margin) {
      left = position.x - w - 16;
    }
    // Clamp left
    left = Math.max(margin, left);
    // Clamp top so it doesn't go above or below viewport
    top = Math.max(margin, Math.min(top, vh - h - margin));

    setStyle({ position: 'fixed', left, top, zIndex: 9999, visibility: 'visible' });
  }, [position.x, position.y, generation]);

  return (
    <div ref={popupRef} className="gen-history-hover-preview" style={style}>
      <div className="gen-history-hover-media">
        {isSequence ? (
          <img src={previewUrl} alt={displayName} />
        ) : isVideo ? (
          <video 
            ref={videoRef}
            src={previewUrl}
            autoPlay
            muted
            loop
            playsInline
          />
        ) : (
          <img src={previewUrl} alt={displayName} />
        )}
      </div>
      <div className="gen-history-hover-info">
        <span className="gen-history-hover-name">{displayName}</span>
        <span className="gen-history-hover-time">{generation.timestamp}</span>
        {generation.subtype && (
          <span className="gen-history-hover-subtype">{generation.subtype}</span>
        )}
      </div>
    </div>
  );
}

// Draggable thumbnail component
function DraggableThumbnail({ generation, onDelete, onTogglePin, isPinned, onHover, onHoverEnd, videoRefPlayback, vote, onVote, onSendToStoryboard, sendToStoryboardEnabled }) {
  const [imageError, setImageError] = useState(false);
  const videoRef = useRef(null);
  const isVideo = generation.type === 'video' || 
                  generation.type === 'interpolate' || 
                  (generation.type === 'preprocess' && generation.subtype === 'video') ||
                  (generation.type === 'upscale' && generation.subtype === 'video') ||
                  (generation.type === 'layers' && generation.subtype === 'video');
  const isSequence = generation.isSequence;
  const previewUrl = buildImageUrl(generation.preview);
  const thumbnailUrl = generation.thumbnailUrl ? buildImageUrl(generation.thumbnailUrl) : null;
  const displayName = generation.name || generation.id || 'Unknown';
  
  // Get the primary content type icon (Camera or Film)
  const getContentTypeIcon = () => {
    if (isSequence) return SequenceIcon;
    return isVideo ? Film : Camera;
  };
  
  // Get the processing type icon (if applicable)
  const getProcessingIcon = () => {
    // Check if this is an upscaled or interpolated item based on path/name
    const path = generation.path || generation.preview || '';
    const name = generation.name || generation.id || '';
    
    // Check directory name or path for processing type
    if (path.includes('upscale_') || name.includes('upscale')) {
      return ArrowUp;
    }
    if (path.includes('interpolate_') || name.includes('interpolate')) {
      return Zap;
    }
    
    // Otherwise use the type field
    switch (generation.type) {
      case 'layers': return Layers;
      case 'preprocess': return Enhance;
      case 'upscale': return ArrowUp;
      case 'interpolate': return Zap;
      case 'export': return Download;
      case 'import': return ImportIcon;
      // Basic image/video types don't need a processing badge
      case 'image':
      case 'video':
        return null;
      default: 
        return null;
    }
  };
  
  // Get the display type for CSS classes (matches detection logic)
  const getProcessingType = () => {
    const path = generation.path || generation.preview || '';
    const name = generation.name || generation.id || '';
    
    // Check directory name or path for processing type
    if (path.includes('upscale_') || name.includes('upscale')) {
      return 'upscale';
    }
    if (path.includes('interpolate_') || name.includes('interpolate')) {
      return 'interpolate';
    }
    
    // Otherwise use the type field
    return generation.type;
  };
  
  // Get subtype badge for layers and preprocess
  const getSubtypeBadge = () => {
    if (!generation.subtype) return null;
    
    const colors = {
      depth: '#3b82f6',
      normals: '#a855f7', 
      crypto: '#f59e0b',
      canny: '#22c55e',
      openpose: '#ef4444',
      video: '#6366f1',
    };
    
    return (
      <span
        className="gallery-badge--subtype"
        style={{ background: colors[generation.subtype] || '#6b7280' }}
      >
        {generation.subtype}
      </span>
    );
  };
  
  // Get frame count badge for sequences
  const getFrameCountBadge = () => {
    if (!isSequence || !generation.frameCount) return null;
    
    return (
      <span className="gallery-badge--subtype gallery-badge--frames">
        {generation.frameCount}f
      </span>
    );
  };
  
  const ContentTypeIcon = getContentTypeIcon();
  const ProcessingIcon = getProcessingIcon();
  const processingType = getProcessingType();
  
  // Debug logging - show detection results
  const path = generation.path || generation.preview || '';
  console.log('Generation DEBUG:', {
    id: generation.id,
    type: generation.type,
    computedType: processingType,
    subtype: generation.subtype,
    path: path,
    name: generation.name,
    hasUpscaleInPath: path.includes('upscale_'),
    hasInterpolateInPath: path.includes('interpolate_'),
    ProcessingIcon: ProcessingIcon?.name || 'none',
  });
  
  const handleDragStart = (e) => {
    // Use preview URL for the path (API returns 'preview' not 'path')
    const dragPath = generation.path || generation.preview;
    e.dataTransfer.setData('text/plain', dragPath);
    e.dataTransfer.setData('application/x-fuk-generation', JSON.stringify({
      ...generation,
      path: dragPath,  // Ensure path is included
    }));
    e.dataTransfer.effectAllowed = 'copy';
    
    const img = e.target.cloneNode(true);
    img.style.width = '80px';
    img.style.height = '80px';
    img.style.opacity = '0.8';
    document.body.appendChild(img);
    e.dataTransfer.setDragImage(img, 40, 40);
    setTimeout(() => document.body.removeChild(img), 0);
    
    // Hide hover preview when dragging starts
    if (onHoverEnd) onHoverEnd();
  };
  
  const handleMouseMove = (e) => {
    if (onHover) {
      onHover(generation, { x: e.clientX, y: e.clientY });
    }
  };
  
  const handleMouseLeave = () => {
    if (onHoverEnd) onHoverEnd();
  };

  return (
    <div
      className={`gen-history-item ${isPinned ? 'pinned' : ''}`}
      draggable
      onDragStart={handleDragStart}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <div className="gen-history-thumb">
        {!imageError ? (
          isSequence ? (
            // Sequence: show first frame as image with sequence indicator
            <div className="gen-history-sequence">
              <img 
                src={previewUrl} 
                alt={displayName}
                onError={() => setImageError(true)}
              />
              <div className="gen-history-sequence-overlay">
                <SequenceIcon />
              </div>
            </div>
          ) : isVideo ? (
            thumbnailUrl ? (
              <img
                src={thumbnailUrl}
                alt={displayName}
                onError={() => setImageError(true)}
              />
            ) : (
              <video
                src={previewUrl}
                muted
                preload="metadata"
                onError={() => setImageError(true)}
              />
            )
          ) : (
            <img 
              src={previewUrl} 
              alt={displayName}
              onError={() => setImageError(true)}
            />
          )
        ) : (
          <div className="gen-history-thumb-error">
            <ContentTypeIcon />
          </div>
        )}
        
        {/* Content type badge (Camera/Film) */}
        <div className={`gen-history-type-badge content ${isVideo ? 'video' : 'image'}`}>
          <ContentTypeIcon />
        </div>
        
        {/* Processing type badge (if applicable) */}
        {ProcessingIcon && (
          <div className={`gen-history-type-badge processing ${getProcessingType()} ${generation.subtype || ''}`}>
            <ProcessingIcon />
          </div>
        )}
        
        {isPinned && (
          <div className="gen-history-pinned-badge">
            <PinIcon />
          </div>
        )}
      </div>
      
      {/* Action bar — always visible, replaces hover overlay + info strip */}
      <div className="gen-history-actions-bar">
        <button 
          className={`gen-history-pin ${isPinned ? 'active' : ''}`}
          onClick={(e) => { e.stopPropagation(); onTogglePin(generation); }}
          title={isPinned ? 'Unpin' : 'Pin to top'}
        >
          <PinIcon />
        </button>
        <div className="gen-history-actions-bar-right">
          <ConformHDButton generation={generation} variant="compact" />
          <SaveButton generation={generation} variant="compact" />
          {onSendToStoryboard && (
            <button
              className="gen-history-pin-storyboard"
              onClick={(e) => { e.stopPropagation(); onSendToStoryboard(generation); }}
              disabled={!sendToStoryboardEnabled}
              title={sendToStoryboardEnabled
                ? 'Pin as storyboard preview for the current shot'
                : 'Open a shot to pin a storyboard preview'}
            >
              <StoryboardPinIcon />
            </button>
          )}
          <button
            className={`gen-history-vote up ${vote === 1 ? 'active' : ''}`}
            onClick={(e) => { e.stopPropagation(); onVote(generation, vote === 1 ? 0 : 1); }}
            title="Good generation"
          >
            <ThumbsUp />
          </button>
          <button
            className={`gen-history-vote down ${vote === -1 ? 'active' : ''}`}
            onClick={(e) => { e.stopPropagation(); onVote(generation, vote === -1 ? 0 : -1); }}
            title="Bad generation"
          >
            <ThumbsDown />
          </button>
          {onDelete && (
            <button
              className="gen-history-delete"
              onClick={(e) => { e.stopPropagation(); onDelete(generation); }}
              title="Delete generation"
            >
              <Trash2 />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── helpers shared by gallery ───────────────────────────────────────────────

function isGenVideo(g) {
  return g.type === 'video' || g.type === 'interpolate' ||
    (g.type === 'preprocess' && g.subtype === 'video') ||
    (g.type === 'upscale'    && g.subtype === 'video') ||
    (g.type === 'layers'     && g.subtype === 'video');
}

function processingIcon(g) {
  const p = g.path || g.preview || '';
  const n = g.name || g.id || '';
  if (p.includes('upscale_')    || n.includes('upscale'))    return ArrowUp;
  if (p.includes('interpolate_')|| n.includes('interpolate')) return Zap;
  switch (g.type) {
    case 'layers':    return Layers;
    case 'preprocess': return Enhance;
    case 'upscale':   return ArrowUp;
    case 'interpolate': return Zap;
    case 'export':    return Download;
    case 'import':    return ImportIcon;
    default:          return null;
  }
}

function processingType(g) {
  const p = g.path || g.preview || '';
  const n = g.name || g.id || '';
  if (p.includes('upscale_')    || n.includes('upscale'))    return 'upscale';
  if (p.includes('interpolate_')|| n.includes('interpolate')) return 'interpolate';
  return g.type;
}

// ─── single gallery thumbnail ─────────────────────────────────────────────────

function GalleryThumb({ generation, isPinned, isSelected, isMultiSelected, compareRole, vote, onSelect, onTogglePin, onVote, onDelete, onSendToStoryboard, sendToStoryboardEnabled }) {
  const [imgErr, setImgErr] = useState(false);
  const video   = isGenVideo(generation);
  const seq     = generation.isSequence;
  const preview = buildImageUrl(generation.preview);
  const thumb   = generation.thumbnailUrl ? buildImageUrl(generation.thumbnailUrl) : null;
  const ProcIcon = processingIcon(generation);
  const procType = processingType(generation);
  const ContentIcon = seq ? SequenceIcon : video ? Film : Camera;

  return (
    <div
      className={`gallery-thumb${isSelected ? ' selected' : ''}${isPinned ? ' pinned' : ''}${isMultiSelected ? ' multi-selected' : ''}${compareRole ? ` compare-${compareRole}` : ''}`}
      onClick={e => onSelect(generation, e.shiftKey)}
    >
      <div className="gallery-thumb-media">
        {!imgErr ? (
          seq ? (
            <img src={preview} alt="" onError={() => setImgErr(true)} />
          ) : video ? (
            thumb
              ? <img src={thumb} alt="" onError={() => setImgErr(true)} />
              : <video src={preview} muted preload="metadata" onError={() => setImgErr(true)} />
          ) : (
            <img src={preview} alt="" onError={() => setImgErr(true)} />
          )
        ) : (
          <div className="gallery-thumb-error"><ContentIcon /></div>
        )}

        {/* content badge */}
        <div className={`gen-history-type-badge content ${video ? 'video' : 'image'}`}>
          <ContentIcon />
        </div>
        {/* processing badge */}
        {ProcIcon && (
          <div className={`gen-history-type-badge processing ${procType} ${generation.subtype || ''}`}>
            <ProcIcon />
          </div>
        )}
        {isPinned && (
          <div className="gen-history-pinned-badge"><PinIcon /></div>
        )}
        {isMultiSelected && (
          <div className="gallery-thumb-check">✓</div>
        )}
        {compareRole && (
          <div className={`gallery-thumb-compare-badge ${compareRole}`}>{compareRole.toUpperCase()}</div>
        )}
      </div>

      {/* action bar */}
      <div className="gen-history-actions-bar">
        <button
          className={`gen-history-pin ${isPinned ? 'active' : ''}`}
          onClick={e => { e.stopPropagation(); onTogglePin(generation); }}
          title={isPinned ? 'Unpin' : 'Pin'}
        ><PinIcon /></button>
        <div className="gen-history-actions-bar-right">
          <SaveButton generation={generation} variant="compact" />
          {onSendToStoryboard && (
            <button
              className="gen-history-pin-storyboard"
              onClick={e => { e.stopPropagation(); onSendToStoryboard(generation); }}
              disabled={!sendToStoryboardEnabled}
              title={sendToStoryboardEnabled
                ? 'Pin as storyboard preview for the current shot'
                : 'Open a shot to pin a storyboard preview'}
            >
              <StoryboardPinIcon />
            </button>
          )}
          <button
            className={`gen-history-vote up ${vote === 1 ? 'active' : ''}`}
            onClick={e => { e.stopPropagation(); onVote(generation, vote === 1 ? 0 : 1); }}
          ><ThumbsUp /></button>
          <button
            className={`gen-history-vote down ${vote === -1 ? 'active' : ''}`}
            onClick={e => { e.stopPropagation(); onVote(generation, vote === -1 ? 0 : -1); }}
          ><ThumbsDown /></button>
          <button
            className="gen-history-delete"
            onClick={e => { e.stopPropagation(); onDelete(generation); }}
            title="Delete"
          ><Trash2 /></button>
        </div>
      </div>
    </div>
  );
}

// ─── generation-settings detail (fullscreen gallery sidebar) ─────────────────

// Scalar generation settings shown as compact label/value cells, in display
// order. `fmt` formats the raw metadata.json value; a field is rendered only
// when present. Anything not listed here (and not hidden below) still shows up
// via the generic pass, so new settings surface without touching this list.
const SETTING_FIELDS = [
  { key: 'image_size',            label: 'Resolution', fmt: v => Array.isArray(v) && v.length >= 2 ? `${v[0]} × ${v[1]}` : String(v) },
  { key: 'seed',                  label: 'Seed' },
  { key: 'infer_steps',           label: 'Steps' },
  { key: 'guidance_scale',        label: 'CFG' },
  { key: 'denoising_strength',    label: 'Denoise' },
  { key: 'sigma_shift',           label: 'Sigma shift' },
  { key: 'switch_dit_boundary',   label: 'DiT boundary' },
  { key: 'exponential_shift_mu',  label: 'Shift μ' },
  { key: 'exponential_shift_mu_used', label: 'Shift μ (auto)', fmt: v => Number(v).toFixed(4) },
  { key: 'video_length',          label: 'Frames' },
  { key: 'sliding_window_size',   label: 'Window' },
  { key: 'sliding_window_stride', label: 'Stride' },
  { key: 'eligen_alpha',          label: 'Eligen α' },
  { key: 'model',                 label: 'Model' },
];

// Keys handled explicitly elsewhere, or too noisy/structured to dump generically.
const SETTING_HIDDEN_KEYS = new Set([
  ...SETTING_FIELDS.map(f => f.key),
  'prompt', 'prompt_source', 'negative_prompt', 'lora', 'loras', 'lora_multiplier',
  'timestamp', 'display_name', 'source_path', 'first_frame_path', 'media_type',
  'is_sequence', 'frame_count', 'frame_pattern', 'first_frame', 'last_frame',
  'available_layers', 'layer_count', 'is_video', 'subtype',
  'prompt_expanded_markers', 'prompt_unknown_markers', 'mood_applied',
  'control_image', 'control_image_urls',
]);

const humanizeKey = (k) => k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

const isPresent = (v) =>
  v !== null && v !== undefined && v !== '' && !(Array.isArray(v) && v.length === 0);

const fmtSettingValue = (v) => {
  if (typeof v === 'boolean') return v ? 'Yes' : 'No';
  if (Array.isArray(v)) return v.join(', ');
  return String(v);
};

// Strip a LoRA name down to its basename (drops long cache/model paths).
const shortLora = (s) => {
  const str = String(s);
  return str.split(/[\\/]/).pop() || str;
};

// Long or path-like values get their own full-width row instead of a grid cell.
const isWideValue = (v) => v.length > 32 || v.includes('/') || v.includes('\\');

// Collapse a metadata object's LoRA fields into a single display string.
// Handles the `loras[]` array (objects or strings) and the legacy single
// `lora` + `lora_multiplier` pair.
function formatLoras(meta) {
  const items = [];
  if (Array.isArray(meta.loras) && meta.loras.length) {
    for (const l of meta.loras) {
      if (typeof l === 'string') { items.push(shortLora(l)); continue; }
      const name = shortLora(l.name || l.path || l.id || 'lora');
      const mult = l.multiplier ?? l.strength ?? l.weight;
      items.push(mult != null ? `${name} · ${mult}` : name);
    }
  } else if (isPresent(meta.lora)) {
    const name = shortLora(meta.lora);
    items.push(meta.lora_multiplier != null ? `${name} · ${meta.lora_multiplier}` : name);
  }
  return items.join(', ');
}

// Flatten a metadata.json into an ordered [{ key, label, value }] list of scalar
// generation settings: curated fields first (SETTING_FIELDS order), then any
// other primitive fields, then a derived LoRA summary. Shared by the single-item
// panel and the A/B compare table so both stay in sync. Prompt/negative are
// handled separately by callers since they render as text blocks.
function collectSettingRows(meta) {
  const rows = [];
  if (!meta) return rows;
  const seen = new Set();

  for (const f of SETTING_FIELDS) {
    if (!isPresent(meta[f.key])) continue;
    seen.add(f.key);
    rows.push({ key: f.key, label: f.label, value: f.fmt ? f.fmt(meta[f.key]) : fmtSettingValue(meta[f.key]) });
  }
  for (const [k, v] of Object.entries(meta)) {
    if (seen.has(k) || SETTING_HIDDEN_KEYS.has(k)) continue;
    if (!isPresent(v) || typeof v === 'object') continue;
    rows.push({ key: k, label: humanizeKey(k), value: fmtSettingValue(v) });
  }
  const lora = formatLoras(meta);
  if (lora) rows.push({ key: 'lora', label: 'LoRA', value: lora });
  return rows;
}

// Shared metadata cache so the large view and both A/B slots reuse fetches.
const _metaCache = new Map();

// Fetch (and cache) the full metadata.json for a generation id.
function useGenerationMetadata(id) {
  const [meta, setMeta] = useState(() => (id && _metaCache.has(id) ? _metaCache.get(id) : null));
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!id) { setMeta(null); return; }
    if (_metaCache.has(id)) { setMeta(_metaCache.get(id)); setLoading(false); return; }

    let cancelled = false;
    setMeta(null);
    setLoading(true);
    // Keep the id's internal slashes real (route is {gen_id:path}); only encode
    // within each segment.
    const encodedId = id.split('/').map(encodeURIComponent).join('/');
    fetch(`${API_URL}/project/generations/${encodedId}/metadata`)
      .then(r => (r.ok ? r.json() : null))
      .then(data => { if (cancelled) return; _metaCache.set(id, data); setMeta(data); })
      .catch(() => { if (!cancelled) setMeta(null); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [id]);

  return { meta, loading };
}

// Renders the full set of generation settings from a loaded metadata.json.
function GenerationSettings({ meta, loading }) {
  if (loading && !meta) {
    return <div className="gallery-detail-settings-status">Loading settings…</div>;
  }
  if (!meta) return null;

  const rows = collectSettingRows(meta);
  const gridRows = rows.filter(r => !isWideValue(r.value));
  const wideRows = rows.filter(r =>  isWideValue(r.value));
  const hasPrompt = isPresent(meta.prompt);
  const hasNeg = isPresent(meta.negative_prompt);

  if (!rows.length && !hasPrompt && !hasNeg) return null;

  return (
    <div className="gallery-detail-settings">
      <div className="gallery-detail-settings-label">Settings</div>

      {gridRows.length > 0 && (
        <div className="gallery-detail-settings-grid">
          {gridRows.map(r => (
            <div className="gallery-setting" key={r.key}>
              <span className="gallery-setting-key">{r.label}</span>
              <span className="gallery-setting-val">{r.value}</span>
            </div>
          ))}
        </div>
      )}

      {wideRows.map(r => (
        <div className="gallery-setting gallery-setting--wide" key={r.key}>
          <span className="gallery-setting-key">{r.label}</span>
          <span className="gallery-setting-val gallery-setting-val--path" title={r.value}>{r.value}</span>
        </div>
      ))}

      {hasPrompt && (
        <div className="gallery-detail-text">
          <span className="gallery-detail-text-label">Prompt</span>
          <div className="gallery-detail-text-body">{meta.prompt}</div>
        </div>
      )}
      {hasNeg && (
        <div className="gallery-detail-text">
          <span className="gallery-detail-text-label">Negative</span>
          <div className="gallery-detail-text-body">{meta.negative_prompt}</div>
        </div>
      )}
    </div>
  );
}

// Side-by-side settings for A/B compare. Builds a label | A | B table over the
// union of both items' settings and highlights rows whose values differ, then
// stacks the prompt/negative prompt for each side (also diff-highlighted).
function CompareSettings({ metaA, metaB, loading }) {
  if (loading && !metaA && !metaB) {
    return <div className="gallery-detail-settings-status">Loading settings…</div>;
  }
  if (!metaA && !metaB) return null;

  const mapA = new Map(collectSettingRows(metaA).map(r => [r.key, r]));
  const mapB = new Map(collectSettingRows(metaB).map(r => [r.key, r]));

  // Union of keys: A's order first, then any keys only present in B.
  const keys = [...mapA.keys()];
  for (const k of mapB.keys()) if (!mapA.has(k)) keys.push(k);

  const rows = keys.map(key => {
    const ra = mapA.get(key);
    const rb = mapB.get(key);
    const va = ra ? ra.value : '—';
    const vb = rb ? rb.value : '—';
    return { key, label: (ra || rb).label, va, vb, diff: va !== vb };
  });

  const promptA = metaA && isPresent(metaA.prompt) ? metaA.prompt : '';
  const promptB = metaB && isPresent(metaB.prompt) ? metaB.prompt : '';
  const negA = metaA && isPresent(metaA.negative_prompt) ? metaA.negative_prompt : '';
  const negB = metaB && isPresent(metaB.negative_prompt) ? metaB.negative_prompt : '';
  const showPrompt = promptA || promptB;
  const showNeg = negA || negB;

  if (!rows.length && !showPrompt && !showNeg) return null;

  const textPair = (label, valA, valB) => (
    <div className="gallery-detail-text">
      <span className="gallery-detail-text-label">{label}</span>
      <div className="gallery-compare-text">
        <div className="gallery-compare-text-side">
          <span className="gallery-compare-tag a">A</span>
          <div className={`gallery-detail-text-body${valA !== valB ? ' diff' : ''}`}>{valA || '—'}</div>
        </div>
        <div className="gallery-compare-text-side">
          <span className="gallery-compare-tag b">B</span>
          <div className={`gallery-detail-text-body${valA !== valB ? ' diff' : ''}`}>{valB || '—'}</div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="gallery-detail-settings">
      <div className="gallery-compare-table-head">
        <span className="gallery-detail-settings-label">Settings</span>
        <span className="gallery-compare-tag a">A</span>
        <span className="gallery-compare-tag b">B</span>
      </div>

      <div className="gallery-compare-table">
        {rows.map(r => (
          <div className={`gallery-compare-row${r.diff ? ' diff' : ''}`} key={r.key}>
            <span className="gallery-compare-cell key">{r.label}</span>
            <span className="gallery-compare-cell val" title={r.va}>{r.va}</span>
            <span className="gallery-compare-cell val" title={r.vb}>{r.vb}</span>
          </div>
        ))}
      </div>

      {showPrompt && textPair('Prompt', promptA, promptB)}
      {showNeg && textPair('Negative', negA, negB)}
    </div>
  );
}

// ─── large preview (top panel in the gallery) ────────────────────────────────

function GalleryLargeView({ generation, generations, isPinned, vote, onTogglePin, onVote, onDelete, onNavigate, multiSelected, onBulkVote, onBulkDelete, onClearSelection, deleteConfirm, onConfirmDelete, onCancelDelete, playbackSpeed }) {
  const preview  = buildImageUrl(generation.preview);
  const video    = isGenVideo(generation);
  const ProcIcon = processingIcon(generation);
  const procType = processingType(generation);
  const currentIdx = generations.findIndex(g => g.id === generation.id);
  const hasPrev = currentIdx > 0;
  const hasNext = currentIdx < generations.length - 1;
  const videoRef = useVideoPlayback(playbackSpeed);

  // Load the full metadata.json for the selected item to show every generation
  // setting (seed, resolution, CFG, steps, LoRAs, …).
  const { meta, loading: metaLoading } = useGenerationMetadata(generation.id);

  const typeColors = {
    depth: '#3b82f6', normals: '#a855f7', crypto: '#f59e0b',
    canny: '#22c55e', openpose: '#ef4444', video: '#6366f1',
  };

  return (
    <div className="gallery-large-view">
      {/* media + nav arrows */}
      <div className="gallery-large-media">
        <button
          className={`gallery-large-nav prev${!hasPrev ? ' disabled' : ''}`}
          onClick={() => hasPrev && onNavigate('prev')}
          disabled={!hasPrev}
          title="Previous (←)"
        >‹</button>

        {video
          ? <video ref={videoRef} key={preview} src={preview} controls autoPlay muted loop playsInline />
          : <ZoomableImage key={preview} src={preview} alt={generation.name || generation.id} defaultZoom={2} />
        }

        <button
          className={`gallery-large-nav next${!hasNext ? ' disabled' : ''}`}
          onClick={() => hasNext && onNavigate('next')}
          disabled={!hasNext}
          title="Next (→)"
        >›</button>
      </div>

      {/* sidebar */}
      <div className="gallery-large-sidebar">
        {/* inline delete confirmation */}
        {deleteConfirm && (
          <div className="gallery-delete-confirm">
            <span>Delete {deleteConfirm.label}?</span>
            <button className="gen-history-delete" onClick={onConfirmDelete}><Trash2 />Yes, delete</button>
            <button className="gallery-bulk-clear" onClick={onCancelDelete}><X />Cancel</button>
          </div>
        )}

        {/* bulk selection controls */}
        {!deleteConfirm && multiSelected && multiSelected.size > 1 && (
          <div className="gallery-bulk-section">
            <div className="gallery-bulk-header">{multiSelected.size} selected</div>
            <div className="gallery-detail-actions">
              <button className="gen-history-vote up" onClick={() => onBulkVote(1)}><ThumbsUp />Good all</button>
              <button className="gen-history-vote down" onClick={() => onBulkVote(-1)}><ThumbsDown />Bad all</button>
              <button className="gen-history-delete" onClick={onBulkDelete}><Trash2 />Delete all</button>
              <button className="gallery-bulk-clear" onClick={onClearSelection}><X />Clear</button>
            </div>
            <div className="gallery-bulk-divider" />
          </div>
        )}

        <div className="gallery-large-name">{generation.name || generation.id}</div>

        <div className="gallery-large-position">{currentIdx + 1} / {generations.length}</div>

        <div className="gallery-detail-badges">
          <span className={`gallery-badge type-${procType}`}>
            {ProcIcon && <ProcIcon />}
            {generation.type}
          </span>
          {generation.subtype && (
            <span className="gallery-badge gallery-badge--frames" style={{ background: typeColors[generation.subtype] || '#6b7280' }}>
              {generation.subtype}
            </span>
          )}
          {generation.isSequence && generation.frameCount && (
            <span className="gallery-badge gallery-badge--frames">
              {generation.frameCount}f
            </span>
          )}
          {vote === 1  && <span className="gallery-badge vote-up"><ThumbsUp /> good</span>}
          {vote === -1 && <span className="gallery-badge vote-down"><ThumbsDown /> bad</span>}
        </div>

        <div className="gallery-detail-rows">
          {generation.date       && <div className="gallery-detail-row"><span>Date</span><span>{generation.date}</span></div>}
          {generation.timestamp  && <div className="gallery-detail-row"><span>Time</span><span>{generation.timestamp}</span></div>}
          {generation.path       && <div className="gallery-detail-row path"><span>Path</span><span title={generation.path}>{generation.path}</span></div>}
          {generation.sourcePath && <div className="gallery-detail-row path"><span>Source</span><span title={generation.sourcePath}>{generation.sourcePath}</span></div>}
        </div>

        <GenerationSettings meta={meta} loading={metaLoading} />

        <div className="gallery-detail-actions">
          <button
            className={`gen-history-pin ${isPinned ? 'active' : ''}`}
            onClick={() => onTogglePin(generation)}
          ><PinIcon />{isPinned ? 'Unpin' : 'Pin'}</button>
          <button
            className={`gen-history-vote up ${vote === 1 ? 'active' : ''}`}
            onClick={() => onVote(generation, vote === 1 ? 0 : 1)}
          ><ThumbsUp />Good</button>
          <button
            className={`gen-history-vote down ${vote === -1 ? 'active' : ''}`}
            onClick={() => onVote(generation, vote === -1 ? 0 : -1)}
          ><ThumbsDown />Bad</button>
          <SaveButton generation={generation} variant="large" />
          <ConformHDButton generation={generation} variant="large" />
          <button
            className="gen-history-delete"
            onClick={() => onDelete(generation)}
          ><Trash2 />Delete</button>
        </div>
      </div>
    </div>
  );
}

// ─── Side-by-side synced video A/B compare ───────────────────────────────────
// Two videos rendered in parallel panes, driven by a single VideoSyncController
// so play/pause/scrub act on both at once. The controller reads each video's
// duration on mount, so we gate it behind both <video>s reporting metadata —
// otherwise it mounts against readyState 0 elements and never picks up duration.

function CompareVideos({ srcA, srcB, labelA = 'A', labelB = 'B', playbackSpeed }) {
  const videoARef = useVideoPlayback(playbackSpeed);
  const videoBRef = useVideoPlayback(playbackSpeed);
  const videoRefs = useMemo(() => [videoARef, videoBRef], []);
  const [loadedCount, setLoadedCount] = useState(0);
  const onLoaded = useCallback(() => setLoadedCount((c) => c + 1), []);

  return (
    <div className="compare-video">
      <div className="compare-video-stage">
        <div className="compare-video-pane">
          <span className="compare-video-label a">{labelA}</span>
          <video ref={videoARef} src={srcA} muted playsInline preload="auto" onLoadedMetadata={onLoaded} />
        </div>
        <div className="compare-video-pane">
          <span className="compare-video-label b">{labelB}</span>
          <video ref={videoBRef} src={srcB} muted playsInline preload="auto" onLoadedMetadata={onLoaded} />
        </div>
      </div>
      {loadedCount >= 2 && <VideoSyncController videoRefs={videoRefs} />}
    </div>
  );
}

// ─── A/B compare view (before/after wipe) ────────────────────────────────────

function CompareLargeView({ a, b, onSwap, onClearA, onClearB, onExit, playbackSpeed }) {
  const srcA = buildImageUrl(a.preview);
  const srcB = buildImageUrl(b.preview);
  const nameA = a.name || a.id;
  const nameB = b.name || b.id;
  // Videos can't be wiped like stills — show them side by side with synced
  // playback instead. Falls back to the wipe slider for images (or mixed pairs).
  const bothVideos = isGenVideo(a) && isGenVideo(b);

  // Full settings for both sides so the sidebar can diff them.
  const { meta: metaA, loading: loadingA } = useGenerationMetadata(a.id);
  const { meta: metaB, loading: loadingB } = useGenerationMetadata(b.id);

  return (
    <div className="gallery-large-view">
      <div className="gallery-large-media">
        {bothVideos ? (
          <CompareVideos
            key={`${srcA}|${srcB}`}
            srcA={srcA}
            srcB={srcB}
            labelA="A"
            labelB="B"
            playbackSpeed={playbackSpeed}
          />
        ) : (
          <CompareImage
            key={`${srcA}|${srcB}`}
            srcA={srcA}
            srcB={srcB}
            labelA="A"
            labelB="B"
          />
        )}
      </div>

      <div className="gallery-large-sidebar gallery-large-sidebar--compare">
        <div className="gallery-compare-header">
          <Columns /> A/B Compare
        </div>
        <div className="gallery-compare-slot">
          <span className="gallery-compare-tag a">A</span>
          <span className="gallery-compare-name" title={nameA}>{nameA}</span>
          <button className="gallery-compare-x" onClick={onClearA} title="Clear A"><X /></button>
        </div>
        <div className="gallery-compare-slot">
          <span className="gallery-compare-tag b">B</span>
          <span className="gallery-compare-name" title={nameB}>{nameB}</span>
          <button className="gallery-compare-x" onClick={onClearB} title="Clear B"><X /></button>
        </div>

        <CompareSettings metaA={metaA} metaB={metaB} loading={loadingA || loadingB} />

        <div className="gallery-detail-actions">
          <button className="gen-history-pin" onClick={onSwap}><Columns />Swap A / B</button>
          <button className="gallery-bulk-clear" onClick={onExit}><X />Exit compare</button>
        </div>

        <div className="gallery-compare-hint">
          {bothVideos
            ? 'Videos are synced · use the controls below the panes to play, pause, and scrub both at once'
            : 'Drag the bar to wipe · scroll to zoom · move mouse to pan'}
        </div>
      </div>
    </div>
  );
}

// ─── fullscreen gallery overlay ───────────────────────────────────────────────

function FullscreenGallery({ generations, pinnedIds, votes, onTogglePin, onVote, onDelete, onClose, hasMore, loading, onLoadMore, onLoadAll, onSendToStoryboard, sendToStoryboardEnabled, playbackSpeed, onPlaybackSpeedChange }) {
  const playbackFPS = Math.round((playbackSpeed || 1.0) * 24);
  const [zoom, setZoom]               = useState(160);
  const [selected, setSelected]       = useState(generations[0] || null);
  const [previewPct, setPreviewPct]   = useState(55);
  const [multiSelected, setMultiSelected] = useState(new Set());
  const [deleteConfirm, setDeleteConfirm] = useState(null); // { ids: [...], label: '' }
  // A/B compare: pick two images and wipe between them
  const [compareMode, setCompareMode] = useState(false);
  const [compareA, setCompareA]       = useState(null);
  const [compareB, setCompareB]       = useState(null);
  const bodyRef      = useRef(null);
  const isDragging   = useRef(false);
  const dragStartY   = useRef(0);
  const dragStartPct = useRef(55);
  const anchorId     = useRef(null);

  // Flat ordered list for range-selection (pinned first, then unpinned)
  const pinnedGens   = useMemo(() => generations.filter(g =>  pinnedIds.includes(g.id)), [generations, pinnedIds]);
  const unpinnedGens = useMemo(() => generations.filter(g => !pinnedIds.includes(g.id)), [generations, pinnedIds]);
  const orderedGens  = useMemo(() => [...pinnedGens, ...unpinnedGens], [pinnedGens, unpinnedGens]);

  // Keyboard: Escape, arrows, +/-
  useEffect(() => {
    const handle = e => {
      if (e.key === 'Escape') { onClose(); return; }
      if ((e.key === '+' || e.key === '=') && e.code !== 'NumpadAdd')      { setZoom(z => Math.min(z + 20, 320)); return; }
      if (e.key === '-'                    && e.code !== 'NumpadSubtract') { setZoom(z => Math.max(z - 20, 80));  return; }
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        setSelected(prev => {
          if (!prev) return generations[0] || null;
          const idx  = generations.findIndex(g => g.id === prev.id);
          const next = e.key === 'ArrowRight' ? idx + 1 : idx - 1;
          return generations[Math.max(0, Math.min(next, generations.length - 1))] || prev;
        });
      }
    };
    window.addEventListener('keydown', handle);
    return () => window.removeEventListener('keydown', handle);
  }, [onClose, generations]);

  // Shortcut events for vote/delete while gallery is open
  const selectedRef = useRef(selected);
  const multiSelectedRef = useRef(multiSelected);
  const onVoteRef = useRef(onVote);
  const onDeleteRef = useRef(onDelete);
  const generationsRef = useRef(generations);
  useEffect(() => { selectedRef.current = selected; }, [selected]);
  useEffect(() => { multiSelectedRef.current = multiSelected; }, [multiSelected]);
  useEffect(() => { onVoteRef.current = onVote; }, [onVote]);
  useEffect(() => { onDeleteRef.current = onDelete; }, [onDelete]);
  useEffect(() => { generationsRef.current = generations; }, [generations]);

  useEffect(() => {
    const onVoteEvent = async (e) => {
      const value = e.detail === 'up' ? 1 : -1;
      const multi = multiSelectedRef.current;
      const gens = generationsRef.current;
      const vote = onVoteRef.current;
      if (multi.size > 1) {
        for (const id of multi) {
          const gen = gens.find(g => g.id === id);
          if (gen) await vote(gen, value);
        }
      } else if (selectedRef.current) {
        await vote(selectedRef.current, value);
      }
    };
    const onDeleteEvent = async () => {
      const multi = multiSelectedRef.current;
      const gens = generationsRef.current;
      const del = onDeleteRef.current;
      const targets = multi.size > 1
        ? [...multi].map(id => gens.find(g => g.id === id)).filter(Boolean)
        : selectedRef.current ? [selectedRef.current] : [];
      const deletedIds = new Set(targets.map(g => g.id));
      for (const gen of targets) await del(gen);
      setMultiSelected(prev => { const n = new Set(prev); deletedIds.forEach(id => n.delete(id)); return n; });
      setSelected(s => (s && deletedIds.has(s.id)) ? null : s);
    };
    window.addEventListener('fuk-history-vote', onVoteEvent);
    window.addEventListener('fuk-history-delete-selected', onDeleteEvent);
    return () => {
      window.removeEventListener('fuk-history-vote', onVoteEvent);
      window.removeEventListener('fuk-history-delete-selected', onDeleteEvent);
    };
  }, []);

  // Draggable divider
  const handleDividerMouseDown = (e) => {
    e.preventDefault();
    isDragging.current   = true;
    dragStartY.current   = e.clientY;
    dragStartPct.current = previewPct;

    const onMove = (e) => {
      if (!isDragging.current || !bodyRef.current) return;
      const bodyH = bodyRef.current.getBoundingClientRect().height;
      const delta = ((e.clientY - dragStartY.current) / bodyH) * 100;
      setPreviewPct(Math.max(15, Math.min(85, dragStartPct.current + delta)));
    };
    const onUp = () => {
      isDragging.current = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  };

  // Assign a picked image/video to the next open A/B slot (or clear it if already picked).
  const handleCompareAssign = useCallback((generation) => {
    if (compareA?.id === generation.id) { setCompareA(null); return; } // toggle A off
    if (compareB?.id === generation.id) { setCompareB(null); return; } // toggle B off
    if (!compareA) { setCompareA(generation); return; }  // fill A first
    if (!compareB) { setCompareB(generation); return; }  // then fill B
    setCompareB(generation);                             // both full → replace B
  }, [compareA, compareB]);

  const toggleCompareMode = useCallback(() => {
    setCompareMode(m => {
      const next = !m;
      if (next) {
        // Seed slot A with the current selection when it's a still image.
        const seed = selected && !isGenVideo(selected) ? selected : null;
        setCompareA(seed);
        setCompareB(null);
        setMultiSelected(new Set());
      }
      return next;
    });
  }, [selected]);

  // Thumb click: in compare mode assign A/B; otherwise shift-click ranges / anchor
  const handleThumbSelect = useCallback((generation, shiftKey) => {
    if (compareMode) { handleCompareAssign(generation); return; }
    if (shiftKey && anchorId.current) {
      const ids = orderedGens.map(g => g.id);
      const aIdx = ids.indexOf(anchorId.current);
      const cIdx = ids.indexOf(generation.id);
      const [lo, hi] = [Math.min(aIdx, cIdx), Math.max(aIdx, cIdx)];
      setMultiSelected(new Set(ids.slice(lo, hi + 1)));
    } else {
      setMultiSelected(new Set());
      anchorId.current = generation.id;
    }
    setSelected(generation);
  }, [orderedGens, compareMode, handleCompareAssign]);

  const handleDelete = useCallback(generation => {
    setDeleteConfirm({ gens: [generation], label: generation.name || generation.id });
  }, []);

  const executePendingDelete = useCallback(async () => {
    if (!deleteConfirm) return;
    const gens = deleteConfirm.gens;
    setDeleteConfirm(null);
    for (const gen of gens) {
      await onDelete(gen);
    }
    const deletedIds = new Set(gens.map(g => g.id));
    setMultiSelected(prev => {
      const next = new Set(prev);
      deletedIds.forEach(id => next.delete(id));
      return next;
    });
    setSelected(s => (s && deletedIds.has(s.id)) ? null : s);
  }, [deleteConfirm, onDelete]);

  const handleNavigate = useCallback((dir) => {
    setMultiSelected(new Set());
    setSelected(prev => {
      if (!prev) return generations[0] || null;
      const idx  = generations.findIndex(g => g.id === prev.id);
      const next = dir === 'next' ? idx + 1 : idx - 1;
      const gen  = generations[Math.max(0, Math.min(next, generations.length - 1))] || prev;
      anchorId.current = gen.id;
      return gen;
    });
  }, [generations]);

  const handleBulkVote = useCallback(async (value) => {
    for (const id of multiSelected) {
      const gen = generations.find(g => g.id === id);
      if (gen) await onVote(gen, value);
    }
  }, [multiSelected, generations, onVote]);

  const handleBulkDelete = useCallback(() => {
    const gens = [...multiSelected].map(id => generations.find(g => g.id === id)).filter(Boolean);
    setDeleteConfirm({ gens, label: `${gens.length} items` });
  }, [multiSelected, generations]);

  return (
    <div className="gallery-overlay">
      {/* header */}
      <div className="gallery-header">
        <div className="gallery-header-left">
          <Clock />
          <span className="gallery-header-title">Gallery</span>
          <span className="gallery-header-count">{generations.length} items</span>
        </div>
        <div className="gallery-header-center">
          <label className="gallery-zoom-label">
            <span>Thumb size</span>
            <input
              type="range" min={80} max={320} step={20}
              value={zoom}
              onChange={e => setZoom(Number(e.target.value))}
              className="gallery-zoom-slider"
            />
            <span>{zoom}px</span>
          </label>
          {onPlaybackSpeedChange && (
            <label className="gallery-playback-label">
              <span>Playback</span>
              <input
                type="range" min={6} max={48} step={6}
                value={playbackFPS}
                onChange={e => onPlaybackSpeedChange(parseInt(e.target.value) / 24)}
                className="gallery-playback-slider"
              />
              <span>{playbackFPS} FPS</span>
            </label>
          )}
        </div>
        <div className="gallery-header-right">
          <button
            className={`gallery-compare-btn${compareMode ? ' active' : ''}`}
            onClick={toggleCompareMode}
            title={compareMode ? 'Exit A/B compare' : 'A/B compare (before/after wipe)'}
          >
            <Columns />
            <span>Compare</span>
          </button>
          <button className="gallery-close-btn" onClick={onClose} title="Close (Esc)"><X /></button>
        </div>
      </div>

      {/* body: vertical split */}
      <div className="gallery-body" ref={bodyRef}>
        {/* large preview */}
        <div className="gallery-large-preview" style={{ height: `${previewPct}%` }}>
          {compareMode ? (
            (compareA && compareB) ? (
              <CompareLargeView
                a={compareA}
                b={compareB}
                onSwap={() => { setCompareA(compareB); setCompareB(compareA); }}
                onClearA={() => setCompareA(null)}
                onClearB={() => setCompareB(null)}
                onExit={() => setCompareMode(false)}
                playbackSpeed={playbackSpeed}
              />
            ) : (
              <div className="gallery-large-empty">
                {!compareA
                  ? 'A/B compare — click an image to set slot A'
                  : 'A/B compare — click a second image to set slot B'}
              </div>
            )
          ) : selected ? (
            <GalleryLargeView
              generation={selected}
              generations={generations}
              isPinned={pinnedIds.includes(selected.id)}
              vote={votes[selected.id] || 0}
              onTogglePin={onTogglePin}
              onVote={onVote}
              onDelete={handleDelete}
              onNavigate={handleNavigate}
              multiSelected={multiSelected}
              onBulkVote={handleBulkVote}
              onBulkDelete={handleBulkDelete}
              onClearSelection={() => setMultiSelected(new Set())}
              deleteConfirm={deleteConfirm}
              onConfirmDelete={executePendingDelete}
              onCancelDelete={() => setDeleteConfirm(null)}
              playbackSpeed={playbackSpeed}
            />
          ) : (
            <div className="gallery-large-empty">Click a thumbnail to preview</div>
          )}
        </div>

        {/* draggable divider */}
        <div className="gallery-drag-handle" onMouseDown={handleDividerMouseDown}>
          <div className="gallery-drag-handle-grip" />
        </div>

        {/* thumbnail grid */}
        <div className="gallery-grid-wrap">
          {pinnedGens.length > 0 && (
            <div className="gallery-section-label"><PinIcon /> Pinned</div>
          )}
          {pinnedGens.length > 0 && (
            <div className="gallery-grid" style={{ '--thumb-w': `${zoom}px` }}>
              {pinnedGens.map(g => (
                <GalleryThumb
                  key={g.id} generation={g}
                  isPinned={true}
                  isSelected={selected?.id === g.id}
                  isMultiSelected={multiSelected.has(g.id)}
                  compareRole={compareMode ? (compareA?.id === g.id ? 'a' : compareB?.id === g.id ? 'b' : null) : null}
                  vote={votes[g.id] || 0}
                  onSelect={handleThumbSelect}
                  onTogglePin={onTogglePin}
                  onVote={onVote}
                  onDelete={handleDelete}
                  onSendToStoryboard={onSendToStoryboard}
                  sendToStoryboardEnabled={sendToStoryboardEnabled}
                />
              ))}
            </div>
          )}
          {unpinnedGens.length > 0 && pinnedGens.length > 0 && (
            <div className="gallery-section-label gallery-section-label--top">Recent</div>
          )}
          <div className="gallery-grid" style={{ '--thumb-w': `${zoom}px` }}>
            {unpinnedGens.map(g => (
              <GalleryThumb
                key={g.id} generation={g}
                isPinned={false}
                isSelected={selected?.id === g.id}
                isMultiSelected={multiSelected.has(g.id)}
                compareRole={compareMode ? (compareA?.id === g.id ? 'a' : compareB?.id === g.id ? 'b' : null) : null}
                vote={votes[g.id] || 0}
                onSelect={handleThumbSelect}
                onTogglePin={onTogglePin}
                onVote={onVote}
                onDelete={handleDelete}
                onSendToStoryboard={onSendToStoryboard}
                sendToStoryboardEnabled={sendToStoryboardEnabled}
              />
            ))}
          </div>
          {(hasMore?.image || hasMore?.video) && (
            <div className="gen-history-load-more gallery-load-more">
              <button onClick={onLoadMore} disabled={loading}>
                Load More (+5 each)
              </button>
              <button onClick={onLoadAll} disabled={loading} className="secondary">
                Load All
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── main component ───────────────────────────────────────────────────────────

export default function GenerationHistory({ project, collapsed, onToggle, galleryOpen, onGalleryOpenChange, playbackSpeed, onPlaybackSpeedChange }) {
  const [generations, setGenerations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imgLimit, setImgLimit] = useState(5);
  const [videoLimit, setVideoLimit] = useState(5);
  const [hasMore, setHasMore] = useState({ image: false, video: false });
  // votes: { [generation.id]: 1 | -1 | 0 }
  const [votes, setVotes] = useState({});
  const setGalleryOpen = onGalleryOpenChange ?? (() => {});

  // Video playback speed refs
  const hoverVideoRef = useVideoPlayback(playbackSpeed);
  const thumbVideoRef = useVideoPlayback(playbackSpeed);

  // Hover preview state
  const [hoveredItem, setHoveredItem] = useState(null);
  const [hoverPosition, setHoverPosition] = useState({ x: 0, y: 0 });
  const hoverTimeoutRef = useRef(null);
  
  const handleItemHover = useCallback((generation, position) => {
    // Clear any pending timeout
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    
    // Show immediately if already showing, otherwise delay slightly
    if (hoveredItem) {
      setHoveredItem(generation);
      setHoverPosition(position);
    } else {
      hoverTimeoutRef.current = setTimeout(() => {
        setHoveredItem(generation);
        setHoverPosition(position);
      }, 300); // 300ms delay before showing
    }
  }, [hoveredItem]);
  
  const handleItemHoverEnd = useCallback(() => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    setHoveredItem(null);
  }, []);
  
  // Pinned items stored in project state
  // Read from project, with fallback to empty array
  const pinnedIds = project?.projectState?.pinnedGenerations || [];
  
  // Update pinned IDs via project state
  const setPinnedIds = useCallback((updater) => {
    if (!project?.updatePinnedGenerations) {
      console.warn('[History] Cannot update pins - no project loaded');
      return;
    }
    
    // Handle both direct values and updater functions
    const newIds = typeof updater === 'function' 
      ? updater(pinnedIds) 
      : updater;
    
    console.log('[History] Updating pinned IDs:', newIds);
    project.updatePinnedGenerations(newIds);
  }, [project?.updatePinnedGenerations, pinnedIds]);

  // Fetch generations from API
  const fetchGenerations = useCallback(async (imgLimitParam = 5, videoLimitParam = 5, force = false) => {
    if (!force && !project?.isProjectLoaded) {
      console.log('[History] Skipping fetch - no project loaded');
      setGenerations([]);
      return;
    }
    
    console.log(`[History] Fetching generations (img_limit=${imgLimitParam}, video_limit=${videoLimitParam})...`);
    setLoading(true);
    setError(null);
    
    try {
      const pinnedParam = pinnedIds.join(',');
      const res = await fetch(`${API_URL}/project/generations?img_limit=${imgLimitParam}&video_limit=${videoLimitParam}&pinned=${encodeURIComponent(pinnedParam)}`);
      console.log('[History] Response status:', res.status);
      
      if (!res.ok) {
        throw new Error(`Failed to fetch: ${res.statusText}`);
      }
      const data = await res.json();
      console.log('[History] Got generations:', data.generations?.length || 0, 'hasMore:', data.hasMore);
      
      if (data.error) {
        console.warn('[History] API returned error:', data.error);
      }
      
      setGenerations(data.generations || []);
      setHasMore(data.hasMore || { image: false, video: false });
      setImgLimit(imgLimitParam);
      setVideoLimit(videoLimitParam);
    } catch (err) {
      console.error('[History] Failed to fetch generations:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [project?.isProjectLoaded, pinnedIds]);

  // Fetch on mount and when project changes
  useEffect(() => {
    fetchGenerations(5, 5);
  }, [project?.currentFilename]);
  
  // Refetch when pins change (to update sort order)
  useEffect(() => {
    if (generations.length > 0) {
      fetchGenerations(imgLimit, videoLimit);
    }
  }, [pinnedIds]);
  
  // Listen for import registration events (from MediaUploader)
  useEffect(() => {
    const handleImportRegistered = (event) => {
      const { id, autoPin } = event.detail;
      console.log('[History] Import registered:', id, 'autoPin:', autoPin);
      
      // Add to pinned if auto-pin is true
      if (autoPin && id) {
        setPinnedIds(prev => {
          if (prev.includes(id)) return prev;
          return [...prev, id];
        });
      }
      
      // Refresh to show the new import
      fetchGenerations(imgLimit, videoLimit, true);

      

    };

    window.addEventListener('fuk-import-registered', handleImportRegistered);
    return () => window.removeEventListener('fuk-import-registered', handleImportRegistered);
  }, [imgLimit, videoLimit, fetchGenerations]);

  useEffect(() => {
    const handleProjectChange = (event) => {
      console.log('[History] Project changed, refreshing...', event.detail);
      setGenerations([]);
      setImgLimit(5);
      setVideoLimit(5);
      fetchGenerations(5, 5, true);
    };
    
    window.addEventListener('fuk-project-changed', handleProjectChange);
    return () => window.removeEventListener('fuk-project-changed', handleProjectChange);
  }, [fetchGenerations]);

  // Auto-refresh when generation completes
  useEffect(() => {
    const handleGenerationComplete = (event) => {
      console.log('[History] Generation complete, auto-refreshing...', event.detail);
      fetchGenerations(imgLimit, videoLimit, true);
    };
    
    window.addEventListener('fuk-generation-complete', handleGenerationComplete);
    return () => window.removeEventListener('fuk-generation-complete', handleGenerationComplete);
  }, [imgLimit, videoLimit, fetchGenerations]);

  // Auto-refresh when an external client (e.g. the Blender addon) saves an entry.
  // Cheap version-counter poll; refetch only when it actually changes.
  useEffect(() => {
    let last = null;
    const id = setInterval(async () => {
      try {
        const r = await fetch('/api/blender/signal');
        if (!r.ok) return;
        const { version } = await r.json();
        if (last === null) { last = version; return; }
        if (version !== last) {
          last = version;
          console.log('[History] Blender saved an entry, auto-refreshing...');
          fetchGenerations(imgLimit, videoLimit, true);
        }
      } catch (_) { /* server down / not ready — ignore */ }
    }, 3000);
    return () => clearInterval(id);
  }, [imgLimit, videoLimit, fetchGenerations]);


  const handleRefresh = () => {
    console.log('[History] Manual refresh clicked');
    fetchGenerations(imgLimit, videoLimit, true);
  };
  
  const handleLoadMore = () => {
    const newImgLimit = imgLimit + 5;
    const newVideoLimit = videoLimit + 5;
    fetchGenerations(newImgLimit, newVideoLimit, true);
  };

  const handleLoadAll = () => {
    // Load a large number to get everything
    fetchGenerations(1000, 1000, true);
  };

  const handleTogglePin = (generation) => {
    const id = generation.id;
    setPinnedIds(prev => {
      if (prev.includes(id)) {
        return prev.filter(p => p !== id);
      } else {
        return [...prev, id];
      }
    });
  };

  // Separate pinned and unpinned
  const pinnedGenerations = generations.filter(g => pinnedIds.includes(g.id));
  const unpinnedGenerations = generations.filter(g => !pinnedIds.includes(g.id));

  // Load votes on mount
  useEffect(() => {
    fetch(`${API_URL}/project/votes`)
      .then(r => r.ok ? r.json() : {})
      .then(data => setVotes(data || {}))
      .catch(() => {}); // Non-fatal — votes file may not exist yet
  }, []);

  const handleVote = useCallback(async (generation, value) => {
    const id = generation.id;
    // Optimistic update
    setVotes(prev => ({ ...prev, [id]: value }));
    try {
      await fetch(`${API_URL}/project/votes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, vote: value }),
      });
    } catch (err) {
      console.error('[History] Vote save failed:', err);
    }
  }, []);

  // Raw delete (no confirm) — callers handle confirmation
  const deleteGeneration = useCallback(async (generation) => {
    try {
      const res = await fetch(`${API_URL}/project/generations/${encodeURIComponent(generation.id)}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        setGenerations(prev => prev.filter(g => g.id !== generation.id));
        setPinnedIds(prev => prev.includes(generation.id) ? prev.filter(p => p !== generation.id) : prev);
      }
    } catch (err) {
      console.error('Failed to delete:', err);
    }
  }, [setPinnedIds]);

  const handleDelete = async (generation) => {
    if (!confirm(`Delete ${generation.name || generation.id}?`)) return;
    await deleteGeneration(generation);
  };

  // Pin a generation as the storyboard panel preview for the current shot.
  // The panel kind (image vs video) is inferred from the generation type.
  const currentShotId = project?.currentFileInfo?.shotNumber || null;
  const handleSendToStoryboard = useCallback(async (generation) => {
    if (!currentShotId) return;
    const kind = isGenVideo(generation) ? 'video' : 'image';
    const path = generation.path || generation.preview;
    if (!path) return;
    try {
      await setPanelPreview(currentShotId, { kind, path });
      // Mirror the generation's prompt into the panel field that drives this
      // kind of output, so the storyboard reflects what actually produced the
      // pinned media. Prefer the raw #marker draft over the resolved string.
      const prompt = generation.promptSource || generation.prompt || '';
      if (prompt) {
        const field = kind === 'image' ? 'imagery_prompt' : 'action_prompt';
        await upsertPanel(currentShotId, { [field]: prompt });
      }
      // Let any mounted storyboard hook refresh its manifest.
      window.dispatchEvent(new CustomEvent('fuk-storyboard-changed', {
        detail: { shotId: currentShotId, kind, path },
      }));
    } catch (e) {
      alert(`Failed to pin to storyboard: ${e.message}`);
    }
  }, [currentShotId]);

  // Count by type for header
  const counts = generations.reduce((acc, g) => {
    acc[g.type] = (acc[g.type] || 0) + 1;
    return acc;
  }, {});

  if (collapsed) {
    return (
      <div className="gen-history-collapsed" onClick={onToggle}>
        <Clock />
        <span>History ({generations.length})</span>
        <ChevronRight />
      </div>
    );
  }

  return (
    <div className="gen-history-panel">
      <div className="gen-history-header">
        <div className="gen-history-title" onClick={onToggle}>
          <ChevronDown />
          <Clock />
          <span>History</span>
          <span className="gen-history-counts">
            {counts.image > 0 && <span className="count-badge image"><Camera /> {counts.image}</span>}
            {counts.video > 0 && <span className="count-badge video"><Film /> {counts.video}</span>}
            {counts.layers > 0 && <span className="count-badge layers"><Layers /> {counts.layers}</span>}
            {counts.preprocess > 0 && <span className="count-badge preprocess"><Enhance /> {counts.preprocess}</span>}
            {counts.upscale > 0 && <span className="count-badge upscale"><ArrowUp /> {counts.upscale}</span>}
            {counts.interpolate > 0 && <span className="count-badge interpolate"><Zap /> {counts.interpolate}</span>}
            {counts.export > 0 && <span className="count-badge export"><Download /> {counts.export}</span>}
            {counts.import > 0 && <span className="count-badge import"><ImportIcon /> {counts.import}</span>}
          </span>
        </div>
        
        <div className="gen-history-header-actions">
          <span className="gen-history-days-label">
            {imgLimit >= 1000 ? 'All' : `${imgLimit}/${videoLimit}`}
          </span>
          <button
            className="gen-history-refresh"
            onClick={() => setGalleryOpen(true)}
            disabled={generations.length === 0}
            title="Open gallery"
          >
            <Maximize2 />
          </button>
          <button
            className="gen-history-refresh"
            onClick={handleRefresh}
            disabled={loading}
            title="Refresh"
          >
            <RefreshCw className={loading ? 'spinning' : ''} />
          </button>
        </div>
      </div>

      <div className="gen-history-content">
        {loading && generations.length === 0 ? (
          <div className="gen-history-loading">Loading...</div>
        ) : error ? (
          <div className="gen-history-error">
            {error}
            <button onClick={handleRefresh}>Retry</button>
          </div>
        ) : generations.length === 0 ? (
          <div className="gen-history-empty">
            {project?.isProjectLoaded 
              ? 'No generations yet. Create some!'
              : 'Open a project to see history'}
          </div>
        ) : (
          <>
            {/* Pinned section */}
            {pinnedGenerations.length > 0 && (
              <div className="gen-history-section pinned">
                <div className="gen-history-section-label">
                  <PinIcon /> Pinned
                </div>
                <div className="gen-history-grid">
                  {pinnedGenerations.map(gen => (
                    <DraggableThumbnail
                      key={gen.id}
                      generation={gen}
                      onDelete={handleDelete}
                      onTogglePin={handleTogglePin}
                      isPinned={true}
                      onHover={handleItemHover}
                      onHoverEnd={handleItemHoverEnd}
                      videoRefPlayback={thumbVideoRef}
                      vote={votes[gen.id] || 0}
                      onVote={handleVote}
                      onSendToStoryboard={handleSendToStoryboard}
                      sendToStoryboardEnabled={!!currentShotId}
                    />
                  ))}
                </div>
              </div>
            )}
            
            {/* Recent section */}
            <div className="gen-history-section recent">
              {pinnedGenerations.length > 0 && (
                <div className="gen-history-section-label">Recent</div>
              )}
              <div className="gen-history-grid">
                {unpinnedGenerations.map(gen => (
                  <DraggableThumbnail
                    key={gen.id}
                    generation={gen}
                    onDelete={handleDelete}
                    onTogglePin={handleTogglePin}
                    isPinned={false}
                    onHover={handleItemHover}
                    onHoverEnd={handleItemHoverEnd}
                    videoRefPlayback={thumbVideoRef}
                    vote={votes[gen.id] || 0}
                    onVote={handleVote}
                    onSendToStoryboard={handleSendToStoryboard}
                    sendToStoryboardEnabled={!!currentShotId}
                  />
                ))}
              </div>
            </div>
            
            {/* Load more */}
            {(hasMore.image || hasMore.video) &&(
              <div className="gen-history-load-more">
                <button onClick={handleLoadMore} disabled={loading}>
                  Load More (+5 each)
                </button>
                <button onClick={handleLoadAll} disabled={loading} className="secondary">
                  Load All
                </button>
              </div>
            )}
          </>
        )}
      </div>

      <div className="gen-history-hint">
        Drag items to input fields • Click bookmark to pin
      </div>
      
      {/* Hover preview popup */}
      {hoveredItem && (
        <HoverPreview
          generation={hoveredItem}
          position={hoverPosition}
          videoRef={hoverVideoRef}
        />
      )}

      {/* Fullscreen gallery overlay */}
      {galleryOpen && (
        <FullscreenGallery
          generations={generations}
          pinnedIds={pinnedIds}
          votes={votes}
          onTogglePin={handleTogglePin}
          onVote={handleVote}
          onDelete={deleteGeneration}
          onClose={() => setGalleryOpen(false)}
          hasMore={hasMore}
          loading={loading}
          onLoadMore={handleLoadMore}
          onLoadAll={handleLoadAll}
          onSendToStoryboard={handleSendToStoryboard}
          sendToStoryboardEnabled={!!currentShotId}
          playbackSpeed={playbackSpeed}
          onPlaybackSpeedChange={onPlaybackSpeedChange}
        />
      )}
    </div>
  );
}

// NOTE: registerImport has been moved to ../utils/historyApi.js
// Import it from there: import { registerImport } from '../utils/historyApi'