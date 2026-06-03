/**
 * Generation History Component
 * Shows past generations with drag-and-drop, pagination, and pinning
 */

import { useState, useEffect, useLayoutEffect, useCallback, useRef, useMemo } from 'react';
import { Film, Camera, Clock, Trash2, RefreshCw, ChevronDown, ChevronRight, Enhance, Zap, ArrowUp, Layers, Download, PinIcon, ImportIcon, SequenceIcon, ThumbsUp, ThumbsDown, Maximize2, X } from './Icons';
import { buildImageUrl, API_URL } from '../utils/constants';
import { useVideoPlayback } from '../hooks/useVideoPlayback';
import ZoomableImage from './ZoomableImage';
import ConformHDButton from './ConformHDButton';



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
function DraggableThumbnail({ generation, onDelete, onTogglePin, isPinned, onHover, onHoverEnd, videoRefPlayback, vote, onVote }) {
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

function GalleryThumb({ generation, isPinned, isSelected, isMultiSelected, vote, onSelect, onTogglePin, onVote, onDelete }) {
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
      className={`gallery-thumb${isSelected ? ' selected' : ''}${isPinned ? ' pinned' : ''}${isMultiSelected ? ' multi-selected' : ''}`}
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
      </div>

      {/* action bar */}
      <div className="gen-history-actions-bar">
        <button
          className={`gen-history-pin ${isPinned ? 'active' : ''}`}
          onClick={e => { e.stopPropagation(); onTogglePin(generation); }}
          title={isPinned ? 'Unpin' : 'Pin'}
        ><PinIcon /></button>
        <div className="gen-history-actions-bar-right">
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

// ─── large preview (top panel in the gallery) ────────────────────────────────

function GalleryLargeView({ generation, generations, isPinned, vote, onTogglePin, onVote, onDelete, onNavigate, multiSelected, onBulkVote, onBulkDelete, onClearSelection, deleteConfirm, onConfirmDelete, onCancelDelete }) {
  const preview  = buildImageUrl(generation.preview);
  const video    = isGenVideo(generation);
  const ProcIcon = processingIcon(generation);
  const procType = processingType(generation);
  const currentIdx = generations.findIndex(g => g.id === generation.id);
  const hasPrev = currentIdx > 0;
  const hasNext = currentIdx < generations.length - 1;

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
          ? <video key={preview} src={preview} controls autoPlay muted loop playsInline />
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

// ─── fullscreen gallery overlay ───────────────────────────────────────────────

function FullscreenGallery({ generations, pinnedIds, votes, onTogglePin, onVote, onDelete, onClose, hasMore, loading, onLoadMore, onLoadAll }) {
  const [zoom, setZoom]               = useState(160);
  const [selected, setSelected]       = useState(generations[0] || null);
  const [previewPct, setPreviewPct]   = useState(55);
  const [multiSelected, setMultiSelected] = useState(new Set());
  const [deleteConfirm, setDeleteConfirm] = useState(null); // { ids: [...], label: '' }
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

  // Thumb click: shift-click ranges, plain click sets anchor
  const handleThumbSelect = useCallback((generation, shiftKey) => {
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
  }, [orderedGens]);

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
        </div>
        <div className="gallery-header-right">
          <button className="gallery-close-btn" onClick={onClose} title="Close (Esc)"><X /></button>
        </div>
      </div>

      {/* body: vertical split */}
      <div className="gallery-body" ref={bodyRef}>
        {/* large preview */}
        <div className="gallery-large-preview" style={{ height: `${previewPct}%` }}>
          {selected ? (
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
                  vote={votes[g.id] || 0}
                  onSelect={handleThumbSelect}
                  onTogglePin={onTogglePin}
                  onVote={onVote}
                  onDelete={handleDelete}
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
                vote={votes[g.id] || 0}
                onSelect={handleThumbSelect}
                onTogglePin={onTogglePin}
                onVote={onVote}
                onDelete={handleDelete}
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

export default function GenerationHistory({ project, collapsed, onToggle, galleryOpen, onGalleryOpenChange, playbackSpeed }) {
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
        />
      )}
    </div>
  );
}

// NOTE: registerImport has been moved to ../utils/historyApi.js
// Import it from there: import { registerImport } from '../utils/historyApi'