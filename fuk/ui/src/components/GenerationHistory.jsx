/**
 * Generation History Component
 * Shows past generations with drag-and-drop, pagination, and pinning
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Film, Camera, Clock, Trash2, RefreshCw, ChevronDown, ChevronRight, Enhance, Zap, ArrowUp, Layers, Download, PinIcon, ImportIcon, SequenceIcon } from './Icons';
import { buildImageUrl, API_URL } from '../utils/constants';
import { useVideoPlayback } from '../hooks/useVideoPlayback';



// Hover preview popup component
function HoverPreview({ generation, position, videoRef }) {
  const isVideo = generation.type === 'video' || generation.type === 'interpolate' || 
                  (generation.type === 'preprocess' && generation.subtype === 'video');
  const isSequence = generation.isSequence;
  const previewUrl = buildImageUrl(generation.preview);
  const displayName = generation.name || generation.id || 'Unknown';
  
  // Calculate position - keep popup in viewport
  const style = {
    position: 'fixed',
    left: position.x + 16,
    top: position.y - 100,
    zIndex: 9999,
  };
  
  // Clamp to viewport
  if (style.left + 320 > window.innerWidth) {
    style.left = position.x - 336; // Show on left side instead
  }
  if (style.top < 8) {
    style.top = 8;
  }
  if (style.top + 200 > window.innerHeight) {
    style.top = window.innerHeight - 208;
  }
  
  return (
    <div className="gen-history-hover-preview" style={style}>
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
function DraggableThumbnail({ generation, onDelete, onTogglePin, isPinned, onHover, onHoverEnd, videoRefPlayback }) {
  const [imageError, setImageError] = useState(false);
  const videoRef = useRef(null);
  const isVideo = generation.type === 'video' || 
                  generation.type === 'interpolate' || 
                  (generation.type === 'preprocess' && generation.subtype === 'video') ||
                  (generation.type === 'upscale' && generation.subtype === 'video');
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
      <span style={{
        fontSize: '0.5rem',
        padding: '0.125rem 0.25rem',
        borderRadius: '0.125rem',
        background: colors[generation.subtype] || '#6b7280',
        color: 'white',
        marginLeft: '0.25rem',
        textTransform: 'uppercase',
        fontWeight: 600,
      }}>
        {generation.subtype}
      </span>
    );
  };
  
  // Get frame count badge for sequences
  const getFrameCountBadge = () => {
    if (!isSequence || !generation.frameCount) return null;
    
    return (
      <span style={{
        fontSize: '0.5rem',
        padding: '0.125rem 0.25rem',
        borderRadius: '0.125rem',
        background: '#6366f1',
        color: 'white',
        marginLeft: '0.25rem',
        fontWeight: 600,
      }}>
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
      title={generation.sourcePath 
        ? `Drag to use as input\n${displayName}\nSource: ${generation.sourcePath}\n${generation.timestamp}`
        : `Drag to use as input\n${displayName}\n${generation.timestamp}`
      }
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
            <video 
              ref={videoRefPlayback}
              src={previewUrl}
              muted
              loop
              preload="none"
              onMouseEnter={(e) => {
                // Load and play only on hover
                if (e.target.readyState === 0) {
                  e.target.load();
                }
                e.target.play().catch(() => {});
              }}
              onMouseLeave={(e) => { 
                e.target.pause(); 
                e.target.currentTime = 0;
              }}
              onError={() => setImageError(true)}
            />
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
      
      <div className="gen-history-info">
        <span className="gen-history-name">
          {displayName.split('/').pop()}
          {getSubtypeBadge()}
          {getFrameCountBadge()}
        </span>
        <span className="gen-history-time">{generation.timestamp}</span>
      </div>
      
      {/* Action buttons - show on hover */}
      <div className="gen-history-actions-overlay">
        <button 
          className={`gen-history-pin ${isPinned ? 'active' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            onTogglePin(generation);
          }}
          title={isPinned ? "Unpin" : "Pin to top"}
        >
          <PinIcon />
        </button>
        
        {onDelete && (
          <button 
            className="gen-history-delete"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(generation);
            }}
            title="Delete generation"
          >
            <Trash2 />
          </button>
        )}
      </div>
    </div>
  );
}

export default function GenerationHistory({ project, collapsed, onToggle, playbackSpeed }) {
  const [generations, setGenerations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imgLimit, setImgLimit] = useState(5);
  const [videoLimit, setVideoLimit] = useState(5);
  const [hasMore, setHasMore] = useState({ image: false, video: false });
  
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

  const handleDelete = async (generation) => {
    if (!confirm(`Delete ${generation.name || generation.id}?`)) return;
    
    try {
      const res = await fetch(`${API_URL}/project/generations/${encodeURIComponent(generation.id)}`, {
        method: 'DELETE'
      });
      
      if (res.ok) {
        setGenerations(prev => prev.filter(g => g.id !== generation.id));
        // Also remove from pinned if it was pinned
        setPinnedIds(prev => prev.filter(p => p !== generation.id));
      }
    } catch (err) {
      console.error('Failed to delete:', err);
    }
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
    </div>
  );
}

// NOTE: registerImport has been moved to ../utils/historyApi.js
// Import it from there: import { registerImport } from '../utils/historyApi'