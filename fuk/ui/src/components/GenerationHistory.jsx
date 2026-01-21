/**
 * Generation History Component
 * Shows past generations with drag-and-drop, pagination, and pinning
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Film, Camera, Clock, Trash2, RefreshCw, ChevronDown, ChevronRight, Enhance, Zap, ArrowUp, Layers, Download, PinIcon, ImportIcon, SequenceIcon } from './Icons';
import { buildImageUrl, API_URL } from '../utils/constants';



// Draggable thumbnail component
function DraggableThumbnail({ generation, onDelete, onTogglePin, isPinned }) {
  const [imageError, setImageError] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const videoRef = useRef(null);
  
  const isVideo = generation.type === 'video' || 
                  generation.type === 'interpolate' || 
                  (generation.type === 'preprocess' && generation.subtype === 'video') ||
                  (generation.type === 'upscale' && generation.subtype === 'video');
  const isSequence = generation.isSequence;
  const previewUrl = buildImageUrl(generation.preview);
  const thumbnailUrl = generation.thumbnailUrl ? buildImageUrl(generation.thumbnailUrl) : null;
  const displayName = generation.name || generation.id || 'Unknown';
  
  // Clean up video when hover ends
  const handleMouseLeave = () => {
    setIsHovering(false);
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.src = '';
      videoRef.current.load(); // Force unload
    }
  };
  
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
  };

  return (
    <div 
      className={`gen-history-item ${isPinned ? 'pinned' : ''}`}
      draggable
      onDragStart={handleDragStart}
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
            // Video: show thumbnail when not hovering, video when hovering
            <div
              style={{ width: '100%', height: '100%', position: 'relative' }}
              onMouseEnter={() => setIsHovering(true)}
              onMouseLeave={handleMouseLeave}
            >
              {!isHovering && thumbnailUrl ? (
                // Show thumbnail image when not hovering
                <img 
                  src={thumbnailUrl} 
                  alt={displayName}
                  onError={() => setImageError(true)}
                />
              ) : (
                // Show video when hovering (or if no thumbnail available)
                <video 
                  ref={videoRef}
                  src={isHovering ? previewUrl : ''}
                  muted
                  loop
                  preload="none"
                  autoPlay={isHovering}
                  onError={() => setImageError(true)}
                />
              )}
            </div>
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

export default function GenerationHistory({ project, collapsed, onToggle }) {
  const [generations, setGenerations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasMore, setHasMore] = useState(false);
  const [daysLoaded, setDaysLoaded] = useState(1);
  
  // Pinned items stored in localStorage
  const [pinnedIds, setPinnedIds] = useState(() => {
    try {
      const saved = localStorage.getItem('fuk_pinned_generations');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  
  // Save pinned to localStorage
  useEffect(() => {
    localStorage.setItem('fuk_pinned_generations', JSON.stringify(pinnedIds));
  }, [pinnedIds]);

  // Fetch generations from API
  const fetchGenerations = useCallback(async (days = 1, force = false) => {
    if (!force && !project?.isProjectLoaded) {
      console.log('[History] Skipping fetch - no project loaded');
      setGenerations([]);
      return;
    }
    
    console.log(`[History] Fetching generations (days=${days})...`);
    setLoading(true);
    setError(null);
    
    try {
      const pinnedParam = pinnedIds.join(',');
      const res = await fetch(`${API_URL}/project/generations?days=${days}&pinned=${encodeURIComponent(pinnedParam)}`);
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
      setHasMore(data.hasMore || false);
      setDaysLoaded(days);
    } catch (err) {
      console.error('[History] Failed to fetch generations:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [project?.isProjectLoaded, pinnedIds]);

  // Fetch on mount and when project changes
  useEffect(() => {
    fetchGenerations(1);
  }, [project?.currentFilename]);
  
  // Refetch when pins change (to update sort order)
  useEffect(() => {
    if (generations.length > 0) {
      fetchGenerations(daysLoaded);
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
      fetchGenerations(daysLoaded, true);

      

    };

    window.addEventListener('fuk-import-registered', handleImportRegistered);
    return () => window.removeEventListener('fuk-import-registered', handleImportRegistered);
  }, [daysLoaded, fetchGenerations]);

  useEffect(() => {
    const handleProjectChange = (event) => {
      console.log('[History] Project changed, refreshing...', event.detail);
      setGenerations([]);
      setDaysLoaded(1);
      fetchGenerations(1, true);
    };
    
    window.addEventListener('fuk-project-changed', handleProjectChange);
    return () => window.removeEventListener('fuk-project-changed', handleProjectChange);
  }, [fetchGenerations]);

  // Auto-refresh when generation completes
  useEffect(() => {
    const handleGenerationComplete = (event) => {
      console.log('[History] Generation complete, auto-refreshing...', event.detail);
      fetchGenerations(daysLoaded, true);
    };
    
    window.addEventListener('fuk-generation-complete', handleGenerationComplete);
    return () => window.removeEventListener('fuk-generation-complete', handleGenerationComplete);
  }, [daysLoaded, fetchGenerations]);


  const handleRefresh = () => {
    console.log('[History] Manual refresh clicked');
    fetchGenerations(daysLoaded, true);
  };
  
  const handleLoadMore = () => {
    const newDays = daysLoaded + 7; // Load another week
    fetchGenerations(newDays, true);
  };
  
  const handleLoadAll = () => {
    fetchGenerations(0, true); // 0 = all
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
            {daysLoaded === 0 ? 'All' : `${daysLoaded}d`}
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
                  />
                ))}
              </div>
            </div>
            
            {/* Load more */}
            {hasMore && (
              <div className="gen-history-load-more">
                <button onClick={handleLoadMore} disabled={loading}>
                  Load More (+7 days)
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
        Drag items to input fields Click bookmark to pin
      </div>
    </div>
  );
}

// NOTE: registerImport has been moved to ../utils/historyApi.js
// Import it from there: import { registerImport } from '../utils/historyApi'