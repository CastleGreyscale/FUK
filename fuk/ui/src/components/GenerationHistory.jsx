/**
 * Generation History Component
 * Shows past generations with drag-and-drop, pagination, and pinning
 */

import { useState, useEffect, useCallback } from 'react';
import { Film, Camera, Clock, Trash2, RefreshCw, ChevronDown, ChevronRight, Enhance, Zap, ArrowUp, Layers, Download } from './Icons';
import { buildImageUrl, API_URL } from '../utils/constants';

// Pin icon (bookmark style)
const PinIcon = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
  </svg>
);

// Import icon (download arrow into folder)
const ImportIcon = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4 4m0 0l-4-4m4 4V4" />
  </svg>
);

// Sequence icon (film strip)
const SequenceIcon = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <rect x="2" y="4" width="20" height="16" rx="2" strokeWidth={2} />
    <line x1="6" y1="4" x2="6" y2="20" strokeWidth={1.5} />
    <line x1="18" y1="4" x2="18" y2="20" strokeWidth={1.5} />
    <line x1="2" y1="9" x2="6" y2="9" strokeWidth={1.5} />
    <line x1="2" y1="15" x2="6" y2="15" strokeWidth={1.5} />
    <line x1="18" y1="9" x2="22" y2="9" strokeWidth={1.5} />
    <line x1="18" y1="15" x2="22" y2="15" strokeWidth={1.5} />
  </svg>
);

// Draggable thumbnail component
function DraggableThumbnail({ generation, onDelete, onTogglePin, isPinned }) {
  const [imageError, setImageError] = useState(false);
  
  const isVideo = generation.type === 'video' || generation.type === 'interpolate' || 
                  (generation.type === 'preprocess' && generation.subtype === 'video');
  const isSequence = generation.isSequence;
  const previewUrl = buildImageUrl(generation.preview);
  const displayName = generation.name || generation.id || 'Unknown';
  
  // Get the appropriate icon based on type and subtype
  const getTypeIcon = () => {
    // Check for sequence first
    if (isSequence) return SequenceIcon;
    
    switch (generation.type) {
      case 'video': return Film;
      case 'interpolate': return Zap;
      case 'upscale': return ArrowUp;
      case 'preprocess': 
        // Return specific icon for preprocess subtypes
        if (generation.subtype === 'video') return Film;
        return Enhance;
      case 'layers': return Layers;
      case 'export': return Download;
      case 'import': return ImportIcon;
      default: return Camera;
    }
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
  
  const TypeIcon = getTypeIcon();
  
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
            <video 
              src={previewUrl}
              muted
              loop
              onMouseEnter={(e) => e.target.play()}
              onMouseLeave={(e) => { e.target.pause(); e.target.currentTime = 0; }}
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
            <TypeIcon />
          </div>
        )}
        
        <div className={`gen-history-type-badge ${generation.type} ${generation.subtype || ''}`}>
          <TypeIcon />
        </div>
        
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
        Drag items to input fields Ã¢â‚¬Â¢ Click bookmark to pin
      </div>
    </div>
  );
}

// NOTE: registerImport has been moved to ../utils/historyApi.js
// Import it from there: import { registerImport } from '../utils/historyApi'