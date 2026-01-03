/**
 * Generation History Component
 * Shows all past generations with drag-and-drop support
 */

import { useState, useEffect, useCallback } from 'react';
import { Film, Camera, Clock, Trash2, RefreshCw, ChevronDown, ChevronRight, Enhance, Zap, ArrowUp } from './Icons';
import { buildImageUrl, API_URL } from '../utils/constants';

// Draggable thumbnail component
function DraggableThumbnail({ generation, onDelete }) {
  const [imageError, setImageError] = useState(false);
  
  const isVideo = generation.type === 'video' || generation.type === 'interpolate';
  const isPreprocess = generation.type === 'preprocess';
  const isUpscale = generation.type === 'upscale';
  const isInterpolate = generation.type === 'interpolate';
  const previewUrl = buildImageUrl(generation.preview);
  
  // Get the appropriate icon based on type
  const getTypeIcon = () => {
    switch (generation.type) {
      case 'video': return Film;
      case 'interpolate': return Zap;
      case 'upscale': return ArrowUp;
      case 'preprocess': return Enhance;
      default: return Camera;
    }
  };
  
  const TypeIcon = getTypeIcon();
  
  const handleDragStart = (e) => {
    // Set the data being dragged - the path that can be used as input
    e.dataTransfer.setData('text/plain', generation.path);
    e.dataTransfer.setData('application/x-fuk-generation', JSON.stringify(generation));
    e.dataTransfer.effectAllowed = 'copy';
    
    // Create a drag image
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
      className="gen-history-item"
      draggable
      onDragStart={handleDragStart}
      title={`Drag to use as input\n${generation.name}\n${generation.timestamp}`}
    >
      <div className="gen-history-thumb">
        {!imageError ? (
          isVideo ? (
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
              alt={generation.name}
              onError={() => setImageError(true)}
            />
          )
        ) : (
          <div className="gen-history-thumb-error">
            <TypeIcon />
          </div>
        )}
        
        <div className={`gen-history-type-badge ${generation.type}`}>
          <TypeIcon />
        </div>
      </div>
      
      <div className="gen-history-info">
        <span className="gen-history-name">{generation.name}</span>
        <span className="gen-history-time">{generation.timestamp}</span>
      </div>
      
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
  );
}

export default function GenerationHistory({ project, collapsed, onToggle }) {
  const [generations, setGenerations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch generations from API
  const fetchGenerations = useCallback(async (force = false) => {
    // Only skip if not forced AND no project loaded
    if (!force && !project?.isProjectLoaded) {
      console.log('[History] Skipping fetch - no project loaded');
      setGenerations([]);
      return;
    }
    
    console.log('[History] Fetching generations...');
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${API_URL}/project/generations`);
      console.log('[History] Response status:', res.status);
      
      if (!res.ok) {
        throw new Error(`Failed to fetch: ${res.statusText}`);
      }
      const data = await res.json();
      console.log('[History] Got generations:', data.generations?.length || 0);
      
      if (data.error) {
        console.warn('[History] API returned error:', data.error);
      }
      
      setGenerations(data.generations || []);
    } catch (err) {
      console.error('[History] Failed to fetch generations:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [project?.isProjectLoaded]);

  // Fetch on mount and when project changes
  useEffect(() => {
    fetchGenerations();
  }, [fetchGenerations, project?.currentFilename]);

  // Manual refresh - always force fetch
  const handleRefresh = () => {
    console.log('[History] Manual refresh clicked');
    fetchGenerations(true);
  };

  // Group by date
  const groupedGenerations = generations.reduce((groups, gen) => {
    const date = gen.date || 'Unknown';
    if (!groups[date]) groups[date] = [];
    groups[date].push(gen);
    return groups;
  }, {});

  const handleDelete = async (generation) => {
    if (!confirm(`Delete ${generation.name}?`)) return;
    
    try {
      const res = await fetch(`${API_URL}/project/generations/${encodeURIComponent(generation.id)}`, {
        method: 'DELETE'
      });
      
      if (res.ok) {
        setGenerations(prev => prev.filter(g => g.id !== generation.id));
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
            {counts.preprocess > 0 && <span className="count-badge preprocess"><Enhance /> {counts.preprocess}</span>}
            {counts.upscale > 0 && <span className="count-badge upscale"><ArrowUp /> {counts.upscale}</span>}
            {counts.interpolate > 0 && <span className="count-badge interpolate"><Zap /> {counts.interpolate}</span>}
          </span>
        </div>
        
        <div className="gen-history-actions">
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
          Object.entries(groupedGenerations).map(([date, gens]) => (
            <div key={date} className="gen-history-group">
              <div className="gen-history-date">{date}</div>
              <div className="gen-history-grid">
                {gens.map(gen => (
                  <DraggableThumbnail 
                    key={gen.id} 
                    generation={gen}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      <div className="gen-history-hint">
        Drag items to input fields
      </div>
    </div>
  );
}