/**
 * Generation History Component
 * Shows all past generations with drag-and-drop support
 */

import { useState, useEffect, useCallback } from 'react';
import { Film, Camera, Clock, Trash2, RefreshCw, ChevronDown, ChevronRight, Enhance } from './Icons';
import { buildImageUrl, API_URL } from '../utils/constants';

// Draggable thumbnail component
function DraggableThumbnail({ generation, onDelete }) {
  const [imageError, setImageError] = useState(false);
  
  const isVideo = generation.type === 'video';
  const isPreprocess = generation.type === 'preprocess';
  const previewUrl = buildImageUrl(generation.preview);
  
  // Get the appropriate icon
  const TypeIcon = isVideo ? Film : isPreprocess ? Enhance : Camera;
  
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
  const fetchGenerations = useCallback(async () => {
    if (!project?.isProjectLoaded) {
      setGenerations([]);
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${API_URL}/project/generations`);
      if (!res.ok) {
        throw new Error(`Failed to fetch: ${res.statusText}`);
      }
      const data = await res.json();
      setGenerations(data.generations || []);
    } catch (err) {
      console.error('Failed to fetch generations:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [project?.isProjectLoaded]);

  // Fetch on mount and when project changes
  useEffect(() => {
    fetchGenerations();
  }, [fetchGenerations, project?.currentFilename]);

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
          </span>
        </div>
        
        <div className="gen-history-actions">
          <button 
            className="gen-history-refresh"
            onClick={fetchGenerations}
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
            <button onClick={fetchGenerations}>Retry</button>
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