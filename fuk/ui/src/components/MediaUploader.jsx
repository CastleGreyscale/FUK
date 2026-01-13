/**
 * Media Uploader Component
 * Handles images, videos, and image sequences with:
 * - Native file dialog via tkinter backend (primary method)
 * - Drag and drop from history panel (passes paths directly)
 * - Image sequence detection (render.####.exr)
 * - Thumbnail previews for all media types
 * 
 * NOTE: Drag-drop from desktop is disabled - use Browse button for local files.
 * This keeps files in-place without copying/uploading.
 */

import { useState, useId, useCallback } from 'react';
import { X, Upload, Film, Layers, Image, FolderOpen } from './Icons';
import { registerImport } from '../utils/historyApi';

const API_URL = '/api';

// Media type detection
const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.mxf'];
const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.exr', '.dpx'];

function getMediaType(path) {
  if (!path) return 'unknown';
  const ext = '.' + path.toLowerCase().split('.').pop();
  if (VIDEO_EXTENSIONS.includes(ext)) return 'video';
  if (path.includes('####') || path.includes('%0')) return 'sequence';
  if (IMAGE_EXTENSIONS.includes(ext)) return 'image';
  return 'unknown';
}

function getMediaIcon(mediaType) {
  switch (mediaType) {
    case 'video': return Film;
    case 'sequence': return Layers;
    case 'image': 
    default: return Image;
  }
}

// Build display URL for media
function buildMediaUrl(path) {
  if (!path) return '';
  // Handle project-relative paths (already formatted as API URLs)
  if (path.startsWith('/api/')) return path;
  if (path.startsWith('api/project/')) return `/${path}`;
  if (path.startsWith('api/')) return `/${path}`;
  // Handle absolute paths that need to go through file serving
  if (path.startsWith('/')) return `/api/project/files${path}`;
  return `/${path}`;
}

export default function MediaUploader({ 
  media,              // Array of media objects: [{path, displayName, mediaType, ...}]
  onMediaChange,      // Callback when media changes
  images,             // LEGACY: Array of string paths (for backward compat with ImageUploader)
  onImagesChange,     // LEGACY: Callback with string paths (for backward compat)
  disabled = false,
  multiple = true,    // Allow multiple files
  accept = 'all',     // 'all', 'images', 'videos', 'exr'
  detectSequences = true,
  id: providedId,
  label = 'Drop media or click to browse',
}) {
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState(null);
  
  const reactId = useId();
  const inputId = providedId || `media-upload-${reactId}`;
  
  // Backward compatibility: handle both object and string array formats
  const isLegacyMode = !!images || !!onImagesChange;
  const rawMedia = media || images || [];
  const handleChange = onMediaChange || onImagesChange;
  
  // Normalize media to array of objects
  const normalizedMedia = rawMedia.map(item => {
    if (typeof item === 'string') {
      return {
        path: item,
        displayName: item.split('/').pop(),
        mediaType: getMediaType(item),
      };
    }
    return {
      ...item,
      mediaType: item.mediaType || getMediaType(item.path),
    };
  });
  
  // Helper to call onChange with correct format
  const callOnChange = useCallback((newMedia) => {
    if (isLegacyMode) {
      // Legacy mode: return array of string paths
      const paths = newMedia.map(item => typeof item === 'string' ? item : item.path);
      handleChange(paths);
    } else {
      // New mode: return array of objects
      handleChange(newMedia);
    }
  }, [isLegacyMode, handleChange]);

  // Open native file dialog via backend (PRIMARY METHOD)
  // Returns absolute paths - no upload/copy needed
  const handleBrowse = useCallback(async () => {
    if (disabled || loading) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/browser/open`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: 'Select Media',
          multiple,
          detect_sequences: detectSequences,
          filter: accept,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Browser failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success && data.files?.length > 0) {
        const newMedia = data.files.map(f => ({
          path: f.path,  // Absolute path - file stays in place
          displayName: f.display_name,
          mediaType: f.media_type,
          firstFrame: f.first_frame,
          lastFrame: f.last_frame,
          frameCount: f.frame_count,
          framePattern: f.frame_pattern,
        }));
        
        if (multiple) {
          callOnChange([...normalizedMedia, ...newMedia]);
        } else {
          callOnChange(newMedia.slice(0, 1));
        }
        
        console.log('[MediaUploader] Selected media (absolute paths):', newMedia);

        // Register imports in history and auto-pin
        for (const file of data.files) {
          const result = await registerImport(file.path, file.display_name, true);
          if (result.success && result.auto_pin && result.id) {
            // Dispatch event to notify GenerationHistory to add pin and refresh
            window.dispatchEvent(new CustomEvent('fuk-import-registered', {
              detail: { id: result.id, autoPin: true }
            }));
          }
        }
      } else if (data.error) {
        throw new Error(data.error);
      }
      // If no files selected (user cancelled), just do nothing
      
    } catch (err) {
      console.error('Browse failed:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [disabled, loading, multiple, detectSequences, accept, normalizedMedia, callOnChange]);

  // Handle drag and drop
  // Only accepts drops from history panel (paths) - NOT desktop files
  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    setDragOver(false);
    
    if (disabled) return;
    
    // Check for path data (drag from history panel or other FUK components)
    const pathData = e.dataTransfer.getData('text/plain');
    const fukGenData = e.dataTransfer.getData('application/x-fuk-generation');
    console.log('[MediaUploader] Drop received:', { pathData, hasFukData: !!fukGenData });
    
    // Accept drag from history (has path data starting with / or api/)
    if (pathData && (pathData.startsWith('/') || pathData.startsWith('api/'))) {
      const newItem = {
        path: pathData,
        displayName: pathData.split('/').pop(),
        mediaType: getMediaType(pathData),
      };
      
      // If we have full generation data, use it for richer info
      if (fukGenData) {
        try {
          const genData = JSON.parse(fukGenData);
          newItem.displayName = genData.name || newItem.displayName;
          newItem.mediaType = genData.type || newItem.mediaType;
        } catch (e) {
          // Ignore parse errors
        }
      }
      
      if (multiple) {
        callOnChange([...normalizedMedia, newItem]);
      } else {
        callOnChange([newItem]);
      }
      
      console.log('[MediaUploader] Dropped from history:', newItem);
      return;
    }
    
    // Desktop file drops are not supported - show helpful message
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      setError('Desktop drag-drop not supported. Click to browse and select files.');
      console.log('[MediaUploader] Desktop drop blocked - use Browse button');
      return;
    }
  }, [disabled, multiple, normalizedMedia, callOnChange]);

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleRemove = (index) => {
    callOnChange(normalizedMedia.filter((_, i) => i !== index));
  };

  return (
    <div className="fuk-media-uploader">
      {/* Drop Zone - Click opens native browse, drag accepts history items */}
      <div 
        className={`fuk-dropzone ${dragOver ? 'fuk-dropzone--dragover' : ''} ${disabled ? 'fuk-dropzone--disabled' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleBrowse}
      >
        <div className="fuk-dropzone-content">
          {loading ? (
            <>
              <div className="fuk-dropzone-spinner" />
              <span className="fuk-dropzone-text">Loading...</span>
            </>
          ) : (
            <>
              <FolderOpen className="fuk-dropzone-icon" />
              <span className="fuk-dropzone-text">{label}</span>
              <span className="fuk-dropzone-hint">
                {normalizedMedia.length > 0 
                  ? `${normalizedMedia.length} item${normalizedMedia.length !== 1 ? 's' : ''} selected`
                  : 'Click to browse or drag from history'
                }
              </span>
            </>
          )}
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="fuk-upload-error">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="fuk-upload-error-dismiss">×</button>
        </div>
      )}

      {/* Media Grid */}
      {normalizedMedia.length > 0 && (
        <div className="fuk-media-grid">
          {normalizedMedia.map((item, index) => {
            const Icon = getMediaIcon(item.mediaType);
            const isVideo = item.mediaType === 'video';
            const isSequence = item.mediaType === 'sequence';
            
            return (
              <div key={index} className="fuk-media-thumb">
                <div className="fuk-media-thumb-preview">
                  {isVideo ? (
                    <video
                      src={buildMediaUrl(item.path)}
                      className="fuk-media-thumb-video"
                      muted
                      preload="metadata"
                    />
                  ) : isSequence ? (
                    <div className="fuk-media-thumb-sequence">
                      <Layers className="fuk-media-thumb-sequence-icon" />
                      <span className="fuk-media-thumb-sequence-info">
                        {item.frameCount || '?'} frames
                      </span>
                    </div>
                  ) : (
                    <img
                      src={buildMediaUrl(item.path)}
                      alt={item.displayName}
                      className="fuk-media-thumb-img"
                    />
                  )}
                  
                  {/* Type badge */}
                  <div className={`fuk-media-thumb-badge fuk-media-thumb-badge--${item.mediaType}`}>
                    <Icon className="fuk-media-thumb-badge-icon" />
                  </div>
                </div>
                
                <div className="fuk-media-thumb-info">
                  <span className="fuk-media-thumb-name" title={item.displayName}>
                    {item.displayName}
                  </span>
                </div>
                
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemove(index);
                  }}
                  className="fuk-media-thumb-remove"
                  title="Remove"
                >
                  <X className="fuk-media-thumb-remove-icon" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Export helper for backward compatibility
export function isVideoFile(path) {
  if (!path) return false;
  const ext = '.' + path.toLowerCase().split('.').pop();
  return VIDEO_EXTENSIONS.includes(ext);
}

export function isSequenceFile(path) {
  if (!path) return false;
  return path.includes('####') || path.includes('%0');
}

export function isImageFile(path) {
  if (!path) return false;
  const ext = '.' + path.toLowerCase().split('.').pop();
  return IMAGE_EXTENSIONS.includes(ext) && !isSequenceFile(path);
}