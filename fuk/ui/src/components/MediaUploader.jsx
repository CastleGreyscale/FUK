/**
 * Media Uploader Component
 * Handles images, videos, and image sequences with:
 * - Native file dialog via tkinter backend
 * - Drag and drop from browser
 * - Image sequence detection (render.####.exr)
 * - Thumbnail previews for all media types
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
  
  // Normalize media to array of objects
  const normalizedMedia = (media || []).map(item => {
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

  // Open native file dialog via backend
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
          path: f.path,
          displayName: f.display_name,
          mediaType: f.media_type,
          firstFrame: f.first_frame,
          lastFrame: f.last_frame,
          frameCount: f.frame_count,
          framePattern: f.frame_pattern,
        }));
        
        if (multiple) {
          onMediaChange([...normalizedMedia, ...newMedia]);
        } else {
          onMediaChange(newMedia.slice(0, 1));
        }
        
        console.log('[MediaUploader] Selected media:', newMedia);

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
  }, [disabled, loading, multiple, detectSequences, accept, normalizedMedia, onMediaChange]);

  // Handle file input change (fallback for systems without tkinter)
  const handleFileInput = useCallback(async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Upload files to server
      const uploadedMedia = [];
      
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_URL}/upload/media`, {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        uploadedMedia.push({
          path: data.path,
          displayName: file.name,
          mediaType: getMediaType(file.name),
        });
      }
      
      if (multiple) {
        onMediaChange([...normalizedMedia, ...uploadedMedia]);
      } else {
        onMediaChange(uploadedMedia.slice(0, 1));
      }
      
      console.log('[MediaUploader] Uploaded media:', uploadedMedia);

      // Register uploads in history and auto-pin
      for (const media of uploadedMedia) {
        const result = await registerImport(media.path, media.displayName, true);
        if (result.success && result.auto_pin && result.id) {
          window.dispatchEvent(new CustomEvent('fuk-import-registered', {
            detail: { id: result.id, autoPin: true }
          }));
        }
      }
      
    } catch (err) {
      console.error('Upload failed:', err);
      setError(err.message);
    } finally {
      setLoading(false);
      e.target.value = '';  // Reset input
    }
  }, [multiple, normalizedMedia, onMediaChange]);

  // Handle drag and drop
  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    setDragOver(false);
    
    if (disabled) return;
    
    // Check for path data (drag from history panel)
    const pathData = e.dataTransfer.getData('text/plain');
    if (pathData && (pathData.startsWith('/') || pathData.startsWith('api/'))) {
      const newItem = {
        path: pathData,
        displayName: pathData.split('/').pop(),
        mediaType: getMediaType(pathData),
      };
      
      if (multiple) {
        onMediaChange([...normalizedMedia, newItem]);
      } else {
        onMediaChange([newItem]);
      }
      return;
    }
    
    // Handle file drops
    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    
    // Filter to accepted types
    const acceptedFiles = files.filter(f => {
      const ext = '.' + f.name.toLowerCase().split('.').pop();
      if (accept === 'images') return IMAGE_EXTENSIONS.includes(ext);
      if (accept === 'videos') return VIDEO_EXTENSIONS.includes(ext);
      if (accept === 'exr') return ext === '.exr' || ext === '.dpx';
      return IMAGE_EXTENSIONS.includes(ext) || VIDEO_EXTENSIONS.includes(ext);
    });
    
    if (acceptedFiles.length === 0) {
      setError('No supported media files found');
      return;
    }
    
    // Create synthetic event for file handler
    const syntheticEvent = { target: { files: acceptedFiles, value: '' } };
    handleFileInput(syntheticEvent);
  }, [disabled, multiple, accept, normalizedMedia, onMediaChange, handleFileInput]);

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleRemove = (index) => {
    onMediaChange(normalizedMedia.filter((_, i) => i !== index));
  };

  // Build accept string for file input
  const getAcceptString = () => {
    switch (accept) {
      case 'images': return IMAGE_EXTENSIONS.join(',');
      case 'videos': return VIDEO_EXTENSIONS.join(',');
      case 'exr': return '.exr,.dpx';
      default: return [...IMAGE_EXTENSIONS, ...VIDEO_EXTENSIONS].join(',');
    }
  };

  return (
    <div className="fuk-media-uploader">
      {/* Drop Zone */}
      <div 
        className={`fuk-dropzone ${dragOver ? 'fuk-dropzone--dragover' : ''} ${disabled ? 'fuk-dropzone--disabled' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleBrowse}
      >
        {/* Hidden file input as fallback */}
        <input
          type="file"
          accept={getAcceptString()}
          multiple={multiple}
          onChange={handleFileInput}
          disabled={disabled || loading}
          style={{ display: 'none' }}
          id={inputId}
        />
        
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
                  : 'Supports images, videos, and sequences'
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
          <button onClick={() => setError(null)} className="fuk-upload-error-dismiss">Ã—</button>
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