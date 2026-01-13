/**
 * Image Uploader Component
 * Handles multi-image selection with thumbnails using native tkinter dialog
 * 
 * NO UPLOAD - Files stay in place, only paths are passed around
 */

import { useState, useId, useCallback } from 'react';
import { X, FolderOpen } from './Icons';

const API_URL = '/api';

export default function ImageUploader({ images, onImagesChange, disabled, id: providedId }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Generate unique ID for this instance (or use provided one)
  const reactId = useId();
  const inputId = providedId || `image-upload-${reactId}`;

  // Open native file dialog via backend (tkinter)
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
          title: 'Select Images',
          multiple: true,
          detect_sequences: false,
          filter: 'images',
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Browser failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.success && data.files?.length > 0) {
        // Extract just the paths from the file objects
        const paths = data.files.map(f => f.path);
        onImagesChange([...images, ...paths]);
        console.log('[ImageUploader] Selected images (absolute paths):', paths);
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
  }, [disabled, loading, images, onImagesChange]);

  const handleRemove = (index) => {
    onImagesChange(images.filter((_, i) => i !== index));
  };

  // Build display URL for images
  const buildImageUrl = (path) => {
    if (!path) return '';
    // Handle absolute paths - serve through files endpoint
    if (path.startsWith('/')) return `/api/project/files${path}`;
    // Handle API URLs
    if (path.startsWith('api/')) return `/${path}`;
    return `/${path}`;
  };

  return (
    <div className="fuk-image-uploader">
      {/* Click Zone */}
      <div 
        className={`fuk-dropzone ${disabled ? 'fuk-dropzone--disabled' : ''}`}
        onClick={handleBrowse}
        style={{ cursor: disabled ? 'not-allowed' : 'pointer' }}
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
              <span className="fuk-dropzone-text">
                Click to browse for images
              </span>
              <span className="fuk-dropzone-hint">
                {images.length} image{images.length !== 1 ? 's' : ''} selected
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

      {/* Thumbnail Grid */}
      {images.length > 0 && (
        <div className="fuk-thumbnail-grid">
          {images.map((path, index) => (
            <div key={index} className="fuk-thumbnail">
              <img
                src={buildImageUrl(path)}
                alt={`Image ${index + 1}`}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemove(index);
                }}
                className="fuk-thumbnail-remove"
                title="Remove image"
              >
                <X style={{ width: '0.75rem', height: '0.75rem' }} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}