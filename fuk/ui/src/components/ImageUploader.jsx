/**
 * Image Uploader Component
 * Handles multi-image upload with thumbnails and drag-drop
 * Supports BOTH file uploads AND internal path drops from GenerationHistory
 */

import { useState } from 'react';
import { uploadControlImage } from '../utils/api';
import { X, Upload } from './Icons';
import { buildImageUrl } from '../utils/constants';

export default function ImageUploader({ 
  images, 
  onImagesChange, 
  disabled,
  accept = 'image/*',  // Can override for video: 'video/*'
  multiple = true,
}) {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const handleUpload = async (files) => {
    if (!files || files.length === 0) return;
    
    setUploading(true);
    const uploadedPaths = [];
    
    try {
      for (const file of files) {
        const data = await uploadControlImage(file);
        uploadedPaths.push(data.path);
      }
      
      if (multiple) {
        onImagesChange([...images, ...uploadedPaths]);
      } else {
        onImagesChange(uploadedPaths.slice(0, 1));
      }
      console.log('✓ Uploaded files:', uploadedPaths);
    } catch (error) {
      console.error('Upload failed:', error);
      alert(`Failed to upload: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleRemove = (index) => {
    onImagesChange(images.filter((_, i) => i !== index));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    
    // First, check for internal path drop from GenerationHistory
    const internalPath = e.dataTransfer.getData('text/plain');
    const generationData = e.dataTransfer.getData('application/x-fuk-generation');
    
    if (internalPath && internalPath.startsWith('api/project/cache/')) {
      // This is an internal drop from history panel
      console.log('[ImageUploader] Internal drop detected:', internalPath);
      
      // Parse generation data for additional info
      let generation = null;
      if (generationData) {
        try {
          generation = JSON.parse(generationData);
          console.log('[ImageUploader] Generation data:', generation);
        } catch (err) {
          console.warn('[ImageUploader] Failed to parse generation data');
        }
      }
      
      // Use the path directly - no upload needed
      if (multiple) {
        onImagesChange([...images, internalPath]);
      } else {
        onImagesChange([internalPath]);
      }
      return;
    }
    
    // Otherwise, handle file drops from computer
    const acceptType = accept.replace('/*', '');
    const files = Array.from(e.dataTransfer.files).filter(f => 
      f.type.startsWith(acceptType)
    );
    
    if (files.length > 0) {
      console.log('[ImageUploader] File drop detected:', files.length, 'files');
      handleUpload(files);
    } else if (internalPath) {
      // Fallback: try to use any path that was dropped
      console.log('[ImageUploader] Using fallback path:', internalPath);
      if (multiple) {
        onImagesChange([...images, internalPath]);
      } else {
        onImagesChange([internalPath]);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  // Determine if path is video based on extension
  const isVideo = (path) => {
    const ext = path.split('.').pop()?.toLowerCase();
    return ['mp4', 'webm', 'mov', 'avi', 'mkv'].includes(ext);
  };

  return (
    <div className="fuk-image-uploader">
      {/* Drop Zone */}
      <div 
        className={`fuk-dropzone ${dragOver ? 'drag-over' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              handleUpload(Array.from(e.target.files));
              e.target.value = '';
            }
          }}
          disabled={disabled || uploading}
          style={{ display: 'none' }}
          id="image-upload-input"
        />
        <label 
          htmlFor="image-upload-input" 
          className="fuk-dropzone-label"
          style={{ cursor: disabled ? 'not-allowed' : 'pointer' }}
        >
          <Upload style={{ width: '1.5rem', height: '1.5rem', opacity: 0.5, marginBottom: '0.5rem' }} />
          <span>
            {uploading ? 'Uploading...' : 'Drop file or drag from History'}
          </span>
          <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
            {images.length} file{images.length !== 1 ? 's' : ''} selected
          </span>
        </label>
      </div>

      {/* Thumbnail Grid */}
      {images.length > 0 && (
        <div className="fuk-thumbnail-grid">
          {images.map((path, index) => (
            <div key={`${path}-${index}`} className="fuk-thumbnail">
              {isVideo(path) ? (
                <video
                  src={buildImageUrl(path)}
                  muted
                  loop
                  onMouseEnter={(e) => e.target.play()}
                  onMouseLeave={(e) => { e.target.pause(); e.target.currentTime = 0; }}
                />
              ) : (
                <img
                  src={buildImageUrl(path)}
                  alt={`Input ${index + 1}`}
                  onError={(e) => {
                    e.target.style.display = 'none';
                  }}
                />
              )}
              <button
                onClick={() => handleRemove(index)}
                className="fuk-thumbnail-remove"
                title="Remove"
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