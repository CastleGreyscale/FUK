/**
 * Image Uploader Component
 * Handles multi-image upload with thumbnails and drag-drop
 * Supports both file uploads AND internal path drops from GenerationHistory
 */

import { useState } from 'react';
import { uploadControlImage } from '../utils/api';
import { buildImageUrl } from '../utils/constants';
import { X, Upload, Link } from './Icons';

export default function ImageUploader({ images, onImagesChange, disabled, accept = "image/*" }) {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [dragType, setDragType] = useState(null); // 'file' or 'internal'

  const handleUpload = async (files) => {
    if (!files || files.length === 0) return;
    
    setUploading(true);
    const uploadedPaths = [];
    
    try {
      for (const file of files) {
        const data = await uploadControlImage(file);
        uploadedPaths.push(data.path);
      }
      
      onImagesChange([...images, ...uploadedPaths]);
      console.log('✓ Uploaded images:', uploadedPaths);
    } catch (error) {
      console.error('Image upload failed:', error);
      alert(`Failed to upload images: ${error.message}`);
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
    setDragType(null);
    
    // Check for internal FUK generation drop first
    const fukData = e.dataTransfer.getData('application/x-fuk-generation');
    if (fukData) {
      try {
        const generation = JSON.parse(fukData);
        const path = generation.path;
        
        // Don't add duplicates
        if (!images.includes(path)) {
          onImagesChange([...images, path]);
          console.log('✓ Added from history:', path);
        }
        return;
      } catch (err) {
        console.error('Failed to parse generation data:', err);
      }
    }
    
    // Check for plain text path (simpler drag)
    const textData = e.dataTransfer.getData('text/plain');
    if (textData && textData.startsWith('api/project/cache/')) {
      if (!images.includes(textData)) {
        onImagesChange([...images, textData]);
        console.log('✓ Added path:', textData);
      }
      return;
    }
    
    // Fall back to file upload
    const files = Array.from(e.dataTransfer.files).filter(f => 
      f.type.startsWith('image/') || (accept !== "image/*" && f.type.startsWith('video/'))
    );
    
    if (files.length > 0) {
      handleUpload(files);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    
    // Detect what's being dragged
    const types = e.dataTransfer.types;
    if (types.includes('application/x-fuk-generation') || 
        (types.includes('text/plain') && !types.includes('Files'))) {
      setDragType('internal');
    } else {
      setDragType('file');
    }
    
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
    setDragType(null);
  };

  // Get display URL for a path
  const getDisplayUrl = (path) => {
    // If it's already a full URL or starts with /, use as-is
    if (path.startsWith('http') || path.startsWith('/')) {
      return path;
    }
    // If it's an api path, build the proper URL
    if (path.startsWith('api/')) {
      return buildImageUrl(path);
    }
    // Otherwise assume it's relative to outputs
    return `/${path}`;
  };

  return (
    <div className="fuk-image-uploader">
      {/* Drop Zone */}
      <div 
        className={`fuk-dropzone ${dragOver ? 'drag-over' : ''} ${dragType === 'internal' ? 'drag-over-internal' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <input
          type="file"
          accept={accept}
          multiple
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
          {dragOver && dragType === 'internal' ? (
            <>
              <Link style={{ width: '1.5rem', height: '1.5rem', color: '#10b981', marginBottom: '0.5rem' }} />
              <span style={{ color: '#10b981' }}>Drop to add from history</span>
            </>
          ) : (
            <>
              <Upload style={{ width: '1.5rem', height: '1.5rem', opacity: 0.5, marginBottom: '0.5rem' }} />
              <span>
                {uploading ? 'Uploading...' : 'Drop images or click to upload'}
              </span>
              <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                {images.length} item{images.length !== 1 ? 's' : ''} selected
              </span>
            </>
          )}
        </label>
      </div>

      {/* Thumbnail Grid */}
      {images.length > 0 && (
        <div className="fuk-thumbnail-grid">
          {images.map((path, index) => (
            <div key={index} className="fuk-thumbnail">
              <img
                src={getDisplayUrl(path)}
                alt={`Control ${index + 1}`}
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'flex';
                }}
              />
              <div className="fuk-thumbnail-error" style={{ display: 'none' }}>
                <span>?</span>
              </div>
              <button
                onClick={() => handleRemove(index)}
                className="fuk-thumbnail-remove"
                title="Remove image"
              >
                <X style={{ width: '0.75rem', height: '0.75rem' }} />
              </button>
              {path.startsWith('api/project/cache/') && (
                <div className="fuk-thumbnail-badge" title="From generation history">
                  <Link style={{ width: '0.5rem', height: '0.5rem' }} />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}