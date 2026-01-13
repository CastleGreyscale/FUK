/**
 * Inline Image Input Component
 * Compact dropzone with inline preview - for use in forms where space is limited
 */

import { useState, useId } from 'react';
import { X, Upload } from './Icons';
import { buildImageUrl } from '../utils/constants';

export default function InlineImageInput({ 
  value,           // Single path string or null
  onChange,        // (path: string | null) => void
  onLayersPackDrop, // Optional: (genData) => void - for handling layers packs
  disabled = false,
  label,           // Optional label
  required = false,
  accept = "image/*",
  placeholder = "Drop or click",
}) {
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const inputId = useId();

  const handleUpload = async (files) => {
    // Upload functionality disabled - files must be referenced from project
    console.log('[InlineImageInput] File upload not supported');
    alert('Please drag images from the history panel.\nDirect file uploads are not supported.');
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    onChange(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    
    // Check for layers pack FIRST (if handler provided)
    if (onLayersPackDrop) {
      const fukGenData = e.dataTransfer.getData('application/x-fuk-generation');
      if (fukGenData) {
        try {
          const genData = JSON.parse(fukGenData);
          console.log('[InlineImageInput] Received generation data:', genData);
          console.log('[InlineImageInput] Type:', genData.type, 'Has metadata:', !!genData.metadata, 'Has available_layers:', !!genData.metadata?.available_layers);
          
          // Check if it's a layers pack with available_layers
          if (genData.type === 'layers' && genData.metadata?.available_layers) {
            console.log('[InlineImageInput] Layers pack detected, delegating to handler');
            onLayersPackDrop(genData);
            return; // Don't process as single image
          }
        } catch (err) {
          console.error('[InlineImageInput] Failed to parse generation data:', err);
        }
      }
    }
    
    // Check for internal drag (from history) - single image/video
    const internalPath = e.dataTransfer.getData('text/plain');
    if (internalPath && internalPath.startsWith('api/')) {
      onChange(internalPath);
      return;
    }
    
    const files = Array.from(e.dataTransfer.files).filter(f => 
      f.type.startsWith('image/') || f.type.startsWith('video/')
    );
    
    if (files.length > 0) {
      handleUpload(files);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const previewUrl = value ? buildImageUrl(value) : null;
  const isVideo = value && (value.endsWith('.mp4') || value.endsWith('.webm'));

  return (
    <div className="inline-image-input">
      {label && (
        <label className="inline-image-label">
          {label}
          {required && <span className="required">*</span>}
        </label>
      )}
      
      <div className="inline-image-row">
        {/* Compact Dropzone */}
        <div 
          className={`inline-dropzone ${dragOver ? 'drag-over' : ''} ${value ? 'has-value' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            type="file"
            accept={accept}
            onChange={(e) => {
              if (e.target.files && e.target.files.length > 0) {
                handleUpload(Array.from(e.target.files));
                e.target.value = '';
              }
            }}
            disabled={disabled || uploading}
            style={{ display: 'none' }}
            id={inputId}
          />
          <label 
            htmlFor={inputId} 
            className="inline-dropzone-label"
          >
            {uploading ? (
              <span className="uploading">Uploading...</span>
            ) : (
              <>
                <Upload className="upload-icon" />
                <span>{placeholder}</span>
              </>
            )}
          </label>
        </div>

        {/* Inline Preview */}
        {value && (
          <div className="inline-preview">
            {isVideo ? (
              <video 
                src={previewUrl}
                muted
                loop
                onMouseEnter={(e) => e.target.play()}
                onMouseLeave={(e) => { e.target.pause(); e.target.currentTime = 0; }}
              />
            ) : (
              <img src={previewUrl} alt="Preview" />
            )}
            <button
              onClick={handleRemove}
              className="inline-preview-remove"
              title="Remove"
              disabled={disabled}
            >
              <X />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}