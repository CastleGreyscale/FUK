/**
 * Image Uploader Component
 * Handles multi-image upload with thumbnails and drag-drop
 */

import { useState } from 'react';
import { uploadControlImage } from '../utils/api';
import { X, Upload } from './Icons';

export default function ImageUploader({ images, onImagesChange, disabled }) {
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
    
    const files = Array.from(e.dataTransfer.files).filter(f => 
      f.type.startsWith('image/')
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
          accept="image/*"
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
          <Upload style={{ width: '1.5rem', height: '1.5rem', opacity: 0.5, marginBottom: '0.5rem' }} />
          <span>
            {uploading ? 'Uploading...' : 'Drop images or click to upload'}
          </span>
          <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
            {images.length} image{images.length !== 1 ? 's' : ''} selected
          </span>
        </label>
      </div>

      {/* Thumbnail Grid */}
      {images.length > 0 && (
        <div className="fuk-thumbnail-grid">
          {images.map((path, index) => (
            <div key={index} className="fuk-thumbnail">
              <img
                src={`/${path}`}
                alt={`Control ${index + 1}`}
              />
              <button
                onClick={() => handleRemove(index)}
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
