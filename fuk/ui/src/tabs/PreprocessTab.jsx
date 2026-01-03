/**
 * Preprocess Tab
 * Image preprocessing for control inputs: Canny, OpenPose, Depth
 */

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Pipeline, Loader2, CheckCircle, Camera } from '../../src/components/Icons';
import TabButton from '../components/TabButton';
import ImageUploader from '../components/ImageUploader';
import { useLocalStorage } from '../../src/hooks/useLocalStorage';
import { buildImageUrl } from '../../src/utils/constants';
import Footer from '../components/Footer';

const API_URL = '/api';

// Default settings for each method
const DEFAULT_SETTINGS = {
  canny: {
    low_threshold: 100,
    high_threshold: 200,
    canny_invert: false,
    blur_kernel: 3,
  },
  openpose: {
    detect_body: true,
    detect_hand: false,
    detect_face: false,
  },
  depth: {
    depth_model: 'depth_anything_v2',
    depth_invert: false,
    depth_normalize: true,
    depth_colormap: 'inferno',
  },
};

export default function PreprocessTab({ config, activeTab, setActiveTab, project }) {
  // UI state
  const [selectedMethod, setSelectedMethod] = useState('canny');
  const [sourceImage, setSourceImage] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // Settings per method (localStorage fallback)
  const [localSettings, setLocalSettings] = useLocalStorage('fuk_preprocess_settings', DEFAULT_SETTINGS);
  
  // Use project state if available, otherwise localStorage
  const settings = useMemo(() => {
    if (project?.projectState?.tabs?.preprocess) {
      return { ...DEFAULT_SETTINGS, ...project.projectState.tabs.preprocess };
    }
    return { ...DEFAULT_SETTINGS, ...localSettings };
  }, [project?.projectState?.tabs?.preprocess, localSettings]);
  
  // Ref to track latest settings for updateSettings callback
  const settingsRef = useRef(settings);
  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);
  
  const updateSettings = useCallback((methodUpdates) => {
    const currentSettings = settingsRef.current;
    const newSettings = { ...currentSettings, ...methodUpdates };
    
    if (project?.isProjectLoaded && project?.updateTabState) {
      project.updateTabState('preprocess', newSettings);
    } else {
      setLocalSettings(newSettings);
    }
  }, [project?.isProjectLoaded, project?.updateTabState, setLocalSettings]);
  
  // Current method settings
  const currentSettings = settings[selectedMethod] || DEFAULT_SETTINGS[selectedMethod];
  
  const setCurrentSettings = (updates) => {
    updateSettings({
      [selectedMethod]: { ...currentSettings, ...updates }
    });
  };
  
  // Handle source image selection
  const handleSourceImageChange = (paths) => {
    setSourceImage(paths[0] || null);
    setResult(null);
    setError(null);
  };
  
  // Run preprocessing
  const handleProcess = async () => {
    if (!sourceImage) {
      alert('Please select a source image');
      return;
    }
    
    setProcessing(true);
    setError(null);
    setResult(null);
    
    try {
      const payload = {
        image_path: sourceImage,
        method: selectedMethod,
        ...currentSettings,
      };
      
      console.log('Preprocessing:', payload);
      
      const res = await fetch(`${API_URL}/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errorData.detail || `Preprocessing failed: ${res.statusText}`);
      }
      
      const data = await res.json();
      console.log('Preprocessing complete:', data);
      
      setResult(data);
      
      // Save to project lastState so ImageTab can access it
      if (project?.updateLastState) {
        project.updateLastState({
          lastPreprocessedImage: data.url,
          lastPreprocessedMethod: selectedMethod,
          lastPreprocessedParams: currentSettings,
        });
      }
      
    } catch (err) {
      console.error('Preprocessing error:', err);
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };
  
  // Render method-specific controls
  const renderMethodControls = () => {
    if (selectedMethod === 'canny') {
      return (
        <>
          <div className="fuk-form-group-compact">
            <label className="fuk-label">Low Threshold</label>
            <div className="fuk-input-inline">
              <input
                type="range"
                className="fuk-input"
                style={{ flex: 2 }}
                value={currentSettings.low_threshold}
                onChange={(e) => setCurrentSettings({ low_threshold: parseInt(e.target.value) })}
                min={0}
                max={500}
                step={10}
              />
              <input
                type="number"
                className="fuk-input"
                style={{ width: '80px' }}
                value={currentSettings.low_threshold}
                onChange={(e) => setCurrentSettings({ low_threshold: parseInt(e.target.value) })}
                min={0}
                max={500}
              />
            </div>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-label">High Threshold</label>
            <div className="fuk-input-inline">
              <input
                type="range"
                className="fuk-input"
                style={{ flex: 2 }}
                value={currentSettings.high_threshold}
                onChange={(e) => setCurrentSettings({ high_threshold: parseInt(e.target.value) })}
                min={0}
                max={500}
                step={10}
              />
              <input
                type="number"
                className="fuk-input"
                style={{ width: '80px' }}
                value={currentSettings.high_threshold}
                onChange={(e) => setCurrentSettings({ high_threshold: parseInt(e.target.value) })}
                min={0}
                max={500}
              />
            </div>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-label">Blur Kernel (reduce noise)</label>
            <select
              className="fuk-select"
              value={currentSettings.blur_kernel}
              onChange={(e) => setCurrentSettings({ blur_kernel: parseInt(e.target.value) })}
            >
              <option value={0}>None</option>
              <option value={3}>3x3</option>
              <option value={5}>5x5</option>
              <option value={7}>7x7</option>
            </select>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-checkbox-group">
              <input
                type="checkbox"
                className="fuk-checkbox"
                checked={currentSettings.canny_invert}
                onChange={(e) => setCurrentSettings({ canny_invert: e.target.checked })}
              />
              <span className="fuk-label" style={{ marginBottom: 0 }}>Invert (white edges on black)</span>
            </label>
          </div>
        </>
      );
    }
    
    if (selectedMethod === 'openpose') {
      return (
        <>
          <div className="fuk-form-group-compact">
            <label className="fuk-checkbox-group">
              <input
                type="checkbox"
                className="fuk-checkbox"
                checked={currentSettings.detect_body}
                onChange={(e) => setCurrentSettings({ detect_body: e.target.checked })}
              />
              <span className="fuk-label" style={{ marginBottom: 0 }}>Detect Body</span>
            </label>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-checkbox-group">
              <input
                type="checkbox"
                className="fuk-checkbox"
                checked={currentSettings.detect_hand}
                onChange={(e) => setCurrentSettings({ detect_hand: e.target.checked })}
              />
              <span className="fuk-label" style={{ marginBottom: 0 }}>Detect Hands</span>
            </label>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-checkbox-group">
              <input
                type="checkbox"
                className="fuk-checkbox"
                checked={currentSettings.detect_face}
                onChange={(e) => setCurrentSettings({ detect_face: e.target.checked })}
              />
              <span className="fuk-label" style={{ marginBottom: 0 }}>Detect Face</span>
            </label>
          </div>
          
          <p style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: '1rem' }}>
            OpenPose detects human pose keypoints for precise control over body position and gesture.
          </p>
        </>
      );
    }
    
    if (selectedMethod === 'depth') {
      return (
        <>
          <div className="fuk-form-group-compact">
            <label className="fuk-label">Depth Model</label>
            <select
              className="fuk-select"
              value={currentSettings.depth_model}
              onChange={(e) => setCurrentSettings({ depth_model: e.target.value })}
            >
              <option value="midas_small">MiDaS Small (Fast)</option>
              <option value="midas_large">MiDaS Large (Balanced)</option>
              <option value="depth_anything_v2">Depth Anything V2 (SOTA)</option>
              <option value="depth_anything_v3">Depth Anything V3 (Latest)</option>
              <option value="zoedepth">ZoeDepth (Metric)</option>
            </select>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-label">Colormap</label>
            <select
              className="fuk-select"
              value={currentSettings.depth_colormap || 'inferno'}
              onChange={(e) => setCurrentSettings({ depth_colormap: e.target.value === 'none' ? null : e.target.value })}
            >
              <option value="none">Grayscale</option>
              <option value="inferno">Inferno</option>
              <option value="viridis">Viridis</option>
              <option value="magma">Magma</option>
              <option value="plasma">Plasma</option>
              <option value="turbo">Turbo</option>
            </select>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-checkbox-group">
              <input
                type="checkbox"
                className="fuk-checkbox"
                checked={currentSettings.depth_normalize}
                onChange={(e) => setCurrentSettings({ depth_normalize: e.target.checked })}
              />
              <span className="fuk-label" style={{ marginBottom: 0 }}>Normalize Depth</span>
            </label>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-checkbox-group">
              <input
                type="checkbox"
                className="fuk-checkbox"
                checked={currentSettings.depth_invert}
                onChange={(e) => setCurrentSettings({ depth_invert: e.target.checked })}
              />
              <span className="fuk-label" style={{ marginBottom: 0 }}>Invert Depth (near = dark)</span>
            </label>
          </div>
          
          <p style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: '1rem' }}>
            {currentSettings.depth_model === 'depth_anything_v2' && 'Ã¢Å“â€œ Recommended: Best quality/speed balance'}
            {currentSettings.depth_model === 'depth_anything_v3' && 'Ã¢Å¡Â¡ Latest model with improved detail'}
            {currentSettings.depth_model === 'midas_small' && 'Ã¢Å¡Â¡ Fastest processing'}
            {currentSettings.depth_model === 'midas_large' && 'Good quality, widely compatible'}
            {currentSettings.depth_model === 'zoedepth' && 'Metric depth estimation'}
          </p>
        </>
      );
    }
  };
  
  return (
    <>
      {/* Preview Area */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-container" style={{ display: 'flex', gap: '2rem', alignItems: 'center', justifyContent: 'center' }}>
          {/* Source Image */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
            <h3 style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#9ca3af', letterSpacing: '0.05em' }}>Source</h3>
            {sourceImage ? (
              <img
                src={buildImageUrl(sourceImage)}
                alt="Source"
                style={{
                  maxWidth: '100%',
                  maxHeight: '60vh',
                  objectFit: 'contain',
                  borderRadius: '0.5rem',
                  border: '1px solid #374151',
                }}
              />
            ) : (
              <div 
                className="fuk-card-dashed" 
                style={{ width: '100%', aspectRatio: '16/9' }}
              >
                <div className="fuk-placeholder">
                  <Camera style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', opacity: 0.3 }} />
                  <p style={{ color: '#6b7280' }}>Select source image below</p>
                </div>
              </div>
            )}
          </div>
          

          
          {/* Processed Result */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
            <h3 style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#9ca3af', letterSpacing: '0.05em' }}>
              {selectedMethod.charAt(0).toUpperCase() + selectedMethod.slice(1)} Result
            </h3>
            {result ? (
              <div style={{ position: 'relative', width: '100%' }}>
                <img
                  src={buildImageUrl(result.url)}
                  alt="Processed"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '60vh',
                    objectFit: 'contain',
                    borderRadius: '0.5rem',
                    border: '1px solid #374151',
                  }}
                />
                <div 
                  style={{
                    position: 'absolute',
                    top: '0.5rem',
                    right: '0.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem',
                    padding: '0.25rem 0.5rem',
                    background: 'rgba(16, 185, 129, 0.9)',
                    borderRadius: '0.375rem',
                    fontSize: '0.75rem',
                  }}
                >
                  <CheckCircle style={{ width: '1rem', height: '1rem' }} />
                  Complete
                </div>
              </div>
            ) : error ? (
              <div 
                className="fuk-card-dashed" 
                style={{ width: '100%', aspectRatio: '16/9', borderColor: '#ef4444' }}
              >
                <div className="fuk-placeholder">
                  <p style={{ color: '#ef4444' }}>Error: {error}</p>
                </div>
              </div>
            ) : (
              <div 
                className="fuk-card-dashed" 
                style={{ width: '100%', aspectRatio: '16/9' }}
              >
                <div className="fuk-placeholder">
                  <Pipeline style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', opacity: 0.3 }} />
                  <p style={{ color: '#6b7280' }}>
                    {processing ? 'Processing...' : 'Click Process to generate'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Source Image Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Source Image</h3>
            
            <ImageUploader
              images={sourceImage ? [sourceImage] : []}
              onImagesChange={handleSourceImageChange}
              disabled={processing}
            />
            
            <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
              Upload or select an image to preprocess
            </p>
          </div>

          {/* Method Selector Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Method</h3>
            
            <div className="fuk-form-group-compact">
              <select
                className="fuk-select"
                value={selectedMethod}
                onChange={(e) => {
                  setSelectedMethod(e.target.value);
                  setResult(null);
                  setError(null);
                }}
                disabled={processing}
              >
                <option value="canny">Canny Edge Detection</option>
                <option value="openpose">OpenPose (Pose)</option>
                <option value="depth">Depth Estimation</option>
              </select>
            </div>
            
            <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(168, 85, 247, 0.1)', borderRadius: '0.375rem', fontSize: '0.75rem', color: '#c084fc' }}>
              {selectedMethod === 'canny' && 'Clean edge detection for line art and contours'}
              {selectedMethod === 'openpose' && 'Detect human pose keypoints for precise control'}
              {selectedMethod === 'depth' && 'Estimate depth map for 3D-aware generation'}
            </div>
          </div>

          {/* Method-Specific Settings Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Settings</h3>
            {renderMethodControls()}
          </div>
        </div>
      </div>

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={processing}
        progress={processing ? { progress: 0.5, phase: 'processing' } : null}
        elapsedSeconds={0}
        onGenerate={handleProcess}
        onCancel={() => setProcessing(false)}
        canGenerate={!!sourceImage}
        generateLabel="Process"
        generatingLabel="Processing..."
      />
    </>
  );
}