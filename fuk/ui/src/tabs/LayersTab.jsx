/**
 * Layers Tab
 * Generate and manage AOV layers (Arbitrary Output Variables)
 * Layers: Depth, Normals, Cryptomatte
 * Supports both images and frame-by-frame video processing
 * 
 * Features:
 * - Persists last result in project state for cross-tab restoration
 * - On-demand loading of previews to prevent UI hangs
 */

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import {CheckCircle, Layers, AlertCircle, Film } from '../components/Icons';
import MediaUploader, { isVideoFile } from '../components/MediaUploader';
import ZoomableImage from '../components/ZoomableImage';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
import Footer from '../components/Footer';

const API_URL = '/api';

const DEFAULT_SETTINGS = {
  // Active layers to generate
  layers: {
    depth: true,
    normals: true,
    crypto: false,
  },
  
  // Depth settings
  depthModel: 'depth_anything_v2',
  depthInvert: false,
  depthNormalize: true,
  depthColormap: 'inferno',
  
  // Normals settings
  normalsMethod: 'from_depth',
  normalsDepthModel: 'depth_anything_v2',
  normalsSpace: 'tangent',
  normalsFlipY: false,
  normalsIntensity: 1.0,
  
  // Cryptomatte settings
  cryptoModel: 'sam2_hiera_large',
  cryptoMaxObjects: 50,
  cryptoMinArea: 500,
  
  // Video output mode
  videoOutputMode: 'mp4',
};

export default function LayersTab({ config, activeTab, setActiveTab, project }) {
  // Settings (localStorage fallback)
  const [localSettings, setLocalSettings] = useLocalStorage('fuk_layers_settings', DEFAULT_SETTINGS);
  
  const settings = useMemo(() => {
    if (project?.projectState?.tabs?.layers) {
      return { ...DEFAULT_SETTINGS, ...project.projectState.tabs.layers };
    }
    return { ...DEFAULT_SETTINGS, ...localSettings };
  }, [project?.projectState?.tabs?.layers, localSettings]);
  
  const updateSettings = useCallback((updates) => {
    const newSettings = { ...settings, ...updates };
    
    if (project?.isProjectLoaded && project.updateTabState) {
      project.updateTabState('layers', newSettings);
    } else {
      setLocalSettings(newSettings);
    }
  }, [project, settings, setLocalSettings]);
  
  // UI state
  const [sourceInput, setSourceInput] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [generatedLayers, setGeneratedLayers] = useState({});
  const [errors, setErrors] = useState({});
  const [elapsedTime, setElapsedTime] = useState(0);
  
  // Timer ref
  const timerRef = useRef(null);
  const mountedRef = useRef(true);
  
  // Track if we've restored from project state
  const hasRestoredRef = useRef(false);
  
  // Detect if input is video
  const isVideo = useMemo(() => isVideoFile(sourceInput), [sourceInput]);
  
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);
  
  // Restore last preview from project state
  useEffect(() => {
    // Only restore once per project file and when no current layers
    if (Object.keys(generatedLayers).length > 0 || hasRestoredRef.current) return;
    
    const lastState = project?.projectState?.lastState;
    if (!lastState?.lastLayersPreview) return;
    
    // Restore the generated layers
    const meta = lastState.lastLayersMeta || {};
    
    // Restore generatedLayers object from meta
    if (meta.generatedLayers && typeof meta.generatedLayers === 'object') {
      setGeneratedLayers(meta.generatedLayers);
    }
    
    // Restore source input if available
    if (meta.sourceInput) {
      setSourceInput(meta.sourceInput);
    }
    
    hasRestoredRef.current = true;
  }, [project?.projectState?.lastState, generatedLayers]);
  
  // Reset restoration flag when project file changes
  useEffect(() => {
    hasRestoredRef.current = false;
    setGeneratedLayers({});
    setErrors({});
  }, [project?.currentFilename]);
  
  // Handle source input (MediaUploader passes array of media objects)
  const handleSourceChange = (media) => {
    const first = media[0];
    setSourceInput(first?.path || first || null);
    setGeneratedLayers({});
    setErrors({});
  };
  
  const updateLayer = (layerName, enabled) => {
    updateSettings({
      layers: {
        ...settings.layers,
        [layerName]: enabled,
      }
    });
  };
  
  const handleGenerate = async () => {
    if (!sourceInput) {
      alert('Please select a source image or video');
      return;
    }
    
    setProcessing(true);
    setGeneratedLayers({});
    setErrors({});
    setElapsedTime(0);
    
    const startTime = Date.now();
    timerRef.current = setInterval(() => {
      if (mountedRef.current) {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
      }
    }, 1000);
    
    try {
      let endpoint, payload;
      
      if (isVideo) {
        // Video layers - frame by frame
        endpoint = `${API_URL}/layers/video/generate`;
        payload = {
          video_path: sourceInput,
          output_mode: settings.videoOutputMode,
          layers: settings.layers,
          
          depth_model: settings.depthModel,
          depth_invert: settings.depthInvert,
          depth_normalize: settings.depthNormalize,
          depth_colormap: settings.depthColormap,
          
          normals_method: settings.normalsMethod,
          normals_depth_model: settings.normalsDepthModel,
          normals_space: settings.normalsSpace,
          normals_flip_y: settings.normalsFlipY,
          normals_intensity: settings.normalsIntensity,
          
          crypto_model: settings.cryptoModel,
          crypto_max_objects: settings.cryptoMaxObjects,
          crypto_min_area: settings.cryptoMinArea,
        };
      } else {
        // Image layers
        endpoint = `${API_URL}/layers/generate`;
        payload = {
          image_path: sourceInput,
          layers: settings.layers,
          
          depth_model: settings.depthModel,
          depth_invert: settings.depthInvert,
          depth_normalize: settings.depthNormalize,
          depth_colormap: settings.depthColormap,
          
          normals_method: settings.normalsMethod,
          normals_depth_model: settings.normalsDepthModel,
          normals_space: settings.normalsSpace,
          normals_flip_y: settings.normalsFlipY,
          normals_intensity: settings.normalsIntensity,
          
          crypto_model: settings.cryptoModel,
          crypto_max_objects: settings.cryptoMaxObjects,
          crypto_min_area: settings.cryptoMinArea,
        };
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }
      
      const result = await response.json();
      
      const layers = {};
      if (result.layers) {
        Object.entries(result.layers).forEach(([name, data]) => {
          layers[name] = {
            url: data.url,
            name: name.charAt(0).toUpperCase() + name.slice(1),
            info: isVideo 
              ? `${data.frame_count || result.frame_count || 0} frames`
              : (data.model || data.method || (data.num_objects ? `${data.num_objects} objects` : '')),
            isVideo,
          };
        });
      }
      
      if (result.source) {
        layers.beauty = {
          url: result.source,
          name: 'Beauty (Source)',
          info: 'Original',
          isVideo,
        };
      }
      
      setGeneratedLayers(layers);
      setErrors(result.errors || {});
      
      // Save to project lastState for cross-tab persistence
      if (project?.updateLastState) {
        // Use first layer URL as the main preview
        const firstLayerUrl = Object.values(layers)[0]?.url || null;
        
        project.updateLastState({
          lastLayersPreview: firstLayerUrl,
          lastLayersMeta: {
            isVideo,
            sourceInput,
            generatedLayers: layers,
            frameCount: result.frame_count || 0,
          },
          activeTab: 'layers',
        });
      }
      // Notify history to refresh  
      window.dispatchEvent(new CustomEvent('fuk-generation-complete', {
        detail: { 
          type: 'layers',
          result: result,
        }
      }));
      
    } catch (err) {
      console.error('Layer generation failed:', err);
      setErrors({ _general: err.message });
    } finally {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setProcessing(false);
    }
  };
  
  const handleCancel = () => {
    setProcessing(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };
  
  const enabledLayersCount = Object.values(settings.layers).filter(v => v).length;
  const hasErrors = Object.keys(errors).length > 0;
  
  // Format elapsed time
  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };
  
  return (
    <>
      {/* Preview Area - Grid Layout for Layers */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-container fuk-preview-container--padded">
          {Object.keys(generatedLayers).length > 0 ? (
            <div className="fuk-preview-grid">
              {Object.entries(generatedLayers).map(([layerName, layerData]) => (
                <div key={layerName} className="fuk-layer-card">
                  <div className="fuk-layer-card-header">
                    <div>
                      <span className="fuk-layer-card-title">
                        {layerData.name || layerName}
                        {layerData.isVideo && <Film className="fuk-icon--sm fuk-ml-2" />}
                      </span>
                      {layerData.info && (
                        <span className="fuk-layer-card-info">{layerData.info}</span>
                      )}
                    </div>
                    <CheckCircle className="fuk-icon--md fuk-icon--success" />
                  </div>
                  <div className="fuk-layer-card-body">
                    {layerData.isVideo ? (
                      <video
                        src={buildImageUrl(layerData.url)}
                        controls
                        preload="metadata"
                        className="fuk-layer-card-media"
                      />
                    ) : (
                      <ZoomableImage
                        src={buildImageUrl(layerData.url)}
                        alt={layerName}
                        className="fuk-layer-card-media"
                        loading="lazy"
                      />
                    )}
                  </div>
                </div>
              ))}
              
              {/* Show errors if any */}
              {hasErrors && (
                <div className="fuk-alert fuk-alert--error fuk-alert--block">
                  <div className="fuk-alert-header">
                    <AlertCircle className="fuk-alert-icon" />
                    <span className="fuk-alert-title">Some layers failed</span>
                  </div>
                  {Object.entries(errors).map(([layer, error]) => (
                    <p key={layer} className="fuk-alert-detail">
                      {layer}: {typeof error === 'string' ? error : JSON.stringify(error)}
                    </p>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="fuk-placeholder-card">
              <div className="fuk-placeholder">
                <Layers className="fuk-placeholder-icon" />
                <p className="fuk-placeholder-text">
                  {processing 
                    ? `Generating layers... ${formatTime(elapsedTime)}`
                    : 'Select source and layers, then click Generate'
                  }
                </p>
                {processing && isVideo && (
                  <p className="fuk-placeholder-subtext">
                    Processing video frame-by-frame
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Source Input Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Source Input</h3>
            
            <MediaUploader
              media={sourceInput ? [{ path: sourceInput }] : []}
              onMediaChange={handleSourceChange}
              disabled={processing}
              accept="all"
            />
            
            <p className="fuk-help-text">
              Drag from History or upload an image/video
            </p>
            
            {isVideo && (
              <div className="fuk-alert fuk-alert--info fuk-mt-3">
                <Film className="fuk-alert-icon" />
                <span className="fuk-alert-text">Video: Processing frame-by-frame</span>
              </div>
            )}
            
            {/* Video output mode */}
            {isVideo && (
              <div className="fuk-form-group-compact fuk-mt-4 fuk-pt-4 fuk-border-top">
                <label className="fuk-label">Output Format</label>
                <select
                  className="fuk-select"
                  value={settings.videoOutputMode}
                  onChange={(e) => updateSettings({ videoOutputMode: e.target.value })}
                  disabled={processing}
                >
                  <option value="mp4">MP4 Video (per layer)</option>
                  <option value="sequence">Image Sequences</option>
                </select>
                <p className="fuk-help-text">
                  {settings.videoOutputMode === 'mp4' 
                    ? 'Each layer as separate video' 
                    : 'Frame sequences for EXR workflow'}
                </p>
              </div>
            )}
          </div>

          {/* Layer Selection Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Active Layers</h3>
            
            <div className="fuk-layer-toggle">
              <label className="fuk-layer-toggle-item">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.layers.depth}
                  onChange={(e) => updateLayer('depth', e.target.checked)}
                  disabled={processing}
                />
                <div className="fuk-layer-toggle-content">
                  <span className="fuk-layer-toggle-name">Depth (Z)</span>
                  <span className="fuk-layer-toggle-desc">Distance from camera</span>
                </div>
              </label>
              
              <label className="fuk-layer-toggle-item">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.layers.normals}
                  onChange={(e) => updateLayer('normals', e.target.checked)}
                  disabled={processing}
                />
                <div className="fuk-layer-toggle-content">
                  <span className="fuk-layer-toggle-name">Normals (N)</span>
                  <span className="fuk-layer-toggle-desc">Surface orientation</span>
                </div>
              </label>
              
              <label className="fuk-layer-toggle-item">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.layers.crypto}
                  onChange={(e) => updateLayer('crypto', e.target.checked)}
                  disabled={processing}
                />
                <div className="fuk-layer-toggle-content">
                  <span className="fuk-layer-toggle-name">Cryptomatte</span>
                  <span className="fuk-layer-toggle-desc">Per-object ID masks</span>
                </div>
              </label>
            </div>
            
            <p className="fuk-help-text fuk-mt-3">
              {enabledLayersCount} layer{enabledLayersCount !== 1 ? 's' : ''} selected
              {isVideo && ` (Ã— video frames)`}
            </p>
          </div>

          {/* Depth Settings */}
          {settings.layers.depth && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Depth Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Depth Model</label>
                <select
                  className="fuk-select"
                  value={settings.depthModel}
                  onChange={(e) => updateSettings({ depthModel: e.target.value })}
                  disabled={processing}
                >
                  <optgroup label="Depth Anything V3 (Latest)">
                    <option value="da3_mono_large">DA3 Mono Large (Best)</option>
                    <option value="da3_metric_large">DA3 Metric (Real-world scale)</option>
                    <option value="da3_large">DA3 Large (Multi-view)</option>
                    <option value="da3_giant">DA3 Giant (Highest quality)</option>
                  </optgroup>
                  <optgroup label="Legacy">
                    <option value="depth_anything_v2">Depth Anything V2</option>
                    <option value="midas_large">MiDaS Large</option>
                    <option value="midas_small">MiDaS Small (Fast)</option>
                    <option value="zoedepth">ZoeDepth</option>
                  </optgroup>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Colormap</label>
                <select
                  className="fuk-select"
                  value={settings.depthColormap || ''}
                  onChange={(e) => updateSettings({ depthColormap: e.target.value || null })}
                  disabled={processing}
                >
                  <option value="">Grayscale</option>
                  <option value="inferno">Inferno (Heat)</option>
                  <option value="viridis">Viridis (Green-Blue)</option>
                  <option value="magma">Magma (Purple-Orange)</option>
                  <option value="plasma">Plasma (Purple-Yellow)</option>
                  <option value="turbo">Turbo (Rainbow)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.depthNormalize}
                    onChange={(e) => updateSettings({ depthNormalize: e.target.checked })}
                    disabled={processing}
                  />
                  <span className="fuk-checkbox-label">Normalize to 0-1 range</span>
                </label>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.depthInvert}
                    onChange={(e) => updateSettings({ depthInvert: e.target.checked })}
                    disabled={processing}
                  />
                  <span className="fuk-checkbox-label">Invert (near = black)</span>
                </label>
              </div>
            </div>
          )}

          {/* Normals Settings */}
          {settings.layers.normals && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Normals Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Estimation Method</label>
                <select
                  className="fuk-select"
                  value={settings.normalsMethod}
                  onChange={(e) => updateSettings({ normalsMethod: e.target.value })}
                  disabled={processing}
                >
                  <option value="from_depth">From Depth (Fast)</option>
                  <option value="dsine">DSINE (Better Quality)</option>
                </select>
              </div>
              
              {settings.normalsMethod === 'from_depth' && (
                <>
                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Depth Model</label>
                    <select
                      className="fuk-select"
                      value={settings.normalsDepthModel}
                      onChange={(e) => updateSettings({ normalsDepthModel: e.target.value })}
                      disabled={processing}
                    >
                      <optgroup label="Depth Anything V3 (Latest)">
                        <option value="da3_mono_large">DA3 Mono Large (Best)</option>
                        <option value="da3_metric_large">DA3 Metric (Real-world scale)</option>
                        <option value="da3_large">DA3 Large (Multi-view)</option>
                        <option value="da3_giant">DA3 Giant (Highest quality)</option>
                      </optgroup>
                      <optgroup label="Legacy">
                        <option value="depth_anything_v2">Depth Anything V2</option>
                        <option value="midas_large">MiDaS Large</option>
                        <option value="midas_small">MiDaS Small (Fast)</option>
                        <option value="zoedepth">ZoeDepth</option>
                      </optgroup>
                    </select>
                  </div>
                  
                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Intensity: {settings.normalsIntensity.toFixed(1)}</label>
                    <input
                      type="range"
                      className="fuk-slider"
                      min="0.1"
                      max="3.0"
                      step="0.1"
                      value={settings.normalsIntensity}
                      onChange={(e) => updateSettings({ normalsIntensity: parseFloat(e.target.value) })}
                      disabled={processing}
                    />
                    <div className="fuk-range-labels">
                      <span>Subtle</span>
                      <span>Strong</span>
                    </div>
                  </div>
                </>
              )}
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Normal Space</label>
                <select
                  className="fuk-select"
                  value={settings.normalsSpace}
                  onChange={(e) => updateSettings({ normalsSpace: e.target.value })}
                  disabled={processing}
                >
                  <option value="tangent">Tangent Space</option>
                  <option value="object">Object Space</option>
                  <option value="world">World Space</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.normalsFlipY}
                    onChange={(e) => updateSettings({ normalsFlipY: e.target.checked })}
                    disabled={processing}
                  />
                  <span className="fuk-checkbox-label">Flip Y (OpenGL ↔ DirectX)</span>
                </label>
              </div>
              
              <p className="fuk-help-text">Normals encoded as RGB (XYZ → RGB)</p>
            </div>
          )}

          {/* Cryptomatte Settings */}
          {settings.layers.crypto && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Cryptomatte Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">SAM2 Model</label>
                <select
                  className="fuk-select"
                  value={settings.cryptoModel}
                  onChange={(e) => updateSettings({ cryptoModel: e.target.value })}
                  disabled={processing}
                >
                  <option value="sam2_hiera_large">Large (Best Quality)</option>
                  <option value="sam2_hiera_base_plus">Base+ (Good)</option>
                  <option value="sam2_hiera_small">Small (Balanced)</option>
                  <option value="sam2_hiera_tiny">Tiny (Fast)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Max Objects: {settings.cryptoMaxObjects}</label>
                <input
                  type="range"
                  className="fuk-slider"
                  min="10"
                  max="100"
                  step="5"
                  value={settings.cryptoMaxObjects}
                  onChange={(e) => updateSettings({ cryptoMaxObjects: parseInt(e.target.value) })}
                  disabled={processing}
                />
                <div className="fuk-range-labels">
                  <span>10</span>
                  <span>100</span>
                </div>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Min Area: {settings.cryptoMinArea}px</label>
                <input
                  type="range"
                  className="fuk-slider"
                  min="100"
                  max="2000"
                  step="100"
                  value={settings.cryptoMinArea}
                  onChange={(e) => updateSettings({ cryptoMinArea: parseInt(e.target.value) })}
                  disabled={processing}
                />
                <div className="fuk-range-labels">
                  <span>Small</span>
                  <span>Large only</span>
                </div>
              </div>
              
              <p className="fuk-help-text">Creates per-object mattes using Segment Anything 2</p>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={processing}
        progress={processing ? { progress: 0.5, phase: 'generating_layers' } : null}
        elapsedSeconds={elapsedTime}
        onGenerate={handleGenerate}
        onCancel={handleCancel}
        canGenerate={!!sourceInput && enabledLayersCount > 0}
        generateLabel={isVideo ? "Generate Video Layers" : "Generate Layers"}
        generatingLabel={isVideo ? "Generating Video Layers..." : "Generating..."}
      />
    </>
  );
}