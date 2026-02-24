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
import GenerationModal from '../components/GenerationModal';
import { useGeneration } from '../hooks/useGeneration';
import { startTask } from '../utils/api';
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
  depthModel: 'da3_mono_large',
  depthInvert: false,
  depthNormalize: true,
  depthRangeMin: 0.0,
  depthRangeMax: 1.0,
  
  // Normals settings
  normalsMethod: 'from_depth',
  normalsDepthModel: 'da3_mono_large',
  normalsSpace: 'tangent',
  normalsFlipY: false,
  normalsIntensity: 1.0,
  
  // Cryptomatte settings
  cryptoModel: 'sam2_hiera_large',
  cryptoMaxObjects: 50,
  cryptoMinArea: 500,
};

export default function LayersTab({ config, activeTab, setActiveTab, project }) {
  // Settings (localStorage fallback)
  const [localSettings, setLocalSettings] = useLocalStorage('fuk_layers_settings', DEFAULT_SETTINGS);
  
  const settings = useMemo(() => {
    // Valid depth models (only DA3 family now)
    const validDepthModels = ['da3_mono_large', 'da3_metric_large', 'da3_large', 'da3_giant'];
    
    let mergedSettings;
    if (project?.projectState?.tabs?.layers) {
      mergedSettings = { ...DEFAULT_SETTINGS, ...project.projectState.tabs.layers };
    } else {
      mergedSettings = { ...DEFAULT_SETTINGS, ...localSettings };
    }
    
    // Validate and migrate depth model settings
    // If stored value is not a valid DA3 model, reset to default
    if (!validDepthModels.includes(mergedSettings.depthModel)) {
      console.warn(`[Layers] Invalid depthModel '${mergedSettings.depthModel}', resetting to 'da3_mono_large'`);
      mergedSettings.depthModel = 'da3_mono_large';
    }
    
    if (!validDepthModels.includes(mergedSettings.normalsDepthModel)) {
      console.warn(`[Layers] Invalid normalsDepthModel '${mergedSettings.normalsDepthModel}', resetting to 'da3_mono_large'`);
      mergedSettings.normalsDepthModel = 'da3_mono_large';
    }
    
    return mergedSettings;
  }, [project?.projectState?.tabs?.layers, localSettings]);
  
  const updateSettings = useCallback((updates) => {
    const newSettings = { ...settings, ...updates };
    
    if (project?.isProjectLoaded && project.updateTabState) {
      project.updateTabState('layers', newSettings);
    } else {
      setLocalSettings(newSettings);
    }
  }, [project, settings, setLocalSettings]);
  
  // Persist migrated settings to localStorage on first load
  useEffect(() => {
    // Only run if not using project state
    if (project?.projectState?.tabs?.layers) return;
    
    // Check if we need to persist migration
    const needsMigration = 
      localSettings.depthModel !== settings.depthModel ||
      localSettings.normalsDepthModel !== settings.normalsDepthModel;
    
    if (needsMigration) {
      console.log('[Layers] Persisting migrated depth model settings to localStorage');
      setLocalSettings(settings);
    }
  }, [settings, localSettings, project?.projectState?.tabs?.layers, setLocalSettings]);
  
  // UI state
  const [sourceInput, setSourceInput] = useState(null);
  const [generatedLayers, setGeneratedLayers] = useState({});
  const [errors, setErrors] = useState({});

  const {
    generating: processing,
    progress,
    result: genResult,
    error: genError,
    elapsedSeconds: elapsedTime,
    consoleLog,
    showModal,
    startGeneration,
    cancel,
    closeModal,
  } = useGeneration();
  
  // Timer ref

  const mountedRef = useRef(true);
  
  // Track if we've restored from project state
  const hasRestoredRef = useRef(false);
  
  // Detect if input is video
  const isVideo = useMemo(() => isVideoFile(sourceInput), [sourceInput]);
  
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
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
    console.log('[Layers] Reset effect fired - currentFilename changed to:', project?.currentFilename);
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

    setGeneratedLayers({});
    setErrors({});

    try {
      let taskType, payload;

      if (isVideo) {
        taskType = 'layers_video';
        payload = {
          video_path: sourceInput,
          layers: settings.layers,
          depth_model: settings.depthModel,
          depth_invert: settings.depthInvert,
          depth_normalize: settings.depthNormalize,
          depth_range_min: settings.depthRangeMin ?? 0.0,
          depth_range_max: settings.depthRangeMax ?? 1.0,
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
        taskType = 'layers';
        payload = {
          image_path: sourceInput,
          layers: settings.layers,
          depth_model: settings.depthModel,
          depth_invert: settings.depthInvert,
          depth_normalize: settings.depthNormalize,
          depth_range_min: settings.depthRangeMin ?? 0.0,
          depth_range_max: settings.depthRangeMax ?? 1.0,
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

      const data = await startTask(taskType, payload);
      startGeneration(data.generation_id);
    } catch (err) {
      console.error('Layer generation failed:', err);
      setErrors({ general: err.message });
    }
  };
  
  const handleCancel = () => {
    cancel();
  };
  
  // Bridge: extract layers from task result
  useEffect(() => {
    if (!genResult || genResult.status !== 'complete' || !genResult.result) return;

    const data = genResult.result;
    console.log('[LayersTab] Task complete:', data);

    if (data.layers) {
      const newLayers = {};
      for (const [name, info] of Object.entries(data.layers)) {
        // Build proper layer object for preview renderer
        // Backend returns { url, path, ... } for both image and video layers
        const layerObj = typeof info === 'string' ? { url: info } : { ...info };
        // Add display name (capitalize first letter)
        layerObj.name = name.charAt(0).toUpperCase() + name.slice(1);
        // Video layers have frame_count; image layers don't
        layerObj.isVideo = isVideo || !!layerObj.frame_count;
        // Build info string from backend metadata
        if (layerObj.frame_count) {
          layerObj.info = `${layerObj.frame_count} frames`;
        } else if (layerObj.model) {
          layerObj.info = layerObj.model;
        } else if (layerObj.method) {
          layerObj.info = layerObj.method;
        } else if (layerObj.num_objects != null) {
          layerObj.info = `${layerObj.num_objects} objects`;
        }
        newLayers[name] = layerObj;
      }
      setGeneratedLayers(newLayers);
    }
    if (data.errors) {
      setErrors(data.errors);
    }

    // Save to project lastState
    if (project?.updateLastState) {
      // Build the same layer objects we stored in state
      const layersForSave = {};
      if (data.layers) {
        for (const [name, info] of Object.entries(data.layers)) {
          const layerObj = typeof info === 'string' ? { url: info } : { ...info };
          layerObj.name = name.charAt(0).toUpperCase() + name.slice(1);
          layerObj.isVideo = isVideo || !!layerObj.frame_count;
          layersForSave[name] = layerObj;
        }
      }
      project.updateLastState({
        lastLayersPreview: data.layers,
        lastLayersMeta: {
          sourceInput,
          isVideo,
          layers: settings.layers,
          generatedLayers: layersForSave,
        },
        activeTab: 'layers',
      });
    }

    window.dispatchEvent(new CustomEvent('fuk-generation-complete', {
      detail: { type: 'layers', result: data }
    }));
  }, [genResult]);

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
                        src={buildImageUrl(layerData.preview_url || layerData.url)}
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
              initialDir={project?.projectState?.lastState?.lastUploadDir}
              onDirectorySelected={(dir) => project?.updateLastState?.({ lastUploadDir: dir })}
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
              {isVideo && ` (video frames)`}
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
                    <option value="da3_mono_large">DA3 Mono Large (Best)</option>
                    <option value="da3_metric_large">DA3 Metric (Real-world scale)</option>
                    <option value="da3_large">DA3 Large (Multi-view)</option>
                    <option value="da3_giant">DA3 Giant (Highest quality)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">
                  Range Min: {(settings.depthRangeMin ?? 0).toFixed(2)}
                </label>
                <div className="fuk-input-inline">
                  <input
                    type="range"
                    className="fuk-input fuk-input--flex-2"
                    value={settings.depthRangeMin ?? 0}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      const updates = { depthRangeMin: val };
                      if (val >= (settings.depthRangeMax ?? 1)) {
                        updates.depthRangeMax = Math.min(val + 0.01, 1.0);
                      }
                      updateSettings(updates);
                    }}
                    min={0}
                    max={1}
                    step={0.01}
                    disabled={processing}
                  />
                  <input
                    type="number"
                    className="fuk-input fuk-input--w-80"
                    value={settings.depthRangeMin ?? 0}
                    onChange={(e) => updateSettings({ depthRangeMin: parseFloat(e.target.value) || 0 })}
                    min={0}
                    max={1}
                    step={0.01}
                    disabled={processing}
                  />
                </div>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">
                  Range Max: {(settings.depthRangeMax ?? 1).toFixed(2)}
                </label>
                <div className="fuk-input-inline">
                  <input
                    type="range"
                    className="fuk-input fuk-input--flex-2"
                    value={settings.depthRangeMax ?? 1}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      const updates = { depthRangeMax: val };
                      if (val <= (settings.depthRangeMin ?? 0)) {
                        updates.depthRangeMin = Math.max(val - 0.01, 0.0);
                      }
                      updateSettings(updates);
                    }}
                    min={0}
                    max={1}
                    step={0.01}
                    disabled={processing}
                  />
                  <input
                    type="number"
                    className="fuk-input fuk-input--w-80"
                    value={settings.depthRangeMax ?? 1}
                    onChange={(e) => updateSettings({ depthRangeMax: parseFloat(e.target.value) || 1 })}
                    min={0}
                    max={1}
                    step={0.01}
                    disabled={processing}
                  />
                </div>
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
                        <option value="da3_mono_large">DA3 Mono Large (Best)</option>
                        <option value="da3_metric_large">DA3 Metric (Real-world scale)</option>
                        <option value="da3_large">DA3 Large (Multi-view)</option>
                        <option value="da3_giant">DA3 Giant (Highest quality)</option>
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
      <GenerationModal
        isOpen={showModal}
        type="layers"
        generating={processing}
        progress={progress}
        elapsedSeconds={elapsedTime}
        consoleLog={consoleLog}
        error={genError}
        onCancel={cancel}
        onClose={closeModal}
      />
    </>
  );
}