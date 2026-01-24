/**
 * Preprocess Tab
 * Image/Video preprocessing for control inputs: Canny, OpenPose, Depth
 * Supports both single images and frame-by-frame video processing
 * 
 * Updated: VideoSyncController for synchronized video comparison
 */

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Pipeline, CheckCircle, Camera, Film, AlertCircle, Folder } from '../components/Icons';
import MediaUploader, { isVideoFile } from '../components/MediaUploader';
import ZoomableImage from '../components/ZoomableImage';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
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
    depth_model: 'da3_mono_large',  // Changed from depth_anything_v2
    depth_invert: false,
    depth_normalize: true,
    depth_colormap: 'inferno',
  },
  // Video-specific settings
  video: {
    output_mode: 'mp4',  // 'mp4' or 'sequence'
  },
};

export default function PreprocessTab({ config, activeTab, setActiveTab, project }) {
  // UI state
  const [selectedMethod, setSelectedMethod] = useState('canny');
  const [sourceInput, setSourceInput] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // Timer state
  const [startTime, setStartTime] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const mountedRef = useRef(true);
  
  // Track if we've restored from project state
  const hasRestoredRef = useRef(false);
  // Flag to prevent clearing result during restoration
  const isRestoringRef = useRef(false);
  
  // Detect if input is video
  const isVideo = useMemo(() => isVideoFile(sourceInput), [sourceInput]);
  
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
  
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);
  
  // Timer effect
  useEffect(() => {
    if (!processing || !startTime) return;
    
    const interval = setInterval(() => {
      if (mountedRef.current) {
        setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [processing, startTime]);
  
  // Restore last preview from project state when tab becomes active
  useEffect(() => {
    // Only run when this tab is active
    if (activeTab !== 'preprocess') return;
    if (result) return;
    
    const lastState = project?.projectState?.lastState;
    if (!lastState?.lastPreprocessPreview) return;
    
  
    // Restore the preview and metadata
    const meta = lastState.lastPreprocessMeta || {};

    console.log('[PreprocessTab] Restoring from lastState:', {
      preview: lastState.lastPreprocessPreview,
      meta
    });
    
    hasRestoredRef.current = true;
    // Set flag to prevent handleSourceInputChange from clearing result
    isRestoringRef.current = true;
    
    // Restore source input FIRST (before result)
    if (meta.sourceInput) {
      setSourceInput(meta.sourceInput);
    }
    
    // Restore selected method if available
    if (meta.method) {
      setSelectedMethod(meta.method);
    }
    
    // Restore result
    setResult({
      url: lastState.lastPreprocessPreview,
      isVideo: meta.isVideo || false,
      isSequence: meta.isSequence || false,
      frame_count: meta.frameCount || 0,
    });
    
    // Clear restoring flag after state has settled
    setTimeout(() => {
      isRestoringRef.current = false;
    }, 100);
    
  }, [activeTab, project?.projectState?.lastState, result]);
  
  // Reset state when project file changes
  useEffect(() => {
    setResult(null);
    setSourceInput(null);
    setError(null);
  }, [project?.currentFilename]);
  
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
  const videoSettings = settings.video || DEFAULT_SETTINGS.video;
  
  const setCurrentSettings = (updates) => {
    updateSettings({
      [selectedMethod]: { ...currentSettings, ...updates }
    });
  };
  
  const setVideoSettings = (updates) => {
    updateSettings({
      video: { ...videoSettings, ...updates }
    });
  };
  
  // Handle source input selection (MediaUploader passes array of media objects)
  const handleSourceInputChange = (media) => {
    // Don't clear result if we're in the middle of restoring
    if (isRestoringRef.current) {
      console.log('[PreprocessTab] Skipping result clear during restore');
      return;
    }
    
    const first = media[0];
    const newPath = first?.path || first || null;
    
    // Only update and clear result if value actually changed
    if (newPath !== sourceInput) {
      setSourceInput(newPath);
      setResult(null);
      setError(null);
    }
  };
  
  // Run preprocessing
  const handleProcess = async () => {
    if (!sourceInput) {
      alert('Please select a source image or video');
      return;
    }
    
    setProcessing(true);
    setStartTime(Date.now());
    setElapsedSeconds(0);
    setError(null);
    setResult(null);
    
    try {
      // Build request payload
      const payload = {
        method: selectedMethod,
        ...currentSettings,
      };
      
      let endpoint;
      
      if (isVideo) {
        // Video processing
        endpoint = `${API_URL}/preprocess/video`;
        payload.video_path = sourceInput;
        payload.output_mode = videoSettings.output_mode;
      } else {
        // Image processing
        endpoint = `${API_URL}/preprocess`;
        payload.image_path = sourceInput;
      }
      
      console.log(`Preprocessing (${isVideo ? 'video' : 'image'}):`, payload);
      
      const res = await fetch(endpoint, {
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
      
      // Handle response - use preview_url for sequences, output_url for videos/images
      const displayUrl = data.is_sequence 
        ? (data.preview_url || data.output_url) 
        : (data.output_url || data.url);
      
      const newResult = {
        ...data,
        isVideo,
        url: displayUrl,
        // Keep is_sequence flag for display logic
        isSequence: data.is_sequence || false,
        // For sequences, store the directory URL separately
        sequenceUrl: data.is_sequence ? data.output_url : null,
        frames: data.frames || [],
      };
      
      setResult(newResult);
      
      // Save to project lastState for cross-tab persistence
      if (project?.updateLastState) {
        project.updateLastState({
          lastPreprocessPreview: displayUrl,
          lastPreprocessMeta: {
            isVideo,
            isSequence: data.is_sequence || false,
            method: selectedMethod,
            sourceInput,
            frameCount: data.frame_count || 0,
            params: currentSettings,
          },
          activeTab: 'preprocess',
        });
      }
      
      // Notify history to refresh
      window.dispatchEvent(new CustomEvent('fuk-generation-complete', {
        detail: { 
          type: 'preprocess',
          method: selectedMethod,
          result: data,
        }
      }));
      
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
                className="fuk-input fuk-input--flex-2"
                value={currentSettings.low_threshold}
                onChange={(e) => setCurrentSettings({ low_threshold: parseInt(e.target.value) })}
                min={0}
                max={500}
                step={10}
              />
              <input
                type="number"
                className="fuk-input fuk-input--w-80"
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
                className="fuk-input fuk-input--flex-2"
                value={currentSettings.high_threshold}
                onChange={(e) => setCurrentSettings({ high_threshold: parseInt(e.target.value) })}
                min={0}
                max={500}
                step={10}
              />
              <input
                type="number"
                className="fuk-input fuk-input--w-80"
                value={currentSettings.high_threshold}
                onChange={(e) => setCurrentSettings({ high_threshold: parseInt(e.target.value) })}
                min={0}
                max={500}
              />
            </div>
          </div>
          
          <div className="fuk-form-group-compact">
            <label className="fuk-label">Blur Kernel</label>
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
              <span className="fuk-checkbox-label">Invert Output</span>
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
              <span className="fuk-checkbox-label">Detect Body</span>
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
              <span className="fuk-checkbox-label">Detect Hands</span>
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
              <span className="fuk-checkbox-label">Detect Face</span>
            </label>
          </div>
          
          <p className="fuk-help-text fuk-mt-4">
            OpenPose detection for human pose estimation. Enable what you need.
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
              value={currentSettings.depth_colormap || ''}
              onChange={(e) => setCurrentSettings({ depth_colormap: e.target.value || null })}
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
                checked={currentSettings.depth_normalize}
                onChange={(e) => setCurrentSettings({ depth_normalize: e.target.checked })}
              />
              <span className="fuk-checkbox-label">Normalize Depth</span>
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
              <span className="fuk-checkbox-label">Invert Depth (near = dark)</span>
            </label>
          </div>
          
          <p className="fuk-help-text fuk-mt-4">
            {currentSettings.depth_model === 'da3_mono_large' && '✓ Recommended: Latest SOTA for single images'}
            {currentSettings.depth_model === 'da3_metric_large' && '📏 Metric depth in meters (real-world scale)'}
            {currentSettings.depth_model === 'da3_large' && '🔄 Supports multi-view consistency'}
            {currentSettings.depth_model === 'da3_giant' && '🏆 Highest quality, 1.15B params'}
            {currentSettings.depth_model === 'depth_anything_v2' && '⚡ V2 fallback, great quality'}
            {currentSettings.depth_model === 'midas_small' && '⚡ Fastest processing'}
            {currentSettings.depth_model === 'midas_large' && 'Good quality, widely compatible'}
            {currentSettings.depth_model === 'zoedepth' && 'Metric depth estimation'}
          </p>
        </>
      );
    }
  };
  
  const methodTitle = selectedMethod.charAt(0).toUpperCase() + selectedMethod.slice(1);
  
  // Format elapsed time
  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };
  
  // Render the result preview - handles video, sequence, and image
  const renderResultPreview = () => {
    if (!result) return null;
    
    // Sequence output - show first frame as image
    if (result.isSequence) {
      return (
        <div className="fuk-media-frame fuk-media-frame--success">
          {result.url ? (
            <ZoomableImage
              src={buildImageUrl(result.url)}
              alt={`${methodTitle} Preview (Frame 1)`}
              className="fuk-preview-media--constrained"
              loading="lazy"
            />
          ) : (
            <div className="fuk-placeholder fuk-placeholder--sm">
              <Folder className="fuk-placeholder-icon" />
              <p className="fuk-placeholder-text">Sequence saved</p>
            </div>
          )}
          <div className="fuk-preview-badge fuk-preview-badge--success">
            <CheckCircle className="fuk-icon--md" />
            {result.frame_count} frames
          </div>
          {/* Sequence info overlay */}
          <div className="fuk-sequence-info">
            <Folder className="fuk-icon--sm" />
            <span>Image Sequence</span>
          </div>
        </div>
      );
    }
    
    // Video output - show as video player
    if (result.isVideo) {
      return (
        <div className="fuk-media-frame fuk-media-frame--success">
          <video
            src={buildImageUrl(result.url)}
            controls
            className="fuk-preview-media--constrained"
            preload="metadata"
          />
          <div className="fuk-preview-badge fuk-preview-badge--success">
            <CheckCircle className="fuk-icon--md" />
            Complete
          </div>
        </div>
      );
    }
    
    // Image output
    return (
      <div className="fuk-media-frame fuk-media-frame--success">
        <ZoomableImage
          src={buildImageUrl(result.url)}
          alt="Processed"
          className="fuk-preview-media--constrained"
          loading="lazy"
        />
        <div className="fuk-preview-badge fuk-preview-badge--success">
          <CheckCircle className="fuk-icon--md" />
          Complete
        </div>
      </div>
    );
  };
  
  return (
    <>
      {/* Preview Area - Side by Side Comparison */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-compare">
          {/* Source Input */}
          <div className="fuk-preview-pane">
            <div className="fuk-preview-pane-header">
              <h3 className="fuk-preview-pane-title">
                Source {isVideo && <Film className="fuk-icon--sm fuk-ml-2" />}
              </h3>
              {isVideo && (
                <span className="fuk-badge fuk-badge--purple">Video</span>
              )}
            </div>
            {sourceInput ? (
              isVideo ? (
                <video
                  src={buildImageUrl(sourceInput)}
                  controls
                  className="fuk-preview-media fuk-preview-media--constrained"
                  preload="metadata"
                />
              ) : (
                <ZoomableImage
                  src={buildImageUrl(sourceInput)}
                  alt="Source"
                  className="fuk-preview-media fuk-preview-media--constrained"
                  loading="lazy"
                />
              )
            ) : (
              <div className="fuk-placeholder-card fuk-placeholder-card--100 fuk-placeholder-card--16x9">
                <div className="fuk-placeholder">
                  <Camera className="fuk-placeholder-icon" />
                  <p className="fuk-placeholder-text">Select source image or video below</p>
                </div>
              </div>
            )}
          </div>
          
          {/* Processed Result */}
          <div className="fuk-preview-pane">
            <div className="fuk-preview-pane-header">
              <h3 className="fuk-preview-pane-title">{methodTitle} Result</h3>
              {result?.isVideo && !result?.isSequence && (
                <span className="fuk-badge fuk-badge--green">
                  {result.frame_count} frames
                </span>
              )}
              {result?.isSequence && (
                <span className="fuk-badge fuk-badge--blue">
                  Sequence: {result.frame_count} frames
                </span>
              )}
            </div>
            {result ? (
              renderResultPreview()
            ) : error ? (
              <div className="fuk-placeholder-card fuk-placeholder-card--100 fuk-placeholder-card--16x9 fuk-placeholder-card--error">
                <div className="fuk-placeholder">
                  <AlertCircle className="fuk-placeholder-icon fuk-placeholder-icon--error" />
                  <p className="fuk-placeholder-text--error">Error: {error}</p>
                </div>
              </div>
            ) : (
              <div className="fuk-placeholder-card fuk-placeholder-card--100 fuk-placeholder-card--16x9">
                <div className="fuk-placeholder">
                  <Pipeline className="fuk-placeholder-icon" />
                  <p className="fuk-placeholder-text">
                    {processing ? `Processing... ${formatTime(elapsedSeconds)}` : 'Click Process to generate'}
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
          {/* Source Input Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Source Input</h3>
            
            <MediaUploader
              media={sourceInput ? [{ path: sourceInput }] : []}
              onMediaChange={handleSourceInputChange}
              disabled={processing}
              accept="all"
            />
            
            <p className="fuk-help-text">
              Upload or select an image or video to preprocess
            </p>
            
            {/* Video-specific settings */}
            {isVideo && (
              <div className="fuk-mt-4 fuk-pt-4 fuk-border-top">
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">Output Format</label>
                  <select
                    className="fuk-select"
                    value={videoSettings.output_mode}
                    onChange={(e) => setVideoSettings({ output_mode: e.target.value })}
                    disabled={processing}
                  >
                    <option value="mp4">MP4 Video</option>
                    <option value="sequence">Image Sequence</option>
                  </select>
                </div>
                <p className="fuk-help-text">
                  {videoSettings.output_mode === 'mp4' 
                    ? 'Outputs a single video file' 
                    : 'Outputs individual PNG frames for EXR workflow'}
                </p>
              </div>
            )}
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
            
            <div className="fuk-method-info fuk-method-info--purple">
              {selectedMethod === 'canny' && 'Clean edge detection for line art and contours'}
              {selectedMethod === 'openpose' && 'Detect human pose keypoints for precise control'}
              {selectedMethod === 'depth' && 'Estimate depth map for 3D-aware generation'}
            </div>
            
            {isVideo && (
              <div className="fuk-alert fuk-alert--info fuk-mt-4">
                <Film className="fuk-alert-icon" />
                <span className="fuk-alert-text">
                  Video mode: Processing frame-by-frame
                </span>
              </div>
            )}
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
        elapsedSeconds={elapsedSeconds}
        onGenerate={handleProcess}
        onCancel={() => setProcessing(false)}
        canGenerate={!!sourceInput}
        generateLabel={isVideo ? "Process Video" : "Process"}
        generatingLabel={isVideo ? "Processing Video..." : "Processing..."}
      />
    </>
  );
}