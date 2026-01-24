/**
 * Post-Process Tab
 * Upscaling (images & videos) and frame interpolation
 * Before/After comparison layout
 * Supports both single images and frame-by-frame video processing
 * 
 * Features:
 * - Persists last result in project state for cross-tab restoration
 * - On-demand loading of previews to prevent UI hangs
 */

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { Enhance, Loader2, CheckCircle, Camera, Film, AlertCircle } from '../components/Icons';
import MediaUploader, { isVideoFile } from '../components/MediaUploader';
import ZoomableImage from '../components/ZoomableImage';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
import Footer from '../components/Footer';

const API_URL = '/api';

const DEFAULT_SETTINGS = {
  // Upscaling
  upscaleMethod: 'realesrgan',
  upscaleFactor: 4,
  denoise: 0.5,
  
  // Frame Interpolation
  interpolationMethod: 'rife',
  targetFramerate: 24,
  sourceFramerate: 16,
  
  // Video output mode
  videoOutputMode: 'mp4',
};

export default function PostprocessTab({ config, activeTab, setActiveTab, project }) {
  // Settings (localStorage fallback)
  const [localSettings, setLocalSettings] = useLocalStorage('fuk_postprocess_settings', DEFAULT_SETTINGS);
  
  const settings = useMemo(() => {
    if (project?.projectState?.tabs?.postprocess) {
      return { ...DEFAULT_SETTINGS, ...project.projectState.tabs.postprocess };
    }
    return { ...DEFAULT_SETTINGS, ...localSettings };
  }, [project?.projectState?.tabs?.postprocess, localSettings]);
  
  // Ref to avoid infinite loop
  const settingsRef = useRef(settings);
  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);
  
  const updateSettings = useCallback((updates) => {
    const newSettings = { ...settingsRef.current, ...updates };
    
    if (project?.isProjectLoaded && project?.updateTabState) {
      project.updateTabState('postprocess', newSettings);
    } else {
      setLocalSettings(newSettings);
    }
  }, [project?.isProjectLoaded, project?.updateTabState, setLocalSettings]);
  
  // UI state
  const [activeProcess, setActiveProcess] = useState('upscale');
  const [sourceInput, setSourceInput] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  
  // Track input dimensions for proper aspect ratio
  const [inputDimensions, setInputDimensions] = useState({ width: 16, height: 9 });
  const [outputDimensions, setOutputDimensions] = useState({ width: 16, height: 9 });
  
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
  
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);
  
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
    if (activeTab !== 'postprocess') return;
    
    // Skip if we already have a result (user is actively working)
    if (result) return;
    
    const lastState = project?.projectState?.lastState;
    if (!lastState?.lastPostprocessPreview) return;
    
    // Restore the preview and metadata
    const meta = lastState.lastPostprocessMeta || {};
    
    console.log('[PostprocessTab] Restoring from lastState:', {
      preview: lastState.lastPostprocessPreview,
      meta
    });
    
    // Set flag to prevent handleSourceChange from clearing result
    isRestoringRef.current = true;
    
    // Restore source input and process type FIRST (before result)
    if (meta.sourceInput) {
      setSourceInput(meta.sourceInput);
    }
    if (meta.processType) {
      setActiveProcess(meta.processType);
    }
    
    // Restore result
    setResult({
      type: meta.isVideo ? 'video' : 'image',
      url: lastState.lastPostprocessPreview,
      scale: meta.scale,
      inputSize: meta.inputSize,
      outputSize: meta.outputSize,
      method: meta.method,
      frameCount: meta.frameCount,
      targetFps: meta.targetFps,
      sourceFps: meta.sourceFps,
      multiplier: meta.multiplier,
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
  
  // Fetch capabilities on mount
  useEffect(() => {
    fetch(`${API_URL}/postprocess/capabilities`)
      .then(res => res.json())
      .then(data => {
        console.log('[PostProcess] Capabilities:', data);
        setCapabilities(data);
      })
      .catch(err => {
        console.warn('[PostProcess] Could not fetch capabilities:', err);
      });
  }, []);
  
  // Load input dimensions when source changes
  useEffect(() => {
    if (!sourceInput) {
      setInputDimensions({ width: 16, height: 9 });
      return;
    }
    
    const url = buildImageUrl(sourceInput);
    
    if (!isVideo) {
      const img = new Image();
      img.onload = () => {
        setInputDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = url;
    } else {
      const video = document.createElement('video');
      video.onloadedmetadata = () => {
        setInputDimensions({ width: video.videoWidth, height: video.videoHeight });
      };
      video.src = url;
    }
  }, [sourceInput, isVideo]);
  
  // Update output dimensions when result changes
  useEffect(() => {
    if (!result) {
      setOutputDimensions(inputDimensions);
      return;
    }
    
    if (result.type === 'image' && result.outputSize) {
      setOutputDimensions({ width: result.outputSize.width, height: result.outputSize.height });
    } else if (result.type === 'video') {
      setOutputDimensions(inputDimensions);
    }
  }, [result, inputDimensions]);
  
  // Handle source input (MediaUploader passes array of media objects)
  const handleSourceChange = (media) => {
    // Don't clear result if we're in the middle of restoring
    if (isRestoringRef.current) {
      console.log('[PostprocessTab] Skipping result clear during restore');
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
  
  // Save result to project lastState
  const saveResultToLastState = useCallback((newResult) => {
    if (!project?.updateLastState) return;
    
    project.updateLastState({
      lastPostprocessPreview: newResult.url,
      lastPostprocessMeta: {
        isVideo: newResult.type === 'video',
        processType: activeProcess,
        sourceInput,
        scale: newResult.scale,
        inputSize: newResult.inputSize,
        outputSize: newResult.outputSize,
        method: newResult.method,
        frameCount: newResult.frameCount,
        targetFps: newResult.targetFps,
        sourceFps: newResult.sourceFps,
        multiplier: newResult.multiplier,
      },
      activeTab: 'postprocess',
    });
  }, [project?.updateLastState, activeProcess, sourceInput]);
  
  const handleUpscale = async () => {
    if (!sourceInput) {
      setError('Please select a source image or video');
      return;
    }
    
    setProcessing(true);
    setStartTime(Date.now());
    setElapsedSeconds(0);
    setProgress({ phase: 'Starting upscale...', progress: 0.1 });
    setError(null);
    
    try {
      let endpoint, payload;
      
      if (isVideo) {
        // Video upscaling - frame by frame
        endpoint = `${API_URL}/postprocess/upscale/video`;
        payload = {
          source_path: sourceInput,
          scale: settings.upscaleFactor,
          model: settings.upscaleMethod,
          denoise: settings.denoise,
          output_mode: settings.videoOutputMode,
        };
      } else {
        // Image upscaling
        endpoint = `${API_URL}/postprocess/upscale`;
        payload = {
          source_path: sourceInput,
          scale: settings.upscaleFactor,
          model: settings.upscaleMethod,
          denoise: settings.denoise,
        };
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Upscale failed');
      }
      
      const data = await response.json();
      console.log('[PostProcess] Upscale result:', data);
      
      let newResult;
      if (isVideo) {
        newResult = {
          type: 'video',
          url: data.output_url,
          path: data.output_path,
          inputSize: data.input_size,
          outputSize: data.output_size,
          scale: data.scale,
          method: data.method,
          frameCount: data.frame_count,
        };
      } else {
        newResult = {
          type: 'image',
          url: data.output_url,
          path: data.output_path,
          inputSize: data.input_size,
          outputSize: data.output_size,
          scale: data.scale,
          method: data.method,
        };
      }
      
      setResult(newResult);
      setProgress({ phase: 'Complete', progress: 1 });
      
      // Save to project lastState
      saveResultToLastState(newResult);
      
      // Notify history to refresh
      window.dispatchEvent(new CustomEvent('fuk-generation-complete', {
        detail: { 
          type: 'upscale',
          result: data,
          elapsed: Math.floor((Date.now() - startTime) / 1000)
        }
      }));
      
    } catch (err) {
      console.error('[PostProcess] Upscale error:', err);
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };
  
  const handleInterpolate = async () => {
    if (!sourceInput) {
      setError('Please select a source video');
      return;
    }
    
    setProcessing(true);
    setStartTime(Date.now());
    setElapsedSeconds(0);
    setProgress({ phase: 'Starting interpolation...', progress: 0.1 });
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/postprocess/interpolate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path: sourceInput,
          source_fps: settings.sourceFramerate,
          target_fps: settings.targetFramerate,
          model: settings.interpolationMethod,
        }),
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Interpolation failed');
      }
      
      const data = await response.json();
      console.log('[PostProcess] Interpolate result:', data);
      
      const newResult = {
        type: 'video',
        url: data.output_url,
        path: data.output_path,
        sourceFps: data.source_fps,
        targetFps: data.target_fps,
        multiplier: data.multiplier,
        method: data.method,
      };
      
      setResult(newResult);
      setProgress({ phase: 'Complete', progress: 1 });
      // Save to project lastState
      saveResultToLastState(newResult);
      // Notify history to refresh
      window.dispatchEvent(new CustomEvent('fuk-generation-complete', {
        detail: { 
          type: 'interpolate',
          result: data,
          elapsed: Math.floor((Date.now() - startTime) / 1000)
        }
      }));
      
    } catch (err) {
      console.error('[PostProcess] Interpolate error:', err);
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };
  
  const handleProcess = async () => {
    if (activeProcess === 'upscale') {
      await handleUpscale();
    } else {
      await handleInterpolate();
    }
  };
  
  const handleCancel = () => {
    setProcessing(false);
    setProgress(null);
    setStartTime(null);
    setElapsedSeconds(0);
  };
  
  const canProcess = sourceInput && !processing;
  const inputAspectRatio = inputDimensions.width / inputDimensions.height;
  const inputAspectStyle = { '--preview-aspect': inputAspectRatio };
  
  const getInputInfo = () => {
    if (!sourceInput) return null;
    if (activeProcess === 'upscale') {
      return `${inputDimensions.width}×${inputDimensions.height}${isVideo ? ' (video)' : ''}`;
    }
    return `${settings.sourceFramerate} fps`;
  };
  
  const getOutputInfo = () => {
    if (!result) return null;
    if (result.type === 'image') {
      return `${result.outputSize.width}×${result.outputSize.height} (${result.scale}x)`;
    }
    if (result.frameCount) {
      return `${result.outputSize?.width || inputDimensions.width}×${result.outputSize?.height || inputDimensions.height} (${result.scale}x, ${result.frameCount} frames)`;
    }
    return `${result.targetFps} fps (${result.multiplier}x frames)`;
  };
  
  // Format elapsed time
  const formatTime = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };
  
  return (
    <>
      {/* Preview Area - Side by Side Comparison */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-compare">
          {/* Input */}
          <div className="fuk-preview-pane">
            <div className="fuk-preview-pane-header">
              <h3 className="fuk-preview-pane-title">
                Input
                {isVideo && <Film className="fuk-icon--sm fuk-ml-2" />}
              </h3>
              {sourceInput && (
                <span className="fuk-preview-pane-info">{getInputInfo()}</span>
              )}
            </div>
            
            {sourceInput ? (
              <div className="fuk-media-frame">
                {isVideo ? (
                  <video
                    src={buildImageUrl(sourceInput)}
                    controls
                    loop
                    className="fuk-preview-media--constrained"
                    preload="metadata"
                  />
                ) : (
                  <img
                    src={buildImageUrl(sourceInput)}
                    alt="Input"
                    className="fuk-preview-media--constrained"
                    loading="lazy"
                  />
                )}
              </div>
            ) : (
              <div 
                className="fuk-placeholder-card fuk-placeholder-card--100 fuk-placeholder-card--16x9"
              >
                <div className="fuk-placeholder">
                  {activeProcess === 'upscale' ? (
                    <Camera className="fuk-placeholder-icon--sm" />
                  ) : (
                    <Film className="fuk-placeholder-icon--sm" />
                  )}
                  <p className="fuk-placeholder-text">
                    {activeProcess === 'upscale' ? 'Select image or video to upscale' : 'Select video to interpolate'}
                  </p>
                </div>
              </div>
            )}
          </div>
          
          {/* Arrow */}
          <div className={`fuk-compare-arrow ${processing ? 'fuk-compare-arrow--active' : ''}`}>
            {processing ? (
              <Loader2 className="fuk-compare-arrow-icon fuk-compare-arrow-icon--spin" />
            ) : (
              <span>→</span>
            )}
            {processing && progress && (
              <span className="fuk-compare-arrow-label">{progress.phase}</span>
            )}
          </div>
          
          {/* Output */}
          <div className="fuk-preview-pane">
            <div className="fuk-preview-pane-header">
              <h3 className="fuk-preview-pane-title">
                {activeProcess === 'upscale' ? 'Upscaled' : 'Interpolated'}
              </h3>
              {result && (
                <div className="fuk-status-complete">
                  <CheckCircle className="fuk-icon--sm" />
                  <span>{getOutputInfo()}</span>
                </div>
              )}
            </div>
            
            {result ? (
              <div className="fuk-media-frame fuk-media-frame--success">
                {result.type === 'image' ? (
                  <img
                    src={buildImageUrl(result.url)}
                    alt="Processed"
                    className="fuk-preview-media--constrained"
                    loading="lazy"
                  />
                ) : (
                  <video
                    src={buildImageUrl(result.url)}
                    controls
                    autoPlay
                    loop
                    className="fuk-preview-media--constrained"
                    preload="metadata"
                  />
                )}
                <span className="fuk-preview-badge fuk-preview-badge--method">
                  {result.method}
                </span>
              </div>
            ) : (
              <div 
                className="fuk-placeholder-card fuk-placeholder-card--100 fuk-placeholder-card--dynamic"
                style={sourceInput ? inputAspectStyle : undefined}
              >
                <div className="fuk-placeholder">
                  <Enhance className="fuk-placeholder-icon--sm" />
                  <p className="fuk-placeholder-text">
                    {processing ? `Processing... ${formatTime(elapsedSeconds)}` : 'Result will appear here'}
                  </p>
                  {processing && isVideo && (
                    <p className="fuk-placeholder-subtext">Processing video frame-by-frame</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
                    {/* Input Source Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">
              {activeProcess === 'upscale' ? 'Image/Video Input' : 'Video Input'}
            </h3>
            
            <MediaUploader
              media={sourceInput ? [{ path: sourceInput }] : []}
              onMediaChange={handleSourceChange}
              disabled={processing}
              accept={activeProcess === 'interpolate' ? 'videos' : 'all'}
            />
            
            <p className="fuk-help-text">
              {activeProcess === 'upscale' 
                ? 'Drag from History or upload an image/video to enhance'
                : 'Drag from History or upload a video to interpolate'
              }
            </p>
            
            {isVideo && activeProcess === 'upscale' && (
              <div className="fuk-alert fuk-alert--info fuk-mt-3">
                <Film className="fuk-alert-icon" />
                <span className="fuk-alert-text">Video: Processing frame-by-frame</span>
              </div>
            )}
            
            {error && (
              <div className="fuk-alert fuk-alert--error fuk-mt-3">
                <AlertCircle className="fuk-alert-icon" />
                <span className="fuk-alert-text">{error}</span>
              </div>
            )}
            
            {/* Video output mode for upscaling */}
            {isVideo && activeProcess === 'upscale' && (
              <div className="fuk-form-group-compact fuk-mt-4 fuk-pt-4 fuk-border-top">
                <label className="fuk-label">Output Format</label>
                <select
                  className="fuk-select"
                  value={settings.videoOutputMode}
                  onChange={(e) => updateSettings({ videoOutputMode: e.target.value })}
                  disabled={processing}
                >
                  <option value="mp4">MP4 Video</option>
                  <option value="sequence">Image Sequence</option>
                </select>
                <p className="fuk-help-text">
                  {settings.videoOutputMode === 'mp4' 
                    ? 'Outputs a single upscaled video' 
                    : 'Outputs individual frames for EXR workflow'}
                </p>
              </div>
            )}
          </div>

          {/* Process Type Selection */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Process Type</h3>
            
            <div className="fuk-radio-card-group">
              <label className={`fuk-radio-card ${activeProcess === 'upscale' ? 'fuk-radio-card--active' : ''}`}>
                <input
                  type="radio"
                  className="fuk-radio"
                  checked={activeProcess === 'upscale'}
                  onChange={() => {
                    setActiveProcess('upscale');
                    setResult(null);
                    setError(null);
                  }}
                />
                <div className="fuk-radio-card-content">
                  <span className="fuk-radio-card-title">
                    <Camera className="fuk-icon" />
                    Upscaling
                  </span>
                  <span className="fuk-radio-card-desc">
                    Increase resolution using AI (images & videos)
                  </span>
                </div>
              </label>
              
              <label className={`fuk-radio-card ${activeProcess === 'interpolate' ? 'fuk-radio-card--active' : ''}`}>
                <input
                  type="radio"
                  className="fuk-radio"
                  checked={activeProcess === 'interpolate'}
                  onChange={() => {
                    setActiveProcess('interpolate');
                    setResult(null);
                    setError(null);
                  }}
                />
                <div className="fuk-radio-card-content">
                  <span className="fuk-radio-card-title">
                    <Film className="fuk-icon" />
                    Frame Interpolation
                  </span>
                  <span className="fuk-radio-card-desc">
                    Generate intermediate frames (RIFE)
                  </span>
                </div>
              </label>
            </div>
            
            {/* Capability indicators */}
            {capabilities && (
              <div className="fuk-capability-box">
                <div className="fuk-capability-label">Available backends:</div>
                <div className="fuk-capability-list">
                  <span className={capabilities.upscaling?.ncnn_available ? 'fuk-capability-item--active' : 'fuk-capability-item'}>
                    {capabilities.upscaling?.ncnn_available ? '✓' : '○'} ESRGAN-NCNN
                  </span>
                  <span className={capabilities.interpolation?.ncnn_available ? 'fuk-capability-item--active' : 'fuk-capability-item'}>
                    {capabilities.interpolation?.ncnn_available ? '✓' : '○'} RIFE-NCNN
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Upscaling Settings */}
          {activeProcess === 'upscale' && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Upscaling Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">AI Model</label>
                <select
                  className="fuk-select"
                  value={settings.upscaleMethod}
                  onChange={(e) => updateSettings({ upscaleMethod: e.target.value })}
                  disabled={processing}
                >
                  <option value="realesrgan">Real-ESRGAN (Recommended)</option>
                  <option value="lanczos">Lanczos (Fast, No AI)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Scale Factor</label>
                <div className="fuk-radio-group">
                  {[2, 4, 8].map(factor => (
                    <label key={factor} className="fuk-radio-option">
                      <input
                        type="radio"
                        className="fuk-radio"
                        checked={settings.upscaleFactor === factor}
                        onChange={() => updateSettings({ upscaleFactor: factor })}
                        disabled={processing}
                      />
                      <span className="fuk-radio-label">{factor}x</span>
                    </label>
                  ))}
                </div>
                <p className="fuk-help-text--inline">
                  {settings.upscaleFactor}x scale = {settings.upscaleFactor * settings.upscaleFactor}x more pixels
                </p>
              </div>
              
              {settings.upscaleMethod === 'realesrgan' && (
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">Denoise Strength</label>
                  <div className="fuk-input-inline">
                    <input
                      type="range"
                      className="fuk-range fuk-input--flex-2"
                      value={settings.denoise}
                      onChange={(e) => updateSettings({ denoise: parseFloat(e.target.value) })}
                      min={0}
                      max={1}
                      step={0.05}
                      disabled={processing}
                    />
                    <input
                      type="number"
                      className="fuk-input fuk-input--w-80"
                      value={settings.denoise}
                      onChange={(e) => updateSettings({ denoise: parseFloat(e.target.value) })}
                      step={0.05}
                      min={0}
                      max={1}
                      disabled={processing}
                    />
                  </div>
                  <p className="fuk-help-text--inline">
                    Higher = more noise reduction (may soften details)
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Frame Interpolation Settings */}
          {activeProcess === 'interpolate' && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Interpolation Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">AI Model</label>
                <select
                  className="fuk-select"
                  value={settings.interpolationMethod}
                  onChange={(e) => updateSettings({ interpolationMethod: e.target.value })}
                  disabled={processing}
                >
                  <option value="rife">RIFE (Recommended)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Source Framerate</label>
                <select
                  className="fuk-select"
                  value={settings.sourceFramerate}
                  onChange={(e) => updateSettings({ sourceFramerate: parseInt(e.target.value) })}
                  disabled={processing}
                >
                  <option value="8">8 fps</option>
                  <option value="12">12 fps</option>
                  <option value="16">16 fps (Wan default)</option>
                  <option value="24">24 fps</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Target Framerate</label>
                <select
                  className="fuk-select"
                  value={settings.targetFramerate}
                  onChange={(e) => updateSettings({ targetFramerate: parseInt(e.target.value) })}
                  disabled={processing}
                >
                  <option value="24">24 fps (Film)</option>
                  <option value="30">30 fps (Video)</option>
                  <option value="60">60 fps (Smooth)</option>
                </select>
              </div>
              
              <div className="fuk-stats-box">
                <div className="fuk-stats-primary">
                  Frame multiplier: <strong>{Math.round(settings.targetFramerate / settings.sourceFramerate)}x</strong>
                </div>
                <div className="fuk-stats-secondary">
                  {settings.sourceFramerate} fps → {settings.targetFramerate} fps
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={processing}
        progress={progress}
        elapsedSeconds={elapsedSeconds}
        onGenerate={handleProcess}
        onCancel={handleCancel}
        canGenerate={canProcess}
        generateLabel={activeProcess === 'upscale' ? (isVideo ? 'Upscale Video' : 'Upscale') : 'Interpolate'}
        generatingLabel={activeProcess === 'upscale' ? (isVideo ? 'Upscaling Video...' : 'Upscaling...') : 'Interpolating...'}
      />
    </>
  );
}