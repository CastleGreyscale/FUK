/**
 * Post-Process Tab
 * Upscaling (images & videos) and frame interpolation
 * Before/After comparison layout
 * Supports both single images and frame-by-frame video generating
 * 
 * Features:
 * - Persists last result in project state for cross-tab restoration
 * - On-demand loading of previews to prevent UI hangs
 */

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { Enhance, Loader2, CheckCircle, Camera, Film, AlertCircle } from '../components/Icons';
import MediaUploader, { isVideoFile } from '../components/MediaUploader';
import ZoomableImage from '../components/ZoomableImage';
import GenerationModal from '../components/GenerationModal';
import { useGeneration } from '../hooks/useGeneration';
import { startTask } from '../utils/api';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
import { useVideoPlayback } from '../hooks/useVideoPlayback';
import Footer from '../components/Footer';

const API_URL = '/api';

const DEFAULT_SETTINGS = {
  // Upscaling — images
  upscaleMethod: 'realesrgan',
  upscaleFactor: 4,

  // Upscaling — video. SeedVR2 is the temporal option and the default when
  // it is installed; realesrgan/lanczos remain available and are per-frame,
  // which visibly flickers on generated footage.
  videoUpscaleMethod: 'seedvr2',
  videoUpscaleFactor: 2,

  // SeedVR2 (ignored by the per-frame models)
  seedvr2Variant: 'seedvr2_7b_fp8',
  seedvr2FrameWindow: 13,
  seedvr2ResolutionCap: 1920,
  seedvr2MemoryMode: 'auto',
  // -1 = derive from the frame window. 0 disables batch blending, which shows
  // up as the picture stepping every frame-window frames.
  seedvr2TemporalOverlap: -1,
  // 0..1. Blends moving pixels back toward the source to keep motion blur.
  seedvr2MotionProtection: 0.7,

  // Frame Interpolation
  interpolationMethod: 'film',
  targetFramerate: 24,
  sourceFramerate: 16,

  // Video output mode
  videoOutputMode: 'mp4',
};

export default function PostprocessTab({ config, activeTab, setActiveTab, project, playbackSpeed }) {
  // Settings (localStorage fallback)
  const [localSettings, setLocalSettings] = useLocalStorage('fuk_postprocess_settings', DEFAULT_SETTINGS);
  const videoRef = useVideoPlayback(playbackSpeed);
  
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
  const [result, setResult] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [localError, setLocalError] = useState(null);

  const {
    generating,
    progress,
    result: genResult,
    error: genError,
    elapsedSeconds,
    consoleLog,
    showModal,
    startGeneration,
    cancel,
    closeModal,
    reset: resetGeneration,
  } = useGeneration();

  const error = genError || localError;
  
  // Track input dimensions for proper aspect ratio
  const [inputDimensions, setInputDimensions] = useState({ width: 16, height: 9 });
  const [outputDimensions, setOutputDimensions] = useState({ width: 16, height: 9 });
  
  
  // Track if we've restored from project state
  const hasRestoredRef = useRef(false);
  // Flag to prevent clearing result during restoration
  const isRestoringRef = useRef(false);
  
  // Detect if input is video
  const isVideo = useMemo(() => isVideoFile(sourceInput), [sourceInput]);

  // These key names have to stay in step with PostProcessorManager
  // .get_capabilities() — they drifted once already when the interpolator was
  // swapped from RIFE to FILM, leaving the badge permanently dark.
  const seedvr2Available = capabilities?.video_restoration?.available ?? false;
  const isSeedVR2 = isVideo && settings.videoUpscaleMethod === 'seedvr2';


  // Reset restoration flag when leaving the tab
  useEffect(() => {
    if (activeTab !== 'postprocess') {
      hasRestoredRef.current = false;
    }
  }, [activeTab]);
  
  // Restore last preview from project state when tab becomes active
  useEffect(() => {
    // Only run when this tab is active
    if (activeTab !== 'postprocess') return;
    
    // Skip if we already have a result (user is actively working)
    if (result) return;
    
    // Only restore once per tab activation
    if (hasRestoredRef.current) return;
    
    const lastState = project?.projectState?.lastState;
    if (!lastState?.lastPostprocessPreview) return;
    
    // Restore the preview and metadata
    const meta = lastState.lastPostprocessMeta || {};
    
    console.log('[PostprocessTab] Restoring from lastState:', {
      preview: lastState.lastPostprocessPreview,
      meta
    });
    
    hasRestoredRef.current = true;
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
    hasRestoredRef.current = false; // Allow restoration for new project
  }, [project?.currentFilename]);
  
  // Fetch capabilities on mount
  useEffect(() => {
    fetch(`${API_URL}/postprocess/capabilities`)
      .then(res => res.json())
      .then(data => {
        console.log('[PostProcess] Capabilities:', data);
        setCapabilities(data);

        // SeedVR2 is the default video model, but it is only present when the
        // vendored engine is installed. Fall back rather than letting the user
        // submit a request that is guaranteed to 500.
        if (!data?.video_restoration?.available &&
            settingsRef.current.videoUpscaleMethod === 'seedvr2') {
          updateSettings({ videoUpscaleMethod: 'realesrgan' });
        }
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
      setLocalError('Please select a source image or video');
      return;
    }

    setLocalError(null);
    setResult(null);

    try {
      let taskType, payload;

      if (isVideo) {
        taskType = 'upscale_video';
        payload = {
          source_path: sourceInput,
          scale: settings.videoUpscaleFactor,
          model: settings.videoUpscaleMethod,
          output_mode: settings.videoOutputMode,
        };
        if (isSeedVR2) {
          payload.variant = settings.seedvr2Variant;
          payload.frame_window = settings.seedvr2FrameWindow;
          payload.resolution_cap = settings.seedvr2ResolutionCap;
          payload.memory_mode = settings.seedvr2MemoryMode;
          payload.temporal_overlap = settings.seedvr2TemporalOverlap;
          payload.motion_protection = settings.seedvr2MotionProtection;
        }
      } else {
        taskType = 'upscale';
        payload = {
          source_path: sourceInput,
          scale: settings.upscaleFactor,
          model: settings.upscaleMethod,
        };
      }

      const data = await startTask(taskType, payload);
      startGeneration(data.generation_id);
    } catch (err) {
      console.error('[PostProcess] Upscale error:', err);
      setLocalError(err.message);
    }
  };

  const handleInterpolate = async () => {
    if (!sourceInput) {
      setLocalError('Please select a source video');
      return;
    }

    setLocalError(null);
    setResult(null);

    try {
      const data = await startTask('interpolate', {
        source_path: sourceInput,
        source_fps: settings.sourceFramerate,
        target_fps: settings.targetFramerate,
        model: settings.interpolationMethod,
      });
      startGeneration(data.generation_id);
    } catch (err) {
      console.error('[PostProcess] Interpolate error:', err);
      setLocalError(err.message);
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
    cancel();
  };

  // Bridge: extract tab-specific result from task completion
  useEffect(() => {
    if (!genResult || genResult.status !== 'complete' || !genResult.result) return;

    const data = genResult.result;
    console.log('[PostProcess] Task complete:', data);

    let newResult;
    if (data.frame_count) {
      // Video result
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
    } else if (data.output_size) {
      // Image upscale result
      newResult = {
        type: 'image',
        url: data.output_url,
        path: data.output_path,
        inputSize: data.input_size,
        outputSize: data.output_size,
        scale: data.scale,
        method: data.method,
      };
    } else {
      // Interpolation result
      newResult = {
        type: 'video',
        url: data.output_url,
        path: data.output_path,
        sourceFps: data.source_fps,
        targetFps: data.target_fps,
        multiplier: data.multiplier,
        method: data.method,
      };
    }

    setResult(newResult);
    saveResultToLastState(newResult);

    window.dispatchEvent(new CustomEvent('fuk-generation-complete', {
      detail: {
        type: activeProcess,
        result: data,
        elapsed: elapsedSeconds,
      }
    }));
  }, [genResult]);

  const canProcess = sourceInput && !generating;
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
                    muted
                    loop
                    className="fuk-preview-media--constrained"
                    preload="metadata"
                  />
                ) : (
                  <ZoomableImage
                    src={buildImageUrl(sourceInput)}
                    alt="Input"
                    className="fuk-preview-media--constrained"
                    loading="lazy"
                  />
                )}
              </div>
            ) : (
              <div className="fuk-placeholder-card">
                <div className="fuk-placeholder">
                  {activeProcess === 'upscale' ? (
                    <Camera className="fuk-placeholder-icon" />
                  ) : (
                    <Film className="fuk-placeholder-icon" />
                  )}
                  <p className="fuk-placeholder-text">
                    {activeProcess === 'upscale' ? 'Select image or video to upscale' : 'Select video to interpolate'}
                  </p>
                </div>
              </div>
            )}
          </div>
          
          {/* Arrow */}
          <div className={`fuk-compare-arrow ${generating ? 'fuk-compare-arrow--active' : ''}`}>
            {generating ? (
              <Loader2 className="fuk-compare-arrow-icon fuk-compare-arrow-icon--spin" />
            ) : (
              <span>→</span>
            )}
            {generating && progress && (
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
              <div className="fuk-media-frame">
                {result.type === 'image' ? (
                  <ZoomableImage
                    src={buildImageUrl(result.url)}
                    alt="Processed"
                    className="fuk-preview-media--constrained"
                    loading="lazy"
                  />
                ) : (
                  <video
                    src={buildImageUrl(result.url)}
                    controls
                    muted
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
                className="fuk-placeholder-card fuk-placeholder-card--ratio"
                style={sourceInput ? inputAspectStyle : undefined}
              >
                <div className="fuk-placeholder">
                  <Enhance className="fuk-placeholder-icon" />
                  <p className="fuk-placeholder-text">
                    {generating ? `generating... ${formatTime(elapsedSeconds)}` : 'Result will appear here'}
                  </p>
                  {generating && isVideo && (
                    <p className="fuk-placeholder-subtext">
                      generating video frame-by-frame
                    </p>
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
              disabled={generating}
              accept={activeProcess === 'interpolate' ? 'videos' : 'all'}
              initialDir={project?.projectState?.lastState?.lastUploadDir}
              onDirectorySelected={(dir) => project?.updateLastState?.({ lastUploadDir: dir })}
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
                <span className="fuk-alert-text">
                  Video: generating frame-by-frame
                </span>
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
                  disabled={generating}
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

                  }}
                />
                <div className="fuk-radio-card-content">
                  <span className="fuk-radio-card-title">
                    <Film className="fuk-icon" />
                    Frame Interpolation
                  </span>
                  <span className="fuk-radio-card-desc">
                    Generate intermediate frames (FILM)
                  </span>
                </div>
              </label>
            </div>
            
            {/* Capability indicators.
                These keys must match PostProcessorManager.get_capabilities().
                They drifted once — the interpolator moved from RIFE to FILM
                and this block kept probing interpolation.ncnn_available, which
                no longer exists, so the badge read as unavailable forever. */}
            {capabilities && (
              <div className="fuk-capability-box">
                <div className="fuk-capability-label">Available backends:</div>
                <div className="fuk-capability-list">
                  <span className={capabilities.upscaling?.ncnn_available ? 'fuk-capability-item--active' : 'fuk-capability-item'}>
                    {capabilities.upscaling?.ncnn_available ? '✓' : '○'} ESRGAN-NCNN
                  </span>
                  <span className={capabilities.video_restoration?.available ? 'fuk-capability-item--active' : 'fuk-capability-item'}>
                    {capabilities.video_restoration?.available ? '✓' : '○'} SeedVR2
                  </span>
                  <span className={capabilities.interpolation?.film_available ? 'fuk-capability-item--active' : 'fuk-capability-item'}>
                    {capabilities.interpolation?.film_available ? '✓' : '○'} FILM
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Upscaling Settings — video */}
          {activeProcess === 'upscale' && isVideo && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Video Upscaling</h3>

              <div className="fuk-form-group-compact">
                <label className="fuk-label">Model</label>
                <select
                  className="fuk-select"
                  value={settings.videoUpscaleMethod}
                  onChange={(e) => updateSettings({ videoUpscaleMethod: e.target.value })}
                  disabled={generating}
                >
                  <option value="seedvr2" disabled={!seedvr2Available}>
                    SeedVR2 — temporal{seedvr2Available ? ' (Recommended)' : ' (unavailable)'}
                  </option>
                  <option value="realesrgan">Real-ESRGAN (per-frame)</option>
                  <option value="lanczos">Lanczos (Fast, No AI)</option>
                </select>
                <p className="fuk-help-text">
                  {isSeedVR2
                    ? 'Restores the sequence as a sequence, so detail stays put between frames'
                    : 'Enhances each frame in isolation — expect detail to crawl between frames'}
                </p>
                {!seedvr2Available && capabilities.video_restoration?.hint && (
                  <p className="fuk-help-text">
                    SeedVR2 unavailable: {(capabilities.video_restoration.missing || []).join(', ')}.
                    {' '}{capabilities.video_restoration.hint}
                  </p>
                )}
              </div>

              <div className="fuk-form-group-compact">
                <label className="fuk-label">Scale Factor</label>
                <div className="fuk-radio-group">
                  {/* SeedVR2 targets an absolute resolution rather than a
                      multiple, and 8x on any real source overshoots every
                      sensible cap. */}
                  {(isSeedVR2 ? [2, 4] : [2, 4, 8]).map(factor => (
                    <label key={factor} className="fuk-radio-option">
                      <input
                        type="radio"
                        className="fuk-radio"
                        checked={settings.videoUpscaleFactor === factor}
                        onChange={() => updateSettings({ videoUpscaleFactor: factor })}
                        disabled={generating}
                      />
                      <span className="fuk-radio-label">{factor}x</span>
                    </label>
                  ))}
                </div>
              </div>

              {isSeedVR2 && (
                <>
                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Quality</label>
                    <select
                      className="fuk-select"
                      value={settings.seedvr2Variant}
                      onChange={(e) => updateSettings({ seedvr2Variant: e.target.value })}
                      disabled={generating}
                    >
                      <option value="seedvr2_7b_fp8">7B fp8 — best quality, slowest</option>
                      <option value="seedvr2_7b_sharp_fp8">7B Sharp fp8 — more detail</option>
                      <option value="seedvr2_7b_q4">7B Q4 — 7B on a tight budget</option>
                      <option value="seedvr2_3b_fp8">3B fp8 — fast, good quality</option>
                      <option value="seedvr2_3b_q8">3B Q8 — fastest, lowest VRAM</option>
                    </select>
                  </div>

                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Frame Window</label>
                    <select
                      className="fuk-select"
                      value={settings.seedvr2FrameWindow}
                      onChange={(e) => updateSettings({ seedvr2FrameWindow: Number(e.target.value) })}
                      disabled={generating}
                    >
                      {/* 4n+1 only — the VAE compresses time 4x plus a key frame */}
                      {[5, 9, 13, 17, 21].map(n => (
                        <option key={n} value={n}>
                          {n} frames{n === 9 || n === 13 ? ' (recommended)' : ''}
                        </option>
                      ))}
                    </select>
                    <p className="fuk-help-text">
                      Frames restored together. 9–13 measures best; both narrower
                      and wider increase stepping, so this is not a "more is
                      better" control. Also the dominant VRAM cost.
                    </p>
                  </div>

                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Batch Blending</label>
                    <select
                      className="fuk-select"
                      value={settings.seedvr2TemporalOverlap}
                      onChange={(e) => updateSettings({ seedvr2TemporalOverlap: Number(e.target.value) })}
                      disabled={generating}
                    >
                      <option value={-1}>Auto — scale to frame window</option>
                      <option value={0}>Off — fastest, visible seams</option>
                      <option value={2}>2 frames</option>
                      <option value={4}>4 frames</option>
                      <option value={6}>6 frames</option>
                      <option value={8}>8 frames</option>
                    </select>
                    <p className="fuk-help-text">
                      Frames shared and crossfaded between batches. Each batch is
                      restored independently, so with this off the picture visibly
                      steps every frame-window frames. More costs proportionally
                      more time.
                    </p>
                  </div>

                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Resolution Cap</label>
                    <select
                      className="fuk-select"
                      value={settings.seedvr2ResolutionCap}
                      onChange={(e) => updateSettings({ seedvr2ResolutionCap: Number(e.target.value) })}
                      disabled={generating}
                    >
                      <option value={1280}>1280px</option>
                      <option value={1920}>1920px (HD)</option>
                      <option value={2560}>2560px</option>
                      <option value={3840}>3840px (4K)</option>
                    </select>
                    <p className="fuk-help-text">Upper bound on the longest output edge</p>
                  </div>

                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Motion Protection</label>
                    <select
                      className="fuk-select"
                      value={settings.seedvr2MotionProtection}
                      onChange={(e) => updateSettings({ seedvr2MotionProtection: Number(e.target.value) })}
                      disabled={generating}
                    >
                      <option value={0}>Off — full restoration</option>
                      <option value={0.4}>0.4 — light</option>
                      <option value={0.7}>0.7 — recommended</option>
                      <option value={1.0}>1.0 — maximum</option>
                    </select>
                    <p className="fuk-help-text">
                      SeedVR2 removes motion blur, and motion blur is what makes
                      24fps read as smooth — so fast-moving subjects come back
                      stepping and ghosting. This blends moving pixels back
                      toward the source to keep their blur; still areas keep the
                      full sharpening. Does nothing on a static shot.
                    </p>
                  </div>

                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Memory Mode</label>
                    <select
                      className="fuk-select"
                      value={settings.seedvr2MemoryMode}
                      onChange={(e) => updateSettings({ seedvr2MemoryMode: e.target.value })}
                      disabled={generating}
                    >
                      <option value="auto">Auto — retry lower on OOM</option>
                      <option value="0">Balanced</option>
                      <option value="1">Narrow window</option>
                      <option value="2">Minimum window + block swap</option>
                      <option value="3">Quantized + everything</option>
                    </select>
                    <p className="fuk-help-text">
                      Auto starts fast and steps down until the clip fits. Pinning
                      a mode reports the out-of-memory error instead of retrying.
                    </p>
                  </div>
                </>
              )}

            </div>
          )}

          {/* Upscaling Settings — stills */}
          {activeProcess === 'upscale' && !isVideo && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Upscaling Settings</h3>

              <div className="fuk-form-group-compact">
                <label className="fuk-label">AI Model</label>
                <select
                  className="fuk-select"
                  value={settings.upscaleMethod}
                  onChange={(e) => updateSettings({ upscaleMethod: e.target.value })}
                  disabled={generating}
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
                        disabled={generating}
                      />
                      <span className="fuk-radio-label">{factor}x</span>
                    </label>
                  ))}
                </div>
                <p className="fuk-help-text--inline">
                  {settings.upscaleFactor}x scale = {settings.upscaleFactor * settings.upscaleFactor}x more pixels
                </p>
              </div>

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
                  disabled={generating}
                >
                  <option value="film">FILM (Recommended)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Source Framerate</label>
                <select
                  className="fuk-select"
                  value={settings.sourceFramerate}
                  onChange={(e) => updateSettings({ sourceFramerate: parseInt(e.target.value) })}
                  disabled={generating}
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
                  disabled={generating}
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
        generating={generating}
        progress={progress}
        elapsedSeconds={elapsedSeconds}
        onGenerate={handleProcess}
        onCancel={cancel}
        canGenerate={canProcess}
        generateLabel={activeProcess === 'upscale' ? (isVideo ? 'Upscale Video' : 'Upscale') : 'Interpolate'}
        generatingLabel={activeProcess === 'upscale' ? (isVideo ? 'Upscaling Video...' : 'Upscaling...') : 'Interpolating...'}
      />
      <GenerationModal
        isOpen={showModal}
        type={activeProcess}
        generating={generating}
        progress={progress}
        elapsedSeconds={elapsedSeconds}
        consoleLog={consoleLog}
        error={genError}
        onCancel={cancel}
        onClose={closeModal}
      />
    </>
  );
}