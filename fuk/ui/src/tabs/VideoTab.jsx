/**
 * Video Generation Tab
 * Wan video generation with I2V, FLF2V support
 * 
 * Features:
 * - Auto-reads dimensions from input image
 * - Scale factor for VRAM management
 * - Manual frame entry with 4n+1 validation
 * - Duration feedback
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Film, CheckCircle, AlertCircle } from '../../src/components/Icons';
import MediaUploader from '../components/MediaUploader.jsx';
import SeedControl from '../components/SeedControl';
import { useGeneration } from '../hooks/useGeneration';
import { useLocalStorage } from '../../src/hooks/useLocalStorage';
import { useSavedSeeds } from '../hooks/useSavedSeeds';
import { startVideoGeneration } from '../../src/utils/api';
import { formatTime } from '../utils/helpers.js';
import { 
  DEFAULT_VIDEO_SETTINGS, 
  buildImageUrl, 
  SEED_MODES, 
  generateRandomSeed 
} from '../../src/utils/constants';
import Footer from '../components/Footer';


// Scale factor presets for VRAM management
const SCALE_FACTORS = [
  { label: '100%', value: 1.0 },
  { label: '75%', value: 0.75 },
  { label: '50%', value: 0.5 },
  { label: '25%', value: 0.25 },
];

// Round to nearest valid 4n+1 frame count
function roundToValid4n1(frames) {
  // Find n such that 4n+1 is closest to frames
  const n = Math.round((frames - 1) / 4);
  return Math.max(5, 4 * n + 1); // Minimum 5 frames
}

// Get duration string from frame count (assuming 24fps output)
function getFrameDuration(frames, fps = 24) {
  const seconds = frames / fps;
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)}ms`;
  }
  return `${seconds.toFixed(2)}s`;
}

export default function VideoTab({ config, activeTab, setActiveTab, project }) {
  const defaults = config?.defaults?.video || config?.defaults || {};
  
  // Build initial defaults by merging config with hardcoded defaults
  const initialDefaults = useMemo(() => ({
    ...DEFAULT_VIDEO_SETTINGS,
    task: 'i2v-A14B', // Default to Wan 2.2
    negative_prompt: defaults.negative_prompt || DEFAULT_VIDEO_SETTINGS.negative_prompt,
    guidance_scale: defaults.guidance_scale ?? DEFAULT_VIDEO_SETTINGS.guidance_scale,
    flow_shift: defaults.flow_shift ?? DEFAULT_VIDEO_SETTINGS.flow_shift,
    lora_multiplier: defaults.lora_multiplier ?? DEFAULT_VIDEO_SETTINGS.lora_multiplier,
    blocks_to_swap: defaults.blocks_to_swap ?? DEFAULT_VIDEO_SETTINGS.blocks_to_swap,
    video_length: defaults.length ?? DEFAULT_VIDEO_SETTINGS.video_length,
    // New fields
    scale_factor: 1.0,
    source_width: null,  // From input image
    source_height: null, // From input image
  }), [defaults]);
  
  // Fallback localStorage for when no project is loaded
  const [localFormData, setLocalFormData] = useLocalStorage('fuk_video_settings', initialDefaults);

  // Use project state if available, otherwise localStorage
  const formData = useMemo(() => {
    if (project?.projectState?.tabs?.video) {
      return { ...initialDefaults, ...project.projectState.tabs.video };
    }
    return { ...initialDefaults, ...localFormData };
  }, [project?.projectState?.tabs?.video, localFormData, initialDefaults]);

  // Ref to track latest formData for setFormData callback
  const formDataRef = useRef(formData);
  useEffect(() => {
    formDataRef.current = formData;
  }, [formData]);

  // Update function that writes to project or localStorage
  const setFormData = useCallback((updater) => {
    const currentData = formDataRef.current;
    const newData = typeof updater === 'function' 
      ? updater(currentData) 
      : updater;
    
    if (project?.isProjectLoaded && project?.updateTabState) {
      project.updateTabState('video', newData);
    } else {
      setLocalFormData(newData);
    }
  }, [project?.isProjectLoaded, project?.updateTabState, setLocalFormData]);

  // Auto-resize textarea handler
  const handleTextareaResize = useCallback((e) => {
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
  }, []);

  // Frame input state (for controlled input before validation)
  const [frameInput, setFrameInput] = useState(String(formData.video_length || 81));
  
  // Sync frame input when formData changes externally
  useEffect(() => {
    setFrameInput(String(formData.video_length || 81));
  }, [formData.video_length]);

  // Generation state
  const {
    generating,
    progress,
    result,
    error,
    elapsedSeconds,
    startGeneration,
    cancel,
    reset: resetGeneration,
  } = useGeneration();

  // Saved seeds hook
  const savedSeedsHook = useSavedSeeds();

  // Reset generation result when switching project files
  useEffect(() => {
    if (project?.currentFilename) {
      resetGeneration();
    }
  }, [project?.currentFilename, resetGeneration]);

  // Get saved preview from project or use generation result
  const previewVideo = useMemo(() => {
    if (result?.outputs?.mp4) {
      return result.outputs.mp4;
    }
    if (project?.projectState?.lastState?.lastVideoPreview) {
      return project.projectState.lastState.lastVideoPreview;
    }
    return null;
  }, [result, project?.projectState?.lastState?.lastVideoPreview]);

  // Update last state when generation completes
  useEffect(() => {
    if (result?.outputs?.mp4) {
      if (project?.updateLastState) {
        project.updateLastState({
          lastVideoPreview: result.outputs.mp4,
          activeTab: 'video',
        });
      }
      
      if (result.seed_used !== undefined && result.seed_used !== null) {
        setFormData(prev => ({
          ...prev,
          lastUsedSeed: result.seed_used,
        }));
      }
    }
  }, [result, project?.updateLastState, setFormData]);

  // Load image dimensions when image_path changes
  useEffect(() => {
    if (formData.image_path) {
      const img = new Image();
      img.onload = () => {
        // Round to nearest 64 for model compatibility
        const width = Math.round(img.width / 64) * 64;
        const height = Math.round(img.height / 64) * 64;
        
        setFormData(prev => ({
          ...prev,
          source_width: width,
          source_height: height,
          // Update output dimensions with scale factor
          width: Math.round(width * prev.scale_factor / 64) * 64,
          height: Math.round(height * prev.scale_factor / 64) * 64,
        }));
      };
      img.onerror = () => {
        console.warn('Could not load image for dimension detection');
      };
      img.src = buildImageUrl(formData.image_path);
    }
  }, [formData.image_path, setFormData]);

  // Update output dimensions when scale factor changes
  const handleScaleChange = (newScale) => {
    const sourceW = formData.source_width || 832;
    const sourceH = formData.source_height || 480;
    
    setFormData(prev => ({
      ...prev,
      scale_factor: newScale,
      width: Math.round(sourceW * newScale / 64) * 64 || 832,
      height: Math.round(sourceH * newScale / 64) * 64 || 480,
    }));
  };

  // Handle frame input change with validation
  const handleFrameInputChange = (value) => {
    setFrameInput(value);
  };

  // Validate and round frames on blur
  const handleFrameInputBlur = () => {
    const parsed = parseInt(frameInput) || 81;
    const valid = roundToValid4n1(parsed);
    setFrameInput(String(valid));
    setFormData(prev => ({ ...prev, video_length: valid }));
  };

  // Determine the seed to use based on mode
  const getEffectiveSeed = useCallback(() => {
    const mode = formData.seedMode || SEED_MODES.RANDOM;
    
    switch (mode) {
      case SEED_MODES.FIXED:
        return formData.seed;
      case SEED_MODES.RANDOM:
        return generateRandomSeed();
      case SEED_MODES.INCREMENT:
        if (formData.lastUsedSeed !== null) {
          return formData.lastUsedSeed + 1;
        }
        return formData.seed !== null ? formData.seed : generateRandomSeed();
      default:
        return formData.seed;
    }
  }, [formData.seedMode, formData.seed, formData.lastUsedSeed]);

  const handleGenerate = async () => {
    const effectiveSeed = getEffectiveSeed();
    
    const payload = {
      ...formData,
      seed: effectiveSeed,
      image_path: formData.image_path ? formData.image_path.replace(/^\/outputs\//, '').replace(/^\//, '') : null,
      end_image_path: formData.end_image_path ? formData.end_image_path.replace(/^\/outputs\//, '').replace(/^\//, '') : null,
    };
    
    console.log('Video generation payload:', payload);
    console.log(`Seed mode: ${formData.seedMode}, using seed: ${effectiveSeed}`);
    
    setFormData(prev => ({
      ...prev,
      lastUsedSeed: effectiveSeed,
      seed: prev.seedMode === SEED_MODES.INCREMENT ? effectiveSeed : prev.seed,
    }));
    
    try {
      const data = await startVideoGeneration(payload);
      startGeneration(data.generation_id);
    } catch (err) {
      console.error('Video generation failed:', err);
      alert('Failed to start video generation');
    }
  };

  const handleStartImageChange = (paths) => {
    setFormData(prev => ({ ...prev, image_path: paths[0] || null }));
  };

  const handleEndImageChange = (paths) => {
    setFormData(prev => ({ ...prev, end_image_path: paths[0] || null }));
  };

  // Check if task requires images
  const requiresStartImage = formData.task.includes('i2v') || formData.task.includes('flf2v');
  const requiresEndImage = formData.task.includes('flf2v');
  
  // Calculate whether current frame input is valid
  const currentFrameValid = (parseInt(frameInput) - 1) % 4 === 0;
  const validFrameCount = roundToValid4n1(parseInt(frameInput) || 81);
  
  // Calculate CSS variable for dynamic aspect ratio
  const aspectRatioStyle = { '--preview-aspect': `${formData.width || 832} / ${formData.height || 480}` };

  return (
    <>
      {/* Preview Area */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-single">
          {previewVideo ? (
            <div className="fuk-preview-container">
              <video
                src={buildImageUrl(previewVideo)}
                controls
                loop
                autoPlay
                className="fuk-preview-media"
              />
              <div className="fuk-preview-info">
                <div className="fuk-preview-info-row">
                  <span>{formData.width}×{formData.height}</span>
                  <span>•</span>
                  <span>{formData.video_length} frames ({getFrameDuration(formData.video_length)})</span>
                  <span>•</span>
                  <span>{formData.steps} steps</span>
                  {result?.outputs?.mp4 && (
                    <>
                      <span>•</span>
                      <span className="fuk-status-complete">
                        <CheckCircle className="fuk-icon--sm" />
                        {formatTime(elapsedSeconds)}
                      </span>
                    </>
                  )}
                  {result?.seed_used && (
                    <>
                      <span>•</span>
                      <span className="fuk-seed-display">Seed: {result.seed_used}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div 
              className="fuk-placeholder-card fuk-placeholder-card--60 fuk-placeholder-card--dynamic"
              style={aspectRatioStyle}
            >
              <div className="fuk-placeholder">
                <Film className="fuk-placeholder-icon" />
                <p className="fuk-placeholder-text">
                  {formData.width || '---'} × {formData.height || '---'} × {formData.video_length} frames
                  <br />
                  <span className="fuk-placeholder-subtext">
                    {formData.source_width ? `Source: ${formData.source_width}×${formData.source_height}` : 'Upload image to set dimensions'}
                  </span>
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Input Images Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Input Images</h3>
            
            {requiresStartImage ? (
              <>
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">
                    Start Image <span className="fuk-label-required">(Required)</span>
                  </label>
                  <MediaUploader
                    images={formData.image_path ? [formData.image_path] : []}
                    onImagesChange={handleStartImageChange}
                    disabled={generating}
                  />
                  {formData.source_width && (
                    <p className="fuk-help-text">
                      Detected: {formData.source_width} × {formData.source_height}
                    </p>
                  )}
                </div>
                
                {requiresEndImage && (
                  <div className="fuk-form-group-compact fuk-mt-4">
                    <label className="fuk-label">
                      End Image <span className="fuk-label-required">(Required for FLF2V)</span>
                    </label>
                    <MediaUploader
                      images={formData.end_image_path ? [formData.end_image_path] : []}
                      onImagesChange={handleEndImageChange}
                      disabled={generating}
                    />
                  </div>
                )}
              </>
            ) : (
              <div className="fuk-empty-state">
                <Film className="fuk-empty-state-icon" />
                <p className="fuk-empty-state-text">
                  Select an I2V or FLF2V model<br />to enable image inputs
                </p>
              </div>
            )}
          </div>
          
          {/* Video Settings Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Video Settings</h3>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Model</label>
              <select
                className="fuk-select"
                value={formData.task}
                onChange={(e) => setFormData({...formData, task: e.target.value})}
              >
                <option value="i2v-A14B">Wan 2.2 I2V (Recommended)</option>
                <option value="i2v-14B">Wan 2.1 I2V</option>
                <option value="flf2v-14B">Wan 2.1 FLF2V (First+Last Frame)</option>
                <option value="i2v-14B-FC">Wan 2.1 I2V Fun Control</option>
              </select>
            </div>
            
            {/* Resolution from input + scale factor */}
            <div className="fuk-form-group-compact">
              <label className="fuk-label">
                Resolution 
                {formData.source_width && (
                  <span className="fuk-label-description">
                    (from {formData.source_width}×{formData.source_height})
                  </span>
                )}
              </label>
              <div className="fuk-input-inline">
                <select
                  className="fuk-select"
                  value={formData.scale_factor}
                  onChange={(e) => handleScaleChange(parseFloat(e.target.value))}
                >
                  {SCALE_FACTORS.map(sf => (
                    <option key={sf.value} value={sf.value}>{sf.label}</option>
                  ))}
                </select>
                <span className="fuk-input-result">
                  → {formData.width || '---'} × {formData.height || '---'}
                </span>
              </div>
              {!formData.source_width && (
                <p className="fuk-help-text fuk-help-text--warning">
                  <AlertCircle className="fuk-icon--sm" />
                  Upload a start image to auto-detect dimensions
                </p>
              )}
            </div>
            
            {/* Frame count with duration feedback */}
            <div className="fuk-form-group-compact">
              <label className="fuk-label">
                Frames
                <span className="fuk-label-description">(must be 4n+1)</span>
              </label>
              <div className="fuk-input-inline">
                <input
                  type="number"
                  className={`fuk-input ${!currentFrameValid ? 'fuk-input--warning' : ''}`}
                  value={frameInput}
                  onChange={(e) => handleFrameInputChange(e.target.value)}
                  onBlur={handleFrameInputBlur}
                  min={5}
                  max={241}
                  step={4}
                />
                <span className="fuk-input-result">
                  ≈ {getFrameDuration(formData.video_length)} @ 24fps
                </span>
              </div>
              {!currentFrameValid && (
                <p className="fuk-help-text fuk-help-text--info">
                  Will round to {validFrameCount} frames ({getFrameDuration(validFrameCount)})
                </p>
              )}
            </div>
          </div>

          {/* Generation Parameters Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Generation</h3>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Steps</label>
              <div className="fuk-radio-group">
                <label className="fuk-radio-option">
                  <input
                    type="radio"
                    className="fuk-radio"
                    checked={formData.stepsMode === 'preset' && formData.steps === 20}
                    onChange={() => setFormData({...formData, stepsMode: 'preset', steps: 20})}
                  />
                  <span className="fuk-radio-label">20</span>
                </label>
                
                <label className="fuk-radio-option">
                  <input
                    type="radio"
                    className="fuk-radio"
                    checked={formData.stepsMode === 'preset' && formData.steps === 40}
                    onChange={() => setFormData({...formData, stepsMode: 'preset', steps: 40})}
                  />
                  <span className="fuk-radio-label">40</span>
                </label>
                
                <label className="fuk-radio-option">
                  <input
                    type="radio"
                    className="fuk-radio"
                    checked={formData.stepsMode === 'custom'}
                    onChange={() => setFormData({...formData, stepsMode: 'custom'})}
                  />
                  <span className="fuk-radio-label">Custom:</span>
                </label>
                
                <input
                  type="number"
                  className="fuk-input fuk-input--w-80"
                  value={formData.steps}
                  onChange={(e) => setFormData({...formData, stepsMode: 'custom', steps: parseInt(e.target.value)})}
                  disabled={formData.stepsMode !== 'custom'}
                  min={1}
                  max={100}
                />
              </div>
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Guidance Scale</label>
              <input
                type="number"
                className="fuk-input"
                value={formData.guidance_scale}
                onChange={(e) => setFormData({...formData, guidance_scale: parseFloat(e.target.value)})}
                step={0.5}
                min={1}
                max={15}
              />
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Flow Shift</label>
              <input
                type="number"
                className="fuk-input"
                value={formData.flow_shift}
                onChange={(e) => setFormData({...formData, flow_shift: parseFloat(e.target.value)})}
                step={0.5}
                min={1}
                max={10}
              />
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">
                Blocks to Swap <span className="fuk-label-description">(VRAM)</span>
              </label>
              <input
                type="number"
                className="fuk-input"
                value={formData.blocks_to_swap}
                onChange={(e) => setFormData({...formData, blocks_to_swap: parseInt(e.target.value)})}
                min={0}
                max={39}
              />
            </div>
          </div>

          {/* Seed Control Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Seed Control</h3>
            <SeedControl
              seed={formData.seed}
              seedMode={formData.seedMode || SEED_MODES.RANDOM}
              lastUsedSeed={formData.lastUsedSeed}
              model={formData.task}
              prompt={formData.prompt}
              savedSeeds={savedSeedsHook.getSeedsForModel(formData.task)}
              isSeedSaved={savedSeedsHook.isSeedSaved}
              onSeedChange={(seed) => setFormData({...formData, seed})}
              onSeedModeChange={(seedMode) => setFormData({...formData, seedMode})}
              onSaveSeed={(seed, note) => savedSeedsHook.saveSeed(formData.task, seed, note)}
              onRemoveSeed={savedSeedsHook.removeSeed}
              onSelectSavedSeed={(seed) => setFormData({...formData, seed})}
              disabled={generating}
            />
          </div>

          {/* Prompt Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Prompt</h3>
            
            <div className="fuk-form-group-compact">
              <textarea
                className="fuk-textarea"
                value={formData.prompt}
                onChange={(e) => {
                  setFormData({...formData, prompt: e.target.value});
                  handleTextareaResize(e);
                }}
                onInput={handleTextareaResize}
                placeholder="A cinematic video of..."
                rows={4}
                style={{ resize: 'none', overflow: 'hidden' }}
              />
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Negative Prompt</label>
              <textarea
                className="fuk-textarea"
                value={formData.negative_prompt}
                onChange={(e) => {
                  setFormData({...formData, negative_prompt: e.target.value});
                  handleTextareaResize(e);
                }}
                onInput={handleTextareaResize}
                placeholder="blurry, low quality..."
                rows={4}
                style={{ resize: 'none', overflow: 'hidden' }}
              />
            </div>
          </div>
        </div>
      </div>
      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={generating}
        progress={progress}
        elapsedSeconds={elapsedSeconds}
        onGenerate={handleGenerate}
        onCancel={cancel}
        canGenerate={!!formData.prompt && (!requiresStartImage || !!formData.image_path) && (!requiresEndImage || !!formData.end_image_path)}
        generateLabel="Generate Video"
        generatingLabel="Generating..."
      />
    </>
  );
}