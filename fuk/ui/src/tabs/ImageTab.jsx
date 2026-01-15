/**
 * Image Generation Tab
 * Main UI for Qwen image generation
 * Refactored: All inline styles moved to CSS classes
 */

import { useEffect, useCallback, useMemo, useRef } from 'react';
import { Camera, CheckCircle, X, Pipeline } from '../../src/components/Icons';
import MediaUploader from '../components/MediaUploader';
import SeedControl from '../components/SeedControl';
import { useGeneration } from '../hooks/useGeneration';
import { useLocalStorage } from '../../src/hooks/useLocalStorage';
import { useSavedSeeds } from '../hooks/useSavedSeeds';
import { startImageGeneration } from '../../src/utils/api';
import { calculateDimensions, formatTime } from '../utils/helpers.js';
import { 
  ASPECT_RATIOS, 
  DEFAULT_IMAGE_SETTINGS, 
  buildImageUrl, 
  SEED_MODES, 
  generateRandomSeed 
} from '../../src/utils/constants';
import Footer from '../components/Footer';

export default function ImageTab({ config, activeTab, setActiveTab, project }) {
  const defaults = config?.defaults || {};
  
  // Build initial defaults by merging config with hardcoded defaults
  const initialDefaults = useMemo(() => ({
    ...DEFAULT_IMAGE_SETTINGS,
    negative_prompt: defaults.negative_prompt || DEFAULT_IMAGE_SETTINGS.negative_prompt,
    guidance_scale: defaults.guidance_scale ?? DEFAULT_IMAGE_SETTINGS.guidance_scale,
    flow_shift: defaults.flow_shift ?? DEFAULT_IMAGE_SETTINGS.flow_shift,
    lora_multiplier: defaults.lora_multiplier ?? DEFAULT_IMAGE_SETTINGS.lora_multiplier,
    blocks_to_swap: defaults.blocks_to_swap ?? DEFAULT_IMAGE_SETTINGS.blocks_to_swap,
  }), [defaults]);
  
  // Fallback localStorage for when no project is loaded
  const [localFormData, setLocalFormData] = useLocalStorage('fuk_image_settings', initialDefaults);

  // Use project state if available, otherwise localStorage
  const formData = useMemo(() => {
    if (project?.projectState?.tabs?.image) {
      return { ...initialDefaults, ...project.projectState.tabs.image };
    }
    return { ...initialDefaults, ...localFormData };
  }, [project?.projectState?.tabs?.image, localFormData, initialDefaults]);

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
      project.updateTabState('image', newData);
    } else {
      setLocalFormData(newData);
    }
  }, [project?.isProjectLoaded, project?.updateTabState, setLocalFormData]);

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
  const previewImage = useMemo(() => {
    if (result?.outputs?.png) {
      return result.outputs.png;
    }
    if (project?.projectState?.lastState?.lastImagePreview) {
      return project.projectState.lastState.lastImagePreview;
    }
    return null;
  }, [result, project?.projectState?.lastState?.lastImagePreview]);

  // Get current dimensions
  const getCurrentDimensions = () => {
    return calculateDimensions(formData.aspectRatio, formData.width || 1024);
  };

  // Update height when aspect ratio changes
  const prevDimsRef = useRef({ aspectRatio: null, width: null });
  useEffect(() => {
    const { aspectRatio, width } = formData;
    const prev = prevDimsRef.current;
    
    if (prev.aspectRatio === aspectRatio && prev.width === width) {
      return;
    }
    
    prevDimsRef.current = { aspectRatio, width };
    
    const dims = calculateDimensions(aspectRatio, width || 1024);
    if (dims.height !== formData.height) {
      setFormData(prev => ({ ...prev, height: dims.height }));
    }
  }, [formData.aspectRatio, formData.width, formData.height, setFormData]);

  // Update last state and seed when generation completes
  useEffect(() => {
    if (result?.outputs?.png) {
      if (project?.updateLastState) {
        project.updateLastState({
          lastImagePreview: result.outputs.png,
          activeTab: 'image',
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
    const dims = calculateDimensions(formData.aspectRatio, formData.width);
    const effectiveSeed = getEffectiveSeed();
    
    const payload = {
      ...formData,
      width: dims.width,
      height: dims.height,
      steps: formData.stepsMode === 'custom' ? formData.steps : parseInt(formData.steps),
      seed: effectiveSeed,
      control_image_paths: formData.control_image_paths,
      control_image_path: formData.control_image_paths.length > 0 
        ? formData.control_image_paths[0] 
        : null
    };
    
    console.log('Generation payload:', payload);
    console.log(`Seed mode: ${formData.seedMode}, using seed: ${effectiveSeed}`);
    
    setFormData(prev => ({
      ...prev,
      lastUsedSeed: effectiveSeed,
      seed: prev.seedMode === SEED_MODES.INCREMENT ? effectiveSeed : prev.seed,
    }));
    
    try {
      const data = await startImageGeneration(payload);
      startGeneration(data.generation_id);
    } catch (err) {
      console.error('Generation failed:', err);
      alert('Failed to start generation');
    }
  };

  const handleImagesChange = (paths) => {
    setFormData(prev => ({ ...prev, control_image_paths: paths }));
  };

  // Calculate CSS variable for dynamic aspect ratio
  const dims = getCurrentDimensions();
  const aspectRatioStyle = { '--preview-aspect': `${dims.width} / ${dims.height}` };

  return (
    <>
      {/* Preview Area */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-single">
          {previewImage ? (
            <div className="fuk-preview-container">
              <img
                src={buildImageUrl(previewImage)}
                alt="Generated"
                className="fuk-preview-media"
              />
              <div className="fuk-preview-info">
                <div className="fuk-preview-info-row">
                  <span>{dims.width}x{dims.height}</span>
                  {result?.outputs?.png && (
                    <>
                      <span>Complete</span>
                      <span className="fuk-status-complete">
                        <CheckCircle className="fuk-icon--sm" />
                        {formatTime(elapsedSeconds)}
                      </span>
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
                <Camera className="fuk-placeholder-icon" />
                <p className="fuk-placeholder-text">
                  {dims.width} x {dims.height}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Preprocessed Image Card */}
          {project?.projectState?.lastState?.lastPreprocessedImage && formData.model === 'qwen_image_2509_edit' && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-card-title--highlight fuk-mb-3">
                Last Preprocessed Image
              </h3>
              
              <div className="fuk-preprocess-preview">
                <div className="fuk-preprocess-preview-thumb">
                  <img
                    src={project.projectState.lastState.lastPreprocessedImage}
                    alt="Preprocessed"
                    className="fuk-preprocess-preview-image"
                  />
                </div>
                
                <div className="fuk-preprocess-preview-info">
                  <div className="fuk-preprocess-preview-method">
                    <strong>Method:</strong> {project.projectState.lastState.lastPreprocessedMethod || 'unknown'}
                  </div>
                  
                  <button
                    className="fuk-btn fuk-btn-secondary"
                    onClick={() => {
                      const preprocessedUrl = project.projectState.lastState.lastPreprocessedImage;
                      const path = preprocessedUrl.replace(/^\/outputs\//, '').replace(/^\//, '');
                      
                      if (!formData.control_image_paths.includes(path)) {
                        setFormData(prev => ({
                          ...prev,
                          control_image_paths: [...prev.control_image_paths, path]
                        }));
                      }
                    }}
                    disabled={generating}
                  >
                    <Pipeline className="fuk-icon--md" />
                    Use as Control Image
                  </button>
                  
                  <p className="fuk-help-text">From Pre-Processors tab</p>
                </div>
              </div>
            </div>
          )}

          {/* Image Tools Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Image Tools</h3>
            
            {formData.model === 'qwen_image_2509_edit' ? (
              <>
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">
                    Control Images ({formData.control_image_paths.length})
                  </label>
                  <MediaUploader
                    images={formData.control_image_paths}
                    onImagesChange={handleImagesChange}
                    disabled={generating}
                  />
                </div>
                
                <div className="fuk-form-group-compact fuk-mt-4">
                  <label className="fuk-label">Edit Strength</label>
                  <div className="fuk-input-inline">
                    <input
                      type="range"
                      className="fuk-input fuk-input--flex-2"
                      value={formData.edit_strength || 0.7}
                      onChange={(e) => setFormData({...formData, edit_strength: parseFloat(e.target.value)})}
                      min={0}
                      max={1}
                      step={0.05}
                    />
                    <input
                      type="number"
                      className="fuk-input fuk-input--w-80"
                      value={formData.edit_strength || 0.7}
                      onChange={(e) => setFormData({...formData, edit_strength: parseFloat(e.target.value)})}
                      step={0.05}
                      min={0}
                      max={1}
                    />
                  </div>
                </div>
                
                <p className="fuk-help-text">
                  Upload one or more images to guide the generation.
                </p>
              </>
            ) : (
              <div className="fuk-empty-state">
                <Camera className="fuk-empty-state-icon" />
                <p className="fuk-empty-state-text">
                  Select "Qwen Edit 2509" model<br />to enable image editing tools
                </p>
              </div>
            )}
          </div>

          {/* Model & LoRA Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Model & Style</h3>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Model</label>
              <select
                className="fuk-select"
                value={formData.model}
                onChange={(e) => setFormData({...formData, model: e.target.value})}
              >
                <option value="qwen_image">Qwen Image</option>
                <option value="qwen_image_2509_edit">Qwen Edit 2509</option>
              </select>
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">LoRA</label>
              <select
                className="fuk-select"
                value={formData.lora || ''}
                onChange={(e) => setFormData({...formData, lora: e.target.value || null})}
              >
                <option value="">None</option>
                {config?.models?.loras?.map(lora => (
                  <option key={lora} value={lora}>
                    {lora.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </option>
                ))}
              </select>
            </div>
            
            {formData.lora && (
              <div className="fuk-form-group-compact">
                <label className="fuk-label">LoRA Strength</label>
                <div className="fuk-input-inline">
                  <input
                    type="range"
                    className="fuk-input fuk-input--flex-2"
                    value={formData.lora_multiplier}
                    onChange={(e) => setFormData({...formData, lora_multiplier: parseFloat(e.target.value)})}
                    min={0}
                    max={2}
                    step={0.1}
                  />
                  <input
                    type="number"
                    className="fuk-input fuk-input--w-80"
                    value={formData.lora_multiplier}
                    onChange={(e) => setFormData({...formData, lora_multiplier: parseFloat(e.target.value)})}
                    step={0.1}
                    min={0}
                    max={2}
                  />
                </div>
              </div>
            )}
          </div>
          
          {/* Generation Settings Card */}
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
                step={0.1}
                min={1}
                max={10}
              />
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Flow Shift</label>
              <input
                type="number"
                className="fuk-input"
                value={formData.flow_shift}
                onChange={(e) => setFormData({...formData, flow_shift: parseFloat(e.target.value)})}
                step={0.1}
                min={1}
                max={5}
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
                max={30}
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
              model={formData.model}
              prompt={formData.prompt}
              savedSeeds={savedSeedsHook.getSeedsForModel(formData.model)}
              isSeedSaved={savedSeedsHook.isSeedSaved}
              onSeedChange={(seed) => setFormData({...formData, seed})}
              onSeedModeChange={(seedMode) => setFormData({...formData, seedMode})}
              onSaveSeed={(seed, note, prompt) => savedSeedsHook.saveSeed(formData.model, seed, note, prompt)}
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
                onChange={(e) => setFormData({...formData, prompt: e.target.value})}
                placeholder="A cinematic shot of..."
                rows={3}
              />
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Negative Prompt</label>
              <textarea
                className="fuk-textarea"
                value={formData.negative_prompt}
                onChange={(e) => setFormData({...formData, negative_prompt: e.target.value})}
                placeholder="blurry, low quality..."
                rows={2}
              />
            </div>
          </div>

          {/* Dimensions Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Dimensions</h3>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Aspect Ratio</label>
              <select
                className="fuk-select"
                value={formData.aspectRatio}
                onChange={(e) => setFormData({...formData, aspectRatio: e.target.value})}
              >
                {ASPECT_RATIOS.map(ar => (
                  <option key={ar.value} value={ar.value}>{ar.label}</option>
                ))}
              </select>
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Width (Height auto-calculated)</label>
              <div className="fuk-input-inline">
                <input
                  type="number"
                  className="fuk-input"
                  value={formData.width || ''}
                  onChange={(e) => {
                    const val = e.target.value === '' ? '' : parseInt(e.target.value);
                    const dims = calculateDimensions(formData.aspectRatio, val || 1024);
                    setFormData({...formData, width: val, height: dims.height});
                  }}
                  onBlur={(e) => {
                    const val = parseInt(e.target.value) || 1024;
                    const clamped = Math.max(512, Math.min(2048, val));
                    const dims = calculateDimensions(formData.aspectRatio, clamped);
                    setFormData({...formData, width: clamped, height: dims.height});
                  }}
                  step={64}
                  min={512}
                  max={2048}
                  placeholder="1024"
                />
                <span className="fuk-input-suffix">
                  = {dims.width}x{dims.height}
                </span>
              </div>
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
        canGenerate={!!formData.prompt}
        generateLabel="Generate Image"
        generatingLabel="Generating..."
      />
    </>
  );
}