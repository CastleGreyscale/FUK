/**
 * Image Generation Tab
 * Main UI for Qwen image generation
 * 
 * Defaults flow: config.defaults.image -> project state overwrites
 */

import { useEffect, useCallback, useMemo, useRef } from 'react';
import { Camera, CheckCircle, X, Pipeline, FolderOpen } from '../../src/components/Icons';
import MediaUploader from '../components/MediaUploader';
import ZoomableImage from '../components/ZoomableImage';
import SeedControl from '../components/SeedControl';
import GenerationModal from '../components/GenerationModal';
import { useGeneration } from '../hooks/useGeneration';
import { useLocalStorage } from '../../src/hooks/useLocalStorage';
import { useSavedSeeds } from '../hooks/useSavedSeeds';
import { startImageGeneration } from '../../src/utils/api';
import { calculateDimensions, formatTime } from '../utils/helpers.js';
import { 
  buildImageUrl, 
  SEED_MODES, 
  generateRandomSeed 
} from '../../src/utils/constants';
import Footer from '../components/Footer';

export default function ImageTab({ config, activeTab, setActiveTab, project }) {
  // Get defaults from backend config
  const imageDefaults = config?.defaults?.image || {};
  const aspectRatios = config?.defaults?.aspect_ratios || [];
  
  // Model metadata from config
  const imageModels = config?.models?.image_models || [];
  
  // Helper: find selected model entry and check supports
  const getSelectedModel = (modelKey) => {
    return imageModels.find(m => m.key === modelKey || m.aliases?.includes(modelKey));
  };
  const modelSupports = (modelKey, feature) => {
    const model = getSelectedModel(modelKey);
    return model?.supports?.includes(feature) || false;
  };
  
  // Resolve a model key or alias to a canonical key (for dropdown matching)
  const resolveModelKey = (modelKey) => {
    if (!modelKey || !imageModels.length) return modelKey;
    if (imageModels.find(m => m.key === modelKey)) return modelKey;
    const aliased = imageModels.find(m => m.aliases?.includes(modelKey));
    return aliased ? aliased.key : modelKey;
  };
  
  // Initial defaults come entirely from backend config
  // These are the values used when no project is loaded and no localStorage exists
  const initialDefaults = useMemo(() => ({
    prompt: imageDefaults.prompt ?? '',
    negative_prompt: imageDefaults.negative_prompt ?? '',
    model: imageDefaults.model ?? 'qwen_image',
    width: imageDefaults.width ?? 1280,
    aspectRatio: imageDefaults.aspectRatio ?? '1.78:1',
    steps: imageDefaults.steps ?? 20,
    stepsMode: imageDefaults.stepsMode ?? 'preset',
    guidance_scale: imageDefaults.guidance_scale ?? 5,
    lora: imageDefaults.lora ?? null,
    lora_multiplier: imageDefaults.lora_multiplier ?? 1.0,
    seed: imageDefaults.seed ?? null,
    seedMode: imageDefaults.seedMode ?? SEED_MODES.RANDOM,
    lastUsedSeed: imageDefaults.lastUsedSeed ?? null,
    output_format: imageDefaults.output_format ?? 'png',
    edit_strength: imageDefaults.edit_strength ?? 0.85,
    exponential_shift_mu: imageDefaults.exponential_shift_mu ?? null,
    control_image_paths: imageDefaults.control_image_paths ?? [],
    eligen_source: imageDefaults.eligen_source ?? '',
    vram_preset: config?.models?.vram_preset_default ?? 'low',
  }), [imageDefaults]);
  
  // Fallback localStorage for when no project is loaded
  const [localFormData, setLocalFormData] = useLocalStorage('fuk_image_settings', initialDefaults);

  // Use project state if available, otherwise localStorage
  // Order: initialDefaults <- localStorage/projectState (overwrites)
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

  // Resolve alias model values to canonical keys when models config loads
  useEffect(() => {
    if (imageModels.length > 0 && formData.model) {
      const resolved = resolveModelKey(formData.model);
      if (resolved !== formData.model) {
        setFormData(prev => ({ ...prev, model: resolved }));
      }
    }
  }, [imageModels.length]);

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

  // Auto-resize textarea handler
  const handleTextareaResize = useCallback((e) => {
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
  }, []);

  // Generation state
  const {
    generating,
    progress,
    result,
    error,
    elapsedSeconds,
    consoleLog,
    showModal,
    startGeneration,
    cancel,
    closeModal,
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

  // Get current dimensions (pass aspectRatios from config)
  const getCurrentDimensions = useCallback(() => {
    return calculateDimensions(formData.aspectRatio, formData.width || 1024, aspectRatios);
  }, [formData.aspectRatio, formData.width, aspectRatios]);

  // Update height when aspect ratio changes
  const prevDimsRef = useRef({ aspectRatio: null, width: null });
  useEffect(() => {
    const { aspectRatio, width } = formData;
    const prev = prevDimsRef.current;
    
    if (prev.aspectRatio === aspectRatio && prev.width === width) {
      return;
    }
    
    prevDimsRef.current = { aspectRatio, width };
    
    const dims = calculateDimensions(aspectRatio, width || 1024, aspectRatios);
    if (dims.height !== formData.height) {
      setFormData(prev => ({ ...prev, height: dims.height }));
    }
  }, [formData.aspectRatio, formData.width, formData.height, aspectRatios, setFormData]);

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
    const dims = calculateDimensions(formData.aspectRatio, formData.width, aspectRatios);
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
        : null,
      // Pass denoising_strength only when control images exist
      denoising_strength: formData.control_image_paths.length > 0 
        ? formData.edit_strength 
        : null,
      // Pass exponential_shift_mu (null = auto-calculate)
      exponential_shift_mu: formData.exponential_shift_mu,
      eligen_source: formData.eligen_source || null,
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
              <ZoomableImage
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
              className="fuk-placeholder-card fuk-placeholder-card--ratio"
              style={aspectRatioStyle}
            >
              <div className="fuk-placeholder">
                <Camera className="fuk-placeholder-icon" />
                <p className="fuk-placeholder-text">
                  {dims.width} × {dims.height}
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
          {project?.projectState?.lastState?.lastPreprocessedImage && modelSupports(formData.model, 'edit_image') && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-card-title--highlight fuk-mb-3">
                Last Preprocessed Image
              </h3>
              
              <div className="fuk-preprocess-preview">
                <div className="fuk-preprocess-preview-thumb">
                  <ZoomableImage
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
            
            {/* Control Images — for edit and control-union models */}
            {(modelSupports(formData.model, 'edit_image') || modelSupports(formData.model, 'context_image')) ? (
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
                
                {formData.control_image_paths.length > 0 && (
                  <div className="fuk-form-group-compact fuk-mt-4">
                    <label className="fuk-label">Edit Strength</label>
                    <div className="fuk-input-inline">
                      <input
                        type="range"
                        className="fuk-input fuk-input--flex-2"
                        value={formData.edit_strength || 0.85}
                        onChange={(e) => setFormData({...formData, edit_strength: parseFloat(e.target.value)})}
                        min={0}
                        max={1}
                        step={0.05}
                      />
                      <input
                        type="number"
                        className="fuk-input fuk-input--w-80"
                        value={formData.edit_strength || 0.85}
                        onChange={(e) => setFormData({...formData, edit_strength: parseFloat(e.target.value)})}
                        step={0.05}
                        min={0}
                        max={1}
                      />
                    </div>
                    <p className="fuk-help-text fuk-mt-1">
                      Controls how much the control images influence generation. 
                      1.0 = full strength, 0.0 = minimal influence.
                    </p>
                  </div>
                )}
                
                <p className="fuk-help-text">
                  Upload one or more images to guide the generation.
                </p>
              </>

            /* EliGen — entity composition masks */
            ) : modelSupports(formData.model, 'eligen') ? (
              <>
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">Entity Masks</label>
                  <div className="fuk-input-inline">
                    <input
                      type="text"
                      className="fuk-input fuk-input--flex-2"
                      value={formData.eligen_source || ''}
                      onChange={(e) => setFormData({...formData, eligen_source: e.target.value})}
                      placeholder="/path/to/masks/ or .psd / .ora file"
                      disabled={generating}
                    />
                  </div>
                  <div className="fuk-input-inline fuk-mt-2" style={{ gap: '0.5rem' }}>
                    <button
                      className="fuk-btn fuk-btn-secondary"
                      style={{ flex: 1 }}
                      disabled={generating}
                      onClick={async () => {
                        try {
                          const res = await fetch('/api/browser/directory', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ title: 'Select EliGen Masks Folder' }),
                          });
                          const data = await res.json();
                          if (data.success && data.directory) {
                            setFormData(prev => ({ ...prev, eligen_source: data.directory }));
                          }
                        } catch (err) { console.error('Browse folder failed:', err); }
                      }}
                    >
                      Browse Folder
                    </button>
                    <button
                      className="fuk-btn fuk-btn-secondary"
                      style={{ flex: 1 }}
                      disabled={generating}
                      onClick={async () => {
                        try {
                          const res = await fetch('/api/browser/open', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              title: 'Select EliGen Mask File (.psd / .ora)',
                              multiple: false,
                              detect_sequences: false,
                              filter: 'all',
                            }),
                          });
                          const data = await res.json();
                          if (data.success && data.files?.length > 0) {
                            setFormData(prev => ({ ...prev, eligen_source: data.files[0].path }));
                          }
                        } catch (err) { console.error('Browse file failed:', err); }
                      }}
                    >
                      Browse File
                    </button>
                  </div>
                </div>
                
                {formData.eligen_source && (
                  <div className="fuk-form-group-compact fuk-mt-2">
                    <div style={{ 
                      padding: '0.5rem 0.75rem',
                      background: 'rgba(255,255,255,0.03)',
                      borderRadius: '4px',
                      fontSize: '0.8rem',
                      color: 'var(--fuk-text-secondary)',
                      wordBreak: 'break-all',
                    }}>
                      {formData.eligen_source}
                    </div>
                  </div>
                )}
                
                <p className="fuk-help-text fuk-mt-2">
                  Point to a folder of mask PNGs (filename = entity prompt) or a 
                  layered file (.psd / .ora — layer name = entity prompt).
                  Paint white where each entity goes, black elsewhere.
                </p>
              </>

            ) : (
              <div className="fuk-empty-state">
                <Camera className="fuk-empty-state-icon" />
                <p className="fuk-empty-state-text">
                  Select an Edit, Control, or EliGen model<br />to enable image tools
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
                {imageModels.map(model => (
                  <option key={model.key} value={model.key}>{model.description}</option>
                ))}
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
                {config?.models?.loras?.map((lora, idx) => (
                  <option key={typeof lora === 'string' ? lora : lora.key || idx} value={typeof lora === 'string' ? lora : lora.key}>
                    {typeof lora === 'string' ? lora : (lora.name || lora.description || lora.key) + (lora.size_mb ? ` (${lora.size_mb}MB)` : '')}
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
              <label className="fuk-label">
                Exponential Shift Mu
                <span className="fuk-help-text-inline"> (auto if blank)</span>
              </label>
              <input
                type="number"
                className="fuk-input"
                value={formData.exponential_shift_mu ?? ''}
                onChange={(e) => setFormData({
                  ...formData, 
                  exponential_shift_mu: e.target.value ? parseFloat(e.target.value) : null
                })}
                placeholder="auto"
                step={0.1}
                min={1}
                max={10}
              />
              <p className="fuk-help-text fuk-mt-1">
                Controls sampling timestep distribution. Leave blank for auto-calculation.
              </p>
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">VRAM Management</label>
              <select
                className="fuk-select"
                value={formData.vram_preset || 'low'}
                onChange={(e) => setFormData({...formData, vram_preset: e.target.value})}
              >
                {(config?.models?.vram_presets || []).map(preset => (
                  <option key={preset.key} value={preset.key} title={preset.description}>
                    {preset.label}
                  </option>
                ))}
              </select>
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
                onChange={(e) => {
                  setFormData({...formData, prompt: e.target.value});
                  handleTextareaResize(e);
                }}
                onInput={handleTextareaResize}
                placeholder="A cinematic still of..."
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
                {aspectRatios.map(ar => (
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
                    const newDims = calculateDimensions(formData.aspectRatio, val || 1024, aspectRatios);
                    setFormData({...formData, width: val, height: newDims.height});
                  }}
                  onBlur={(e) => {
                    const val = parseInt(e.target.value) || 1024;
                    const clamped = Math.max(512, Math.min(2048, val));
                    const newDims = calculateDimensions(formData.aspectRatio, clamped, aspectRatios);
                    setFormData({...formData, width: clamped, height: newDims.height});
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

      {/* Generation Modal */}
      <GenerationModal
        isOpen={showModal}
        type="image"
        generating={generating}
        progress={progress}
        elapsedSeconds={elapsedSeconds}
        consoleLog={consoleLog}
        error={error}
        onCancel={cancel}
        onClose={closeModal}
      />
    </>
  );
}