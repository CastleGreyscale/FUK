/**
 * Video Generation Tab
 * Wan (I2V, FLF2V, VACE, Animate), LTX-2 and MiniMax-H3 video generation
 *
 * Defaults flow: config.defaults.video (shared UI shape)
 *                -> config.defaults[model.defaults_section] (per-family sampling)
 *                -> project state / localStorage overwrites
 *
 * Every hard limit — frame lattice, latent grid, fps, which shift knob exists,
 * which controls are relevant at all — comes from the selected model's
 * `constraints` block, served alongside the model list by /api/config/models
 * and applied identically on the server. Nothing here assumes Wan's numbers.
 *
 * Features:
 * - Auto-reads dimensions from input image, snapped to the model's latent grid
 * - Scale factor for VRAM management
 * - Manual frame entry validated against the model's frame lattice
 * - Duration feedback at the model's own frame rate
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Film, CheckCircle, AlertCircle, Info } from '../../src/components/Icons';
import MediaUploader from '../components/MediaUploader.jsx';
import SeedControl from '../components/SeedControl';
import GenerationModal from '../components/GenerationModal';
import { useGeneration } from '../hooks/useGeneration';
import { useLocalStorage } from '../../src/hooks/useLocalStorage';
import { useSavedSeeds } from '../hooks/useSavedSeeds';
import { useVideoPlayback } from '../hooks/useVideoPlayback';
import { startVideoGeneration } from '../../src/utils/api';
import { formatTime, snapFrames, snapDimension, frameLattice } from '../utils/helpers.js';
import { 
  buildImageUrl, 
  SEED_MODES, 
  generateRandomSeed,
  API_URL,
} from '../../src/utils/constants';
import Footer from '../components/Footer';
import PromptPanel from '../components/PromptPanel';


// Scale factor presets for VRAM management
const SCALE_FACTORS = [
  { label: '100%', value: 1.0 },
  { label: '75%', value: 0.75 },
  { label: '50%', value: 0.5 },
  { label: '25%', value: 0.25 },
];

// The frame lattice, latent grid and shift knob all differ per model — Wan is
// 4n+1 on a /16 grid with sigma_shift, LTX-2 is 8n+1 on /32 with no shift knob
// at all, MiniMax-H3 is 17n+5 on /32 with flow_shift plus a separate audio
// shift. Everything below reads the selected model's `constraints` block
// (served by /api/config/models) rather than assuming Wan's numbers.
const DEFAULT_CONSTRAINTS = {
  spatial_multiple: 16,
  frame_factor: 4,
  frame_remainder: 1,
  min_frames: 5,
  fps: 24,
  shift_param: 'sigma_shift',
  shift_label: 'Sigma Shift',
  shift_default: 5.0,
  shift_min: 1.0,
  shift_max: 10.0,
  shift_step: 0.5,
  cfg_min: 1,
  cfg_max: 15,
  supports_denoise: true,
  supports_lora: true,
  hidden_controls: [],
};

// Get duration string from frame count
function getFrameDuration(frames, fps = 24) {
  const seconds = frames / fps;
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)}ms`;
  }
  return `${seconds.toFixed(2)}s`;
}

export default function VideoTab({ config, activeTab, setActiveTab, project, playbackSpeed }) {
  const videoRef = useVideoPlayback(playbackSpeed);
  // Get defaults from backend config
  const videoDefaults = config?.defaults?.video || {};
  
  // Model metadata from config
  const videoModels = config?.models?.video_models || [];
  
  // Initial defaults come entirely from backend config
  // These are the values used when no project is loaded and no localStorage exists
  const initialDefaults = useMemo(() => ({
    prompt: videoDefaults.prompt ?? '',
    negative_prompt: videoDefaults.negative_prompt ?? '',
    task: videoDefaults.task ?? 'i2v-A14B',
    video_length: videoDefaults.video_length ?? 41,
    scale_factor: videoDefaults.scale_factor ?? 1.0,
    steps: videoDefaults.steps ?? 20,
    stepsMode: videoDefaults.stepsMode ?? 'preset',
    guidance_scale: videoDefaults.guidance_scale ?? 5.0,
    sigma_shift: videoDefaults.sigma_shift ?? 5.0,
    switch_dit_boundary: videoDefaults.switch_dit_boundary ?? 0.875,
    sliding_window_size: videoDefaults.sliding_window_size ?? null,
    sliding_window_stride: videoDefaults.sliding_window_stride ?? null,
    tea_cache_l1_thresh: videoDefaults.tea_cache_l1_thresh ?? null,
    audio_flow_shift: videoDefaults.audio_flow_shift ?? null,
    denoising_strength: videoDefaults.denoising_strength ?? 1.0,
    loras: videoDefaults.loras ?? (videoDefaults.lora ? [{ key: videoDefaults.lora, multiplier: videoDefaults.lora_multiplier ?? 1.0, bypass: videoDefaults.lora_bypass ?? false }] : []),
    seed: videoDefaults.seed ?? null,
    seedMode: videoDefaults.seedMode ?? SEED_MODES.RANDOM,
    lastUsedSeed: videoDefaults.lastUsedSeed ?? null,
    image_path: videoDefaults.image_path ?? null,
    end_image_path: videoDefaults.end_image_path ?? null,
    animate_pose_video: videoDefaults.animate_pose_video ?? null,
    animate_face_video: videoDefaults.animate_face_video ?? null,
    animate_inpaint_video: videoDefaults.animate_inpaint_video ?? null,
    animate_mask_video: videoDefaults.animate_mask_video ?? null,
    width: videoDefaults.width ?? null,
    height: videoDefaults.height ?? null,
    source_width: videoDefaults.source_width ?? null,
    source_height: videoDefaults.source_height ?? null,
    vram_preset: config?.models?.vram_preset_default ?? 'low',
    batchCount: 1,
    frame_inherit: videoDefaults.frame_inherit ?? true,
    trim_frames: videoDefaults.trim_frames ?? null,
  }), [videoDefaults]);
  
  // Fallback localStorage for when no project is loaded
  const [localFormData, setLocalFormData] = useLocalStorage('fuk_video_settings', initialDefaults);

  // Sampling defaults for a model's own family. `defaults.video` supplies the
  // shared UI shape (scale factor, seed mode, batch count); this overlays the
  // values that genuinely differ per model, so picking LTX-2 gives 121 frames /
  // 30 steps / cfg 3.0 instead of Wan's 41 / 40 / 4.
  const familyDefaultsFor = useCallback((task) => {
    const model = (config?.models?.video_models || []).find(m => m.key === task);
    if (!model) return {};
    const section = model.defaults_section;
    const fam = (section && section !== 'video') ? (config?.defaults?.[section] || {}) : {};
    const c = model.constraints || {};
    const out = {};
    if (fam.video_length != null) out.video_length = fam.video_length;
    if (fam.steps != null) {
      out.steps = fam.steps;
      // The steps radio only offers 20 and 40; LTX-2 wants 30 and MiniMax 50.
      out.stepsMode = [20, 40].includes(fam.steps) ? 'preset' : 'custom';
    }
    if (fam.cfg_scale != null) out.guidance_scale = fam.cfg_scale;
    if (fam.negative_prompt != null) out.negative_prompt = fam.negative_prompt;
    if (fam.denoising_strength != null) out.denoising_strength = fam.denoising_strength;
    if (c.shift_default != null) out.sigma_shift = c.shift_default;
    if (c.audio_shift_default != null) out.audio_flow_shift = c.audio_shift_default;
    return out;
  }, [config?.models?.video_models, config?.defaults]);

  // Use project state if available, otherwise localStorage.
  // New format: tabs.video = { activeModel, modelSettings: { [task]: {...} } }
  // Old format: tabs.video = flat object (backward compat — detected by absence of modelSettings)
  const formData = useMemo(() => {
    const tabState = project?.projectState?.tabs?.video;
    if (tabState?.modelSettings) {
      const activeModel = tabState.activeModel || initialDefaults.task;
      const modelData = tabState.modelSettings[activeModel] || {};
      // Family defaults sit between the global shape and this model's own saved
      // settings, so a model picked for the first time opens on its own numbers
      // while anything the user has already tuned still wins.
      return { ...initialDefaults, ...familyDefaultsFor(activeModel), ...modelData, task: activeModel };
    }
    if (tabState) {
      return { ...initialDefaults, ...familyDefaultsFor(tabState.task ?? initialDefaults.task), ...tabState };
    }
    return { ...initialDefaults, ...familyDefaultsFor(localFormData?.task ?? initialDefaults.task), ...localFormData };
  }, [project?.projectState?.tabs?.video, localFormData, initialDefaults, familyDefaultsFor]);

  // The selected model's hard parameter limits. Drives the frame lattice, the
  // resolution grid, which sampling controls render at all, and their ranges.
  const constraints = useMemo(() => ({
    ...DEFAULT_CONSTRAINTS,
    ...((config?.models?.video_models || []).find(m => m.key === formData.task)?.constraints || {}),
  }), [config?.models?.video_models, formData.task]);

  const hiddenControls = useMemo(
    () => new Set(constraints.hidden_controls || []),
    [constraints]
  );
  const shiftParam = constraints.shift_param ?? null;
  const audioShiftParam = constraints.audio_shift_param ?? null;
  const spatialMultiple = constraints.spatial_multiple ?? 16;
  const modelFps = constraints.fps ?? 24;
  const lattice = useMemo(() => frameLattice(constraints), [constraints]);

  // Migrate old single lora fields to loras array
  const effectiveLoras = useMemo(() => {
    if (formData.loras?.length > 0) return formData.loras;
    if (formData.lora) return [{ key: formData.lora, multiplier: formData.lora_multiplier ?? 1.0, bypass: formData.lora_bypass ?? false }];
    return [];
  }, [formData.loras, formData.lora, formData.lora_multiplier, formData.lora_bypass]);

  // Ref to track latest formData for setFormData callback
  const formDataRef = useRef(formData);
  useEffect(() => {
    formDataRef.current = formData;
  }, [formData]);

  // Ref to access current tab state in setFormData without adding it to deps
  const projectTabRef = useRef(null);
  useEffect(() => {
    projectTabRef.current = project?.projectState?.tabs?.video ?? null;
  }, [project?.projectState?.tabs?.video]);

  // Update function that writes to project or localStorage.
  // On model switch: saves current model's settings, then activates new model's saved settings.
  const setFormData = useCallback((updater) => {
    const currentData = formDataRef.current;
    const newData = typeof updater === 'function' ? updater(currentData) : updater;

    if (project?.isProjectLoaded && project?.updateTabState) {
      const currentModel = currentData.task;
      const newModel = newData.task;
      const existingModelSettings = projectTabRef.current?.modelSettings || {};

      if (newModel !== currentModel) {
        // Save current model's state, switch activeModel — formData recomputes from new model's saved settings
        project.updateTabState('video', {
          activeModel: newModel,
          modelSettings: { ...existingModelSettings, [currentModel]: currentData },
        });
      } else {
        project.updateTabState('video', {
          activeModel: currentModel,
          modelSettings: { ...existingModelSettings, [currentModel]: newData },
        });
      }
    } else {
      // No project: one flat blob, so a model switch has to re-seed the
      // sampling values itself — the per-model merge in formData only runs on
      // the project path.
      setLocalFormData(
        newData.task !== currentData.task
          ? { ...newData, ...familyDefaultsFor(newData.task) }
          : newData
      );
    }
  }, [project?.isProjectLoaded, project?.updateTabState, setLocalFormData, familyDefaultsFor]);

  // Frame input state (for controlled input before validation)
  const [frameInput, setFrameInput] = useState(String(formData.video_length || 81));

  // Source video metadata (populated when control_path is a video)
  const [sourceVideoInfo, setSourceVideoInfo] = useState(null);

  // Metadata reload from history drag
  const [metaDragOver, setMetaDragOver] = useState(false);
  const [droppedPreview, setDroppedPreview] = useState(null);
  const [metaLoadedFrom, setMetaLoadedFrom] = useState(null);
  
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
    consoleLog,
    showModal,
    startGeneration,
    cancel,
    closeModal,
    reset: resetGeneration,
    setKeepModalOpen,
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
        // Snap up to the selected model's latent grid (/16 Wan, /32 LTX-2 and
        // MiniMax-H3). Rounding up rather than to-nearest means the frame is
        // never cropped below the source, and matches what the runner does.
        const width = snapDimension(img.width, spatialMultiple);
        const height = snapDimension(img.height, spatialMultiple);

        setFormData(prev => ({
          ...prev,
          source_width: width,
          source_height: height,
          // Update output dimensions with scale factor
          width: snapDimension(width * prev.scale_factor, spatialMultiple),
          height: snapDimension(height * prev.scale_factor, spatialMultiple),
        }));
      };
      img.onerror = () => {
        console.warn('Could not load image for dimension detection');
      };
      img.src = buildImageUrl(formData.image_path);
    }
  }, [formData.image_path, setFormData, spatialMultiple]);

  // Fetch video info when control_path changes — drives frame count inheritance
  useEffect(() => {
    if (!formData.control_path) {
      setSourceVideoInfo(null);
      return;
    }
    const cleanPath = formData.control_path.replace(/^\/outputs\//, '').replace(/^\//, '');
    fetch(`/api/video/info?path=${encodeURIComponent(cleanPath)}`)
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(info => {
        setSourceVideoInfo(info);
        const base = snapFrames(info.frame_count, constraints);
        const frames = formData.trim_frames
          ? snapFrames(Math.min(base, formData.trim_frames), constraints)
          : base;
        setFrameInput(String(frames));
        setFormData(prev => ({ ...prev, video_length: frames }));
      })
      .catch(() => setSourceVideoInfo(null));
  }, [formData.control_path, constraints]); // eslint-disable-line react-hooks/exhaustive-deps

  // Update frame count when trim changes — always derives from source when a control video is loaded
  const handleTrimChange = (value) => {
    const trimVal = value ? parseInt(value) : null;
    setFormData(prev => {
      const newData = { ...prev, trim_frames: trimVal };
      if (sourceVideoInfo) {
        const base = snapFrames(sourceVideoInfo.frame_count, constraints);
        const frames = trimVal
          ? snapFrames(Math.min(base, trimVal), constraints)
          : base;
        setFrameInput(String(frames));
        return { ...newData, video_length: frames };
      }
      return newData;
    });
  };

  // Update output dimensions when scale factor changes
  const handleScaleChange = (newScale) => {
    const sourceW = formData.source_width || 832;
    const sourceH = formData.source_height || 480;
    
    setFormData(prev => ({
      ...prev,
      scale_factor: newScale,
      width: snapDimension(sourceW * newScale, spatialMultiple),
      height: snapDimension(sourceH * newScale, spatialMultiple),
    }));
  };

  // Handle frame input change with validation
  const handleFrameInputChange = (value) => {
    setFrameInput(value);
  };

  // Validate and round frames on blur
  const handleFrameInputBlur = () => {
    const parsed = parseInt(frameInput) || formData.video_length || lattice.minFrames;
    const valid = snapFrames(parsed, constraints);
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

  // --- Batch generation ---
  const batchQueueRef = useRef(null);
  const [batchProgress, setBatchProgress] = useState(null); // { current, total }

  const buildBatchSeeds = useCallback((seedMode, startSeed, count) => {
    if (seedMode === SEED_MODES.RANDOM) {
      return Array.from({ length: count }, () => generateRandomSeed());
    }
    if (seedMode === SEED_MODES.INCREMENT) {
      const base = startSeed ?? 0;
      return Array.from({ length: count }, (_, i) => base + i);
    }
    // FIXED — same seed every time
    return Array.from({ length: count }, () => startSeed);
  }, []);

  // Chain next generation when previous completes
  useEffect(() => {
    if (!result || result.error) return;
    if (!batchQueueRef.current) return;

    const queue = batchQueueRef.current;
    queue.index += 1;

    if (queue.index >= queue.seeds.length) {
      batchQueueRef.current = null;
      setBatchProgress(null);
      setKeepModalOpen(false);
      return;
    }

    const nextSeed = queue.seeds[queue.index];
    setBatchProgress({ current: queue.index + 1, total: queue.seeds.length });
    setKeepModalOpen(true);

    startVideoGeneration({ ...queue.basePayload, seed: nextSeed })
      .then(data => startGeneration(data.generation_id))
      .catch(err => {
        console.error('Batch video generation failed:', err);
        batchQueueRef.current = null;
        setBatchProgress(null);
        setKeepModalOpen(false);
      });
  }, [result]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleGenerate = async () => {
    const effectiveSeed = getEffectiveSeed();
    const count = formData.batchCount || 1;
    
    const basePayload = {
      ...formData,
      image_path: formData.image_path ? formData.image_path.replace(/^\/outputs\//, '') : null,
      end_image_path: formData.end_image_path ? formData.end_image_path.replace(/^\/outputs\//, '') : null,
      control_path: formData.control_path ? formData.control_path.replace(/^\/outputs\//, '') : null,
      animate_pose_video: formData.animate_pose_video ? formData.animate_pose_video.replace(/^\/outputs\//, '') : null,
      animate_face_video: formData.animate_face_video ? formData.animate_face_video.replace(/^\/outputs\//, '') : null,
      animate_inpaint_video: formData.animate_inpaint_video ? formData.animate_inpaint_video.replace(/^\/outputs\//, '') : null,
      animate_mask_video: formData.animate_mask_video ? formData.animate_mask_video.replace(/^\/outputs\//, '') : null,
      denoising_strength: 1.0,
      lora: null,
      lora_multiplier: 1.0,
      lora_bypass: undefined,
      // Don't post values the selected model has no knob for — they would only
      // land in the runner's **kwargs and then be recorded in the generation
      // metadata as settings that were never applied.
      loras: constraints.supports_lora === false ? [] : effectiveLoras,
      sigma_shift: shiftParam ? formData.sigma_shift : null,
      audio_flow_shift: audioShiftParam ? formData.audio_flow_shift : null,
    };

    // Build seed queue for batch
    if (count > 1) {
      const seeds = buildBatchSeeds(formData.seedMode, effectiveSeed, count);
      batchQueueRef.current = { seeds, index: 0, basePayload };
      setBatchProgress({ current: 1, total: count });
      setKeepModalOpen(true);
      basePayload.seed = seeds[0];
    } else {
      batchQueueRef.current = null;
      setBatchProgress(null);
      setKeepModalOpen(false);
      basePayload.seed = effectiveSeed;
    }
    
    console.log('Video generation payload:', basePayload);
    console.log(`Seed mode: ${formData.seedMode}, batch: ${count}, using seed: ${basePayload.seed}`);
    
    setFormData(prev => ({
      ...prev,
      lastUsedSeed: effectiveSeed,
      seed: prev.seedMode === SEED_MODES.INCREMENT ? effectiveSeed : prev.seed,
    }));
    
    try {
      const data = await startVideoGeneration(basePayload);
      startGeneration(data.generation_id);
    } catch (err) {
      console.error('Video generation failed:', err);
      batchQueueRef.current = null;
      setBatchProgress(null);
      alert('Failed to start video generation');
    }
  };

  // Keyboard shortcut event listeners (dispatched by useKeyboardShortcuts in App)
  const generateRef = useRef(handleGenerate);
  const cancelRef = useRef(cancel);
  const seedModeRef = useRef(formData.seedMode);
  useEffect(() => { generateRef.current = handleGenerate; }, [handleGenerate]);
  useEffect(() => { cancelRef.current = cancel; }, [cancel]);
  useEffect(() => { seedModeRef.current = formData.seedMode; }, [formData.seedMode]);

  useEffect(() => {
    const onGenerate = () => generateRef.current();
    const onCancel = () => cancelRef.current();
    const onSeedMode = (e) => {
      const mode = e.detail;
      if (mode === 'cycle') {
        const cycle = [SEED_MODES.RANDOM, SEED_MODES.FIXED, SEED_MODES.INCREMENT];
        const next = cycle[(cycle.indexOf(seedModeRef.current) + 1) % cycle.length];
        setFormData(prev => ({ ...prev, seedMode: next }));
      } else {
        setFormData(prev => ({ ...prev, seedMode: mode }));
      }
    };
    const onPlayPause = () => {
      const v = videoRef.current;
      if (!v) return;
      v.paused ? v.play() : v.pause();
    };
    const onStepForward = () => {
      const v = videoRef.current;
      if (!v) return;
      v.pause();
      const fps = 24;
      v.currentTime = Math.min(v.duration, v.currentTime + 1 / fps);
    };
    const onStepBack = () => {
      const v = videoRef.current;
      if (!v) return;
      v.pause();
      const fps = 24;
      v.currentTime = Math.max(0, v.currentTime - 1 / fps);
    };
    window.addEventListener('fuk-shortcut-generate', onGenerate);
    window.addEventListener('fuk-shortcut-cancel', onCancel);
    window.addEventListener('fuk-shortcut-seed-mode', onSeedMode);
    window.addEventListener('fuk-shortcut-video-play-pause', onPlayPause);
    window.addEventListener('fuk-shortcut-video-step-forward', onStepForward);
    window.addEventListener('fuk-shortcut-video-step-back', onStepBack);
    return () => {
      window.removeEventListener('fuk-shortcut-generate', onGenerate);
      window.removeEventListener('fuk-shortcut-cancel', onCancel);
      window.removeEventListener('fuk-shortcut-seed-mode', onSeedMode);
      window.removeEventListener('fuk-shortcut-video-play-pause', onPlayPause);
      window.removeEventListener('fuk-shortcut-video-step-forward', onStepForward);
      window.removeEventListener('fuk-shortcut-video-step-back', onStepBack);
    };
  }, [setFormData]);

  const handleStartImageChange = (paths) => {
    setFormData(prev => ({ ...prev, image_path: paths[0] || null }));
  };

  const handleEndImageChange = (paths) => {
    setFormData(prev => ({ ...prev, end_image_path: paths[0] || null }));
  };

  const handleControlPathChange = (paths) => {
    setFormData(prev => ({ ...prev, control_path: paths[0] || null }));
  };

  const handleAnimatePoseVideoChange = (paths) => {
    setFormData(prev => ({ ...prev, animate_pose_video: paths[0] || null }));
  };

  const handleAnimateFaceVideoChange = (paths) => {
    setFormData(prev => ({ ...prev, animate_face_video: paths[0] || null }));
  };

  const handleAnimateInpaintVideoChange = (paths) => {
    setFormData(prev => ({ ...prev, animate_inpaint_video: paths[0] || null }));
  };

  const handleAnimateMaskVideoChange = (paths) => {
    setFormData(prev => ({ ...prev, animate_mask_video: paths[0] || null }));
  };

  // Check if task requires images or control video — always check both model supports and task
  // name so neither source alone can cause a false negative during config load or key mismatches.
  const selectedModel = videoModels.find(m => m.key === formData.task);
  const modelSupports = selectedModel?.supports || [];
  // Each pipeline family names its conditioning differently, but they all
  // arrive over the same three form fields (image_path, end_image_path,
  // control_path), so the gating maps every family's vocabulary onto those:
  //   Wan      input_image / vace_reference_image / vace_video / end_image
  //   LTX-2    input_images (first frame) / in_context_videos (IC-LoRA driver)
  //   MiniMax  keyframes (first+last) / references (Ref2VA subject)
  const requiresControlVideo = modelSupports.includes('vace_video')
    || modelSupports.includes('in_context_videos')
    || modelSupports.includes('references')
    || formData.task?.includes('-FC');
  const isFunControl = modelSupports.includes('vace_video') || formData.task?.includes('-FC');

  // The video slot means something different per family, and only Wan's is
  // actually required, so it is labelled from the model rather than assumed.
  const videoSlot = modelSupports.includes('in_context_videos')
    ? {
        label: 'In-Context Video',
        required: false,
        help: 'Driving video for the IC-LoRAs — depth/pose/edge for Union-Control, '
            + 'or a low-detail clip for Detailer. Only used when one is loaded.',
      }
    : modelSupports.includes('references')
    ? {
        label: 'Reference Video',
        required: false,
        help: 'Reference clip the prompt can address as <Video 1>. Its own '
            + 'soundtrack comes along as <Audio 1> when it has one.',
      }
    : {
        label: 'Control Video',
        required: true,
        help: 'Video or image sequence for pose/motion control',
      };
  const requiresStartImage = modelSupports.includes('input_image')
    || modelSupports.includes('vace_reference_image')
    || modelSupports.includes('input_images')
    || modelSupports.includes('keyframes')
    || modelSupports.includes('references')
    || formData.task?.includes('i2v')
    || formData.task?.includes('flf2v')
    || formData.task?.includes('inp')
    || modelSupports.includes('animate_pose_video');
  // MiniMax FL2VA interpolates between a first and last frame, the same shape
  // as Wan's flf2v.
  const requiresEndImage = modelSupports.includes('end_image')
    || modelSupports.includes('keyframes')
    || formData.task?.includes('flf2v');

  const requiresAnimatePoseVideo = modelSupports.includes('animate_pose_video');
  const requiresAnimateFaceVideo = modelSupports.includes('animate_face_video');
  const requiresAnimateInpaintVideo = modelSupports.includes('animate_inpaint_video');
  const requiresAnimateMaskVideo = modelSupports.includes('animate_mask_video');

  // Showing a slot is not the same as demanding it. MiniMax generates happily
  // from a prompt alone — keyframes and references are both optional — so
  // gating Generate on an image would block valid work.
  const imageInputsOptional = modelSupports.includes('keyframes')
    || modelSupports.includes('references');
  const startImageLabel = (modelSupports.includes('references') || requiresAnimatePoseVideo)
    ? 'Reference Image'
    : 'Start Image';
  
  // Calculate whether current frame input sits on the selected model's lattice
  const parsedFrames = parseInt(frameInput);
  const validFrameCount = snapFrames(
    Number.isNaN(parsedFrames) ? (formData.video_length || lattice.minFrames) : parsedFrames,
    constraints
  );
  const currentFrameValid = !Number.isNaN(parsedFrames) && parsedFrames === validFrameCount;
  
  // Calculate CSS variable for dynamic aspect ratio
  const aspectRatioStyle = { '--preview-aspect': `${formData.width || 832} / ${formData.height || 480}` };

  // Reload generation settings from a history item dropped onto the preview panel
  const handleMetaDrop = useCallback(async (e) => {
    e.preventDefault();
    setMetaDragOver(false);

    const fukGenData = e.dataTransfer.getData('application/x-fuk-generation');
    if (!fukGenData) return;

    let gen;
    try { gen = JSON.parse(fukGenData); } catch { return; }

    try {
      const res = await fetch(`${API_URL}/project/generations/${encodeURIComponent(gen.id)}/metadata`);
      if (!res.ok) return;
      const meta = await res.json();

      const updates = {};
      // Restore the RAW draft (with #markers intact), not the resolved string —
      // the resolved `prompt` has mood + expanded markers baked in, which would
      // re-bake into the editable field and stack "Mood: …" on the next gen.
      const draftPrompt = meta.prompt_source || meta.prompt;
      if (draftPrompt)                     updates.prompt              = draftPrompt;
      if (meta.negative_prompt)            updates.negative_prompt     = meta.negative_prompt;
      // Video model is stored as 'model' but the tab uses 'task'
      if (meta.model)                      updates.task                = meta.model;
      if (meta.guidance_scale != null)     updates.guidance_scale      = meta.guidance_scale;
      if (meta.infer_steps != null)        updates.steps               = meta.infer_steps;
      if (meta.video_length != null)       updates.video_length        = meta.video_length;
      if (meta.loras?.length > 0) {
        updates.loras = meta.loras;
      } else if (meta.lora != null) {
        updates.loras = [{ key: meta.lora, multiplier: meta.lora_multiplier ?? 1.0 }];
      }
      if (meta.sigma_shift != null)        updates.sigma_shift         = meta.sigma_shift;
      if (meta.switch_dit_boundary != null) updates.switch_dit_boundary = meta.switch_dit_boundary;
if (meta.denoising_strength != null) updates.denoising_strength  = meta.denoising_strength;
      if (meta.sliding_window_size != null)   updates.sliding_window_size   = meta.sliding_window_size;
      if (meta.sliding_window_stride != null) updates.sliding_window_stride = meta.sliding_window_stride;
      // Restored explicitly rather than via the `!= null` idiom above: null is
      // the meaningful "cache off" value, so a baseline run must clear it.
      if ('tea_cache_l1_thresh' in meta) updates.tea_cache_l1_thresh = meta.tea_cache_l1_thresh;
      if (meta.seed != null) {
        updates.seed     = meta.seed;
        updates.seedMode = SEED_MODES.FIXED;
      }
      if (meta.image_size?.[0] && meta.image_size?.[1]) {
        updates.source_width  = meta.image_size[0];
        updates.source_height = meta.image_size[1];
        updates.width         = meta.image_size[0];
        updates.height        = meta.image_size[1];
      }
      // Restore animate video inputs if present
      if (meta.animate_pose_video)    updates.animate_pose_video    = meta.animate_pose_video;
      if (meta.animate_face_video)    updates.animate_face_video    = meta.animate_face_video;
      if (meta.animate_inpaint_video) updates.animate_inpaint_video = meta.animate_inpaint_video;
      if (meta.animate_mask_video)    updates.animate_mask_video    = meta.animate_mask_video;

      setFormData(prev => ({ ...prev, ...updates }));

      if (meta.video_length != null) setFrameInput(String(meta.video_length));

      const previewPath = gen.preview || gen.path;
      setDroppedPreview(previewPath ? buildImageUrl(previewPath) : null);
      setMetaLoadedFrom(gen.name || gen.id?.split('/').pop() || 'history');

    } catch (err) {
      console.error('[VideoTab] Metadata reload failed:', err);
    }
  }, [setFormData]);

  return (
    <>
      {/* Preview Area — also acts as metadata drop target */}
      <div
        className={`fuk-preview-area ${metaDragOver ? 'meta-drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setMetaDragOver(true); }}
        onDragLeave={() => setMetaDragOver(false)}
        onDrop={handleMetaDrop}
      >
        <div className="fuk-preview-single">
          {/* droppedPreview takes priority — shows reference gen while settings are loaded */}
          {droppedPreview ? (
            <div className="fuk-preview-container fuk-preview-container--meta">
              {droppedPreview.includes('.mp4') || droppedPreview.includes('.webm') ? (
                <video
                  src={droppedPreview}
                  muted
                  loop
                  autoPlay
                  playsInline
                  className="fuk-preview-media fuk-preview-media--meta"
                />
              ) : (
                <img
                  src={droppedPreview}
                  alt="Loaded from history"
                  className="fuk-preview-media fuk-preview-media--meta"
                />
              )}
              <div className="fuk-preview-info">
                <div className="fuk-preview-info-row">
                  <span className="fuk-meta-loaded-badge">↩ Settings from: {metaLoadedFrom}</span>
                  <button
                    className="fuk-meta-dismiss"
                    onClick={() => { setDroppedPreview(null); setMetaLoadedFrom(null); }}
                    title="Dismiss"
                  >×</button>
                </div>
              </div>
            </div>
          ) : previewVideo ? (
            <div className="fuk-preview-container">
              <video
                ref={videoRef}
                src={buildImageUrl(previewVideo)}
                controls
                muted
                loop
                autoPlay
                className="fuk-preview-media"
              />
              <div className="fuk-preview-info">
                <div className="fuk-preview-info-row">
                  <span>{formData.width}×{formData.height}</span>
                  <span>•</span>
                  <span>{formData.video_length} frames ({getFrameDuration(formData.video_length, modelFps)})</span>
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
              className="fuk-placeholder-card fuk-placeholder-card--ratio"
              style={aspectRatioStyle}
            >
              <div className="fuk-placeholder">
                <Film className="fuk-placeholder-icon" />
                <p className="fuk-placeholder-text">
                  {formData.width || '---'} × {formData.height || '---'} × {formData.video_length} frames
                </p>
                <p className="fuk-placeholder-subtext">
                  {formData.source_width ? `Source: ${formData.source_width}×${formData.source_height}` : 'Upload image to set dimensions'}
                </p>
                <p className="fuk-placeholder-subtext fuk-meta-hint">
                  Drop a history item here to reload its settings
                </p>
              </div>
            </div>
          )}
        </div>
        {metaDragOver && (
          <div className="fuk-meta-drop-overlay">
            <span>Load Settings</span>
          </div>
        )}

        <PromptPanel
          prompt={formData.prompt}
          negativePrompt={formData.negative_prompt}
          onChange={(keyOrPatch, value) => setFormData(prev => (
            typeof keyOrPatch === 'object' && keyOrPatch !== null
              ? { ...prev, ...keyOrPatch }
              : { ...prev, [keyOrPatch]: value }
          ))}
          disabled={generating}
          model={formData.task}
          loras={effectiveLoras}
          mode="video"
        />
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Input Images Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">
              {requiresAnimatePoseVideo ? 'Animate Inputs' : isFunControl ? 'Control Inputs' : 'Input Images'}
            </h3>

            {/* Control Video — Wan VACE/Fun-Control, or an LTX-2 in-context driver */}
            {requiresControlVideo && (
              <div className="fuk-form-group-compact">
                <label className="fuk-label">
                  {videoSlot.label}{' '}
                  {videoSlot.required
                    ? <span className="fuk-label-required">(Required)</span>
                    : <span className="fuk-label-description">(Optional)</span>}
                </label>
                <MediaUploader
                  images={formData.control_path ? [formData.control_path] : []}
                  onImagesChange={handleControlPathChange}
                  disabled={generating}
                  multiple={false}
                  accept="all"
                  label="Drop video or click to browse"
                />
                <p className="fuk-help-text">{videoSlot.help}</p>
              </div>
            )}

            {/* Animate Pose Video */}
            {requiresAnimatePoseVideo && (
              <div className="fuk-form-group-compact">
                <label className="fuk-label">
                  Pose Video <span className="fuk-label-required">(Required)</span>
                </label>
                <MediaUploader
                  images={formData.animate_pose_video ? [formData.animate_pose_video] : []}
                  onImagesChange={handleAnimatePoseVideoChange}
                  disabled={generating}
                  multiple={false}
                  accept="all"
                  label="Drop video or click to browse"
                />
                <p className="fuk-help-text">
                  Pose keypoints video for animation control
                </p>
              </div>
            )}

            {/* Animate Face Video */}
            {requiresAnimateFaceVideo && (
              <div className="fuk-form-group-compact fuk-mt-4">
                <label className="fuk-label">
                  Face Video <span className="fuk-label-required">(Required)</span>
                </label>
                <MediaUploader
                  images={formData.animate_face_video ? [formData.animate_face_video] : []}
                  onImagesChange={handleAnimateFaceVideoChange}
                  disabled={generating}
                  multiple={false}
                  accept="all"
                  label="Drop video or click to browse"
                />
                <p className="fuk-help-text">
                  Facial animation video for expression control
                </p>
              </div>
            )}

            {/* Animate Inpaint Video (Replace mode only) */}
            {requiresAnimateInpaintVideo && (
              <div className="fuk-form-group-compact fuk-mt-4">
                <label className="fuk-label">
                  Inpaint Video <span className="fuk-label-description">(Optional for Replace)</span>
                </label>
                <MediaUploader
                  images={formData.animate_inpaint_video ? [formData.animate_inpaint_video] : []}
                  onImagesChange={handleAnimateInpaintVideoChange}
                  disabled={generating}
                  multiple={false}
                  accept="all"
                  label="Drop video or click to browse"
                />
                <p className="fuk-help-text">
                  Inpaint content video for texture replacement
                </p>
              </div>
            )}

            {/* Animate Mask Video (Replace mode only) */}
            {requiresAnimateMaskVideo && (
              <div className="fuk-form-group-compact fuk-mt-4">
                <label className="fuk-label">
                  Mask Video <span className="fuk-label-description">(Optional for Replace)</span>
                </label>
                <MediaUploader
                  images={formData.animate_mask_video ? [formData.animate_mask_video] : []}
                  onImagesChange={handleAnimateMaskVideoChange}
                  disabled={generating}
                  multiple={false}
                  accept="all"
                  label="Drop video or click to browse"
                />
                <p className="fuk-help-text">
                  Mask video for selective replacement areas
                </p>
              </div>
            )}

            {/* Start Image - for I2V modes (optional in Fun Control, required for Animate) */}
            {(requiresStartImage || isFunControl) ? (
              <>
                <div className={`fuk-form-group-compact ${(requiresControlVideo || requiresAnimatePoseVideo) ? 'fuk-mt-4' : ''}`}>
                  <label className="fuk-label">
                    {startImageLabel}
                    {(isFunControl || imageInputsOptional)
                      ? <span className="fuk-label-description">(Optional)</span>
                      : <span className="fuk-label-required">(Required)</span>}
                  </label>
                  <MediaUploader
                    images={formData.image_path ? [formData.image_path] : []}
                    onImagesChange={handleStartImageChange}
                    disabled={generating}
                    multiple={false}
                    accept="all"
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
                      End Image{' '}
                      {imageInputsOptional
                        ? <span className="fuk-label-description">(Optional)</span>
                        : <span className="fuk-label-required">(Required for FLF2V)</span>}
                    </label>
                    <MediaUploader
                      images={formData.end_image_path ? [formData.end_image_path] : []}
                      onImagesChange={handleEndImageChange}
                      disabled={generating}
                      multiple={false}
                      accept="all"
                    />
                  </div>
                )}

              </>
            ) : (
              <div className="fuk-empty-state">
                <Film className="fuk-empty-state-icon" />
                <p className="fuk-empty-state-text">
                  Select an I2V, FLF2V, Animate, Fun Control, LTX-2<br />or MiniMax-H3 model to enable control inputs
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
                {videoModels.length > 0 ? (
                  videoModels.map(model => (
                    <option key={model.key} value={model.key}>{model.description}</option>
                  ))
                ) : (
                  <>
                    <option value="i2v-A14B">Wan 2.2 I2V (Recommended)</option>
                    <option value="i2v-14B">Wan 2.1 I2V</option>
                    <option value="flf2v-14B">Wan 2.1 FLF2V (First+Last Frame)</option>
                    <option value="i2v-14B-FC">Wan 2.1 I2V Fun Control</option>
                  </>
                )}
              </select>
            </div>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">
                Resolution
                {formData.source_width && (
                  <span className="fuk-label-description">
                    (from {formData.source_width}×{formData.source_height})
                  </span>
                )}
              </label>
              <select
                className="fuk-select"
                value={formData.scale_factor}
                onChange={(e) => handleScaleChange(parseFloat(e.target.value))}
              >
                {SCALE_FACTORS.map(sf => (
                  <option key={sf.value} value={sf.value}>{sf.label}</option>
                ))}
              </select>
              <p className="fuk-help-text fuk-mt-1">
                → {formData.width || '---'} × {formData.height || '---'}
              </p>
              {!formData.source_width && (
                <p className="fuk-help-text fuk-help-text--warning">
                  <AlertCircle className="fuk-icon--sm" />
                  Upload a start image to auto-detect dimensions
                </p>
              )}
            </div>

            <div className="fuk-form-group-compact">
              <label className="fuk-label">
                Frames
                <span className="fuk-label-description">
                  (must be {lattice.factor}n+{lattice.remainder})
                </span>
                {sourceVideoInfo && (
                  <span className="fuk-label-description">
                    · Source: {sourceVideoInfo.frame_count} @ {sourceVideoInfo.fps.toFixed(1)}fps
                  </span>
                )}
              </label>

              <div className="fuk-input-inline">
                <input
                  type="number"
                  className={`fuk-input fuk-input--w-80 ${!currentFrameValid ? 'fuk-input--warning' : ''}`}
                  value={frameInput}
                  onChange={(e) => handleFrameInputChange(e.target.value)}
                  onBlur={handleFrameInputBlur}
                  min={lattice.minFrames}
                  max={241}
                  step={lattice.factor}
                />
                <span className="fuk-input-result">
                  ≈ {getFrameDuration(formData.video_length, modelFps)} @ {modelFps}fps
                </span>
                {sourceVideoInfo && (
                  <>
                    <label className="fuk-label fuk-label--nowrap">Trim to</label>
                    <input
                      type="number"
                      className="fuk-input fuk-input--w-80"
                      value={formData.trim_frames ?? ''}
                      onChange={(e) => handleTrimChange(e.target.value)}
                      placeholder="∞"
                      min={lattice.minFrames}
                      step={lattice.factor}
                    />
                  </>
                )}
              </div>
              {!currentFrameValid && (
                <p className="fuk-help-text fuk-help-text--info">
                  Will round up to {validFrameCount} frames ({getFrameDuration(validFrameCount, modelFps)})
                </p>
              )}
            </div>

            {!hiddenControls.has('sliding_window') && (
            <div className="fuk-form-pair">
              <div className="fuk-form-group-compact">
                <label className="fuk-label" title="Number of frames per sliding window chunk. Leave blank to disable.">
                  Window Size <Info className="fuk-label-info" />
                </label>
                <input
                  type="number"
                  className="fuk-input"
                  value={formData.sliding_window_size ?? ''}
                  onChange={(e) => setFormData({
                    ...formData,
                    sliding_window_size: e.target.value ? parseInt(e.target.value) : null
                  })}
                  placeholder="disabled"
                  step={1}
                  min={1}
                />
              </div>
              <div className="fuk-form-group-compact">
                <label className="fuk-label" title="Step size between sliding windows. Leave blank to disable.">
                  Window Stride <Info className="fuk-label-info" />
                </label>
                <input
                  type="number"
                  className="fuk-input"
                  value={formData.sliding_window_stride ?? ''}
                  onChange={(e) => setFormData({
                    ...formData,
                    sliding_window_stride: e.target.value ? parseInt(e.target.value) : null
                  })}
                  placeholder="disabled"
                  step={1}
                  min={1}
                />
              </div>
            </div>
            )}

            {constraints.supports_lora !== false && (
            <div className="fuk-form-group-compact">
              <div className="lora-header">
                <label className="fuk-label">LoRA</label>
                <button
                  type="button"
                  className="lora-add-btn"
                  onClick={() => setFormData({...formData, loras: [...effectiveLoras, { key: '', multiplier: 1.0 }]})}
                >+ Add</button>
              </div>
              {effectiveLoras.map((entry, idx) => (
                <div key={idx} className={`lora-row${entry.bypass ? ' lora-row--bypassed' : ''}`}>
                  <select
                    className="fuk-select lora-row-select"
                    value={entry.key || ''}
                    onChange={(e) => {
                      const selected = config?.models?.loras?.find(l => (typeof l === 'string' ? l : l.key) === e.target.value);
                      const strength = (selected && typeof selected !== 'string' && selected.default_strength != null)
                        ? selected.default_strength
                        : entry.multiplier;
                      const updated = effectiveLoras.map((l, i) => i === idx ? { ...l, key: e.target.value, multiplier: strength } : l);
                      setFormData({...formData, loras: updated});
                    }}
                  >
                    <option value="">None</option>
                    {config?.models?.loras
                      ?.filter(l => {
                        // A curated entry's "model" is a list — one LoRA often
                        // applies to several registry keys — while a scanned one
                        // has no model at all and is offered everywhere. The
                        // selected model key is formData.task here, not
                        // formData.model: this tab names the field after the
                        // generation task.
                        if (typeof l === 'string' || !l.model) return true;
                        const models = Array.isArray(l.model) ? l.model : [l.model];
                        return models.includes(formData.task);
                      })
                      .map((lora, i) => (
                      <option key={typeof lora === 'string' ? lora : lora.key || i} value={typeof lora === 'string' ? lora : lora.key}>
                        {typeof lora === 'string' ? lora : (lora.name || lora.description || lora.key) + (lora.size_mb ? ` (${lora.size_mb}MB)` : '')}
                      </option>
                    ))}
                  </select>
                  <input
                    type="number"
                    className="fuk-input lora-row-multiplier"
                    value={entry.multiplier}
                    onChange={(e) => {
                      const updated = effectiveLoras.map((l, i) => i === idx ? { ...l, multiplier: parseFloat(e.target.value) } : l);
                      setFormData({...formData, loras: updated});
                    }}
                    step={0.05}
                    min={0}
                    max={2}
                  />
                  <button
                    type="button"
                    title={entry.bypass ? 'Enable LoRA' : 'Bypass LoRA'}
                    className={`lora-row-btn${entry.bypass ? ' lora-row-btn--active' : ''}`}
                    onClick={() => {
                      const updated = effectiveLoras.map((l, i) => i === idx ? { ...l, bypass: !l.bypass } : l);
                      setFormData({...formData, loras: updated});
                    }}
                  >⊘</button>
                  <button
                    type="button"
                    className="lora-row-btn"
                    onClick={() => setFormData({...formData, loras: effectiveLoras.filter((_, i) => i !== idx)})}
                  >×</button>
                </div>
              ))}
            </div>
            )}
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
            
            <div className="fuk-form-pair">
              <div className="fuk-form-group-compact">
                <label
                  className="fuk-label"
                  title={constraints.cfg_min >= 1 && constraints.cfg_max <= 10 && constraints.cfg_min === 1
                    ? 'Classifier-free guidance. 1.0 disables CFG (and the negative prompt with it) — which is what MiniMax-H3 ships as its default.'
                    : 'Classifier-free guidance strength.'}
                >
                  Guidance Scale <Info className="fuk-label-info" />
                </label>
                <input
                  type="number"
                  className="fuk-input"
                  value={formData.guidance_scale}
                  onChange={(e) => setFormData({...formData, guidance_scale: parseFloat(e.target.value)})}
                  step={0.5}
                  min={constraints.cfg_min ?? 1}
                  max={constraints.cfg_max ?? 15}
                />
              </div>
              {/* LTX-2 derives its noise schedule from sequence length and takes
                  no shift argument, so the control is absent rather than inert. */}
              {shiftParam && (
              <div className="fuk-form-group-compact">
                <label className="fuk-label" title={constraints.shift_help || 'Controls sampling timestep distribution.'}>
                  {constraints.shift_label || 'Sigma Shift'} <Info className="fuk-label-info" />
                </label>
                <input
                  type="number"
                  className="fuk-input"
                  value={formData.sigma_shift ?? constraints.shift_default ?? ''}
                  onChange={(e) => setFormData({...formData, sigma_shift: parseFloat(e.target.value)})}
                  step={constraints.shift_step ?? 0.5}
                  min={constraints.shift_min ?? 1}
                  max={constraints.shift_max ?? 10}
                />
              </div>
              )}
              {/* MiniMax-H3 denoises picture and sound together on two
                  schedules, so the audio branch has its own shift. */}
              {audioShiftParam && (
              <div className="fuk-form-group-compact">
                <label className="fuk-label" title={constraints.audio_shift_help || 'Noise-schedule shift for the audio branch.'}>
                  {constraints.audio_shift_label || 'Audio Shift'} <Info className="fuk-label-info" />
                </label>
                <input
                  type="number"
                  className="fuk-input"
                  value={formData.audio_flow_shift ?? constraints.audio_shift_default ?? ''}
                  onChange={(e) => setFormData({...formData, audio_flow_shift: parseFloat(e.target.value)})}
                  step={constraints.audio_shift_step ?? 0.5}
                  min={constraints.audio_shift_min ?? 1}
                  max={constraints.audio_shift_max ?? 10}
                />
              </div>
              )}
            </div>

            {(!hiddenControls.has('switch_dit_boundary') || !hiddenControls.has('tea_cache')) && (
            <div className="fuk-form-pair">
              {!hiddenControls.has('switch_dit_boundary') && (
              <div className="fuk-form-group-compact">
                <label className="fuk-label" title="Timestep fraction (×1000) where Wan 2.2 switches from the high-noise DiT to the low-noise DiT. Higher = more steps on the high-noise expert. Dual-DiT (A14B) models only. Default: 0.875">
                  DiT Switch Boundary <Info className="fuk-label-info" />
                </label>
                <input
                  type="number"
                  className="fuk-input"
                  value={formData.switch_dit_boundary}
                  onChange={(e) => setFormData({...formData, switch_dit_boundary: parseFloat(e.target.value)})}
                  step={0.025}
                  min={0}
                  max={1}
                />
              </div>
              )}
              {!hiddenControls.has('tea_cache') && (
              <div className="fuk-form-group-compact">
                <label className="fuk-label" title="TeaCache: skips the DiT forward pass on steps whose timestep embedding is close enough to the previous one. Blank = off. Lower = more conservative, higher = faster but risks motion stutter. Try 0.08–0.15 before going higher. Unvalidated on our shots — A/B against a blank run before trusting it on a final.">
                  TeaCache Threshold <Info className="fuk-label-info" />
                </label>
                <input
                  type="number"
                  className="fuk-input"
                  value={formData.tea_cache_l1_thresh ?? ''}
                  onChange={(e) => setFormData({
                    ...formData,
                    tea_cache_l1_thresh: e.target.value ? parseFloat(e.target.value) : null
                  })}
                  placeholder="off"
                  step={0.01}
                  min={0}
                  max={1}
                />
              </div>
              )}
            </div>
            )}
            {!hiddenControls.has('tea_cache') && formData.tea_cache_l1_thresh != null && (
              <p className="fuk-help-text fuk-help-text--info">
                TeaCache on — speed/quality tradeoff is unvalidated on our shots.
                Watch the DiT switch step for artifacts.
              </p>
            )}

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
        canGenerate={!!formData.prompt
          && (imageInputsOptional || !requiresStartImage || !!formData.image_path)
          && (imageInputsOptional || !requiresEndImage || !!formData.end_image_path)
          && (!requiresAnimatePoseVideo || !!formData.animate_pose_video)
          && (!requiresAnimateFaceVideo || !!formData.animate_face_video)}
        generateLabel="Generate Video"
        generatingLabel={batchProgress ? `Generating ${batchProgress.current}/${batchProgress.total}...` : 'Generating...'}
        batchCount={formData.batchCount}
        onBatchCountChange={(val) => setFormData(prev => ({ ...prev, batchCount: val }))}
      />

      {/* Generation Modal */}
      <GenerationModal
        isOpen={showModal}
        type="video"
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