/**
 * FUK Project Manager
 * Handles project file discovery, loading, saving, and versioning
 */

/**
 * Deep merge two objects
 * Recursively merges nested objects, with source taking precedence
 */
function deepMerge(target, source) {
  const result = { ...target };
  
  for (const key in source) {
    if (source[key] !== undefined) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        // Recursively merge objects
        result[key] = deepMerge(target[key] || {}, source[key]);
      } else {
        // Direct assignment for primitives and arrays
        result[key] = source[key];
      }
    }
  }
  
  return result;
}

// Default config - can be overridden by user config
const DEFAULT_PROJECT_CONFIG = {
  versionFormat: 'date',  // 'date' (251229) or 'sequential' (v01)
  cacheFolder: 'cache',   // Relative to Fuk project folder
  autoSaveInterval: null, // null = disabled, or milliseconds
};

/**
 * Generate version string based on format
 */
export function generateVersion(format = 'date') {
  if (format === 'date') {
    const now = new Date();
    const yy = String(now.getFullYear()).slice(-2);
    const mm = String(now.getMonth() + 1).padStart(2, '0');
    const dd = String(now.getDate()).padStart(2, '0');
    return `${yy}${mm}${dd}`;
  } else {
    // Sequential - caller needs to provide context
    return 'v01';
  }
}

/**
 * Generate next sequential version
 */
export function getNextSequentialVersion(existingVersions) {
  const versionNumbers = existingVersions
    .filter(v => v.startsWith('v'))
    .map(v => parseInt(v.slice(1), 10))
    .filter(n => !isNaN(n));
  
  const maxVersion = versionNumbers.length > 0 ? Math.max(...versionNumbers) : 0;
  return `v${String(maxVersion + 1).padStart(2, '0')}`;
}

/**
 * Parse a project filename into components
 * Format: projectname_shot##_version.json
 */
export function parseProjectFilename(filename) {
  // Remove .json extension
  const base = filename.replace(/\.json$/, '');
  
  // Try to match pattern: anything_shot##_version
  const match = base.match(/^(.+)_shot(\d+)_(.+)$/i);
  
  if (match) {
    return {
      projectName: match[1],
      shotNumber: match[2],
      version: match[3],
      filename: filename,
    };
  }
  
  // Fallback: try simpler patterns
  const simpleMatch = base.match(/^(.+)_(.+)$/);
  if (simpleMatch) {
    return {
      projectName: simpleMatch[1],
      shotNumber: '01',
      version: simpleMatch[2],
      filename: filename,
    };
  }
  
  return null;
}

/**
 * Generate a project filename from components
 */
export function generateProjectFilename(projectName, shotNumber, version) {
  const shot = String(shotNumber).padStart(2, '0');
  return `${projectName}_shot${shot}_${version}.json`;
}

/**
 * Create empty project state structure
 * This is what gets saved to the .json file
 * 
 * Field names must match defaults.json exactly (camelCase).
 * 
 * @param {Object} defaults - User defaults from defaults.json (optional)
 */
export function createEmptyProjectState(defaults = {}) {
  const imageDefaults = defaults.image || {};
  const videoDefaults = defaults.video || {};
  const preprocessDefaults = defaults.preprocess || {};
  const postprocessDefaults = defaults.postprocess || {};
  const layersDefaults = defaults.layers || {};
  const exportDefaults = defaults.export || {};
  
  return {
    // Metadata
    meta: {
      version: '1.0',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      fukVersion: '1.0.0',
    },
    
    // Project info (derived from filename, but stored for reference)
    project: {
      name: '',
      shot: '01',
      version: '',
    },
    
    // Tab states - all generation settings
    // Field names match defaults.json keys (camelCase)
    tabs: {
      image: {
        prompt: imageDefaults.prompt || '',
        negative_prompt: imageDefaults.negative_prompt || '',
        model: imageDefaults.model || 'qwen_image',
        aspectRatio: imageDefaults.aspectRatio || '1.78:1',
        width: imageDefaults.width || 1280,
        height: null,  // Calculated from width and aspect ratio
        steps: imageDefaults.steps || 20,
        stepsMode: imageDefaults.stepsMode || 'preset',
        guidance_scale: imageDefaults.guidance_scale ?? 4,
        cfg_scale: imageDefaults.cfg_scale ?? 5.0,
        denoising_strength: imageDefaults.denoising_strength ?? 1,
        seed: imageDefaults.seed ?? null,
        seedMode: imageDefaults.seedMode || 'random',
        lastUsedSeed: null,
        lora: imageDefaults.lora ?? null,
        lora_multiplier: imageDefaults.lora_multiplier ?? 1.0,
        output_format: imageDefaults.output_format || 'png',
        edit_strength: imageDefaults.edit_strength ?? 1,
        exponential_shift_mu: imageDefaults.exponential_shift_mu ?? null,
        save_latent: imageDefaults.save_latent ?? true,
        control_image_paths: imageDefaults.control_image_paths || [],
      },
      video: {
        prompt: videoDefaults.prompt || '',
        negative_prompt: videoDefaults.negative_prompt || '',
        model: videoDefaults.model || 'wan_t2v_14b',
        task: videoDefaults.task || 'i2v-A14B',
        video_length: videoDefaults.video_length || 41,
        scale_factor: videoDefaults.scale_factor ?? 1.0,
        steps: videoDefaults.steps || 20,
        stepsMode: videoDefaults.stepsMode || 'preset',
        guidance_scale: videoDefaults.guidance_scale ?? 5.0,
        cfg_scale: videoDefaults.cfg_scale ?? 6.0,
        denoising_strength: videoDefaults.denoising_strength ?? 1.0,
        sigma_shift: videoDefaults.sigma_shift ?? 5.0,
        sliding_window_size: videoDefaults.sliding_window_size ?? null,
        sliding_window_stride: videoDefaults.sliding_window_stride ?? null,
        motion_bucket_id: videoDefaults.motion_bucket_id ?? null,
        seed: videoDefaults.seed ?? null,
        seedMode: videoDefaults.seedMode || 'random',
        lastUsedSeed: null,
        lora: videoDefaults.lora ?? null,
        lora_multiplier: videoDefaults.lora_multiplier ?? 1.0,
        save_latent: videoDefaults.save_latent ?? true,
        width: videoDefaults.width ?? null,
        height: videoDefaults.height ?? null,
        source_width: videoDefaults.source_width ?? null,
        source_height: videoDefaults.source_height ?? null,
        image_path: null,
        end_image_path: null,
        control_path: null,
      },
      preprocess: preprocessDefaults,
      postprocess: postprocessDefaults,
      layers: layersDefaults,
      export: exportDefaults,
    },
    
    // Imported assets (control images, reference images, etc.)
    assets: {
      controlImages: [],
      referenceImages: [],
      inputVideos: [],
    },
    
    // Last state for "restore where you left off"
    lastState: {
      activeTab: 'image',
      lastCacheFolder: null,
      lastImagePreview: null,
      lastVideoPreview: null,
      lastPreprocessPreview: null,
      lastPreprocessMeta: null,
      lastPostprocessPreview: null,
      lastPostprocessMeta: null,
      lastLayersPreview: null,
      lastLayersMeta: null,
      lastExportPath: null,
    },
    
    notes: '',
    pinnedGenerations: [],
  };
}
/**
 * Merge loaded state with defaults (handles version upgrades)
 * Uses deep merge to properly handle nested configuration objects
 * 
 * @param {Object} loadedState - State loaded from .json file
 * @param {Object} defaults - User defaults from defaults.json (optional)
 */
export function mergeWithDefaults(loadedState, defaults = {}) {
  const emptyState = createEmptyProjectState(defaults);
  
  // Deep merge each top-level section
  return {
    meta: deepMerge(emptyState.meta, loadedState.meta || {}),
    project: deepMerge(emptyState.project, loadedState.project || {}),
    tabs: {
      image: deepMerge(emptyState.tabs.image, loadedState.tabs?.image || {}),
      video: deepMerge(emptyState.tabs.video, loadedState.tabs?.video || {}),
      preprocess: deepMerge(emptyState.tabs.preprocess, loadedState.tabs?.preprocess || {}),
      postprocess: deepMerge(emptyState.tabs.postprocess, loadedState.tabs?.postprocess || {}),
      layers: deepMerge(emptyState.tabs.layers, loadedState.tabs?.layers || {}),
      export: deepMerge(emptyState.tabs.export, loadedState.tabs?.export || {}),
    },
    assets: deepMerge(emptyState.assets, loadedState.assets || {}),
    lastState: deepMerge(emptyState.lastState, loadedState.lastState || {}),
    notes: loadedState.notes || '',
    pinnedGenerations: loadedState.pinnedGenerations || [],
  };
}

/**
 * Group project files by shot and version
 */
export function organizeProjectFiles(files) {
  const shots = {};
  
  files.forEach(file => {
    const parsed = parseProjectFilename(file.name);
    if (!parsed) return;
    
    const shotKey = parsed.shotNumber;
    if (!shots[shotKey]) {
      shots[shotKey] = {
        number: shotKey,
        versions: [],
      };
    }
    
    shots[shotKey].versions.push({
      version: parsed.version,
      filename: file.name,
      path: file.path,
      modifiedAt: file.modifiedAt,
    });
  });
  
  // Sort versions within each shot (newest first for dates, highest first for sequential)
  Object.values(shots).forEach(shot => {
    shot.versions.sort((a, b) => {
      // Try to sort as dates first (YYMMDD)
      if (/^\d{6}$/.test(a.version) && /^\d{6}$/.test(b.version)) {
        return b.version.localeCompare(a.version); // Descending
      }
      // Try sequential (v01, v02)
      if (a.version.startsWith('v') && b.version.startsWith('v')) {
        return parseInt(b.version.slice(1)) - parseInt(a.version.slice(1));
      }
      // Fallback to string compare
      return b.version.localeCompare(a.version);
    });
  });
  
  return shots;
}

/**
 * Generate cache folder path for a new generation
 */
export function generateCachePath(baseCacheFolder, generationType = 'img_gen') {
  const now = new Date();
  const timestamp = now.toISOString()
    .replace(/[-:]/g, '')
    .replace('T', '_')
    .slice(0, 15); // YYYYMMDD_HHMMSS
  
  return `${baseCacheFolder}/${generationType}_${timestamp}`;
}