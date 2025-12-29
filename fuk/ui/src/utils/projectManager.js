/**
 * FUK Project Manager
 * Handles project file discovery, loading, saving, and versioning
 */

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
 */
export function createEmptyProjectState() {
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
    tabs: {
      image: {
        prompt: '',
        model: 'qwen_image',
        negative_prompt: '',
        aspectRatio: '16:9',
        width: 1344,
        height: 756,
        steps: 20,
        stepsMode: 'preset',
        guidance_scale: 2.1,
        flow_shift: 2.1,
        seed: null,
        seedMode: 'random',  // 'random', 'fixed', 'increment'
        lastUsedSeed: null,
        lora: null,
        lora_multiplier: 1.0,
        blocks_to_swap: 10,
        output_format: 'png',
        edit_strength: 0.7,
        control_image_paths: [],
      },
      video: {
        prompt: '',
        task: 'i2v-14B',
        negative_prompt: '',
        width: 832,
        height: 480,
        video_length: 81,
        steps: 20,
        guidance_scale: 5.0,
        flow_shift: 5.0,
        seed: null,
        seedMode: 'random',
        lastUsedSeed: null,
        lora: null,
        lora_multiplier: 1.0,
        blocks_to_swap: 15,
        image_path: null,
        end_image_path: null,
      },
      preprocess: {
        // Placeholder for future
      },
      postprocess: {
        // Placeholder for future
      },
      export: {
        outputFormat: 'exr',
        colorSpace: 'linear',
        // More to come
      },
    },
    
    // Imported assets (control images, reference images, etc.)
    assets: {
      controlImages: [],      // Paths relative to project
      referenceImages: [],
      inputVideos: [],
    },
    
    // Last state for "restore where you left off"
    lastState: {
      activeTab: 'image',
      lastImagePreview: null,   // Path to last generated image
      lastVideoPreview: null,   // Path to last generated video
      lastCacheFolder: null,    // Last cache folder used
    },
    
    // Notes - free text for artist notes
    notes: '',
  };
}

/**
 * Merge loaded state with defaults (handles version upgrades)
 */
export function mergeWithDefaults(loadedState) {
  const defaults = createEmptyProjectState();
  
  return {
    meta: { ...defaults.meta, ...loadedState.meta },
    project: { ...defaults.project, ...loadedState.project },
    tabs: {
      image: { ...defaults.tabs.image, ...loadedState.tabs?.image },
      video: { ...defaults.tabs.video, ...loadedState.tabs?.video },
      preprocess: { ...defaults.tabs.preprocess, ...loadedState.tabs?.preprocess },
      postprocess: { ...defaults.tabs.postprocess, ...loadedState.tabs?.postprocess },
      export: { ...defaults.tabs.export, ...loadedState.tabs?.export },
    },
    assets: { ...defaults.assets, ...loadedState.assets },
    lastState: { ...defaults.lastState, ...loadedState.lastState },
    notes: loadedState.notes || '',
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
