/**
 * FUK UI Constants
 */

// API base - in dev, Vite proxies /api to localhost:8000
export const API_URL = '/api';

// Aspect ratio presets for image generation
export const ASPECT_RATIOS = [
  { label: '1:1 (Square)', value: '1:1', ratio: 100/100 },      
  { label: '1.33:1 (Fullscreen)', value: '1.33:1', ratio: 133/100 },
  { label: '1.85:1 (Academy)', value: '1.85:1', ratio: 185/100 },
  { label: '1.78:1 (Widescreen)', value: '1.78:1', ratio: 178/100 },
  { label: '2.39:1 (Anamorphic)', value: '2.39:1', ratio: 239/100 },
  { label: '2.75:1 (Panavision)', value: '2.75:1', ratio: 276/100 },
];

// Video length presets (must be 4n+1)
export const VIDEO_LENGTHS = [
  { label: '1 sec (25 frames)', value: 25 },
  { label: '2 sec (49 frames)', value: 49 },
  { label: '3 sec (81 frames)', value: 81 },
  { label: '4 sec (97 frames)', value: 97 },
  { label: '5 sec (121 frames)', value: 121 },
];

// Step presets
export const STEP_PRESETS = [20, 40];

// Seed mode options (ComfyUI-style)
export const SEED_MODES = {
  FIXED: 'fixed',       // Use exact seed value
  RANDOM: 'random',     // Generate new random seed each time
  INCREMENT: 'increment', // Increment seed by 1 after each generation
};

// Generate a random seed (0 to 2^32-1)
export function generateRandomSeed() {
  return Math.floor(Math.random() * 4294967295);
}

// Default negative prompt (fallback if not loaded from backend)
export const DEFAULT_NEGATIVE_PROMPT = 
  'blurry, low quality, distorted, deformed, ugly, bad anatomy, worst quality, ' +
  'low resolution, jpeg artifacts, watermark, signature, username, text';

// Default form values
export const DEFAULT_IMAGE_SETTINGS = {
  prompt: '',
  model: 'qwen_image',
  negative_prompt: DEFAULT_NEGATIVE_PROMPT,
  aspectRatio: '1.78:1',
  width: 1344,
  steps: 20,
  stepsMode: 'preset',
  guidance_scale: 2.1,
  flow_shift: 2.1,
  seed: null,
  seedMode: SEED_MODES.RANDOM,  // Default to random
  lastUsedSeed: null,           // Track last seed used for saving
  lora: null,
  lora_multiplier: 1.0,
  blocks_to_swap: 10,
  output_format: 'png',
  edit_strength: 0.7,
  control_image_paths: [],
};

export const DEFAULT_VIDEO_SETTINGS = {
  prompt: '',
  task: 'i2v-14B',
  negative_prompt: DEFAULT_NEGATIVE_PROMPT,
  width: 832,
  height: 480,
  video_length: 81,
  steps: 20,
  guidance_scale: 5.0,
  flow_shift: 5.0,
  seed: null,
  seedMode: SEED_MODES.RANDOM,
  lastUsedSeed: null,
  lora: null,
  lora_multiplier: 1.0,
  blocks_to_swap: 15,
  image_path: null,
  end_image_path: null,
};

/**
 * Build proper image URL from backend path
 * Handles both project-cache paths and output paths
 */
export function buildImageUrl(path) {
  if (!path) return null;
  
  // New dynamic cache endpoint (project-specific)
  if (path.startsWith('api/project/cache/')) {
    return `/${path}`;
  }
  
  // External files endpoint (for imports) - new format
  if (path.startsWith('api/project/files/')) {
    return `/${path}`;
  }
  
  // Legacy external files endpoint (for backwards compatibility)
  if (path.startsWith('api/files/')) {
    // Convert to new format
    const absolutePath = path.replace('api/files/', '');
    return `/api/project/files/${absolutePath}`;
  }
  
  // Legacy static cache mount (fallback)
  if (path.startsWith('project-cache/')) {
    return `/${path}`;
  }
  
  // Absolute paths need to go through api/project/files endpoint
  if (path.startsWith('/')) {
    // Strip leading slash for URL path segment
    return `/api/project/files${path}`;
  }
  
  return `/outputs/${path}`;
}