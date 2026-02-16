/**
 * FUK UI Helper Functions
 */



/**
 * Format seconds as MM:SS
 */
export function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Calculate dimensions based on aspect ratio and width
 * Width rounds to nearest 64 (safe for all diffusion models)
 * Height rounds to nearest 16 (finer granularity, prevents ratio collisions)
 * 
 * @param {string} aspectRatioValue - The aspect ratio value (e.g., "1.78:1")
 * @param {number} width - Target width
 * @param {Array} aspectRatios - Array of aspect ratio objects from config
 * @returns {{width: number, height: number}}
 */
export function calculateDimensions(aspectRatioValue, width, aspectRatios = []) {
  const ratio = aspectRatios.find(ar => ar.value === aspectRatioValue)?.ratio || 1;
  const rawHeight = width / ratio;
  return {
    width: Math.round(width / 16) * 16,
    height: Math.round(rawHeight / 16) * 16
  };
}

/**
 * Validate video length is 4n+1
 */
export function isValidVideoLength(length) {
  return (length - 1) % 4 === 0;
}

/**
 * Get nearest valid video length
 */
export function nearestValidVideoLength(length) {
  const remainder = (length - 1) % 4;
  if (remainder === 0) return length;
  return length + (4 - remainder);
}

/**
 * Format file size
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Debounce function
 */
export function debounce(fn, delay) {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}
