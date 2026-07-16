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
 * Calculate dimensions based on aspect ratio and width.
 * Both width and height are rounded UP to the next 16-multiple —
 * Qwen's ShapeChecker requires a multiple of 16, and all other supported
 * models accept any multiple of 16 (it's always a multiple of 8 too).
 * We always round up (ceil, never nearest) so the output is at least the
 * requested size: users can crop off the few extra pixels rather than
 * being forced to upscale a too-small image.
 *
 * @param {string} aspectRatioValue - The aspect ratio value (e.g., "1.78:1")
 * @param {number} width - Target width
 * @param {Array} aspectRatios - Array of aspect ratio objects from config
 * @returns {{width: number, height: number}}
 */
export function calculateDimensions(aspectRatioValue, width, aspectRatios = []) {
  const ratio = aspectRatios.find(ar => ar.value === aspectRatioValue)?.ratio || 1;
  const rawHeight = width / ratio;
  // Round up to the next 16-multiple, but tolerate a sub-pixel epsilon first so
  // a value already sitting on a boundary (e.g. 1440.04 from an approximate
  // ratio) isn't pushed up a needless extra 16px.
  const ceil16 = (v) => Math.ceil(v / 16 - 1e-2) * 16;
  return {
    width: ceil16(width),
    height: ceil16(rawHeight)
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
