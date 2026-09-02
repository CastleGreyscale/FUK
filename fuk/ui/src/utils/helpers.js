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
 * The frame lattice each video model requires, as `factor`n + `remainder`.
 * Wan is 4n+1, LTX-2 is 8n+1, MiniMax-H3 is 17n+5 — so these come from the
 * model's `constraints` block (served by /api/config/models), never hardcoded.
 * Falls back to Wan's lattice when no constraints are available yet.
 */
export const DEFAULT_FRAME_LATTICE = { factor: 4, remainder: 1, minFrames: 5 };

export function frameLattice(constraints) {
  const c = constraints || {};
  return {
    factor: c.frame_factor ?? DEFAULT_FRAME_LATTICE.factor,
    remainder: c.frame_remainder ?? DEFAULT_FRAME_LATTICE.remainder,
    minFrames: c.min_frames ?? DEFAULT_FRAME_LATTICE.minFrames,
  };
}

export function isValidVideoLength(length, constraints) {
  const { factor, remainder, minFrames } = frameLattice(constraints);
  return length >= minFrames && length % factor === remainder % factor;
}

/**
 * Snap a frame count onto the model's lattice, rounding UP — matching what the
 * pipelines do internally, so the user never silently gets a shorter clip than
 * the number shown in the field.
 */
export function snapFrames(length, constraints) {
  const { factor, remainder, minFrames } = frameLattice(constraints);
  const n = Math.max(1, Math.round(length) || 0);
  const snapped = Math.ceil((n - remainder) / factor) * factor + remainder;
  return Math.max(minFrames, snapped);
}

/**
 * Snap a pixel dimension up to the model's latent grid (16 for Wan, 32 for
 * LTX-2 and MiniMax-H3, 64 for LTX-2 two-stage).
 */
export function snapDimension(px, multiple = 16) {
  const m = Math.max(1, multiple);
  return Math.max(m, Math.ceil((Math.round(px) || 0) / m) * m);
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
