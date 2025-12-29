/**
 * FUK UI Helper Functions
 */

import { ASPECT_RATIOS } from './constants';

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
 */
export function calculateDimensions(aspectRatio, width) {
  const ratio = ASPECT_RATIOS.find(ar => ar.value === aspectRatio)?.ratio || 1;
  const height = Math.round(width / ratio);
  // Round to nearest 64 (required by diffusion models)
  return {
    width: Math.round(width / 64) * 64,
    height: Math.round(height / 64) * 64
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
