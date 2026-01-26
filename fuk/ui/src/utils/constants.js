/**
 * FUK UI Constants
 * 
 * Static values only - user-configurable defaults come from backend /api/config/defaults
 */

// API base - in dev, Vite proxies /api to localhost:8000
export const API_URL = '/api';

// Seed mode options (enum - structure won't change)
export const SEED_MODES = {
  FIXED: 'fixed',       // Use exact seed value
  RANDOM: 'random',     // Generate new random seed each time
  INCREMENT: 'increment', // Increment seed by 1 after each generation
};

// Generate a random seed (0 to 2^32-1)
export function generateRandomSeed() {
  return Math.floor(Math.random() * 4294967295);
}

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