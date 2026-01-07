/**
 * History API utilities
 * Separated from GenerationHistory component to fix Vite HMR issues
 */

import { API_URL } from '../utils/constants';

/**
 * Register an imported asset in history
 * Call this from any component when assets are imported
 * 
 * @param {string} path - Absolute path to the file
 * @param {string} name - Display name (optional)
 * @param {boolean} autoPin - Whether to auto-pin (default: true)
 * @returns {Promise<{success, id, path, auto_pin}>}
 */
export async function registerImport(path, name = null, autoPin = true) {
  try {
    const res = await fetch(`${API_URL}/project/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        path,
        name,
        auto_pin: autoPin,
      }),
    });
    
    if (!res.ok) {
      throw new Error(`Import failed: ${res.statusText}`);
    }
    
    const data = await res.json();
    console.log('[History] Registered import:', data);
    
    return data;
  } catch (err) {
    console.error('[History] Failed to register import:', err);
    return { success: false, error: err.message };
  }
}

/**
 * Delete a generation from history
 * 
 * @param {string} generationId - ID of the generation to delete
 * @returns {Promise<{success: boolean}>}
 */
export async function deleteGeneration(generationId) {
  try {
    const res = await fetch(`${API_URL}/project/generations/${encodeURIComponent(generationId)}`, {
      method: 'DELETE'
    });
    
    if (!res.ok) {
      throw new Error(`Delete failed: ${res.statusText}`);
    }
    
    return { success: true };
  } catch (err) {
    console.error('[History] Failed to delete generation:', err);
    return { success: false, error: err.message };
  }
}

/**
 * Fetch generations from history
 * 
 * @param {number} days - Number of days to fetch (0 = all)
 * @param {string[]} pinnedIds - Array of pinned generation IDs
 * @returns {Promise<{generations: Array, hasMore: boolean}>}
 */
export async function fetchGenerations(days = 1, pinnedIds = []) {
  const params = new URLSearchParams({ days: days.toString() });
  
  if (pinnedIds.length > 0) {
    params.set('pinned', pinnedIds.join(','));
  }
  
  const res = await fetch(`${API_URL}/project/generations?${params}`);
  
  if (!res.ok) {
    throw new Error(`Failed to fetch: ${res.statusText}`);
  }
  
  return await res.json();
}