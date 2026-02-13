/**
 * History API utilities for registering imports
 */

const API_URL = '/api';

/**
 * Register an imported file in history
 * 
 * @param {string} path - File path (absolute, or printf pattern for sequences)
 * @param {string} displayName - Display name for the file
 * @param {boolean} autoPin - Whether to auto-pin the import
 * @param {object} sequenceInfo - Optional sequence metadata
 * @param {number} sequenceInfo.firstFrame - First frame number
 * @param {number} sequenceInfo.lastFrame - Last frame number  
 * @param {number} sequenceInfo.frameCount - Total frame count
 * @param {string} sequenceInfo.framePattern - Pattern with #### notation
 * @returns {Promise<object>} Import result
 */
export async function registerImport(path, displayName, autoPin = true, sequenceInfo = null) {
  try {
    const body = {
      path,
      name: displayName,
      auto_pin: autoPin,
    };
    
    // Add sequence metadata if provided
    if (sequenceInfo) {
      body.first_frame = sequenceInfo.firstFrame;
      body.last_frame = sequenceInfo.lastFrame;
      body.frame_count = sequenceInfo.frameCount;
      body.frame_pattern = sequenceInfo.framePattern;
    }
    
    const response = await fetch(`${API_URL}/project/import`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      console.error('[HistoryAPI] Import failed:', error);
      return { success: false, error: error.detail || 'Import failed' };
    }
    
    const result = await response.json();
    return result;
    
  } catch (err) {
    console.error('[HistoryAPI] Import error:', err);
    return { success: false, error: err.message };
  }
}