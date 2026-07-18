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
/**
 * Save a generation's output file (png/mp4/…) to a user-chosen location.
 *
 * Opens a native "Save As" dialog on the server (this is a local app) and copies
 * the file there — for pulling finished generations out of the project cache to
 * share / post for approval.
 *
 * @param {object} generation - History item (uses .path/.preview + .id/.name)
 * @returns {Promise<{success: boolean, path?: string, cancelled?: boolean, error?: string}>}
 */
export async function saveGenerationToLocation(generation) {
  const source = generation.path || generation.preview;
  if (!source) return { success: false, error: 'No file to save' };

  // Build a discoverable default filename: "<project>_<gen>.<ext>".
  const ext = (source.split('?')[0].split('.').pop() || '').toLowerCase();
  const base = String(generation.id || generation.name || 'generation').replace(/[\\/]+/g, '_');
  const name = ext && ext.length <= 5 ? `${base}.${ext}` : base;

  try {
    const response = await fetch(`${API_URL}/project/generations/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, name }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      return { success: false, error: error.detail || 'Save failed' };
    }
    return await response.json();
  } catch (err) {
    console.error('[HistoryAPI] Save error:', err);
    return { success: false, error: err.message };
  }
}

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