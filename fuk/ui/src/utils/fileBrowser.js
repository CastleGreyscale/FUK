/**
 * In-app file browser API
 *
 * browseFiles() / browseDirectory() open the FileBrowser modal (mounted once
 * in App via FileBrowserHost) and resolve with the same shapes the old native
 * dialog endpoints (/api/browser/open, /api/browser/directory) returned, so a
 * caller swaps its fetch for a call:
 *
 *   browseFiles     → { success: true, files: [{ path, display_name, media_type,
 *                       first_frame, last_frame, frame_count, frame_pattern }] }
 *   browseDirectory → { success: true, directory }
 *
 * Cancelling resolves { success: false, cancelled: true, files: [], directory: null }.
 */

let openHandler = null;

export const CANCELLED = { success: false, cancelled: true, files: [], directory: null };

/** Called by FileBrowserHost. Returns an unregister function. */
export function registerFileBrowser(handler) {
  openHandler = handler;
  return () => { if (openHandler === handler) openHandler = null; };
}

function openBrowser(request) {
  if (!openHandler) {
    return Promise.resolve({ ...CANCELLED, error: 'File browser is not mounted' });
  }
  return new Promise(resolve => openHandler({ ...request, resolve }));
}

/**
 * Pick files.
 * @param {object} opts
 * @param {string} opts.filter - all | images | videos | exr | layered | any
 * @param {boolean} opts.detectSequences - collapse frame sequences into one entry
 */
export function browseFiles({
  title = 'Select Media',
  multiple = true,
  filter = 'all',
  detectSequences = true,
  initialDir = null,
} = {}) {
  return openBrowser({ mode: 'files', title, multiple, filter, detectSequences, initialDir });
}

/** Pick a folder. */
export function browseDirectory({ title = 'Select Folder', initialDir = null } = {}) {
  return openBrowser({ mode: 'directory', title, initialDir, filter: 'all', detectSequences: true });
}
