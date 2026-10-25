/**
 * PhotoCraft bridge API (server routes in fuk/ui/photocraft_endpoints.py)
 *
 * PhotoCraft can't call out, so the FUK server drives it over its control
 * channel. Pulls copy the document into an import entry and return an
 * `api/project/cache/...` URL the Image tab inputs take directly; nothing here
 * starts a generation.
 */

const API_URL = '/api/photocraft';

async function call(path, body) {
  const response = await fetch(`${API_URL}/${path}`, body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await response.json(); } catch { /* empty body */ }
  if (!response.ok) {
    throw new Error(data.detail || `PhotoCraft bridge: HTTP ${response.status}`);
  }
  for (const warning of data.warnings || []) console.warn('[PhotoCraft]', warning);
  return data;
}

/** Connection, open documents and the active document's layers. Never throws on "not running". */
export const getPhotoCraftStatus = () => call('status');

/** Start PhotoCraft with its control port open (no-op if it is already reachable). */
export const launchPhotoCraft = () => call('launch', {});

/**
 * Composite or one layer of a document.
 * @param {{source?: 'composite'|'active'|'layer', layer?: string, doc_index?: number, max_side?: number}} opts
 */
export const pullFromPhotoCraft = (opts = {}) => call('pull', opts);

/** Composite + mask from the selection (or a `_mask` layer). */
export const pullInpaintFromPhotoCraft = (opts = {}) => call('pull-inpaint', opts);

/** `@<prompt>` layers → an EliGen mask folder; returns {eligen_source, entities}. */
export const pullEligenFromPhotoCraft = (opts = {}) => call('pull-eligen', opts);

/** Depth of the composite (or a layer) placed back as a layer. */
export const depthToPhotoCraftLayer = (opts = {}) => call('depth-to-layer', opts);

/** Any FUK image (URL or absolute path) → a new layer in the active document. */
export const sendToPhotoCraft = (path, opts = {}) => call('place', { path, ...opts });
