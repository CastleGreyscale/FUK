/**
 * Layer Stack API Client
 * Non-destructive latent editing endpoints
 *
 * Layer edits now run through the normal /api/generate/image endpoint
 * (with stack_id + layer_name in the request). This file handles
 * stack lifecycle (init, toggle, flatten, etc.) only.
 */

import { API_URL } from './constants';

const BASE = `${API_URL}/layers`;

async function _post(url, body = {}) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function _patch(url, body = {}) {
  const res = await fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

async function _delete(url) {
  const res = await fetch(url, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

/** List all layer stacks */
export async function listStacks() {
  const res = await fetch(`${BASE}/list`);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

/**
 * Initialize a new layer stack from a source image.
 * Creates img_edit_XXX dir, copies/encodes base latent.
 * @param {string} sourceImageUrl  e.g. "/api/project/cache/fuk_shot/img_gen_054/generated.png"
 * @param {string} [name]          Optional human-readable name
 * @returns {{ success, stack_id, stack, base_preview_url }}
 */
export async function initStack(sourceImageUrl, name) {
  return _post(`${BASE}/init`, { source_image_url: sourceImageUrl, name });
}

/** Get a stack manifest by ID */
export async function getStack(stackId) {
  const res = await fetch(`${BASE}/${stackId}`);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

/** Toggle a layer on or off */
export async function toggleLayer(stackId, layerId, enabled) {
  return _patch(`${BASE}/${stackId}/layer/${layerId}/toggle`, { enabled });
}

/** Switch a layer to a specific version */
export async function switchVersion(stackId, layerId, version) {
  return _patch(`${BASE}/${stackId}/layer/${layerId}/version`, { version });
}

/** Reorder layers by providing the full ordered list of layer_ids */
export async function reorderLayers(stackId, layerIds) {
  return _post(`${BASE}/${stackId}/reorder`, { layer_ids: layerIds });
}

/** Remove a layer from the stack */
export async function removeLayer(stackId, layerId) {
  return _delete(`${BASE}/${stackId}/layer/${layerId}`);
}

/** Flatten all enabled layers → returns { preview_url, stack } */
export async function flattenStack(stackId) {
  return _post(`${BASE}/${stackId}/flatten`);
}