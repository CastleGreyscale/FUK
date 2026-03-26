/**
 * Layer Stack API Client
 * Non-destructive latent editing endpoints
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

/** Create a new stack from an existing generation */
export async function createStack(imageUrl, name) {
  return _post(`${BASE}/create`, { image_url: imageUrl, name });
}

/** Get a stack manifest by ID */
export async function getStack(stackId) {
  const res = await fetch(`${BASE}/${stackId}`);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

/**
 * Run a Qwen edit and add result as a new layer.
 * @param {string} stackId
 * @param {object} params  { name, prompt, model, control_image, seed, ... }
 */
export async function addLayer(stackId, params) {
  return _post(`${BASE}/${stackId}/add`, params);
}

/**
 * Re-run a layer (optionally with param overrides) → new version.
 * @param {string} stackId
 * @param {object} params  { layer_id, version_id?, prompt?, seed?, ... }
 */
export async function rerunLayer(stackId, params) {
  return _post(`${BASE}/${stackId}/rerun`, params);
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
