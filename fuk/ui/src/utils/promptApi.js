import { API_URL } from './constants';

/**
 * Anyone editing the prompt vocabulary (workspace tags, project subjects) can
 * dispatch this event after a successful write so token consumers (the
 * MarkerTextarea autocomplete, PromptPanel's Slot Picker) refetch.
 */
export const PROMPT_TOKENS_CHANGED_EVENT = 'fuk-prompt-tokens-changed';
export function notifyPromptTokensChanged() {
  window.dispatchEvent(new CustomEvent(PROMPT_TOKENS_CHANGED_EVENT));
}

export async function fetchPromptTokens({ model, activeLoras } = {}) {
  const params = new URLSearchParams();
  if (model) params.set('model', model);
  if (activeLoras?.length) params.set('active_loras', activeLoras.join(','));
  const qs = params.toString();
  const res = await fetch(`${API_URL}/prompt/tokens${qs ? '?' + qs : ''}`);
  if (!res.ok) throw new Error(`Failed to load prompt tokens: ${res.status}`);
  return res.json();
}

export async function expandPrompt({ text, model, activeLoras, styleChips, mode, intent }) {
  const res = await fetch(`${API_URL}/prompt/expand`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text ?? '',
      model: model || null,
      active_loras: activeLoras || [],
      style_chips: styleChips || [],
      mode: mode || 'image',
      intent: intent || null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Expand failed: ${res.statusText}`);
  }
  return res.json();
}

// ----------------------------------------------------------------------------
// Workspace tags: `#marker` vocabulary shared across all projects.
// Project-level subjects (the storyboard) shadow these on marker collision.
// ----------------------------------------------------------------------------

async function _jsonOrThrow(res, fallback) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || fallback);
  }
  return res.json();
}

export async function fetchTags() {
  const res = await fetch(`${API_URL}/tags`);
  return _jsonOrThrow(res, 'Failed to load tags');
}

export async function fetchTagCategories() {
  const res = await fetch(`${API_URL}/tags/categories`);
  return _jsonOrThrow(res, 'Failed to load tag categories');
}

export async function createTag({ name, value, category }) {
  const res = await fetch(`${API_URL}/tags`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, value, category: category || null }),
  });
  const data = await _jsonOrThrow(res, 'Failed to create tag');
  notifyPromptTokensChanged();
  return data;
}

export async function updateTag(id, { name, value, category }) {
  const res = await fetch(`${API_URL}/tags/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, value, category: category || null }),
  });
  const data = await _jsonOrThrow(res, 'Failed to update tag');
  notifyPromptTokensChanged();
  return data;
}

export async function deleteTag(id) {
  const res = await fetch(`${API_URL}/tags/${id}`, { method: 'DELETE' });
  const data = await _jsonOrThrow(res, 'Failed to delete tag');
  notifyPromptTokensChanged();
  return data;
}

export async function fetchTagHistory(id) {
  const res = await fetch(`${API_URL}/tags/${id}/history`);
  return _jsonOrThrow(res, 'Failed to load tag history');
}

export async function fetchDeletedTags() {
  const res = await fetch(`${API_URL}/tags/deleted`);
  return _jsonOrThrow(res, 'Failed to load deleted tags');
}

export async function restoreTagVersion(id, vid) {
  const res = await fetch(`${API_URL}/tags/${id}/history/${vid}/restore`, { method: 'POST' });
  const data = await _jsonOrThrow(res, 'Failed to restore tag');
  notifyPromptTokensChanged();
  return data;
}

/**
 * Copy (or move) workspace tags into the active project's storyboard.
 * Returns `{added, replaced, unchanged, conflicts, moved}`; conflicts are
 * project markers that already exist with a different value, left alone
 * unless `overwrite` is set.
 */
export async function pushTagsToStoryboard(ids, { move = false, overwrite = false } = {}) {
  const res = await fetch(`${API_URL}/tags/push-to-storyboard`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids, move, overwrite }),
  });
  const data = await _jsonOrThrow(res, 'Failed to push tags to storyboard');
  notifyPromptTokensChanged();
  window.dispatchEvent(new CustomEvent('fuk-storyboard-changed'));
  return data;
}

export async function compilePrompt({ text, model, activeLoras, styleChips, styleLabel }) {
  const res = await fetch(`${API_URL}/prompt/compile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text ?? '',
      model: model || null,
      active_loras: activeLoras || [],
      style_chips: styleChips || [],
      style_label: styleLabel || 'Style',
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Compile failed: ${res.statusText}`);
  }
  return res.json();
}
