import { API_URL } from './constants';

async function jsonOrThrow(res, fallback) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || fallback);
  }
  return res.json();
}

export async function fetchManifest() {
  const res = await fetch(`${API_URL}/storyboard`);
  return jsonOrThrow(res, 'Failed to load storyboard');
}

export async function saveManifest(manifest) {
  const res = await fetch(`${API_URL}/storyboard`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(manifest || {}),
  });
  return jsonOrThrow(res, 'Failed to save storyboard');
}

export async function saveSpecs(specs) {
  const res = await fetch(`${API_URL}/storyboard/specs`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(specs || {}),
  });
  return jsonOrThrow(res, 'Failed to save specs');
}

export async function createSubject(payload) {
  const res = await fetch(`${API_URL}/storyboard/globals/subjects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow(res, 'Failed to create subject');
}

export async function updateSubject(id, payload) {
  const res = await fetch(`${API_URL}/storyboard/globals/subjects/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return jsonOrThrow(res, 'Failed to update subject');
}

export async function deleteSubject(id) {
  const res = await fetch(`${API_URL}/storyboard/globals/subjects/${id}`, {
    method: 'DELETE',
  });
  return jsonOrThrow(res, 'Failed to delete subject');
}

export async function saveMood(mood) {
  const res = await fetch(`${API_URL}/storyboard/globals/mood`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mood: mood || '' }),
  });
  return jsonOrThrow(res, 'Failed to save mood');
}

export async function upsertPanel(shotId, patch) {
  const res = await fetch(`${API_URL}/storyboard/panels/${shotId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch || {}),
  });
  return jsonOrThrow(res, 'Failed to save panel');
}

export async function deletePanel(shotId) {
  const res = await fetch(`${API_URL}/storyboard/panels/${shotId}`, {
    method: 'DELETE',
  });
  return jsonOrThrow(res, 'Failed to delete panel');
}

export async function reorderSequence(sequence) {
  const res = await fetch(`${API_URL}/storyboard/sequence`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sequence }),
  });
  return jsonOrThrow(res, 'Failed to reorder sequence');
}

export async function sendPanelToImage(shotId) {
  const res = await fetch(`${API_URL}/storyboard/panels/${shotId}/send/image`, { method: 'POST' });
  return jsonOrThrow(res, 'Failed to send to Image tab');
}

export async function sendPanelToVideo(shotId) {
  const res = await fetch(`${API_URL}/storyboard/panels/${shotId}/send/video`, { method: 'POST' });
  return jsonOrThrow(res, 'Failed to send to Video tab');
}

export async function resolvePromptPreview({ text, model, activeLoras, applyMood }) {
  const res = await fetch(`${API_URL}/prompt/resolve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text || '',
      model: model || null,
      active_loras: activeLoras || [],
      apply_mood: applyMood !== false,
    }),
  });
  return jsonOrThrow(res, 'Failed to resolve prompt');
}
