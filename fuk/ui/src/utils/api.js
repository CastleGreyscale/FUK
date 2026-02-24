/**
 * FUK API Client
 */

import { API_URL } from './constants';

/**
 * Fetch config from backend
 */
export async function fetchConfig() {
  const [defaultsRes, modelsRes] = await Promise.all([
    fetch(`${API_URL}/config/defaults`),
    fetch(`${API_URL}/config/models`)
  ]);
  
  if (!defaultsRes.ok || !modelsRes.ok) {
    throw new Error('Failed to load config');
  }
  
  const defaults = await defaultsRes.json();
  const models = await modelsRes.json();
  
  return { defaults, models };
}

/**
 * Start image generation
 */
export async function startImageGeneration(payload) {
  const res = await fetch(`${API_URL}/generate/image`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  
  if (!res.ok) {
    throw new Error(`Generation failed: ${res.statusText}`);
  }
  
  return res.json();
}

/**
 * Start video generation
 */
export async function startVideoGeneration(payload) {
  const res = await fetch(`${API_URL}/generate/video`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  
  if (!res.ok) {
    throw new Error(`Generation failed: ${res.statusText}`);
  }
  
  return res.json();
}

/**
 * Start a generic background task (preprocess, upscale, interpolate, layers, export)
 * Returns { generation_id, status } — poll via SSE at /api/progress/{generation_id}
 *
 * @param {string} taskType - One of: preprocess, preprocess_video, upscale, upscale_video,
 *                            interpolate, layers, layers_video, export_exr, export_exr_sequence
 * @param {object} payload  - Task-specific parameters (same shape as the original endpoint body)
 */
export async function startTask(taskType, payload) {
  const res = await fetch(`${API_URL}/task/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_type: taskType, payload })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Task start failed: ${res.statusText}`);
  }

  return res.json();
}

/**
 * Cancel a generation
 */
export async function cancelGeneration(generationId) {
  const res = await fetch(`${API_URL}/cancel/${generationId}`, {
    method: 'POST'
  });
  
  return res.json();
}

/**
 * Create SSE connection for progress updates
 */
export function createProgressStream(generationId, onProgress, onComplete, onError) {
  const url = `${API_URL}/progress/${generationId}`;
  console.log('[SSE] Connecting to:', url);
  
  const eventSource = new EventSource(url);
  
  eventSource.onopen = () => {
    console.log('[SSE] Connection opened');
  };
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      if (data.status === 'complete') {
        console.log('[SSE] Status: complete');
        onComplete(data);
        eventSource.close();
      } else if (data.status === 'failed') {
        console.log('[SSE] Status: failed', data.error);
        onError(data.error || 'Unknown error');
        eventSource.close();
      } else if (data.status === 'cancelled') {
        console.log('[SSE] Status: cancelled');
        onError('Generation cancelled');
        eventSource.close();
      } else {
        onProgress(data);
      }
    } catch (e) {
      console.error('[SSE] Failed to parse message:', e, event.data);
    }
  };
  
  eventSource.onerror = (error) => {
    //console.error('[SSE] Connection error:', error);
    // Don't close on error immediately - SSE will try to reconnect
    // Only report error if readyState is CLOSED (2)
    if (eventSource.readyState === EventSource.CLOSED) {
      onError('Connection lost');
    }
  };
  
  return eventSource;
}

/**
 * Get generation history
 */
export async function getImageHistory(limit = 20) {
  const res = await fetch(`${API_URL}/history/images?limit=${limit}`);
  return res.json();
}

export async function getVideoHistory(limit = 20) {
  const res = await fetch(`${API_URL}/history/videos?limit=${limit}`);
  return res.json();
}