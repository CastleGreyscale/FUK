import { API_URL } from './constants';

export async function fetchPromptTokens({ model, activeLoras } = {}) {
  const params = new URLSearchParams();
  if (model) params.set('model', model);
  if (activeLoras?.length) params.set('active_loras', activeLoras.join(','));
  const qs = params.toString();
  const res = await fetch(`${API_URL}/prompt/tokens${qs ? '?' + qs : ''}`);
  if (!res.ok) throw new Error(`Failed to load prompt tokens: ${res.status}`);
  return res.json();
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
