/**
 * LayerStackPanel
 *
 * Non-destructive latent layer editing UI.
 *
 * Anatomy:
 *   ┌─────────────────────────────────────┐
 *   │  LAYER STACK          [Flatten] [+] │
 *   ├─────────────────────────────────────┤
 *   │  ● orange cat    v1 ▾  [↺] [✕]     │
 *   │    └─ v0  "orange cat orig"  [load] │
 *   │    └─ v1  "orange cat warm"  [load] │
 *   │  ○ lamp edit     v0 ▾  [↺] [✕]     │
 *   │  ● hand          v0 ▾  [↺] [✕]     │
 *   ├─────────────────────────────────────┤
 *   │  [preview image]                    │
 *   └─────────────────────────────────────┘
 */

import { useState, useEffect, useCallback } from 'react';
import {
  getStack,
  addLayer,
  rerunLayer,
  toggleLayer,
  switchVersion,
  removeLayer,
  flattenStack,
} from '../utils/layerStackApi';
import '../styles/layer-stack.css';

// ---------------------------------------------------------------------------
// LayerStackPanel
// ---------------------------------------------------------------------------

export default function LayerStackPanel({ stackId, currentImagePath, onFlattenComplete }) {
  const [stack, setStack]               = useState(null);
  const [previewUrl, setPreviewUrl]     = useState(null);
  const [expandedLayers, setExpanded]   = useState({});   // layer_id → bool
  const [loading, setLoading]           = useState(false);
  const [activeOp, setActiveOp]         = useState(null); // human-readable op label
  const [error, setError]               = useState(null);

  // Add-layer form state
  const [showAddForm, setShowAddForm]   = useState(false);
  const [addForm, setAddForm]           = useState({
    name: '',
    prompt: '',
    model: 'qwen_edit',
    seed: '',
  });

  // ---------------------------------------------------------------------------
  // Load stack on mount / stackId change
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!stackId) return;
    loadStack();
  }, [stackId]);

  async function loadStack() {
    try {
      const data = await getStack(stackId);
      setStack(data.stack);
    } catch (e) {
      setError(`Failed to load stack: ${e.message}`);
    }
  }

  // ---------------------------------------------------------------------------
  // Operations
  // ---------------------------------------------------------------------------

  const withLoading = useCallback(async (label, fn) => {
    setLoading(true);
    setActiveOp(label);
    setError(null);
    try {
      const result = await fn();
      if (result?.stack) setStack(result.stack);
      return result;
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setActiveOp(null);
    }
  }, []);

  async function handleToggle(layerId, currentEnabled) {
    await withLoading('Toggling...', () =>
      toggleLayer(stackId, layerId, !currentEnabled)
    );
  }

  async function handleSwitchVersion(layerId, v) {
    await withLoading('Switching version...', () =>
      switchVersion(stackId, layerId, v)
    );
  }

  async function handleRerun(layer) {
    const activeV = layer.versions[layer.active_version];
    await withLoading(`Re-running "${layer.name}"...`, () =>
      rerunLayer(stackId, {
        layer_id: layer.layer_id,
        version_id: layer.active_version,
        // Pre-fill with stored params; user can override via a future modal
        prompt: activeV.prompt,
        seed: activeV.seed,
      })
    );
  }

  async function handleRemove(layerId, name) {
    if (!confirm(`Remove layer "${name}"? The latent files stay on disk.`)) return;
    await withLoading('Removing...', () =>
      removeLayer(stackId, layerId)
    );
  }

  async function handleFlatten() {
    const result = await withLoading('Flattening layers...', () =>
      flattenStack(stackId)
    );
    if (result?.preview_url) {
      setPreviewUrl(result.preview_url + '?t=' + Date.now()); // cache-bust
      onFlattenComplete?.(result.preview_url);
    }
  }

  async function handleAddLayer(e) {
    e.preventDefault();
    if (!addForm.name || !addForm.prompt) return;

    const params = {
      name:          addForm.name,
      prompt:        addForm.prompt,
      model:         addForm.model,
      control_image: currentImagePath || null,
      seed:          addForm.seed ? parseInt(addForm.seed, 10) : null,
    };

    const result = await withLoading(`Adding layer "${addForm.name}"...`, () =>
      addLayer(stackId, params)
    );

    if (result?.stack) {
      setAddForm({ name: '', prompt: '', model: 'qwen_edit', seed: '' });
      setShowAddForm(false);
    }
  }

  function toggleExpand(layerId) {
    setExpanded(prev => ({ ...prev, [layerId]: !prev[layerId] }));
  }

  // ---------------------------------------------------------------------------
  // Render helpers
  // ---------------------------------------------------------------------------

  function renderVersion(layer, version, isActive) {
    return (
      <div
        key={version.v}
        className={`ls-version ${isActive ? 'ls-version--active' : ''}`}
      >
        <span className="ls-version-label">v{version.v}</span>
        <span className="ls-version-prompt" title={version.prompt}>
          {version.prompt?.slice(0, 48) || '—'}
        </span>
        {version.seed != null && (
          <span className="ls-version-seed">#{version.seed}</span>
        )}
        {!isActive && (
          <button
            className="ls-btn ls-btn--ghost ls-btn--xs"
            onClick={() => handleSwitchVersion(layer.layer_id, version.v)}
            disabled={loading}
            title="Switch to this version"
          >
            load
          </button>
        )}
      </div>
    );
  }

  function renderLayer(layer) {
    const isExpanded = !!expandedLayers[layer.layer_id];
    const activeVersion = layer.versions[layer.active_version];

    return (
      <div
        key={layer.layer_id}
        className={`ls-layer ${layer.enabled ? 'ls-layer--on' : 'ls-layer--off'}`}
      >
        {/* Layer header row */}
        <div className="ls-layer-row">
          {/* Toggle */}
          <button
            className={`ls-eye ${layer.enabled ? 'ls-eye--on' : 'ls-eye--off'}`}
            onClick={() => handleToggle(layer.layer_id, layer.enabled)}
            disabled={loading}
            title={layer.enabled ? 'Disable layer' : 'Enable layer'}
          >
            {layer.enabled ? '●' : '○'}
          </button>

          {/* Name */}
          <span className="ls-layer-name">{layer.name}</span>

          {/* Active version badge + expand */}
          <button
            className="ls-version-badge"
            onClick={() => toggleExpand(layer.layer_id)}
            title="Show version history"
          >
            v{layer.active_version}
            <span className="ls-chevron">{isExpanded ? '▴' : '▾'}</span>
          </button>

          {/* Re-run */}
          <button
            className="ls-btn ls-btn--ghost ls-btn--icon"
            onClick={() => handleRerun(layer)}
            disabled={loading}
            title="Re-run this layer (adds new version)"
          >
            ↺
          </button>

          {/* Remove */}
          <button
            className="ls-btn ls-btn--ghost ls-btn--icon ls-btn--danger"
            onClick={() => handleRemove(layer.layer_id, layer.name)}
            disabled={loading}
            title="Remove layer"
          >
            ✕
          </button>
        </div>

        {/* Active version prompt preview */}
        {activeVersion && (
          <div className="ls-layer-prompt-preview">
            {activeVersion.prompt?.slice(0, 72) || '—'}
          </div>
        )}

        {/* Version history (collapsed by default) */}
        {isExpanded && (
          <div className="ls-versions">
            {layer.versions.map(v =>
              renderVersion(layer, v, v.v === layer.active_version)
            )}
          </div>
        )}
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Empty state — no stackId
  // ---------------------------------------------------------------------------

  if (!stackId) {
    return (
      <div className="ls-panel ls-panel--empty">
        <p className="ls-empty-msg">
          Generate an image, then click <strong>New Layer Stack</strong> to start
          non-destructive editing.
        </p>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="ls-panel">
      {/* Header */}
      <div className="ls-header">
        <span className="ls-title">
          <span className="ls-title-dot" />
          LAYER STACK
        </span>
        <div className="ls-header-actions">
          <button
            className="ls-btn ls-btn--primary ls-btn--sm"
            onClick={handleFlatten}
            disabled={loading || !stack?.layers?.length}
            title="Decode all enabled layers to preview"
          >
            {loading && activeOp === 'Flattening layers...' ? 'Flattening…' : 'Flatten'}
          </button>
          <button
            className="ls-btn ls-btn--ghost ls-btn--sm"
            onClick={() => setShowAddForm(v => !v)}
            disabled={loading}
            title="Add a new edit layer"
          >
            {showAddForm ? '✕ Cancel' : '+ Layer'}
          </button>
        </div>
      </div>

      {/* Loading indicator */}
      {loading && (
        <div className="ls-loading-bar">
          <span className="ls-loading-label">{activeOp}</span>
          <div className="ls-loading-track"><div className="ls-loading-fill" /></div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="ls-error" onClick={() => setError(null)}>
          ⚠ {error}
        </div>
      )}

      {/* Add layer form */}
      {showAddForm && (
        <div className="ls-add-form">
          <div className="ls-form-row">
            <label className="ls-label">Name</label>
            <input
              className="ls-input"
              placeholder="e.g. orange cat"
              value={addForm.name}
              onChange={e => setAddForm(f => ({ ...f, name: e.target.value }))}
            />
          </div>
          <div className="ls-form-row">
            <label className="ls-label">Prompt</label>
            <textarea
              className="ls-input ls-textarea"
              placeholder="Describe the edit…"
              value={addForm.prompt}
              onChange={e => setAddForm(f => ({ ...f, prompt: e.target.value }))}
              rows={3}
            />
          </div>
          <div className="ls-form-row ls-form-row--inline">
            <div>
              <label className="ls-label">Model</label>
              <select
                className="ls-input ls-select"
                value={addForm.model}
                onChange={e => setAddForm(f => ({ ...f, model: e.target.value }))}
              >
                <option value="qwen_edit">qwen_edit</option>
                <option value="qwen_image_2512">qwen_image_2512</option>
              </select>
            </div>
            <div>
              <label className="ls-label">Seed</label>
              <input
                className="ls-input"
                type="number"
                placeholder="random"
                value={addForm.seed}
                onChange={e => setAddForm(f => ({ ...f, seed: e.target.value }))}
              />
            </div>
          </div>
          <button
            className="ls-btn ls-btn--primary ls-btn--full"
            onClick={handleAddLayer}
            disabled={loading || !addForm.name || !addForm.prompt}
          >
            {loading ? activeOp : 'Run Edit + Add Layer'}
          </button>
        </div>
      )}

      {/* Layer list */}
      <div className="ls-layers">
        {!stack?.layers?.length ? (
          <div className="ls-empty-layers">No layers yet. Add an edit to get started.</div>
        ) : (
          stack.layers.map(renderLayer)
        )}
      </div>

      {/* Flatten preview */}
      {previewUrl && (
        <div className="ls-preview">
          <div className="ls-preview-label">Flattened result</div>
          <img
            className="ls-preview-img"
            src={previewUrl}
            alt="Flattened layer stack"
          />
        </div>
      )}
    </div>
  );
}
