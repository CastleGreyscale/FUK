/**
 * Describe & Tag — Utilities sub-tab
 *
 * Two related tools in one panel:
 *   1. Image describe — vision LLM produces a cinematographer-style description.
 *   2. Prompt tags — save a description fragment as a named shorthand so it
 *      can be re-injected into prompts later (storyboard / prompt construction).
 *      The tag editor is the shared TagManager, also available as a popup
 *      from any tab (TagManagerModal).
 *
 * The description is shown in a textarea so the user can edit before selecting
 * a span to promote into a tag. Selection is captured via the textarea's
 * native selectionStart/selectionEnd — works without contentEditable hacks.
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { FolderOpen, X } from '../components/Icons';
import TagManager from '../components/TagManager';

const API_URL = '/api';

const VIDEO_EXTS = ['.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v'];

// Quick-focus chips. Clicking one appends its phrase to the focus field (or
// removes it if already present), letting the description concentrate on that
// aspect. The field stays free-text — type anything in addition to these.
const FOCUS_PRESETS = [
  { label: 'Subject', phrase: 'the main subject' },
  { label: 'Wardrobe', phrase: 'wardrobe and clothing' },
  { label: 'Lighting', phrase: 'lighting' },
  { label: 'Setting', phrase: 'the setting and environment' },
  { label: 'Camera', phrase: 'camera angle, lens, and framing' },
  { label: 'Color', phrase: 'color palette' },
];

// Split the free-text focus into discrete comma-separated pieces (trimmed).
function focusPieces(focus) {
  return focus.split(',').map(s => s.trim()).filter(Boolean);
}

function buildImageUrl(path) {
  if (!path) return '';
  if (path.startsWith('/')) return `/api/project/files${path}`;
  return `/${path}`;
}

function isVideoPath(path) {
  if (!path) return false;
  const lower = path.toLowerCase();
  return VIDEO_EXTS.some(ext => lower.endsWith(ext));
}

export default function ImageDescribeTool({ project }) {
  const [imagePath, setImagePath] = useState(null);
  const [browsing, setBrowsing] = useState(false);
  const [running, setRunning] = useState(false);
  const [focus, setFocus] = useState('');
  const [description, setDescription] = useState('');
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [copied, setCopied] = useState(false);
  const [selectionError, setSelectionError] = useState(null);

  const descRef = useRef(null);
  const tagManagerRef = useRef(null);

  // ---- LLM health ---------------------------------------------------------

  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/llm/health`);
      const data = await res.json();
      setHealth(data);
    } catch (e) {
      setHealth({ available: false, error: e.message });
    }
  }, []);

  useEffect(() => { checkHealth(); }, [checkHealth]);

  // ---- Selection → tag draft ---------------------------------------------

  const captureSelection = useCallback(() => {
    const el = descRef.current;
    if (!el) return '';
    const { selectionStart, selectionEnd } = el;
    if (selectionStart == null || selectionEnd == null) return '';
    return el.value.slice(selectionStart, selectionEnd).trim();
  }, []);

  const handleUseSelection = useCallback(() => {
    const sel = captureSelection();
    if (!sel) {
      setSelectionError('Select some text in the description first.');
      return;
    }
    setSelectionError(null);
    tagManagerRef.current?.startDraft(sel);
  }, [captureSelection]);

  const handleUseFullDescription = useCallback(() => {
    if (!description.trim()) return;
    setSelectionError(null);
    tagManagerRef.current?.startDraft(description.trim());
  }, [description]);

  // ---- Browse / describe / copy ------------------------------------------

  const handleBrowse = useCallback(async () => {
    if (browsing || running) return;
    setBrowsing(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/browser/open`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: 'Select Image or Video to Describe',
          multiple: false,
          detect_sequences: false,
          filter: 'all',
        }),
      });
      const data = await res.json();
      if (data.success && data.files?.length > 0) {
        setImagePath(data.files[0].path);
        setDescription('');
      } else if (data.error) {
        throw new Error(data.error);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBrowsing(false);
    }
  }, [browsing, running]);

  const toggleFocusPreset = useCallback((phrase) => {
    setFocus(prev => {
      const pieces = focusPieces(prev);
      const idx = pieces.findIndex(p => p.toLowerCase() === phrase.toLowerCase());
      if (idx >= 0) pieces.splice(idx, 1);
      else pieces.push(phrase);
      return pieces.join(', ');
    });
  }, []);

  const handleDescribe = useCallback(async () => {
    if (!imagePath || running) return;
    setRunning(true);
    setError(null);
    setDescription('');
    try {
      const res = await fetch(`${API_URL}/llm/describe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_path: imagePath, focus: focus.trim() || null }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || `Request failed: ${res.status}`);
      setDescription(data.description || '');
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  }, [imagePath, running, focus]);

  const handleCopy = useCallback(async () => {
    if (!description) return;
    try {
      await navigator.clipboard.writeText(description);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      setError(`Copy failed: ${e.message}`);
    }
  }, [description]);

  const handleClear = useCallback(() => {
    setImagePath(null);
    setDescription('');
    setError(null);
  }, []);

  // ---- Derived ------------------------------------------------------------

  const healthOk = health?.available && health?.model_present;
  const healthMsg = !health ? 'Checking Ollama…'
    : !health.available ? `Ollama unavailable: ${health.error || 'no response'}`
    : !health.model_present ? `Model ${health.model} not pulled. Run: ollama pull ${health.model}`
    : `Ready · ${health.model}`;

  // ---- Render -------------------------------------------------------------

  return (
    <div className="image-describe-tool">

      <div className="image-describe-header">
        <div>
          <h3 className="image-describe-title">Describe &amp; Tag</h3>
          <p className="image-describe-subtitle">
            Describe an image or short video clip, then save fragments as prompt-shorthand tags.
          </p>
        </div>
        <div className={`image-describe-health ${healthOk ? 'image-describe-health--ok' : 'image-describe-health--bad'}`}>
          <span className="image-describe-health-dot" />
          <span>{healthMsg}</span>
          <button className="fuk-btn fuk-btn-secondary image-describe-recheck" onClick={checkHealth}>
            Recheck
          </button>
        </div>
      </div>

      <div className="image-describe-body">

        {/* Image / video / describe */}
        <div className="fuk-card image-describe-card">
          <div className="image-describe-card-header">
            <span className="fuk-label">{isVideoPath(imagePath) ? 'Video' : 'Image'}</span>
            {imagePath && (
              <button className="image-describe-clear" onClick={handleClear} title="Clear">
                <X />
              </button>
            )}
          </div>

          {!imagePath ? (
            <div
              className={`fuk-dropzone ${browsing ? 'fuk-dropzone--disabled' : ''}`}
              onClick={handleBrowse}
            >
              <div className="fuk-dropzone-content">
                {browsing ? (
                  <>
                    <div className="fuk-dropzone-spinner" />
                    <span className="fuk-dropzone-text">Loading…</span>
                  </>
                ) : (
                  <>
                    <FolderOpen className="fuk-dropzone-icon" />
                    <span className="fuk-dropzone-text">Click to browse for image or video</span>
                    <span className="fuk-dropzone-hint">
                      Videos are sampled into 6 frames for description
                    </span>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="image-describe-preview">
              {isVideoPath(imagePath) ? (
                <video
                  src={buildImageUrl(imagePath)}
                  controls
                  muted
                  preload="metadata"
                  className="image-describe-video"
                />
              ) : (
                <img src={buildImageUrl(imagePath)} alt="Selected" />
              )}
              <div className="image-describe-path" title={imagePath}>{imagePath}</div>
              <div className="image-describe-preview-actions">
                <button
                  className="fuk-btn fuk-btn-secondary image-describe-mini"
                  onClick={handleBrowse}
                  disabled={browsing || running}
                >
                  {browsing ? 'Loading…' : 'Change image'}
                </button>
                <button
                  className="fuk-btn fuk-btn-secondary image-describe-mini"
                  onClick={handleClear}
                  disabled={running}
                >
                  Remove
                </button>
              </div>
            </div>
          )}

          <div className="image-describe-focus">
            <span className="fuk-label">
              Focus on
              <span className="image-describe-focus-hint">
                optional — concentrates the description on this; everything else is reduced to a brief aside
              </span>
            </span>
            <div className="image-describe-focus-row">
              <input
                type="text"
                className="fuk-input image-describe-focus-input"
                placeholder="e.g. the creature's scaled hide and posture"
                value={focus}
                onChange={(e) => setFocus(e.target.value)}
              />
              {focus && (
                <button
                  className="image-describe-focus-clear"
                  onClick={() => setFocus('')}
                  title="Clear focus"
                >
                  <X />
                </button>
              )}
            </div>
            <div className="image-describe-focus-presets">
              {FOCUS_PRESETS.map(({ label, phrase }) => {
                const active = focusPieces(focus).some(
                  p => p.toLowerCase() === phrase.toLowerCase()
                );
                return (
                  <button
                    key={label}
                    type="button"
                    className={`image-describe-focus-chip ${active ? 'image-describe-focus-chip--active' : ''}`}
                    onClick={() => toggleFocusPreset(phrase)}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>

          <button
            className="fuk-btn fuk-btn-primary fuk-btn-full image-describe-run"
            onClick={handleDescribe}
            disabled={!imagePath || running || !healthOk}
          >
            {running
              ? (isVideoPath(imagePath) ? 'Sampling frames & describing…' : 'Describing…')
              : (isVideoPath(imagePath) ? 'Describe Video' : 'Describe Image')}
          </button>
        </div>

        {/* Description + selection-to-tag */}
        <div className="fuk-card image-describe-card image-describe-result-card">
          <div className="image-describe-card-header">
            <span className="fuk-label">Description</span>
            <div className="image-describe-result-actions">
              {description && (
                <button className="fuk-btn fuk-btn-secondary image-describe-mini" onClick={handleUseSelection}>
                  Tag selection
                </button>
              )}
              {description && (
                <button className="fuk-btn fuk-btn-secondary image-describe-mini" onClick={handleUseFullDescription}>
                  Tag all
                </button>
              )}
              {description && (
                <button className="fuk-btn fuk-btn-secondary image-describe-mini" onClick={handleCopy}>
                  {copied ? 'Copied' : 'Copy'}
                </button>
              )}
            </div>
          </div>

          {error ? (
            <div className="image-describe-error">{error}</div>
          ) : running ? (
            <div className="image-describe-placeholder">Generating description…</div>
          ) : (
            <textarea
              ref={descRef}
              className="fuk-input image-describe-textarea"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Description will appear here. Select any span and click 'Tag selection' to save it as a prompt shorthand."
              spellCheck={false}
            />
          )}
        </div>

      </div>

      {/* Tag manager — same component as the popup (TagManagerModal) */}
      <div className="fuk-card image-describe-tag-panel">
        <div className="image-describe-tag-panel-header">
          <span className="fuk-label">Prompt Tags</span>
          {selectionError && <span className="image-describe-error">{selectionError}</span>}
        </div>
        <TagManager ref={tagManagerRef} project={project} />
      </div>

    </div>
  );
}
