/**
 * 3D Reconstruct — Utilities sub-tab
 *
 * Images in, proxy geometry out. Two tiers: TRELLIS for a single image,
 * VGGT for a multi-view set. The first-class input is the LoRA Dataset
 * Builder's orbital views — synthetic multi-view → 3D is the workflow this
 * exists for.
 *
 * Runs its own Reconstruct button rather than the footer generation bar,
 * matching PreprocessTab, and reports through GenerationModal.
 *
 * See docs/3D_RECONSTRUCTION_SYSTEM.md
 */

import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Folder, FolderOpen, Download, AlertCircle, CheckCircle, Layers, Info, Trash2,
} from '../components/Icons';
import GenerationModal from '../components/GenerationModal';
import MeshViewer from '../components/MeshViewer';
import { useGeneration } from '../hooks/useGeneration';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { startTask } from '../utils/api';
import { buildImageUrl } from '../utils/constants';

const API_URL = '/api';

const DISPLAY_MODES = [
  { key: 'solid',     label: 'Solid' },
  { key: 'wireframe', label: 'Wireframe' },
  { key: 'points',    label: 'Point Cloud' },
];

const DEFAULT_SETTINGS = {
  model: 'vggt',
  export_glb: true,
  export_ply: false,
  export_obj: false,
  // VGGT
  conf_percentile: 50,
  poisson_depth: 9,
  build_mesh: true,
  max_points: 500000,
  remove_background: 'auto',
  bg_luma: 0.62,
  // TRELLIS
  seed: 42,
  steps: 12,
  cfg_strength: 7.5,
  simplify: 0.95,
  texture_size: 1024,
  fill_holes: true,
};

// Filmstrip column width. Wide enough for two thumbnail columns by default.
const DEFAULT_STRIP_WIDTH = 190;
const MIN_STRIP_WIDTH = 96;
const MAX_STRIP_WIDTH = 620;

// Minimum thumbnail track width. The grid uses auto-fit, so this is a floor
// rather than a fixed size — thumbnails grow to share whatever width is left
// once the strip has fit as many columns as it can.
const DEFAULT_THUMB_SIZE = 74;

// Mirrors the backend guidance: angular spread matters more than raw count,
// but below six views VGGT has too little to work with for a clean surface.
function coverageAdvice(count) {
  if (count === 0)  return null;
  if (count < 4)    return { level: 'error',   text: `${count} image${count === 1 ? '' : 's'} — too few for multi-view. Use TRELLIS for a single image.` };
  if (count < 6)    return { level: 'warning', text: `${count} images — rough proxy only. 8–12 recommended for a clean mesh.` };
  if (count < 8)    return { level: 'warning', text: `${count} images — usable, but 8–12 gives noticeably better geometry.` };
  if (count < 16)   return { level: 'ok',      text: `${count} images — solid object reconstruction.` };
  return { level: 'ok', text: `${count} images — full coverage. Best results.` };
}

export default function ThreeDReconstruct({ config, project }) {
  const [models, setModels] = useState({});
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [inputImages, setInputImages] = useState([]);
  const [inputLabel, setInputLabel] = useState('');
  // Paths the user has switched off in the filmstrip. Tracked as the excluded
  // set rather than the included one so a freshly loaded folder is all-on.
  const [excluded, setExcluded] = useState(() => new Set());
  const [result, setResult] = useState(null);
  const [displayMode, setDisplayMode] = useState('solid');
  const [viewerFormat, setViewerFormat] = useState('glb');
  const [localError, setLocalError] = useState(null);
  const [showDatasets, setShowDatasets] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [settings, setSettings] = useLocalStorage('fuk_threed_settings', DEFAULT_SETTINGS);
  const update = (patch) => setSettings((prev) => ({ ...prev, ...patch }));

  // Splitter between the filmstrip and the viewport. Persisted so the layout
  // survives a tab switch — the viewer remounts, the chosen width shouldn't
  // reset with it.
  const [stripWidth, setStripWidth] = useLocalStorage('fuk_threed_strip_width', DEFAULT_STRIP_WIDTH);
  const [thumbSize, setThumbSize] = useLocalStorage('fuk_threed_thumb_size', DEFAULT_THUMB_SIZE);
  const [dragging, setDragging] = useState(false);
  const dragRef = useRef(null);

  const startDrag = useCallback((event) => {
    event.preventDefault();
    dragRef.current = { startX: event.clientX, startWidth: stripWidth };
    setDragging(true);
  }, [stripWidth]);

  useEffect(() => {
    if (!dragging) return;

    const onMove = (event) => {
      const { startX, startWidth } = dragRef.current || {};
      if (startX === undefined) return;
      const next = startWidth + (event.clientX - startX);
      setStripWidth(Math.min(MAX_STRIP_WIDTH, Math.max(MIN_STRIP_WIDTH, next)));
    };
    const onUp = () => setDragging(false);

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    // Without these the drag keeps selecting text and flickering the cursor
    // as the pointer crosses the viewport and the thumbnails.
    const previousCursor = document.body.style.cursor;
    const previousSelect = document.body.style.userSelect;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousSelect;
    };
  }, [dragging, setStripWidth]);

  const {
    generating, progress, result: genResult, error: genError,
    elapsedSeconds, consoleLog, showModal, startGeneration, cancel, closeModal,
  } = useGeneration();

  const error = genError || localError;
  const activeModel = models[settings.model];
  const isSingleImage = activeModel?.type === 'single_image';

  // --- Load model availability + dataset list -------------------------------
  useEffect(() => {
    fetch(`${API_URL}/threed/models`)
      .then((r) => r.json())
      .then((data) => {
        setModels(data.models || {});
        setModelsLoaded(true);
      })
      .catch((err) => {
        console.error('[ThreeD] Failed to load models:', err);
        setModelsLoaded(true);
      });

    fetch(`${API_URL}/threed/datasets`)
      .then((r) => r.json())
      .then((data) => setDatasets(data.datasets || []))
      .catch((err) => console.error('[ThreeD] Failed to load datasets:', err));
  }, []);

  // --- Bridge task completion into local result state -----------------------
  useEffect(() => {
    if (!genResult || genResult.status !== 'complete' || !genResult.result) return;
    const data = genResult.result;
    setResult(data);
    // Prefer the mesh; fall back to the point cloud when meshing was skipped.
    setViewerFormat(data.urls?.glb ? 'glb' : (data.urls?.ply ? 'ply' : 'glb'));
    setDisplayMode(data.has_mesh ? 'solid' : 'points');
  }, [genResult]);

  // Only the enabled views are sent to the backend — the whole point of the
  // filmstrip is dropping views that poison the solve without reloading.
  const activeImages = useMemo(
    () => inputImages.filter((path) => !excluded.has(path)),
    [inputImages, excluded],
  );

  const coverage = useMemo(
    () => (isSingleImage ? null : coverageAdvice(activeImages.length)),
    [activeImages.length, isSingleImage],
  );

  const toggleImage = useCallback((path) => {
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const setAllActive = useCallback(() => setExcluded(new Set()), []);
  const setNoneActive = useCallback(
    () => setExcluded(new Set(inputImages)), [inputImages],
  );
  const invertActive = useCallback(() => {
    setExcluded((prev) => new Set(inputImages.filter((p) => !prev.has(p))));
  }, [inputImages]);

  const exportFormats = useMemo(() => {
    const formats = [];
    if (settings.export_glb) formats.push('glb');
    if (settings.export_ply) formats.push('ply');
    if (settings.export_obj) formats.push('obj');
    return formats;
  }, [settings.export_glb, settings.export_ply, settings.export_obj]);

  // --- Input pickers --------------------------------------------------------
  const handleBrowseFiles = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/browser/open`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: 'Select images for reconstruction',
          multiple: true,
          filter: 'images',
          detect_sequences: false,
        }),
      });
      const data = await res.json();
      if (data.success && data.files?.length) {
        const paths = data.files.map((f) => f.path);
        setInputImages(paths);
        setInputLabel(`${paths.length} file${paths.length === 1 ? '' : 's'} selected`);
        setExcluded(new Set());
        setLocalError(null);
      }
    } catch (err) {
      setLocalError(`File browser failed: ${err.message}`);
    }
  }, []);

  const handleBrowseFolder = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/browser/directory`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'Select a folder of images' }),
      });
      const data = await res.json();
      if (!data.success || !data.directory) return;

      // Expand to real paths rather than sending the directory as one entry —
      // the coverage warning is only meaningful with the true image count.
      const scanRes = await fetch(`${API_URL}/browser/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          directory: data.directory,
          recursive: true,
          detect_sequences: false,
        }),
      });
      const scan = await scanRes.json();
      const images = (scan.files || []).filter((f) => f.media_type === 'image');

      if (!images.length) {
        setLocalError(`No images found in ${data.directory}`);
        return;
      }
      setInputImages(images.map((f) => f.path));
      setInputLabel(`${data.directory} — ${images.length} images`);
      setExcluded(new Set());
      setLocalError(null);
    } catch (err) {
      setLocalError(`Folder browser failed: ${err.message}`);
    }
  }, []);

  const handlePickDataset = (dataset) => {
    setInputImages(dataset.images);
    setInputLabel(`${dataset.subject_name} — ${dataset.image_count} ${dataset.source} views`);
    setExcluded(new Set());
    setShowDatasets(false);
    setLocalError(null);
    // Orbital sets are inherently multi-view; switch tier if TRELLIS was active.
    if (isSingleImage && dataset.image_count > 1) update({ model: 'vggt' });
  };

  const handleClearInput = () => {
    setInputImages([]);
    setInputLabel('');
    setExcluded(new Set());
    setResult(null);
  };

  // --- Run ------------------------------------------------------------------
  const handleReconstruct = async () => {
    if (!activeImages.length) {
      setLocalError(
        inputImages.length
          ? 'Every view is switched off — enable at least one in the strip'
          : 'Select at least one input image',
      );
      return;
    }
    if (!exportFormats.length) {
      setLocalError('Select at least one export format');
      return;
    }
    if (activeModel && !activeModel.available) {
      setLocalError(
        `${activeModel.name} is not installed — missing: ${activeModel.missing.join(', ')}`
        + (activeModel.hint ? `. ${activeModel.hint}` : ''),
      );
      return;
    }

    setLocalError(null);
    setResult(null);

    try {
      const payload = {
        input_images: activeImages,
        model: settings.model,
        export_formats: exportFormats,
        conf_percentile: Number(settings.conf_percentile),
        poisson_depth: Number(settings.poisson_depth),
        build_mesh: !!settings.build_mesh,
        max_points: Number(settings.max_points),
        remove_background: settings.remove_background ?? 'auto',
        bg_luma: Number(settings.bg_luma ?? 0.62),
        seed: Number(settings.seed),
        steps: Number(settings.steps),
        cfg_strength: Number(settings.cfg_strength),
        simplify: Number(settings.simplify),
        texture_size: Number(settings.texture_size),
        fill_holes: settings.fill_holes !== false,
      };
      const data = await startTask('threed_reconstruct', payload);
      startGeneration(data.generation_id);
    } catch (err) {
      setLocalError(err.message);
    }
  };

  const viewerUrl = result?.urls?.[viewerFormat]
    ? buildImageUrl(result.urls[viewerFormat])
    : null;

  const canReconstruct =
    !generating && activeImages.length > 0 && exportFormats.length > 0;

  // ==========================================================================

  return (
    <div className="threed-tab">

      {/* ── Left: controls ── */}
      <div className="threed-controls">

        {/* Input */}
        <div className="fuk-card threed-card">
          <div className="threed-card-header">
            <span className="fuk-label">Input Images</span>
            {inputImages.length > 0 && (
              <button className="threed-clear-btn" onClick={handleClearInput} title="Clear selection">
                <Trash2 className="fuk-icon--sm" />
              </button>
            )}
          </div>

          <div className="threed-input-row">
            <button className="fuk-btn fuk-btn-secondary threed-btn-sm" onClick={handleBrowseFiles}>
              <FolderOpen /> Files
            </button>
            <button className="fuk-btn fuk-btn-secondary threed-btn-sm" onClick={handleBrowseFolder}>
              <Folder /> Folder
            </button>
            <button
              className={`fuk-btn threed-btn-sm ${showDatasets ? 'fuk-btn-primary' : 'fuk-btn-secondary'}`}
              onClick={() => setShowDatasets((v) => !v)}
              title="Use orbital views generated by the LoRA Dataset Builder"
            >
              <Layers /> Dataset Builder
            </button>
          </div>

          {showDatasets && (
            <div className="threed-dataset-list">
              {datasets.length === 0 ? (
                <div className="threed-dataset-empty">
                  No dataset builder output found. Generate orbital views in the
                  LoRA Dataset Builder first.
                </div>
              ) : (
                datasets.map((dataset) => (
                  <button
                    key={dataset.job_id}
                    className="threed-dataset-item"
                    onClick={() => handlePickDataset(dataset)}
                  >
                    <span className="threed-dataset-name">{dataset.subject_name}</span>
                    <span className="threed-dataset-meta">
                      {dataset.image_count} {dataset.source}
                      {dataset.subject_type ? ` · ${dataset.subject_type}` : ''}
                    </span>
                  </button>
                ))
              )}
            </div>
          )}

          <div className={`threed-input-summary ${inputImages.length ? 'threed-input-summary--set' : ''}`}>
            {inputLabel || 'No input selected'}
          </div>

          {coverage && (
            <div className={`threed-coverage threed-coverage--${coverage.level}`}>
              {coverage.text}
            </div>
          )}
        </div>

        {/* Model */}
        <div className="fuk-card threed-card">
          <span className="fuk-label">Model</span>
          <div className="threed-model-list">
            {!modelsLoaded && <div className="threed-dataset-empty">Loading models…</div>}
            {Object.values(models).map((model) => (
              <button
                key={model.key}
                className={`threed-model-option ${settings.model === model.key ? 'threed-model-option--active' : ''} ${!model.available ? 'threed-model-option--unavailable' : ''}`}
                onClick={() => update({ model: model.key })}
              >
                <div className="threed-model-head">
                  <span className="threed-model-name">{model.name}</span>
                  <span className="threed-model-tier">
                    {model.type === 'single_image' ? 'Single image' : 'Multi-view'}
                  </span>
                </div>
                <div className="threed-model-desc">{model.description}</div>
                <div className="threed-model-foot">
                  {model.vram_gb_estimate && <span>~{model.vram_gb_estimate}GB VRAM</span>}
                  {model.max_input_dim && <span>{model.max_input_dim}px inference</span>}
                  {!model.available && (
                    <span className="threed-model-missing">
                      not installed — {model.missing?.join(', ')}
                    </span>
                  )}
                </div>
              </button>
            ))}
          </div>

          {activeModel && !activeModel.available && activeModel.hint && (
            <div className="threed-hint">
              <Info className="fuk-icon--sm" />
              <code>{activeModel.hint}</code>
            </div>
          )}
        </div>

        {/* Export formats */}
        <div className="fuk-card threed-card">
          <span className="fuk-label">Export Formats</span>
          <div className="threed-format-row">
            <label>
              <input
                type="checkbox" className="fuk-checkbox"
                checked={settings.export_glb}
                onChange={(e) => update({ export_glb: e.target.checked })}
              />
              GLB <span className="threed-format-note">mesh</span>
            </label>
            <label>
              <input
                type="checkbox" className="fuk-checkbox"
                checked={settings.export_ply}
                onChange={(e) => update({ export_ply: e.target.checked })}
              />
              PLY <span className="threed-format-note">point cloud</span>
            </label>
            <label>
              <input
                type="checkbox" className="fuk-checkbox"
                checked={settings.export_obj}
                onChange={(e) => update({ export_obj: e.target.checked })}
              />
              OBJ <span className="threed-format-note">Nuke / Natron</span>
            </label>
          </div>
        </div>

        {/* Advanced */}
        <div className="fuk-card threed-card">
          <button
            className="threed-advanced-toggle"
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? '▾' : '▸'} Advanced — {isSingleImage ? 'TRELLIS' : 'VGGT'}
          </button>

          {showAdvanced && !isSingleImage && (
            <div className="threed-advanced">
              {/* VGGT reconstructs the whole scene, so a plain studio backdrop
                  comes back as invented geometry wrapped around the subject —
                  measured at 69% of the point cloud on a 64-view orbit. */}
              <label className="threed-field">
                <span>Background</span>
                <select
                  className="fuk-select"
                  value={settings.remove_background ?? 'auto'}
                  onChange={(e) => update({ remove_background: e.target.value })}
                >
                  <option value="auto">Remove studio backdrop (recommended)</option>
                  <option value="alpha">Use source alpha matte</option>
                  <option value="none">Keep everything (full scene)</option>
                </select>
                <span className="threed-field-note">
                  {settings.remove_background === 'none'
                    ? 'Correct for environments; on an object the backdrop distorts the mesh.'
                    : settings.remove_background === 'alpha'
                      ? 'Falls back to backdrop keying if the images carry no alpha.'
                      : 'Keys out bright, near-neutral pixels before meshing.'}
                </span>
              </label>

              {(settings.remove_background ?? 'auto') !== 'none' && (
                <label className="threed-field">
                  <span>Backdrop brightness cutoff: {Number(settings.bg_luma ?? 0.62).toFixed(2)}</span>
                  <input
                    type="range" className="fuk-slider" min="0.3" max="0.95" step="0.01"
                    value={settings.bg_luma ?? 0.62}
                    onChange={(e) => update({ bg_luma: Number(e.target.value) })}
                  />
                  <span className="threed-field-note">
                    Lower cuts more background — raise it if a pale subject is being eaten.
                  </span>
                </label>
              )}

              <label className="threed-field">
                <span>
                  Confidence cutoff: keep top {100 - settings.conf_percentile}%
                </span>
                <input
                  type="range" className="fuk-slider" min="0" max="90" step="5"
                  value={settings.conf_percentile}
                  onChange={(e) => update({ conf_percentile: Number(e.target.value) })}
                />
              </label>

              <label className="threed-field threed-field--row">
                <input
                  type="checkbox" className="fuk-checkbox"
                  checked={settings.build_mesh}
                  onChange={(e) => update({ build_mesh: e.target.checked })}
                />
                <span>Build mesh from point cloud (Poisson)</span>
              </label>

              <label className="threed-field">
                <span>Poisson depth: {settings.poisson_depth}</span>
                <input
                  type="range" className="fuk-slider" min="6" max="11" step="1"
                  value={settings.poisson_depth}
                  disabled={!settings.build_mesh}
                  onChange={(e) => update({ poisson_depth: Number(e.target.value) })}
                />
                <span className="threed-field-note">
                  Higher captures more detail and more noise.
                </span>
              </label>

              <label className="threed-field">
                <span>Max points: {Number(settings.max_points).toLocaleString()}</span>
                <input
                  type="range" className="fuk-slider" min="100000" max="2000000" step="100000"
                  value={settings.max_points}
                  onChange={(e) => update({ max_points: Number(e.target.value) })}
                />
              </label>
            </div>
          )}

          {showAdvanced && isSingleImage && (
            <div className="threed-advanced">
              <label className="threed-field">
                <span>Seed</span>
                <input
                  type="number" className="fuk-input" min="0"
                  value={settings.seed}
                  onChange={(e) => update({ seed: Number(e.target.value) })}
                />
              </label>
              <label className="threed-field">
                <span>Sampler steps: {settings.steps}</span>
                <input
                  type="range" className="fuk-slider" min="8" max="50" step="1"
                  value={settings.steps}
                  onChange={(e) => update({ steps: Number(e.target.value) })}
                />
              </label>
              <label className="threed-field">
                <span>Guidance: {settings.cfg_strength}</span>
                <input
                  type="range" className="fuk-slider" min="1" max="15" step="0.5"
                  value={settings.cfg_strength}
                  onChange={(e) => update({ cfg_strength: Number(e.target.value) })}
                />
              </label>
              <label className="threed-field">
                <span>Simplify: remove {Math.round(settings.simplify * 100)}% of faces</span>
                <input
                  type="range" className="fuk-slider" min="0" max="0.98" step="0.01"
                  value={settings.simplify}
                  onChange={(e) => update({ simplify: Number(e.target.value) })}
                />
              </label>
              {/* to_glb fills holes by rendering the mesh from 1000
                  viewpoints. On a dense SLAT mesh that stage, not sampling,
                  is what makes a run take minutes. */}
              <label className="threed-field threed-field--row">
                <input
                  type="checkbox" className="fuk-checkbox"
                  checked={settings.fill_holes !== false}
                  onChange={(e) => update({ fill_holes: e.target.checked })}
                />
                <span>Fill holes — slow on dense meshes</span>
              </label>

              <label className="threed-field">
                <span>Texture size</span>
                <select
                  className="fuk-select"
                  value={settings.texture_size}
                  onChange={(e) => update({ texture_size: Number(e.target.value) })}
                >
                  {[512, 1024, 2048].map((size) => (
                    <option key={size} value={size}>{size}px</option>
                  ))}
                </select>
              </label>
            </div>
          )}
        </div>

        {/* Run */}
        <button
          className="fuk-btn fuk-btn-primary fuk-btn-full threed-run-btn"
          onClick={handleReconstruct}
          disabled={!canReconstruct}
        >
          {generating
            ? 'Reconstructing…'
            : `Reconstruct${activeImages.length ? ` — ${activeImages.length} view${activeImages.length === 1 ? '' : 's'}` : ''}`}
        </button>

        {error && (
          <div className="threed-error">
            <AlertCircle className="fuk-icon--sm" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* ── Middle: vertical input filmstrip ──
          Toggle individual views in and out of the solve. Finding a working
          set is mostly a process of elimination: bad angles, inconsistent
          scale, and views Qwen rendered differently all drag the
          reconstruction down, and spotting them is far quicker than
          re-picking the input from scratch. */}
      {inputImages.length > 0 && (
        <div className="threed-strip" style={{ width: `${stripWidth}px` }}>
          <div className="threed-strip-header">
            <span className="threed-strip-title">
              Input Views
              <span className="threed-strip-count">
                {activeImages.length} / {inputImages.length}
              </span>
            </span>
            <div className="threed-strip-actions">
              <button
                className="fuk-btn fuk-btn-secondary threed-btn-sm"
                onClick={setAllActive}
                disabled={excluded.size === 0}
              >
                All
              </button>
              <button
                className="fuk-btn fuk-btn-secondary threed-btn-sm"
                onClick={setNoneActive}
                disabled={activeImages.length === 0}
              >
                None
              </button>
              <button
                className="fuk-btn fuk-btn-secondary threed-btn-sm"
                onClick={invertActive}
              >
                Invert
              </button>
            </div>
          </div>

          {isSingleImage && activeImages.length > 1 && (
            <div className="threed-strip-note">
              TRELLIS uses the first active view
            </div>
          )}

          <label className="threed-thumb-size">
            <span>Size</span>
            <input
              type="range" className="fuk-slider" min={48} max={260} step={4}
              value={thumbSize}
              onChange={(e) => setThumbSize(Number(e.target.value))}
            />
          </label>

          {/* auto-fit rather than auto-fill: empty tracks collapse, so the
              thumbnails stretch to share the strip's width instead of sitting
              at their minimum with dead space beside them. */}
          <div
            className="threed-strip-scroll"
            style={{ '--threed-thumb': `${thumbSize}px` }}
          >
            {inputImages.map((path, index) => {
              const isActive = !excluded.has(path);
              // Dataset views live in a per-variation folder, so the parent
              // directory is the meaningful label; loose files use the
              // filename.
              const parts = path.split('/');
              const label = parts[parts.length - 1] === 'generated.png'
                ? parts[parts.length - 2]
                : parts[parts.length - 1];
              const isPrimary = isSingleImage && path === activeImages[0];

              return (
                <button
                  key={path}
                  type="button"
                  className={`threed-strip-item ${isActive ? '' : 'threed-strip-item--off'} ${isPrimary ? 'threed-strip-item--primary' : ''}`}
                  onClick={() => toggleImage(path)}
                  title={`${label}\n${isActive ? 'Click to exclude' : 'Click to include'}`}
                >
                  {/* The ratio lives on this wrapper, not the <img>: an
                      aspect-ratio'd replaced element inside a button inside a
                      grid track resolves its height inconsistently, which
                      collapsed the tiles into thin slices. A wrapper with the
                      image absolutely filling it is unambiguous. */}
                  <span className="threed-strip-thumb">
                    <img
                      src={buildImageUrl(path)}
                      alt={label}
                      loading="lazy"
                      decoding="async"
                    />
                    <span className="threed-strip-index">{index + 1}</span>
                    {!isActive && <span className="threed-strip-x">✕</span>}
                  </span>
                  <span className="threed-strip-label">{label}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Splitter — drag to trade filmstrip width against viewport width.
          A wide strip fits more thumbnails per row for scanning a large set;
          a narrow one gives the geometry more room. */}
      {inputImages.length > 0 && (
        <div
          className={`threed-splitter ${dragging ? 'threed-splitter--active' : ''}`}
          onMouseDown={startDrag}
          onDoubleClick={() => setStripWidth(DEFAULT_STRIP_WIDTH)}
          role="separator"
          aria-orientation="vertical"
          title="Drag to resize · double-click to reset"
        >
          <span className="threed-splitter-grip" />
        </div>
      )}

      {/* ── Right: preview ── */}
      <div className="threed-preview-panel">
        <div className="threed-preview-header">
          <div className="threed-display-modes">
            {DISPLAY_MODES.map((mode) => (
              <button
                key={mode.key}
                className={`fuk-btn fuk-btn-secondary threed-btn-sm ${displayMode === mode.key ? 'threed-mode--active' : ''}`}
                onClick={() => setDisplayMode(mode.key)}
                disabled={!result}
              >
                {mode.label}
              </button>
            ))}
          </div>

          {result && Object.keys(result.urls || {}).length > 1 && (
            <div className="threed-format-switch">
              {Object.keys(result.urls).map((format) => (
                <button
                  key={format}
                  className={`fuk-btn fuk-btn-secondary threed-btn-sm ${viewerFormat === format ? 'threed-mode--active' : ''}`}
                  onClick={() => setViewerFormat(format)}
                >
                  {format.toUpperCase()}
                </button>
              ))}
            </div>
          )}
        </div>

        <MeshViewer
          url={viewerUrl}
          displayMode={displayMode}
          className="threed-viewer"
        />

        {result && (
          <div className="threed-result-bar">
            <div className="threed-result-stats">
              <span>{result.model}</span>
              <span>{result.input_count} input{result.input_count === 1 ? '' : 's'}</span>
              {result.vertex_count > 0 && (
                <span>{result.vertex_count.toLocaleString()} verts</span>
              )}
              {result.face_count > 0 && (
                <span>{result.face_count.toLocaleString()} faces</span>
              )}
              {result.point_count > 0 && (
                <span>{result.point_count.toLocaleString()} points</span>
              )}
              <span>{result.elapsed}s</span>
              {result.peak_vram_gb && <span>{result.peak_vram_gb}GB peak</span>}
            </div>

            <div className="threed-downloads">
              {Object.entries(result.urls || {}).map(([format, url]) => (
                <a
                  key={format}
                  className="fuk-btn fuk-btn-secondary threed-btn-sm"
                  href={buildImageUrl(url)}
                  download
                >
                  <Download /> {format.toUpperCase()}
                </a>
              ))}
            </div>
          </div>
        )}

        {!result && !generating && (
          <div className="threed-proxy-note">
            Output is proxy geometry — collision meshes, depth reference,
            compositing guides. Not survey-grade photogrammetry.
          </div>
        )}
      </div>

      <GenerationModal
        isOpen={showModal}
        type="threed_reconstruct"
        generating={generating}
        progress={progress}
        elapsedSeconds={elapsedSeconds}
        consoleLog={consoleLog}
        error={genError}
        onCancel={cancel}
        onClose={closeModal}
      />
    </div>
  );
}
