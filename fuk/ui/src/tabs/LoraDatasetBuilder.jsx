/**
 * LoRA Dataset Builder
 * Three-phase tool: Setup → Running → Curation
 * Uses qwen_edit to produce variation sets for LoRA training.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import ImageUploader from '../components/ImageUploader';
import { ThumbsUp, ThumbsDown, CheckCircle } from '../components/Icons';
import { buildImageUrl } from '../utils/constants';

const toLabel = key => key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

// ============================================================================
// Helpers
// ============================================================================


function variationImageUrl(path) {
  if (!path) return null;
  return buildImageUrl(path);
}

// ============================================================================
// Setup Phase
// ============================================================================

function SetupPhase({ config, onStart }) {
  const d = config?.defaults?.lora_dataset ?? {};

  const [sources, setSources]           = useState([]);
  const [subjectName, setSubjectName]   = useState('');
  const [subjectType, setSubjectType]   = useState('character');
  const [packs, setPacks]               = useState({ character: {}, object: {}, environment: {} });
  const [deselectedVars, setDeselectedVars] = useState({ character: {}, object: {}, environment: {} });
  const [expandedPacks, setExpandedPacks]   = useState({});
  const [seedStrategy, setSeedStrategy] = useState(d.seed_strategy ?? 'fixed');
  const [seed, setSeed]                 = useState(d.seed ?? 509327136);
  const [lora, setLora]                 = useState('');
  const [loraAlpha, setLoraAlpha]       = useState(0.8);
  const [steps, setSteps]               = useState(d.steps ?? 28);
  const [cfg, setCfg]                   = useState(d.cfg_scale ?? 2.5);
  const [starting, setStarting]         = useState(false);
  const [error, setError]               = useState(null);
  const [sourceDragOver, setSourceDragOver] = useState(false);
  const [serverPresets, setServerPresets]   = useState(null);

  // Load live presets so pack counts are always accurate (no manual JSX update needed)
  useEffect(() => {
    fetch('/api/dataset/presets')
      .then(r => r.json())
      .then(setServerPresets)
      .catch(() => {});
  }, []);

  // Reset packs when subject type changes (keep params as-is)
  useEffect(() => {
    setPacks(prev => ({ ...prev, [subjectType]: {} }));
  }, [subjectType]);

  // ── Drag-drop from history panel ──
  const handleSourceDrop = useCallback((e) => {
    e.preventDefault();
    setSourceDragOver(false);

    const fukGenData = e.dataTransfer.getData('application/x-fuk-generation');
    if (!fukGenData) return;

    let gen;
    try { gen = JSON.parse(fukGenData); } catch { return; }
    if (gen.type === 'video' || gen.type === 'interpolate') return;

    const imgPath = gen.path || gen.preview;
    if (imgPath) setSources(prev => prev.includes(imgPath) ? prev : [...prev, imgPath]);
  }, []);

  const togglePack = (type, packKey) => {
    const nowOn = !packs[type]?.[packKey];
    setPacks(prev => ({
      ...prev,
      [type]: { ...prev[type], [packKey]: nowOn },
    }));
    if (nowOn) setExpandedPacks(prev => ({ ...prev, [packKey]: true }));
  };

  const toggleVar = (packKey, varId) => {
    setDeselectedVars(prev => {
      const packDesel = prev[subjectType]?.[packKey] ?? {};
      return {
        ...prev,
        [subjectType]: {
          ...prev[subjectType],
          [packKey]: { ...packDesel, [varId]: !packDesel[varId] },
        },
      };
    });
  };

  const selectedPacks = Object.entries(packs[subjectType] || {})
    .filter(([, on]) => on)
    .map(([k]) => k);

  const totalVariations = selectedPacks.reduce((acc, pk) => {
    const allVars = serverPresets?.[subjectType]?.[pk]?.variations ?? [];
    const desel = deselectedVars[subjectType]?.[pk] ?? {};
    return acc + allVars.filter(v => !desel[v.id]).length;
  }, 0);

  const totalGenerations = totalVariations * Math.max(sources.length, 1);

  const canStart = sources.length > 0 && subjectName.trim() && selectedPacks.length > 0 && totalVariations > 0;

  const handleStart = async () => {
    if (!canStart || starting) return;
    setStarting(true);
    setError(null);
    try {
      const excludedVariationIds = selectedPacks.flatMap(pk => {
        const desel = deselectedVars[subjectType]?.[pk] ?? {};
        return Object.keys(desel).filter(id => desel[id]);
      });

      const res = await fetch('/api/dataset/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_name:           subjectName.trim(),
          subject_type:           subjectType,
          source_paths:           sources,
          selected_packs:         selectedPacks,
          excluded_variation_ids: excludedVariationIds,
          params: {
            seed_strategy:      seedStrategy,
            seed:               seed,
            steps:              steps,
            cfg_scale:          cfg,
            lora:               lora || null,
            lora_alpha:         loraAlpha,
          },
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to start job');
      onStart(data.job_id);
    } catch (err) {
      setError(err.message);
      setStarting(false);
    }
  };

  const currentPacks = Object.entries(serverPresets?.[subjectType] ?? {}).reduce((acc, [key, val]) => {
    acc[key] = {
      label: val.label ?? toLabel(key),
      count: val.variations?.length ?? 0,
    };
    return acc;
  }, {});
  const loras = config?.models?.loras || [];

  return (
    <div className="dataset-setup">

      {/* Left column */}
      <div className="dataset-setup-col">

        {/* Subject */}
        <div className="fuk-card dataset-card">
          <span className="dataset-label">Subject Name</span>
          <input
            className="fuk-input"
            placeholder="e.g. john_doe"
            value={subjectName}
            onChange={e => setSubjectName(e.target.value)}
          />
          <span className="dataset-label">Subject Type</span>
          <select className="fuk-select" value={subjectType} onChange={e => setSubjectType(e.target.value)}>
            <option value="character">Character</option>
            <option value="object">Object</option>
            <option value="environment">Environment</option>
          </select>
        </div>

        {/* Source images */}
        <div
          className={`fuk-card dataset-card dataset-source-card ${sourceDragOver ? 'dataset-source-card--dragover' : ''}`}
          onDragOver={e => { e.preventDefault(); setSourceDragOver(true); }}
          onDragLeave={() => setSourceDragOver(false)}
          onDrop={handleSourceDrop}
        >
          <span className="dataset-label">Source Images</span>
          <ImageUploader images={sources} onImagesChange={setSources} />
          {sourceDragOver && (
            <div className="dataset-drag-overlay">
              <span>Drop to add source</span>
            </div>
          )}
        </div>

        {/* Generation params */}
        <div className="fuk-card dataset-card">
          <span className="dataset-label">Seed Strategy</span>
          <div className="dataset-seed-row">
            {['fixed', 'random'].map(s => (
              <button
                key={s}
                className={`fuk-btn ${seedStrategy === s ? 'fuk-btn-primary' : 'fuk-btn-secondary'}`}
                onClick={() => setSeedStrategy(s)}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>

          {seedStrategy === 'fixed' && (
            <>
              <span className="dataset-label">Seed</span>
              <input
                type="number"
                className="fuk-input"
                value={seed}
                onChange={e => setSeed(parseInt(e.target.value) || 0)}
              />
            </>
          )}

          <div className="dataset-params-row">
            <div className="dataset-params-col">
              <span className="dataset-label">Steps</span>
              <input type="number" className="fuk-input" value={steps} min={1} max={100}
                onChange={e => setSteps(parseInt(e.target.value) || 28)} />
            </div>
            <div className="dataset-params-col">
              <span className="dataset-label">CFG</span>
              <input type="number" className="fuk-input" value={cfg} min={1} max={20} step={0.5}
                onChange={e => setCfg(parseFloat(e.target.value) || 5)} />
            </div>
          </div>
        </div>

        {/* Optional LoRA */}
        <div className="fuk-card dataset-card">
          <span className="dataset-label">Style LoRA (optional)</span>
          <select className="fuk-select" value={lora} onChange={e => setLora(e.target.value)}>
            <option value="">None</option>
            {loras.map((l, i) => {
              const key = typeof l === 'string' ? l : l.key || i;
              const name = typeof l === 'string' ? l : (l.name || l.description || l.key);
              return <option key={key} value={key}>{name}</option>;
            })}
          </select>
          {lora && (
            <>
              <span className="dataset-label">LoRA Alpha — {loraAlpha.toFixed(1)}</span>
              <input type="range" className="fuk-slider" min={0} max={2} step={0.1} value={loraAlpha}
                onChange={e => setLoraAlpha(parseFloat(e.target.value))} />
            </>
          )}
        </div>
      </div>

      {/* Right column: pack selection + action */}
      <div className="dataset-setup-right">

        <div className="fuk-card dataset-card dataset-packs-card">
          <span className="dataset-label">Variation Packs</span>

          <div className="dataset-packs-list">
            {Object.entries(currentPacks).map(([packKey, packMeta]) => {
              const on = !!packs[subjectType]?.[packKey];
              const expanded = !!expandedPacks[packKey];
              const packVars = serverPresets?.[subjectType]?.[packKey]?.variations ?? [];
              const desel = deselectedVars[subjectType]?.[packKey] ?? {};
              const enabledCount = packVars.filter(v => !desel[v.id]).length;
              return (
                <div key={packKey} className={`dataset-pack-item ${on ? 'dataset-pack-item--active' : ''}`}>
                  <div className="dataset-pack-header" onClick={() => togglePack(subjectType, packKey)}>
                    <div className="dataset-pack-header-left">
                      <input type="checkbox" className="fuk-checkbox" checked={on} onChange={() => {}} onClick={e => e.stopPropagation()} />
                      <span className="dataset-pack-name">{packMeta.label}</span>
                    </div>
                    <div className="dataset-pack-header-right">
                      {on && (
                        <span className={`dataset-pack-count ${enabledCount < packMeta.count ? 'dataset-pack-count--partial' : ''}`}>
                          {enabledCount}/{packMeta.count}
                        </span>
                      )}
                      {!on && (
                        <span className="dataset-pack-count">
                          {packMeta.count} var{packMeta.count !== 1 ? 's' : ''}
                        </span>
                      )}
                      {on && packVars.length > 0 && (
                        <button
                          className="dataset-pack-expand-btn"
                          onClick={e => { e.stopPropagation(); setExpandedPacks(prev => ({ ...prev, [packKey]: !prev[packKey] })); }}
                          title={expanded ? 'Collapse' : 'Expand to filter variations'}
                        >
                          {expanded ? '▲' : '▼'}
                        </button>
                      )}
                    </div>
                  </div>

                  {on && expanded && packVars.length > 0 && (
                    <div className="dataset-pack-vars">
                      {packVars.map(v => {
                        const checked = !desel[v.id];
                        return (
                          <label key={v.id} className={`dataset-var-label ${!checked ? 'dataset-var-label--deselected' : ''}`}>
                            <input
                              type="checkbox"
                              className="fuk-checkbox dataset-var-checkbox"
                              checked={checked}
                              onChange={() => toggleVar(packKey, v.id)}
                            />
                            <span>{v.label}</span>
                          </label>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {totalVariations > 0 && (
            <div className="dataset-var-count">
              {sources.length > 1
                ? `${totalVariations} variation${totalVariations !== 1 ? 's' : ''} × ${sources.length} sources = ${totalGenerations} images`
                : `${totalVariations} variation${totalVariations !== 1 ? 's' : ''} selected`
              }
            </div>
          )}
        </div>

        {error && <div className="dataset-error">{error}</div>}

        <button
          className="fuk-btn fuk-btn-primary dataset-start-btn"
          onClick={handleStart}
          disabled={!canStart || starting}
        >
          {starting ? 'Starting…' : `Generate Dataset — ${totalGenerations} image${totalGenerations !== 1 ? 's' : ''}`}
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Running Phase
// ============================================================================

function RunningPhase({ jobId, onComplete }) {
  const [state, setState]       = useState(null);
  const [cancelling, setCancelling] = useState(false);
  const [thumbSize, setThumbSize]   = useState(130);
  const esRef = useRef(null);

  const handleCancel = async () => {
    setCancelling(true);
    try {
      await fetch(`/api/dataset/${jobId}/cancel`, { method: 'POST' });
    } catch (_) {}
  };

  useEffect(() => {
    if (!jobId) return;
    let done = false;

    const handleData = (data) => {
      if (done) return;
      setState(data);
      if (data.status === 'complete') {
        cleanup();
        onComplete(jobId);
      } else if (data.status === 'failed' || data.status === 'cancelled') {
        cleanup();
      }
    };

    // Primary: SSE stream (fast, push-based). Do NOT close on error — let
    // EventSource auto-reconnect so a transient drop heals itself.
    const es = new EventSource(`/api/dataset/${jobId}/stream`);
    esRef.current = es;
    es.onmessage = (e) => {
      try { handleData(JSON.parse(e.data)); } catch (_) {}
    };

    // Fallback: poll full job state as a backstop so the UI keeps advancing
    // (and still reaches 'complete') even if the stream stalls entirely.
    const pollId = setInterval(async () => {
      try {
        const r = await fetch(`/api/dataset/${jobId}`);
        if (r.ok) handleData(await r.json());
      } catch (_) {}
    }, 4000);

    function cleanup() {
      if (done) return;
      done = true;
      es.close();
      clearInterval(pollId);
    }

    return cleanup;
  }, [jobId]);

  if (!state) {
    return <div className="dataset-connecting">Connecting to job stream…</div>;
  }

  const done = state.variations?.filter(v => v.status === 'completed').length ?? 0;
  const failed = state.variations?.filter(v => v.status === 'failed').length ?? 0;
  const total = state.total ?? 0;
  const pct = Math.round((state.progress ?? 0) * 100);

  return (
    <div className="dataset-running">

      {/* Status bar */}
      <div className="fuk-card dataset-status-card">
        <div className="dataset-status-header">
          <span className="dataset-status-label">
            {state.status === 'running' && state.current_label
              ? `Generating ${(state.current_idx ?? 0) + 1} of ${total} — ${state.current_label}`
              : state.status === 'complete'  ? 'Complete'
              : state.status === 'failed'    ? 'Failed'
              : state.status === 'cancelled' ? 'Cancelled'
              : 'Starting…'
            }
          </span>
          <div className="dataset-status-right">
            <span className="dataset-status-count">
              {done}/{total}{failed > 0 ? ` · ${failed} failed` : ''}
            </span>
            <label className="dataset-thumb-size-label">
              <span>Size</span>
              <input
                type="range" className="fuk-slider" min={80} max={280} step={20}
                value={thumbSize}
                onChange={e => setThumbSize(Number(e.target.value))}
              />
              <span className="dataset-thumb-size-value">{thumbSize}px</span>
            </label>
            {state.status === 'running' && (
              <button
                className="fuk-btn fuk-btn-secondary dataset-cancel-btn"
                onClick={handleCancel}
                disabled={cancelling}
              >
                {cancelling ? 'Cancelling…' : 'Cancel'}
              </button>
            )}
          </div>
        </div>
        <div className="dataset-progress-track">
          <div
            className={`dataset-progress-fill ${state.status === 'failed' ? 'dataset-progress-fill--failed' : ''}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Thumbnail grid */}
      <div className="dataset-thumb-grid-wrap">
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fill, minmax(${thumbSize}px, 1fr))`, gap: '0.5rem' }}>
          {(state.variations ?? []).map(v => (
            <VariationThumbnail key={v.id} variation={v} />
          ))}
        </div>
      </div>
    </div>
  );
}

function VariationThumbnail({ variation }) {
  // image_url now arrives directly on the streamed/polled variation, so the
  // thumbnail renders as soon as the image exists — no separate fetch needed.
  const imgUrl = variation.image_url ? variationImageUrl(variation.image_url) : null;
  const status = variation.status;

  return (
    <div className="dataset-thumb">
      <div className="dataset-thumb-img-area">
        {imgUrl ? (
          <img src={imgUrl} alt={variation.label} />
        ) : (
          <div className="dataset-thumb-pending" />
        )}
        {status === 'running' && <div className="dataset-thumb-running-ring" />}
      </div>
      <div className="dataset-thumb-footer">
        <div className={`dataset-thumb-dot dataset-thumb-dot--${status}`} />
        <span className="dataset-thumb-label">{variation.label}</span>
      </div>
    </div>
  );
}

// ============================================================================
// Curation Phase
// ============================================================================

function CurationPhase({ jobId }) {
  const [job, setJob]               = useState(null);
  const [exportDir, setExportDir]   = useState('');
  const [exporting, setExporting]   = useState(false);
  const [exportResult, setExportResult] = useState(null);
  const [thumbSize, setThumbSize]   = useState(180);
  const [rerunning, setRerunning]   = useState(new Set());
  const pollRef = useRef(null);

  const fetchJob = useCallback(async () => {
    try {
      const data = await fetch(`/api/dataset/${jobId}`).then(r => r.json());
      setJob(data);
      return data;
    } catch (e) { console.error(e); }
  }, [jobId]);

  useEffect(() => { fetchJob(); }, [fetchJob]);

  // Poll while any variation is being rerun
  useEffect(() => {
    if (rerunning.size === 0) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    if (pollRef.current) return;
    pollRef.current = setInterval(() => {
      fetchJob().then(data => {
        if (!data) return;
        setRerunning(prev => {
          const still = new Set([...prev].filter(id => {
            const v = data.variations?.find(vv => vv.id === id);
            return v?.status === 'running' || v?.status === 'pending';
          }));
          return still;
        });
      });
    }, 1500);
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, [rerunning.size, fetchJob]);

  const handleRerun = async (variationId) => {
    try {
      const res = await fetch(`/api/dataset/${jobId}/rerun/${variationId}`, { method: 'POST' });
      if (!res.ok) return;
      setRerunning(prev => new Set([...prev, variationId]));
    } catch (_) {}
  };

  const handleApprove = async (variationId, approved) => {
    await fetch(`/api/dataset/${jobId}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ variation_id: variationId, approved }),
    });
    fetchJob();
  };

  const handleApproveAll = async (approved) => {
    const eligible = job?.variations?.filter(v => v.status === 'completed' && !rerunning.has(v.id)) ?? [];
    await Promise.all(eligible.map(v =>
      fetch(`/api/dataset/${jobId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variation_id: v.id, approved }),
      })
    ));
    fetchJob();
  };

  const handleBrowseExportDir = async () => {
    try {
      const res = await fetch('/api/browser/directory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: 'Choose export folder for approved images' }),
      });
      const data = await res.json();
      if (data.success && data.directory) setExportDir(data.directory);
    } catch (_) {}
  };

  const handleExport = async () => {
    setExporting(true);
    setExportResult(null);
    try {
      const res = await fetch(`/api/dataset/${jobId}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_dir: exportDir || null }),
      });
      const data = await res.json();
      setExportResult(data);
    } catch (err) {
      setExportResult({ ok: false, error: err.message });
    } finally {
      setExporting(false);
    }
  };

  if (!job) {
    return <div className="dataset-loading">Loading…</div>;
  }

  const approvedCount = job.approved_count ?? 0;
  const completedVariations = job.variations?.filter(v => v.status === 'completed' || v.status === 'running' || v.status === 'failed') ?? [];

  return (
    <div className="dataset-curation">

      {/* Header */}
      <div className="dataset-curation-header">
        <div className="dataset-curation-title">
          <span className="fuk-label">{job.subject_name}</span>
          <span className="dataset-curation-type">{job.subject_type}</span>
          <span className="dataset-curation-approved-count">{approvedCount} approved</span>
        </div>
        <div className="dataset-curation-actions">
          <button
            className="fuk-btn fuk-btn-secondary dataset-curation-approve-all-btn"
            onClick={() => handleApproveAll(true)}
            title="Approve all completed images"
          >
            Approve All
          </button>
          <button
            className="fuk-btn fuk-btn-secondary dataset-curation-reject-all-btn"
            onClick={() => handleApproveAll(false)}
            title="Reject all completed images"
          >
            Reject All
          </button>
          <label className="dataset-thumb-size-label">
            <span>Size</span>
            <input
              type="range" className="fuk-slider" min={100} max={320} step={20}
              value={thumbSize}
              onChange={e => setThumbSize(Number(e.target.value))}
            />
            <span className="dataset-thumb-size-value">{thumbSize}px</span>
          </label>
          {exportResult && (
            <span className={`dataset-export-result ${exportResult.ok ? 'dataset-export-result--ok' : 'dataset-export-result--error'}`}>
              {exportResult.ok ? `✓ ${exportResult.exported} → ${exportResult.path}` : `✗ ${exportResult.error}`}
            </span>
          )}
          <div
            className={`fuk-input dataset-export-dir ${exportDir ? 'dataset-export-dir--set' : 'dataset-export-dir--empty'}`}
            onClick={handleBrowseExportDir}
            title="Click to choose export folder (defaults to job's approved/ subfolder)"
          >
            {exportDir || 'Default: approved/'}
          </div>
          <button className="fuk-btn fuk-btn-secondary dataset-browse-btn" onClick={handleBrowseExportDir}>Browse</button>
          <button
            className="fuk-btn fuk-btn-primary dataset-export-btn"
            onClick={handleExport}
            disabled={approvedCount === 0 || exporting}
          >
            {exporting ? 'Exporting…' : `Export (${approvedCount})`}
          </button>
        </div>
      </div>

      {/* Grid */}
      <div className="dataset-curation-grid-wrap">
        {completedVariations.length === 0 ? (
          <div className="dataset-curation-empty">No completed variations to review.</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fill, minmax(${thumbSize}px, 1fr))`, gap: '0.75rem' }}>
            {completedVariations.map(v => (
              <CurationCard key={v.id} variation={v} onApprove={handleApprove} onRerun={handleRerun} isRerunning={rerunning.has(v.id)} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CurationCard({ variation, onApprove, onRerun, isRerunning }) {
  const imgUrl = variation.image_url ? variationImageUrl(variation.image_url) : null;
  const approved = variation.approved;
  const isFailed = variation.status === 'failed';
  const isRunning = variation.status === 'running' || isRerunning;

  const borderState = isRunning ? 'running' : isFailed ? 'failed' : approved === true ? 'approved' : approved === false ? 'rejected' : '';

  return (
    <div className={`dataset-curation-card ${borderState ? `dataset-curation-card--${borderState}` : ''}`}>
      <div className="dataset-card-media">
        {imgUrl && !isRunning ? (
          <img src={imgUrl} alt={variation.label} />
        ) : (
          <div className="dataset-card-placeholder">
            {isRunning ? (
              <>
                {imgUrl && <img src={imgUrl} alt={variation.label} className="dataset-card-rerunning-bg" />}
                <div className="dataset-card-spinner" />
                <span className="dataset-card-rerunning-label">Regenerating…</span>
              </>
            ) : (
              <span className={`dataset-card-empty-label ${isFailed ? 'dataset-card-empty-label--failed' : ''}`}>
                {isFailed ? 'Failed' : 'No image'}
              </span>
            )}
          </div>
        )}
      </div>
      <div className="dataset-card-footer">
        <div className="dataset-card-label">{variation.label}</div>
        <div className="dataset-card-actions">
          <button
            className={`dataset-vote-btn ${approved === true ? 'dataset-vote-btn--approved' : ''}`}
            onClick={() => onApprove(variation.id, true)}
            title="Approve"
            disabled={isRunning || isFailed}
          >
            <ThumbsUp />
          </button>
          <button
            className={`dataset-vote-btn ${approved === false ? 'dataset-vote-btn--rejected' : ''}`}
            onClick={() => onApprove(variation.id, false)}
            title="Reject"
            disabled={isRunning || isFailed}
          >
            <ThumbsDown />
          </button>
          <button
            className="dataset-rerun-btn"
            onClick={() => onRerun(variation.id)}
            title="Rerun this generation"
            disabled={isRunning}
          >
            ↺
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Main export
// ============================================================================

export default function LoraDatasetBuilder({ config }) {
  const [phase, setPhase]   = useState('setup');  // setup | running | curation
  const [jobId, setJobId]   = useState(null);

  const handleStart = (newJobId) => {
    setJobId(newJobId);
    setPhase('running');
  };

  const handleComplete = () => {
    setPhase('curation');
  };

  const handleReset = () => {
    setJobId(null);
    setPhase('setup');
  };

  return (
    <div className="dataset-layout">

      {/* Phase indicator */}
      <div className="dataset-phase-bar">
        {['setup', 'running', 'curation'].map((p, i) => {
          const labels = ['Setup', 'Generating', 'Curation'];
          const active = phase === p;
          const done = ['setup', 'running', 'curation'].indexOf(phase) > i;
          return (
            <div key={p} className="dataset-phase-item">
              <div className={`dataset-phase-label ${active ? 'dataset-phase-label--active' : ''} ${done ? 'dataset-phase-label--done' : ''}`}>
                {done && <CheckCircle />}
                {labels[i]}
              </div>
              {i < 2 && <span className="dataset-phase-separator">›</span>}
            </div>
          );
        })}
        {phase !== 'setup' && (
          <button
            className="fuk-btn fuk-btn-secondary dataset-phase-reset"
            onClick={handleReset}
          >
            New Dataset
          </button>
        )}
      </div>

      {/* Phase content */}
      <div className="dataset-content">
        {phase === 'setup'    && <SetupPhase   config={config} onStart={handleStart} />}
        {phase === 'running'  && <RunningPhase jobId={jobId}   onComplete={handleComplete} />}
        {phase === 'curation' && <CurationPhase jobId={jobId} />}
      </div>
    </div>
  );
}
