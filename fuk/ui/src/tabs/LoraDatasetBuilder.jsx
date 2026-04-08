/**
 * LoRA Dataset Builder
 * Three-phase tool: Setup → Running → Curation
 * Uses qwen_edit to produce variation sets for LoRA training.
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import ImageUploader from '../components/ImageUploader';
import { ThumbsUp, ThumbsDown, CheckCircle } from '../components/Icons';
import { buildImageUrl } from '../utils/constants';

// Pack labels — used when presets haven't loaded yet (avoids blank UI on first render)
const PACK_LABELS = {
  character:   { angles: 'Angles', expressions: 'Expressions', environments: 'Environments', lighting: 'Lighting' },
  product:     { angles: 'Angles', surfaces: 'Surfaces', lighting: 'Lighting' },
  environment: { time_of_day: 'Time of Day', weather: 'Weather', season: 'Season' },
};

// Fallback defaults if not in config
const DEFAULT_DENOISE = { character: 1.0, product: 1.0, environment: 1.0 };

// ============================================================================
// Helpers
// ============================================================================

const lbl = { fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: '0.2rem', display: 'block' };
const card = { padding: '0.6rem 0.75rem' };

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
  const [packs, setPacks]               = useState({ character: {}, product: {}, environment: {} });
  const [denoising, setDenoising]       = useState(d.denoising_strength ?? DEFAULT_DENOISE.character);
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
  const handleSourceDrop = useCallback(async (e) => {
    e.preventDefault();
    setSourceDragOver(false);

    const fukGenData = e.dataTransfer.getData('application/x-fuk-generation');
    if (!fukGenData) return;

    let gen;
    try { gen = JSON.parse(fukGenData); } catch { return; }
    if (gen.type === 'video' || gen.type === 'interpolate') return;

    // Add the output image as a source
    const imgPath = gen.path || gen.preview;
    if (imgPath) setSources(prev => prev.includes(imgPath) ? prev : [...prev, imgPath]);

    // Inherit generation params from metadata
    try {
      const res = await fetch(`/api/project/generations/${encodeURIComponent(gen.id)}/metadata`);
      if (!res.ok) return;
      const meta = await res.json();
      if (meta.seed != null) {
        setSeed(meta.seed);
        setSeedStrategy('fixed');
      }
      if (meta.guidance_scale != null) setCfg(meta.guidance_scale);
    } catch (_) {}
  }, []);

  const togglePack = (type, packKey) => {
    setPacks(prev => ({
      ...prev,
      [type]: { ...prev[type], [packKey]: !prev[type][packKey] },
    }));
  };

  const selectedPacks = Object.entries(packs[subjectType] || {})
    .filter(([, on]) => on)
    .map(([k]) => k);

  const totalVariations = selectedPacks.reduce((acc, pk) => {
    return acc + (serverPresets?.[subjectType]?.[pk]?.variations?.length ?? 0);
  }, 0);

  const canStart = sources.length > 0 && subjectName.trim() && selectedPacks.length > 0;

  const handleStart = async () => {
    if (!canStart || starting) return;
    setStarting(true);
    setError(null);
    try {
      const res = await fetch('/api/dataset/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject_name:   subjectName.trim(),
          subject_type:   subjectType,
          source_paths:   sources,
          selected_packs: selectedPacks,
          params: {
            denoising_strength: denoising,
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

  // Derive pack list from server presets (live count) with PACK_LABELS as fallback labels
  const currentPacks = Object.entries(
    serverPresets?.[subjectType] ?? PACK_LABELS[subjectType] ?? {}
  ).reduce((acc, [key, val]) => {
    acc[key] = {
      label: val.label ?? PACK_LABELS[subjectType]?.[key] ?? key,
      count: val.variations?.length ?? 0,
    };
    return acc;
  }, {});
  const loras = config?.models?.loras || [];

  return (
    <div style={{ display: 'flex', gap: '1.25rem', padding: '1rem 1.25rem', height: '100%', boxSizing: 'border-box', overflowY: 'auto' }}>

      {/* Left column */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', width: '340px', flexShrink: 0 }}>

        {/* Subject */}
        <div className="fuk-card" style={card}>
          <span style={lbl}>Subject Name</span>
          <input
            className="fuk-input"
            placeholder="e.g. john_doe"
            value={subjectName}
            onChange={e => setSubjectName(e.target.value)}
            style={{ width: '100%', marginBottom: '0.5rem' }}
          />
          <span style={lbl}>Subject Type</span>
          <select
            className="fuk-select"
            value={subjectType}
            onChange={e => setSubjectType(e.target.value)}
            style={{ width: '100%' }}
          >
            <option value="character">Character</option>
            <option value="product">Product</option>
            <option value="environment">Environment</option>
          </select>
        </div>

        {/* Source images — also a drop target for history items */}
        <div
          className="fuk-card"
          style={{
            ...card,
            outline: sourceDragOver ? '2px solid var(--accent-color, #a855f7)' : '2px solid transparent',
            transition: 'outline 0.1s',
            position: 'relative',
          }}
          onDragOver={e => { e.preventDefault(); setSourceDragOver(true); }}
          onDragLeave={() => setSourceDragOver(false)}
          onDrop={handleSourceDrop}
        >
          <span style={lbl}>Source Images</span>
          <ImageUploader
            images={sources}
            onImagesChange={setSources}
          />
          {sourceDragOver && (
            <div style={{
              position: 'absolute', inset: 0, borderRadius: '4px',
              background: 'var(--accent-color, #a855f7)18',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              pointerEvents: 'none',
            }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--accent-color, #a855f7)', fontWeight: 600 }}>
                Drop to add source + inherit settings
              </span>
            </div>
          )}
        </div>

        {/* Generation params */}
        <div className="fuk-card" style={card}>
          <span style={lbl}>Denoising Strength — {denoising.toFixed(2)}</span>
          <input
            type="range" min={0.3} max={1.0} step={0.01}
            value={denoising}
            onChange={e => setDenoising(parseFloat(e.target.value))}
            style={{ width: '100%', marginBottom: '0.6rem' }}
          />

          <span style={lbl}>Seed Strategy</span>
          <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.5rem' }}>
            {['fixed', 'random'].map(s => (
              <button
                key={s}
                className={`fuk-btn ${seedStrategy === s ? 'fuk-btn-primary' : 'fuk-btn-secondary'}`}
                style={{ flex: 1, fontSize: '0.78rem', padding: '0.25rem 0' }}
                onClick={() => setSeedStrategy(s)}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>

          {seedStrategy === 'fixed' && (
            <>
              <span style={lbl}>Seed</span>
              <input
                type="number"
                className="fuk-input"
                value={seed}
                onChange={e => setSeed(parseInt(e.target.value) || 0)}
                style={{ width: '100%', marginBottom: '0.5rem' }}
              />
            </>
          )}

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <div style={{ flex: 1 }}>
              <span style={lbl}>Steps</span>
              <input type="number" className="fuk-input" value={steps} min={1} max={100}
                onChange={e => setSteps(parseInt(e.target.value) || 28)} style={{ width: '100%' }} />
            </div>
            <div style={{ flex: 1 }}>
              <span style={lbl}>CFG</span>
              <input type="number" className="fuk-input" value={cfg} min={1} max={20} step={0.5}
                onChange={e => setCfg(parseFloat(e.target.value) || 5)} style={{ width: '100%' }} />
            </div>
          </div>
        </div>

        {/* Optional LoRA */}
        <div className="fuk-card" style={card}>
          <span style={lbl}>Style LoRA (optional)</span>
          <select className="fuk-select" value={lora} onChange={e => setLora(e.target.value)} style={{ width: '100%', marginBottom: '0.5rem' }}>
            <option value="">None</option>
            {loras.map((l, i) => {
              const key = typeof l === 'string' ? l : l.key || i;
              const name = typeof l === 'string' ? l : (l.name || l.description || l.key);
              return <option key={key} value={key}>{name}</option>;
            })}
          </select>
          {lora && (
            <>
              <span style={lbl}>LoRA Alpha — {loraAlpha.toFixed(1)}</span>
              <input type="range" min={0} max={2} step={0.1} value={loraAlpha}
                onChange={e => setLoraAlpha(parseFloat(e.target.value))} style={{ width: '100%' }} />
            </>
          )}
        </div>
      </div>

      {/* Right column: pack selection + action */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', flex: 1, minWidth: 0 }}>

        <div className="fuk-card" style={{ ...card, flex: 1 }}>
          <span style={lbl}>Variation Packs</span>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {Object.entries(currentPacks).map(([packKey, packMeta]) => {
              const on = !!packs[subjectType]?.[packKey];
              return (
                <label key={packKey} style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '0.45rem 0.6rem',
                  background: on ? 'var(--accent-color, #a855f7)22' : 'var(--bg-tertiary, #1e1e1e)',
                  borderRadius: '4px',
                  border: `1px solid ${on ? 'var(--accent-color, #a855f7)' : 'var(--border-color)'}`,
                  cursor: 'pointer',
                  transition: 'all 0.1s',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={() => togglePack(subjectType, packKey)}
                    />
                    <span style={{ fontSize: '0.84rem' }}>{packMeta.label}</span>
                  </div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {packMeta.count} variation{packMeta.count !== 1 ? 's' : ''}
                  </span>
                </label>
              );
            })}
          </div>

          {totalVariations > 0 && (
            <div style={{ marginTop: '0.75rem', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
              {totalVariations} total variation{totalVariations !== 1 ? 's' : ''} selected
            </div>
          )}
        </div>

        {error && (
          <div style={{ color: 'var(--error-color, #ef4444)', fontSize: '0.8rem', padding: '0.4rem 0.6rem', background: '#ef444422', borderRadius: '4px' }}>
            {error}
          </div>
        )}

        <button
          className="fuk-btn fuk-btn-primary"
          onClick={handleStart}
          disabled={!canStart || starting}
          style={{ width: '100%', padding: '0.6rem', fontSize: '0.9rem' }}
        >
          {starting ? 'Starting…' : `Generate Dataset — ${totalVariations} image${totalVariations !== 1 ? 's' : ''}`}
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
  const esRef = useRef(null);

  const handleCancel = async () => {
    setCancelling(true);
    try {
      await fetch(`/api/dataset/${jobId}/cancel`, { method: 'POST' });
    } catch (_) {}
  };

  useEffect(() => {
    if (!jobId) return;
    const es = new EventSource(`/api/dataset/${jobId}/stream`);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setState(data);
        if (data.status === 'complete') {
          es.close();
          onComplete(jobId);
        } else if (data.status === 'failed' || data.status === 'cancelled') {
          es.close();
        }
      } catch (_) {}
    };

    es.onerror = () => {
      es.close();
    };

    return () => { es.close(); };
  }, [jobId]);

  if (!state) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>
        Connecting to job stream…
      </div>
    );
  }

  const done = state.variations?.filter(v => v.status === 'completed').length ?? 0;
  const failed = state.variations?.filter(v => v.status === 'failed').length ?? 0;
  const total = state.total ?? 0;
  const pct = Math.round((state.progress ?? 0) * 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '1rem 1.25rem', height: '100%', boxSizing: 'border-box', overflow: 'hidden' }}>

      {/* Status bar */}
      <div className="fuk-card" style={{ padding: '0.6rem 0.75rem', flexShrink: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
          <span style={{ fontSize: '0.85rem' }}>
            {state.status === 'running' && state.current_label
              ? `Generating ${(state.current_idx ?? 0) + 1} of ${total} — ${state.current_label}`
              : state.status === 'complete'  ? 'Complete'
              : state.status === 'failed'    ? 'Failed'
              : state.status === 'cancelled' ? 'Cancelled'
              : 'Starting…'
            }
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontFamily: 'monospace', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              {done}/{total}{failed > 0 ? ` · ${failed} failed` : ''}
            </span>
            {state.status === 'running' && (
              <button
                className="fuk-btn fuk-btn-secondary"
                onClick={handleCancel}
                disabled={cancelling}
                style={{ fontSize: '0.72rem', padding: '0.15rem 0.5rem' }}
              >
                {cancelling ? 'Cancelling…' : 'Cancel'}
              </button>
            )}
          </div>
        </div>
        <div style={{ height: '4px', background: 'var(--bg-tertiary, #333)', borderRadius: '2px', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${pct}%`,
            background: state.status === 'failed' ? 'var(--error-color, #ef4444)' : 'var(--accent-color, #a855f7)',
            transition: 'width 0.4s ease',
          }} />
        </div>
      </div>

      {/* Thumbnail grid */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))', gap: '0.5rem' }}>
          {(state.variations ?? []).map(v => (
            <VariationThumbnail key={v.id} variation={v} jobId={jobId} />
          ))}
        </div>
      </div>
    </div>
  );
}

function VariationThumbnail({ variation, jobId }) {
  const [imgUrl, setImgUrl] = useState(null);

  // Poll for image once variation completes
  useEffect(() => {
    if (variation.status !== 'completed') { setImgUrl(null); return; }
    fetch(`/api/dataset/${jobId}`)
      .then(r => r.json())
      .then(data => {
        const v = data.variations?.find(vv => vv.id === variation.id);
        if (v?.image_url) setImgUrl(variationImageUrl(v.image_url));
      })
      .catch(() => {});
  }, [variation.status, variation.id, jobId]);

  const statusColor = {
    pending:   'var(--text-muted)',
    running:   'var(--accent-color, #a855f7)',
    completed: 'var(--success-color, #22c55e)',
    failed:    'var(--error-color, #ef4444)',
  }[variation.status] ?? 'var(--text-muted)';

  return (
    <div style={{
      background: 'var(--bg-secondary)',
      borderRadius: '4px',
      overflow: 'hidden',
      border: '1px solid var(--border-color)',
    }}>
      <div style={{ aspectRatio: '1', background: 'var(--bg-tertiary, #1a1a1a)', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
        {imgUrl ? (
          <img src={imgUrl} alt={variation.label} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <div style={{ width: '20px', height: '20px', borderRadius: '50%', border: `2px solid ${statusColor}`, opacity: 0.5 }} />
        )}
        {variation.status === 'running' && (
          <div style={{
            position: 'absolute', inset: 0,
            border: '2px solid var(--accent-color, #a855f7)',
            borderRadius: '4px',
            animation: 'pulse 1.5s ease-in-out infinite',
          }} />
        )}
      </div>
      <div style={{ padding: '0.2rem 0.35rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
        <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: statusColor, flexShrink: 0 }} />
        <span style={{ fontSize: '0.62rem', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {variation.label}
        </span>
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

  const fetchJob = useCallback(() => {
    fetch(`/api/dataset/${jobId}`)
      .then(r => r.json())
      .then(setJob)
      .catch(console.error);
  }, [jobId]);

  useEffect(() => { fetchJob(); }, [fetchJob]);

  const handleApprove = async (variationId, approved) => {
    await fetch(`/api/dataset/${jobId}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ variation_id: variationId, approved }),
    });
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
    return <div style={{ padding: '2rem', color: 'var(--text-muted)', textAlign: 'center' }}>Loading…</div>;
  }

  const approvedCount = job.approved_count ?? 0;
  const completedVariations = job.variations?.filter(v => v.status === 'completed') ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0.6rem 1.25rem',
        borderBottom: '1px solid var(--border-color)',
        flexShrink: 0,
        gap: '1rem',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.75rem' }}>
          <span className="fuk-label">{job.subject_name}</span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{job.subject_type}</span>
          <span style={{ fontSize: '0.72rem', color: 'var(--success-color, #22c55e)' }}>
            {approvedCount} approved
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          {exportResult && (
            <span style={{
              fontSize: '0.72rem', fontFamily: 'monospace',
              color: exportResult.ok ? 'var(--success-color, #22c55e)' : 'var(--error-color, #ef4444)',
              overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '280px', whiteSpace: 'nowrap',
            }}>
              {exportResult.ok ? `✓ ${exportResult.exported} → ${exportResult.path}` : `✗ ${exportResult.error}`}
            </span>
          )}
          {/* Directory picker */}
          <div
            className="fuk-input"
            onClick={handleBrowseExportDir}
            title="Click to choose export folder (defaults to job's approved/ subfolder)"
            style={{
              fontSize: '0.72rem', cursor: 'pointer', maxWidth: '200px',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              color: exportDir ? 'var(--text-primary)' : 'var(--text-muted)',
              userSelect: 'none',
            }}
          >
            {exportDir || 'Default: approved/'}
          </div>
          <button
            className="fuk-btn fuk-btn-secondary"
            onClick={handleBrowseExportDir}
            style={{ fontSize: '0.72rem', padding: '0.15rem 0.5rem', flexShrink: 0 }}
          >
            Browse
          </button>
          <button
            className="fuk-btn fuk-btn-primary"
            onClick={handleExport}
            disabled={approvedCount === 0 || exporting}
            style={{ fontSize: '0.8rem', flexShrink: 0 }}
          >
            {exporting ? 'Exporting…' : `Export (${approvedCount})`}
          </button>
        </div>
      </div>

      {/* Grid */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem 1.25rem' }}>
        {completedVariations.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', textAlign: 'center', paddingTop: '3rem', fontSize: '0.85rem' }}>
            No completed variations to review.
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '0.75rem' }}>
            {completedVariations.map(v => (
              <CurationCard key={v.id} variation={v} onApprove={handleApprove} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CurationCard({ variation, onApprove }) {
  const imgUrl = variation.image_url ? variationImageUrl(variation.image_url) : null;
  const approved = variation.approved;

  const borderColor =
    approved === true  ? 'var(--success-color, #22c55e)' :
    approved === false ? 'var(--error-color, #ef4444)' :
    'var(--border-color)';

  return (
    <div style={{
      background: 'var(--bg-secondary)',
      borderRadius: '6px',
      overflow: 'hidden',
      border: `2px solid ${borderColor}`,
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ aspectRatio: '1', background: 'var(--bg-tertiary, #1a1a1a)', overflow: 'hidden' }}>
        {imgUrl ? (
          <img src={imgUrl} alt={variation.label}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
        ) : (
          <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>No image</span>
          </div>
        )}
      </div>
      <div style={{ padding: '0.35rem 0.5rem' }}>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginBottom: '0.35rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {variation.label}
        </div>
        <div style={{ display: 'flex', gap: '0.3rem' }}>
          <button
            onClick={() => onApprove(variation.id, true)}
            title="Approve"
            style={{
              flex: 1, padding: '0.3rem', border: 'none', borderRadius: '3px', cursor: 'pointer',
              background: approved === true ? 'var(--success-color, #22c55e)' : 'var(--bg-tertiary, #333)',
              color: approved === true ? '#fff' : 'var(--text-muted)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.1s',
            }}
          >
            <ThumbsUp style={{ width: '0.9rem', height: '0.9rem' }} />
          </button>
          <button
            onClick={() => onApprove(variation.id, false)}
            title="Reject"
            style={{
              flex: 1, padding: '0.3rem', border: 'none', borderRadius: '3px', cursor: 'pointer',
              background: approved === false ? 'var(--error-color, #ef4444)' : 'var(--bg-tertiary, #333)',
              color: approved === false ? '#fff' : 'var(--text-muted)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 0.1s',
            }}
          >
            <ThumbsDown style={{ width: '0.9rem', height: '0.9rem' }} />
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
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* Phase indicator */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '0', padding: '0.4rem 1.25rem',
        borderBottom: '1px solid var(--border-color)',
        background: 'var(--bg-secondary)',
        flexShrink: 0,
      }}>
        {['setup', 'running', 'curation'].map((p, i) => {
          const labels = ['Setup', 'Generating', 'Curation'];
          const active = phase === p;
          const done = ['setup', 'running', 'curation'].indexOf(phase) > i;
          return (
            <div key={p} style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '0.35rem',
                padding: '0.2rem 0.5rem',
                fontSize: '0.75rem',
                color: active ? 'var(--accent-color, #a855f7)' : done ? 'var(--success-color, #22c55e)' : 'var(--text-muted)',
                fontWeight: active ? '600' : '400',
              }}>
                {done && <CheckCircle style={{ width: '0.75rem', height: '0.75rem' }} />}
                {labels[i]}
              </div>
              {i < 2 && <span style={{ color: 'var(--border-color)', margin: '0 0.1rem' }}>›</span>}
            </div>
          );
        })}
        {phase !== 'setup' && (
          <button
            className="fuk-btn fuk-btn-secondary"
            onClick={handleReset}
            style={{ marginLeft: 'auto', fontSize: '0.72rem', padding: '0.15rem 0.5rem' }}
          >
            New Dataset
          </button>
        )}
      </div>

      {/* Phase content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {phase === 'setup'    && <SetupPhase   config={config} onStart={handleStart} />}
        {phase === 'running'  && <RunningPhase jobId={jobId}   onComplete={handleComplete} />}
        {phase === 'curation' && <CurationPhase jobId={jobId} />}
      </div>
    </div>
  );
}
