/**
 * Model Manager — Utilities → Models
 *
 * Replaces download_models.sh as the primary way to acquire and enable models.
 * The script stays for headless runs.
 *
 * Two sections, deliberately different:
 *
 *   Tools    Read-only status. Green or red, straight from the availability
 *            check each feature already uses, so this panel cannot disagree
 *            with the feature itself. They install via setup.sh, not from here.
 *
 *   Models   Download button plus an active checkbox. Unchecking unregisters a
 *            model — it leaves the generation dropdowns but stays listed here,
 *            so re-enabling is one click and never needs a re-download.
 *
 * Downloads prefer HuggingFace, with a per-component fallback to ModelScope for
 * the repos HF does not carry.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Download, CheckCircle, AlertCircle, Loader2, RefreshCw, Link, Info,
} from '../components/Icons';

const API_URL = '/api';

const CATEGORY_LABELS = {
  image: 'Image models',
  video: 'Video models',
  threed: '3D reconstruction',
  other: 'Other',
};

const fmtBytes = (n) => {
  if (!n) return '—';
  const gb = n / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(n / 1024 ** 2).toFixed(0)} MB`;
};

// ============================================================================
// Tools — status only
// ============================================================================

function ToolRow({ tool }) {
  return (
    <div className={`mm-tool ${tool.available ? 'mm-tool--ok' : 'mm-tool--bad'}`}>
      <span className={`mm-dot ${tool.available ? 'mm-dot--ok' : 'mm-dot--bad'}`} />
      <div className="mm-tool-body">
        <div className="mm-tool-head">
          <span className="mm-tool-name">{tool.name}</span>
          {tool.docs_url && (
            <a className="mm-link" href={tool.docs_url} target="_blank" rel="noreferrer">
              <Link className="fuk-icon--sm" />
            </a>
          )}
        </div>
        <div className="mm-tool-desc">{tool.description}</div>
        {tool.available && tool.detail && (
          <div className="mm-tool-detail">{tool.detail}</div>
        )}
        {!tool.available && (
          <div className="mm-tool-missing">
            missing: {tool.missing.join(', ') || 'unknown'}
            {tool.hint && <> — <code>{tool.hint}</code></>}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Models — download + active
// ============================================================================

function ModelRow({ model, job, onToggle, onDownload, busy }) {
  const [open, setOpen] = useState(false);

  const state = !model.downloadable ? 'na'
    : model.downloaded ? 'complete'
    : model.partial ? 'partial'
    : 'missing';

  const running = job && job.status === 'running';

  return (
    <div className={`mm-model ${model.enabled ? '' : 'mm-model--off'}`}>
      <div className="mm-model-main">
        {/* Active checkbox — controls whether this appears in the dropdowns */}
        <label className="mm-check" title={model.enabled ? 'Registered' : 'Unregistered'}>
          <input
            type="checkbox"
            className="fuk-checkbox"
            checked={model.enabled}
            disabled={busy}
            onChange={(e) => onToggle(model.key, e.target.checked)}
          />
        </label>

        <div className="mm-model-body">
          <div className="mm-model-head">
            <span className="mm-model-name">{model.key}</span>
            <span className={`mm-state mm-state--${state}`}>
              {state === 'complete' && <><CheckCircle className="fuk-icon--sm" /> downloaded</>}
              {state === 'partial' && <><AlertCircle className="fuk-icon--sm" />
                {` ${model.components_present}/${model.components_total}`}</>}
              {state === 'missing' && 'not downloaded'}
              {state === 'na' && 'installed separately'}
            </span>
            {model.size_on_disk_bytes > 0 && (
              <span className="mm-size">{fmtBytes(model.size_on_disk_bytes)}</span>
            )}
            {state === 'missing' && model.size_gb_estimate && (
              <span className="mm-size">~{model.size_gb_estimate} GB</span>
            )}
          </div>

          <div className="mm-model-desc">{model.description}</div>

          {running && (
            <div className="mm-progress">
              <div
                className="mm-progress-bar"
                style={{ width: `${(job.completed / Math.max(1, job.total)) * 100}%` }}
              />
              <span className="mm-progress-text">
                {job.completed}/{job.total} — {job.current || 'starting…'}
              </span>
            </div>
          )}
          {job && job.status === 'partial' && (
            <div className="mm-model-missing">
              {job.failed.length} component(s) failed — retry, or fetch them by hand from the
              links below.
            </div>
          )}

          <button className="mm-disclose" onClick={() => setOpen((v) => !v)}>
            {open ? 'Hide' : 'Where do the weights go?'}
          </button>

          {open && (
            <div className="mm-detail">
              {model.repos.map((r) => (
                <div key={r.model_id} className="mm-repo">
                  <div className="mm-repo-id">{r.model_id}</div>
                  <div className="mm-repo-links">
                    <a href={r.huggingface_url} target="_blank" rel="noreferrer">HuggingFace</a>
                    <a href={r.modelscope_url} target="_blank" rel="noreferrer">ModelScope</a>
                  </div>
                  <code className="mm-path">{r.install_path}</code>
                </div>
              ))}
              {model.components.length > 0 && (
                <table className="mm-parts">
                  <tbody>
                    {model.components.map((c) => (
                      <tr key={`${c.model_id}:${c.pattern}`}
                          className={c.present ? '' : 'mm-part--missing'}>
                        <td>{c.present ? '✓' : '·'}</td>
                        <td>{c.label}</td>
                        <td><code>{c.pattern}</code></td>
                        <td>{c.present ? fmtBytes(c.size_bytes) : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}
        </div>

        {model.downloadable && (
          <button
            className="fuk-btn fuk-btn-secondary fuk-btn-sm mm-dl"
            disabled={running || busy}
            onClick={() => onDownload(model.key)}
          >
            {running
              ? <><Loader2 className="fuk-icon--sm mm-spin" /> downloading</>
              : <><Download className="fuk-icon--sm" /> {model.downloaded ? 'Verify' : 'Download'}</>}
          </button>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Panel
// ============================================================================

export default function ModelManager() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  // model key -> job. Kept out of `data` so a refresh cannot drop live progress.
  const [jobs, setJobs] = useState({});
  const pollRef = useRef(null);

  const load = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/models/manage`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Poll only while something is downloading, then refresh the listing once so
  // the newly-present files show up as downloaded.
  useEffect(() => {
    const active = Object.values(jobs).filter((j) => j && j.status === 'running');
    if (active.length === 0) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    if (pollRef.current) return;

    pollRef.current = setInterval(async () => {
      const updates = {};
      let anyFinished = false;
      for (const [key, job] of Object.entries(jobs)) {
        if (!job || job.status !== 'running') continue;
        try {
          const res = await fetch(`${API_URL}/models/manage/jobs/${job.id}`);
          if (!res.ok) continue;
          const fresh = await res.json();
          updates[key] = fresh;
          if (fresh.status !== 'running') anyFinished = true;
        } catch { /* transient — next tick retries */ }
      }
      if (Object.keys(updates).length) setJobs((prev) => ({ ...prev, ...updates }));
      if (anyFinished) load();
    }, 1500);

    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    };
  }, [jobs, load]);

  const handleToggle = async (key, enabled) => {
    setBusy(true);
    // Optimistic: the checkbox should not lag a round trip.
    setData((d) => ({
      ...d,
      models: d.models.map((m) => (m.key === key ? { ...m, enabled } : m)),
    }));
    try {
      const res = await fetch(`${API_URL}/models/manage/${key}/enabled`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (e) {
      setError(`Could not update ${key}: ${e}`);
      load();   // rewind the optimistic change to whatever the server actually has
    } finally {
      setBusy(false);
    }
  };

  const handleDownload = async (key) => {
    try {
      const res = await fetch(`${API_URL}/models/manage/${key}/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const { job_id, total } = await res.json();
      setJobs((prev) => ({
        ...prev,
        [key]: { id: job_id, status: 'running', completed: 0, total, current: null, failed: [] },
      }));
    } catch (e) {
      setError(`Download failed to start for ${key}: ${e}`);
    }
  };

  if (error && !data) {
    return <div className="mm-error"><AlertCircle className="fuk-icon--sm" /> {error}</div>;
  }
  if (!data) {
    return <div className="mm-loading"><Loader2 className="fuk-icon--sm mm-spin" /> Loading…</div>;
  }

  const grouped = data.models.reduce((acc, m) => {
    (acc[m.category] ||= []).push(m);
    return acc;
  }, {});

  const enabledCount = data.models.filter((m) => m.enabled).length;

  return (
    <div className="mm-panel">
      {error && (
        <div className="mm-error"><AlertCircle className="fuk-icon--sm" /> {error}</div>
      )}

      <div className="mm-header">
        <div>
          <span className="fuk-label">Models &amp; Tools</span>
          <div className="mm-sub">
            {enabledCount} of {data.models.length} models registered · downloads prefer
            {' '}{data.download_source}
          </div>
        </div>
        <button className="fuk-btn fuk-btn-secondary fuk-btn-sm" onClick={load}>
          <RefreshCw className="fuk-icon--sm" /> Refresh
        </button>
      </div>

      {/* Tools */}
      <div className="fuk-card mm-card">
        <span className="fuk-label">Tools</span>
        <div className="mm-sub mm-sub--card">
          Installed by <code>setup.sh</code>, not from here. Status comes from each
          feature&apos;s own check.
        </div>
        <div className="mm-tools">
          {data.tools.map((t) => <ToolRow key={t.key} tool={t} />)}
        </div>
      </div>

      {/* Models, grouped */}
      {Object.entries(grouped).map(([cat, models]) => (
        <div className="fuk-card mm-card" key={cat}>
          <span className="fuk-label">{CATEGORY_LABELS[cat] || cat}</span>
          {cat === 'threed' && (
            <div className="mm-sub mm-sub--card">
              <Info className="fuk-icon--sm" /> These acquire weights through
              {' '}<code>install_trellis_env.sh</code> or on first use, so they have no
              download button here.
            </div>
          )}
          <div className="mm-models">
            {models.map((m) => (
              <ModelRow
                key={m.key}
                model={m}
                job={jobs[m.key]}
                onToggle={handleToggle}
                onDownload={handleDownload}
                busy={busy}
              />
            ))}
          </div>
        </div>
      ))}

      <div className="mm-footnote">
        Weights live under <code>{data.models_root}</code>. Sizes are per model, so
        models sharing a repo — the Qwen text encoder, the Wan VAE — report the same
        files more than once.
      </div>
    </div>
  );
}
