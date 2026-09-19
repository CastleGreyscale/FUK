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
 *            Models whose function variants ship as LoRAs — LTX-2's camera moves
 *            and in-context controls — list those under the model row, picked
 *            and downloaded one at a time.
 *
 *   Storage  Read-speed check and refresh pass over the weights. Flash loses
 *            charge, so files written once and never touched read slower every
 *            month; rewriting them restores full speed.
 *
 * Downloads prefer HuggingFace, with a per-component fallback to ModelScope for
 * the repos HF does not carry.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Download, CheckCircle, AlertCircle, Loader2, RefreshCw, Link, Info, Trash2, Zap,
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

// LTX-2's camera moves and in-context controls: function variants that live as
// LoRAs on the shared base rather than as models of their own. Listed under the
// model they belong to, downloadable one at a time, and never part of whether
// the model itself is complete.
function AddonLoras({ model, job, sizes, sizesLoading, onDownload, onExpand, busy }) {
  const [open, setOpen] = useState(false);
  // null means "everything still missing", which is what most people want —
  // unticking is how you take a subset, rather than starting from nothing.
  const [picked, setPicked] = useState(null);

  const addons = model.addons;
  const missing = addons.filter((a) => !a.present);
  const missingPaths = new Set(missing.map((a) => a.path));
  const present = addons.length - missing.length;
  const running = job && job.status === 'running';

  const selected = picked ?? missingPaths;

  const toggle = (path) => setPicked((prev) => {
    const next = new Set(prev ?? missingPaths);
    if (next.has(path)) next.delete(path); else next.add(path);
    return next;
  });

  // Filtered against what is still missing, so a selection made before a
  // download does not survive it as a request to fetch the same files again.
  const selection = [...selected].filter((p) => missingPaths.has(p));
  const selectedBytes = selection.reduce(
    (n, p) => n + (sizes?.[p]?.total_bytes || 0), 0);

  return (
    <div className="mm-addons">
      {/* Sizes are fetched on expand, not with the panel: one hub round trip per LoRA. */}
      <button
        className="mm-disclose mm-addons-toggle"
        onClick={() => { setOpen((v) => !v); if (!open) onExpand(model.key); }}
      >
        {open ? 'Hide' : 'Show'} optional LoRAs — {present}/{addons.length} installed
      </button>

      {open && (
        <>
          <div className="mm-addons-sub">
            Function variants on the shared base: one small file per camera move or
            control mode, not another copy of the model. They appear in the LoRA
            picker once downloaded.
          </div>

          <div className="mm-addon-list">
            {addons.map((a) => (
              <label
                key={a.path}
                className={`mm-addon ${a.present ? 'mm-addon--have' : ''}`}
                title={a.install_path || a.path}
              >
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  disabled={a.present || running || busy}
                  checked={!a.present && selected.has(a.path)}
                  onChange={() => toggle(a.path)}
                />
                <span className="mm-addon-name">{a.name}</span>
                {a.trigger_word && <code className="mm-addon-trigger">{a.trigger_word}</code>}
                <span className="mm-addon-size">
                  {a.present
                    ? <><CheckCircle className="fuk-icon--sm" /> {fmtBytes(a.size_bytes)}</>
                    : sizes?.[a.path]
                      ? fmtBytes(sizes[a.path].total_bytes)
                      : (sizesLoading ? 'sizing…' : '—')}
                </span>
                {a.huggingface_url && (
                  <a className="mm-link" href={a.huggingface_url} target="_blank" rel="noreferrer"
                     onClick={(e) => e.stopPropagation()}>
                    <Link className="fuk-icon--sm" />
                  </a>
                )}
              </label>
            ))}
          </div>

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
              {job.failed.length} LoRA(s) failed — retry, or fetch them by hand from the
              links above.
            </div>
          )}

          <button
            className="fuk-btn fuk-btn-secondary fuk-btn-sm mm-dl"
            disabled={running || busy || selection.length === 0}
            onClick={() => onDownload(model.key, selection)}
          >
            {running
              ? <><Loader2 className="fuk-icon--sm mm-spin" /> downloading</>
              : <><Download className="fuk-icon--sm" />
                  {missing.length === 0 ? ' All installed'
                    : selection.length === 0 ? ' Nothing selected'
                    : ` Download ${selection.length} LoRA${selection.length > 1 ? 's' : ''}`}
                  {selectedBytes > 0 && ` · ${fmtBytes(selectedBytes)}`}</>}
          </button>
        </>
      )}
    </div>
  );
}

function ModelRow({ model, job, loraJob, size, loraSizes, sizesLoading, loraSizesLoading,
                    onToggle, onDownload, onDownloadLoras, onExpandLoras, onDelete, busy }) {
  const [open, setOpen] = useState(false);
  // Delete is two-step: the server returns a plan, the row shows it, and only
  // an explicit second click removes anything.
  const [plan, setPlan] = useState(null);
  const [planning, setPlanning] = useState(false);

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
              <span className="mm-size">{fmtBytes(model.size_on_disk_bytes)} on disk</span>
            )}
            {/* Projected download. `remaining` rather than `total` is the number
                that matters: a model sharing the Qwen VAE or a Wan encoder with
                something you already have pulls far less than its full size. */}
            {size && !model.downloaded && size.remaining_bytes > 0 && (
              <span className="mm-size mm-size--fetch">
                {fmtBytes(size.remaining_bytes)} to fetch
                {size.remaining_bytes !== size.total_bytes &&
                  ` of ${fmtBytes(size.total_bytes)}`}
                {!size.complete && ' +unknown'}
              </span>
            )}
            {sizesLoading && !size && !model.downloaded && model.downloadable && (
              <span className="mm-size">sizing…</span>
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

          {model.addons?.length > 0 && (
            <AddonLoras
              model={model}
              job={loraJob}
              sizes={loraSizes}
              sizesLoading={loraSizesLoading}
              onDownload={onDownloadLoras}
              onExpand={onExpandLoras}
              busy={busy}
            />
          )}

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
          <div className="mm-actions">
            <button
              className="fuk-btn fuk-btn-secondary fuk-btn-sm mm-dl"
              disabled={running || busy}
              onClick={() => onDownload(model.key)}
            >
              {running
                ? <><Loader2 className="fuk-icon--sm mm-spin" /> downloading</>
                : <><Download className="fuk-icon--sm" /> {model.downloaded ? 'Verify' : 'Download'}</>}
            </button>
            {(model.downloaded || model.partial) && (
              <button
                className="fuk-btn fuk-btn-secondary fuk-btn-sm mm-del"
                disabled={running || busy || planning}
                onClick={async () => {
                  if (plan) { setPlan(null); return; }   // second click on "Cancel"
                  setPlanning(true);
                  setPlan(await onDelete(model.key, false));
                  setPlanning(false);
                }}
              >
                <Trash2 className="fuk-icon--sm" /> {plan ? 'Cancel' : 'Delete'}
              </button>
            )}
          </div>
        )}
      </div>

      {/* Deletion plan — shown before anything is removed. Shared components
          are never deleted, and saying so here is the whole point: otherwise
          removing Krea-2 looks like it would take the Qwen VAE with it. */}
      {plan && (
        <div className="mm-delete-plan">
          <div className="mm-delete-head">
            <AlertCircle className="fuk-icon--sm" />
            Delete {fmtBytes(plan.reclaim_bytes)} from disk? This cannot be undone —
            only re-downloaded.
          </div>
          {plan.removable.length > 0 && (
            <ul className="mm-delete-list">
              {plan.removable.map((r) => (
                <li key={`${r.model_id}:${r.pattern}`}>
                  <code>{r.model_id} → {r.pattern}</code> <span>{fmtBytes(r.size_bytes)}</span>
                </li>
              ))}
            </ul>
          )}
          {plan.kept_shared.length > 0 && (
            <div className="mm-delete-kept">
              Kept — shared with other models:
              <ul className="mm-delete-list">
                {plan.kept_shared.map((k) => (
                  <li key={`${k.model_id}:${k.pattern}`}>
                    <code>{k.pattern}</code>{' '}
                    <span>{fmtBytes(k.size_bytes)} · used by {k.shared_with.join(', ')}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {plan.removable.length === 0 ? (
            <div className="mm-delete-kept">
              Nothing to remove — every component is shared with another model.
            </div>
          ) : (
            <button
              className="fuk-btn fuk-btn-sm mm-del-confirm"
              disabled={busy}
              onClick={async () => { await onDelete(model.key, true); setPlan(null); }}
            >
              <Trash2 className="fuk-icon--sm" /> Confirm delete {fmtBytes(plan.reclaim_bytes)}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Storage maintenance — NAND refresh
// ============================================================================

// Flash loses charge. Weights are written once and read forever, so they age
// until a load that took seconds takes minutes, with nothing corrupt and SMART
// perfectly clean. Rewriting a file puts it back on fresh cells. Scan measures
// and changes nothing; Refresh rewrites what came back slow.
function StorageCard({ state, onStart, onCancel, busy }) {
  const [open, setOpen] = useState(false);
  // Refresh is an hour of disk work and hundreds of GB of writes, so it asks
  // once. Scan is free and does not.
  const [confirming, setConfirming] = useState(false);

  if (!state) return null;

  const job = state.job;
  const running = job && job.status === 'running';
  const report = state.last_refresh || state.last_scan;
  // What a refresh would rewrite, if anything has been measured yet.
  const aged = state.last_scan?.aged_bytes ?? state.last_refresh?.aged_bytes ?? 0;
  // Bytes, not file count: one 38GB checkpoint is a quarter of the sweep on its
  // own, so a file-counted bar would sit still and then jump.
  const pct = running && job.bytes_total
    ? Math.min(100, (job.bytes_done / job.bytes_total) * 100) : 0;

  if (!state.supported) {
    return (
      <div className="fuk-card mm-card">
        <span className="fuk-label">Storage maintenance</span>
        <div className="mm-sub mm-sub--card">
          <Info className="fuk-icon--sm" /> Not applicable here — {state.reason}.
        </div>
      </div>
    );
  }

  return (
    <div className="fuk-card mm-card">
      <span className="fuk-label">Storage maintenance</span>
      <div className="mm-sub mm-sub--card">
        Flash cells leak charge, so weights that sit unread for months come back
        at a fraction of their original speed — nothing is corrupt, reads just get
        slow. Rewriting a file restores it. <b>Check</b> measures and changes
        nothing; <b>Refresh</b> rewrites whatever reads below
        {' '}{state.defaults.threshold_mbps} MB/s.
      </div>

      <div className="mm-maint-actions">
        <button
          className="fuk-btn fuk-btn-secondary fuk-btn-sm"
          disabled={running || busy}
          onClick={() => onStart('scan')}
        >
          {running && job.mode === 'scan'
            ? <><Loader2 className="fuk-icon--sm mm-spin" /> checking</>
            : <><Zap className="fuk-icon--sm" /> Check read speed</>}
        </button>
        <button
          className="fuk-btn fuk-btn-secondary fuk-btn-sm"
          disabled={running || busy}
          onClick={() => setConfirming((v) => !v)}
        >
          {running && job.mode === 'refresh'
            ? <><Loader2 className="fuk-icon--sm mm-spin" /> refreshing</>
            : <><RefreshCw className="fuk-icon--sm" /> {confirming ? 'Cancel' : 'Refresh aged files'}</>}
        </button>
        {running && (
          <button className="fuk-btn fuk-btn-secondary fuk-btn-sm" onClick={onCancel}>
            Stop
          </button>
        )}
        <span className="mm-maint-free">
          {state.device && <>{state.fstype} on <code>{state.device}</code> · </>}
          {fmtBytes(state.free_bytes)} free
        </span>
      </div>

      {confirming && !running && (
        <div className="mm-maint-confirm">
          Every file that reads below {state.defaults.threshold_mbps} MB/s gets copied
          and renamed over itself — {aged
            ? <>about {fmtBytes(aged)} of writes, from the last check</>
            : <>up to the whole library, since nothing has been checked yet</>}
          , and tens of minutes. Safe to run while the server is up: a model already
          loading finishes from the old copy. It can be stopped at any point, and
          what was already rewritten stays rewritten.
          <button
            className="fuk-btn fuk-btn-sm mm-maint-go"
            disabled={busy}
            onClick={() => { setConfirming(false); onStart('refresh'); }}
          >
            <RefreshCw className="fuk-icon--sm" /> Start refresh
          </button>
        </div>
      )}

      {running && (
        <div className="mm-progress">
          <div className="mm-progress-bar" style={{ width: `${pct}%` }} />
          <span className="mm-progress-text">
            {job.completed}/{job.total} · {fmtBytes(job.bytes_done)} of{' '}
            {fmtBytes(job.bytes_total)}
            {job.rewritten > 0 && ` · ${job.rewritten} rewritten`}
            {' — '}{job.current || 'starting…'}
          </span>
        </div>
      )}

      {job && job.status === 'failed' && (
        <div className="mm-model-missing">Sweep failed — {job.error}</div>
      )}

      {!running && report && (
        <div className="mm-maint-report">
          <div>
            <b>{report.files_measured}</b> files measured
            {' · '}median <b>{report.median_mbps} MB/s</b>
            {' · '}slowest <b>{report.slowest_mbps} MB/s</b>
            {report.files_aged > 0
              ? <> · <span className="mm-maint-aged">{report.files_aged} aged
                  ({fmtBytes(report.aged_bytes)})</span></>
              : <> · <span className="mm-maint-ok">
                  <CheckCircle className="fuk-icon--sm" /> none aged</span></>}
          </div>
          {report.files_rewritten > 0 && (
            <div>
              Rewrote {report.files_rewritten} file(s),{' '}
              {fmtBytes(report.bytes_rewritten)} in{' '}
              {(report.elapsed_s / 60).toFixed(0)} min
              {report.mean_gain && ` — ${report.mean_gain}× faster on average`}
            </div>
          )}
          <div className="mm-maint-when">
            {report.mode === 'refresh' ? 'Refreshed' : 'Checked'} {report.finished_at}
            {report.cancelled && ' (stopped early)'}
          </div>

          {/* The slowest rows, when a sweep just ran in this session. The report
              on disk keeps the summary only, so this is empty after a reload. */}
          {job && job.files?.length > 0 && (
            <>
              <button className="mm-disclose" onClick={() => setOpen((v) => !v)}>
                {open ? 'Hide' : 'Show'} the slowest files
              </button>
              {open && (
                <table className="mm-parts mm-maint-table">
                  <tbody>
                    {job.files.slice(0, 15).map((f) => (
                      <tr key={f.path}
                          className={f.before_mbps < report.threshold_mbps
                            ? 'mm-part--missing' : ''}>
                        <td>{f.rewritten ? '↻' : '·'}</td>
                        <td>{f.before_mbps ?? '?'}{f.after_mbps && ` → ${f.after_mbps}`} MB/s</td>
                        <td>{fmtBytes(f.size)}</td>
                        <td><code>{f.rel}</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
        </div>
      )}
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
  // model key -> remote size. Fetched separately because each one is a network
  // round trip to the hub; the panel renders immediately and fills these in.
  const [sizes, setSizes] = useState({});
  const [sizesLoading, setSizesLoading] = useState(false);
  // model key -> { lora path -> size }. Fetched when a LoRA list is expanded,
  // because each add-on is its own single-file repo and therefore its own
  // round trip.
  const [loraSizes, setLoraSizes] = useState({});
  const [loraSizesLoading, setLoraSizesLoading] = useState({});
  // Storage maintenance: filesystem facts, the running sweep, the last report.
  const [maint, setMaint] = useState(null);
  const pollRef = useRef(null);
  const maintPollRef = useRef(null);

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

  const loadSizes = useCallback(async (keys) => {
    if (!keys.length) return;
    setSizesLoading(true);
    try {
      const res = await fetch(`${API_URL}/models/manage/sizes?keys=${keys.join(',')}`);
      if (res.ok) {
        const { sizes: fresh } = await res.json();
        setSizes((prev) => ({ ...prev, ...fresh }));
      }
    } catch { /* sizes are advisory — a failure just leaves them blank */ }
    finally { setSizesLoading(false); }
  }, []);

  const loadMaint = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/models/manage/maintenance`);
      if (res.ok) setMaint(await res.json());
    } catch { /* the card just does not render — never blocks the panel */ }
  }, []);

  useEffect(() => { load(); loadMaint(); }, [load, loadMaint]);

  // A sweep started before this tab was opened — or in another browser tab —
  // is still running server-side, so the card picks it up rather than looking idle.
  useEffect(() => {
    if (maint?.job?.status !== 'running') {
      if (maintPollRef.current) {
        clearInterval(maintPollRef.current); maintPollRef.current = null;
      }
      return undefined;
    }
    if (maintPollRef.current) return undefined;
    maintPollRef.current = setInterval(loadMaint, 1000);
    return () => {
      if (maintPollRef.current) {
        clearInterval(maintPollRef.current); maintPollRef.current = null;
      }
    };
  }, [maint, loadMaint]);

  // Size anything not fully downloaded. Downloaded models already report their
  // real on-disk size, so asking the hub about them would be a wasted request.
  useEffect(() => {
    if (!data) return;
    const need = data.models
      .filter((m) => m.downloadable && !m.downloaded && !sizes[m.key])
      .map((m) => m.key);
    if (need.length) loadSizes(need);
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // confirm=false returns the plan and deletes nothing; confirm=true executes.
  const handleDelete = async (key, confirm) => {
    if (confirm) setBusy(true);
    try {
      const res = await fetch(`${API_URL}/models/manage/${key}/delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const result = await res.json();
      if (confirm) {
        if (result.failed?.length) {
          setError(`${key}: ${result.failed.length} path(s) could not be removed — `
            + result.failed.map((f) => f.error).join('; '));
        }
        // The freed space changes both the on-disk figure and, for anything
        // that shared files with it, the projected download.
        setSizes({});
        await load();
      }
      return result;
    } catch (e) {
      setError(`Delete failed for ${key}: ${e}`);
      return null;
    } finally {
      if (confirm) setBusy(false);
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

  // Filed under "<key>:loras" so add-on progress and base-model progress can
  // run at once without one overwriting the other's bar.
  const handleDownloadLoras = async (key, paths) => {
    try {
      const res = await fetch(`${API_URL}/models/manage/${key}/loras/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paths }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const { job_id, total } = await res.json();
      setJobs((prev) => ({
        ...prev,
        [`${key}:loras`]: {
          id: job_id, status: 'running', completed: 0, total, current: null, failed: [],
        },
      }));
    } catch (e) {
      setError(`LoRA download failed to start for ${key}: ${e}`);
    }
  };

  const handleMaintStart = async (mode) => {
    try {
      const res = await fetch(`${API_URL}/models/manage/maintenance/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const { job } = await res.json();
      // Seed the job locally so the progress bar and the poll start on this
      // render rather than a second later.
      setMaint((m) => ({ ...m, job }));
      setError(null);
    } catch (e) {
      setError(`Could not start the ${mode}: ${e}`);
    }
  };

  const handleMaintCancel = async () => {
    try {
      await fetch(`${API_URL}/models/manage/maintenance/cancel`, { method: 'POST' });
    } catch { /* the sweep either stops or it does not; the poll will say */ }
    loadMaint();
  };

  const handleExpandLoras = useCallback(async (key) => {
    if (loraSizes[key]) return;
    setLoraSizesLoading((prev) => ({ ...prev, [key]: true }));
    try {
      const res = await fetch(`${API_URL}/models/manage/${key}/loras/sizes`);
      if (res.ok) {
        const { sizes: fresh } = await res.json();
        setLoraSizes((prev) => ({ ...prev, [key]: fresh }));
      }
    } catch { /* advisory, same as the model sizes */ }
    finally { setLoraSizesLoading((prev) => ({ ...prev, [key]: false })); }
  }, [loraSizes]);

  // Wrapped in .mm-panel so these inherit the same padding and scroll container
  // as the loaded state — otherwise they render flush against the tab edge.
  if (error && !data) {
    return (
      <div className="mm-panel">
        <div className="mm-error"><AlertCircle className="fuk-icon--sm" /> {error}</div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="mm-panel">
        <div className="mm-loading"><Loader2 className="fuk-icon--sm mm-spin" /> Loading…</div>
      </div>
    );
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
                loraJob={jobs[`${m.key}:loras`]}
                size={sizes[m.key]}
                loraSizes={loraSizes[m.key]}
                sizesLoading={sizesLoading}
                loraSizesLoading={!!loraSizesLoading[m.key]}
                onToggle={handleToggle}
                onDownload={handleDownload}
                onDownloadLoras={handleDownloadLoras}
                onExpandLoras={handleExpandLoras}
                onDelete={handleDelete}
                busy={busy}
              />
            ))}
          </div>
        </div>
      ))}

      <StorageCard
        state={maint}
        onStart={handleMaintStart}
        onCancel={handleMaintCancel}
        busy={busy}
      />

      <div className="mm-footnote">
        Weights live under <code>{data.models_root}</code>. Sizes are per model, so
        models sharing a repo — the Qwen text encoder, the Wan VAE — report the same
        files more than once.
        {data.loras_root && (
          <> Add-on LoRAs are linked into <code>{data.loras_root}</code> from that same
          cache, so they cost their size once.</>
        )}
      </div>
    </div>
  );
}
