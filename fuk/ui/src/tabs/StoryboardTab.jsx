/**
 * Storyboard Tab
 *
 * Project-level pre-production: shot list, panel grid, live globals (subjects
 * + mood). Subjects become `#markers` available across all shots; mood is
 * appended to every prompt at generation time.
 *
 * The storyboard does NOT replace per-shot files — it sits above them. Each
 * panel maps to one shot. "Send to Image/Video" pushes the panel's prompt
 * draft into the corresponding shot's positive_prompt (raw, markers intact)
 * and switches the active shot + tab.
 *
 * All inputs autosave on a short debounce — no manual Save buttons.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Footer from '../components/Footer';
import { Plus, Trash2, Camera, Film, Loader2, AlertCircle, CheckCircle, Image as ImageIcon, X } from '../components/Icons';
import { useStoryboard } from '../hooks/useStoryboard';
import { sendPanelToImage, sendPanelToVideo, snapshotStoryboard, fetchSnapshots, restoreSnapshot } from '../utils/storyboardApi';
import MarkerTextarea from '../components/MarkerTextarea';
import { buildImageUrl, generateRandomSeed } from '../utils/constants';

// Debounce window for autosave. Short enough to feel snappy, long enough to
// coalesce a paragraph of typing into one PUT.
const AUTOSAVE_MS = 700;

/**
 * Run `effect(value)` after the value has been still for `delay` ms.
 * Skips the initial mount so we don't auto-write when a card first renders.
 */
function useDebouncedEffect(value, delay, effect) {
  const firstRef = useRef(true);
  useEffect(() => {
    if (firstRef.current) {
      firstRef.current = false;
      return;
    }
    const t = setTimeout(() => effect(value), delay);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, delay]);
}

export default function StoryboardTab({ activeTab, setActiveTab, project, config }) {
  const sb = useStoryboard(project);
  const { manifest, loading, error } = sb;

  const projectShots = project?.shots || [];

  // Panels we know about — union of manifest sequence and project shot files
  // so the tab works as a shot index even before the user creates any panels.
  const panelOrder = useMemo(() => {
    const seq = manifest?.sequence || [];
    const merged = [...seq];
    for (const shot of projectShots) {
      if (!merged.includes(shot)) merged.push(shot);
    }
    return merged;
  }, [manifest, projectShots]);

  if (!project?.projectFolder) {
    return (
      <>
        <div className="fuk-tab-empty">
          <p>Open a project folder to start a storyboard.</p>
        </div>
        <Footer activeTab={activeTab} setActiveTab={setActiveTab} />
      </>
    );
  }

  return (
    <>
      <div className="storyboard-tab">
        {error && (
          <div className="storyboard-error">
            <AlertCircle /> {error}
          </div>
        )}
        {loading && !manifest && (
          <div className="storyboard-loading"><Loader2 className="animate-spin" /> Loading storyboard…</div>
        )}

        {manifest && (
          <>
            <SnapshotControl sb={sb} />
            <GlobalsSection sb={sb} manifest={manifest} config={config} />
            <PanelGrid
              sb={sb}
              manifest={manifest}
              panelOrder={panelOrder}
              project={project}
              setActiveTab={setActiveTab}
              config={config}
            />
          </>
        )}
      </div>
      <Footer activeTab={activeTab} setActiveTab={setActiveTab} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Status pip — tiny inline saving/saved indicator
// ---------------------------------------------------------------------------

function SaveStatus({ state }) {
  if (state === 'saving') return <span className="storyboard-save-pip storyboard-save-pip--busy"><Loader2 className="animate-spin" /> saving</span>;
  if (state === 'saved') return <span className="storyboard-save-pip storyboard-save-pip--ok"><CheckCircle /> saved</span>;
  if (state === 'error') return <span className="storyboard-save-pip storyboard-save-pip--err">save failed</span>;
  return null;
}

// ---------------------------------------------------------------------------
// Snapshot control — storyboard versioning
// ---------------------------------------------------------------------------

function SnapshotControl({ sb }) {
  const [busy, setBusy] = useState(false);
  const [snapshots, setSnapshots] = useState([]);
  const [selected, setSelected] = useState('');
  const [lastSaved, setLastSaved] = useState(null);
  const [err, setErr] = useState(null);

  const loadSnapshots = useCallback(async () => {
    try {
      const { snapshots: list } = await fetchSnapshots();
      setSnapshots(list);
    } catch (e) {
      // Non-fatal — the dropdown will just be empty
    }
  }, []);

  useEffect(() => { loadSnapshots(); }, [loadSnapshots]);

  const handleSnapshot = useCallback(async () => {
    setBusy(true);
    setErr(null);
    try {
      const { filename } = await snapshotStoryboard();
      setLastSaved(filename);
      await loadSnapshots();
      setTimeout(() => setLastSaved(null), 3000);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }, [loadSnapshots]);

  const handleLoad = useCallback(async () => {
    if (!selected) return;
    if (!confirm(`Load "${selected}"?`)) return;
    setBusy(true);
    setErr(null);
    try {
      await restoreSnapshot(selected);
      await sb.refresh();
      setSelected('');
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }, [selected, sb]);

  return (
    <div className="storyboard-snapshot-bar">
      <button
        type="button"
        className="fuk-btn fuk-btn-secondary"
        onClick={handleSnapshot}
        disabled={busy}
        title="Save a dated copy of the storyboard"
      >
        {busy && !selected ? <Loader2 className="animate-spin" /> : null}
        Snapshot
      </button>

      <select
        className="fuk-input storyboard-snapshot-select"
        value={selected}
        onChange={e => setSelected(e.target.value)}
        disabled={snapshots.length === 0 || busy}
      >
        <option value="">{snapshots.length === 0 ? 'No snapshots' : 'Load snapshot…'}</option>
        {snapshots.map(s => (
          <option key={s.filename} value={s.filename}>
            {s.filename}
          </option>
        ))}
      </select>

      <button
        type="button"
        className="fuk-btn fuk-btn-secondary"
        onClick={handleLoad}
        disabled={!selected || busy}
      >
        {busy && selected ? <Loader2 className="animate-spin" /> : null}
        Load
      </button>

      {lastSaved && (
        <span className="storyboard-snapshot-ok">
          <CheckCircle /> {lastSaved}
        </span>
      )}
      {err && <span className="storyboard-error" style={{ fontSize: '0.8rem' }}>{err}</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Globals: subjects + mood + specs
// ---------------------------------------------------------------------------

function GlobalsSection({ sb, manifest, config }) {
  return (
    <section className="storyboard-globals">
      <h2 className="storyboard-section-title">Globals</h2>
      <p className="storyboard-section-hint">
        Project tags become <code>#markers</code> available in every shot. Mood is appended to every prompt at generation. Active LoRAs surface their trigger markers and caption phrases in <code>#</code> autocomplete across the project. Type <code>#</code> in any field below to reference an existing marker.
      </p>
      <div className="storyboard-globals-grid">
        <TagsEditor sb={sb} tags={manifest.globals?.tags || []} />
        <MoodEditor sb={sb} mood={manifest.globals?.mood || ''} />
      </div>
      <GlobalSeedsEditor
        sb={sb}
        imageSeed={manifest.globals?.image_seed ?? null}
        videoSeed={manifest.globals?.video_seed ?? null}
      />
      <ActiveLorasPicker
        sb={sb}
        active={manifest.globals?.active_loras || []}
        available={config?.models?.loras || []}
      />
    </section>
  );
}

// ---------------------------------------------------------------------------
// Active LoRAs picker — drives caption autocomplete project-wide.
// ---------------------------------------------------------------------------

function ActiveLorasPicker({ sb, active, available }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  // Normalize available entries; some installs ship them as bare strings.
  const items = useMemo(() => (available || []).map(l => (
    typeof l === 'string'
      ? { key: l, name: l, model: null, trigger_word: null }
      : { key: l.key, name: l.name || l.description || l.key, model: l.model || null, trigger_word: l.trigger_word || null }
  )).filter(l => l.key), [available]);

  // Group by model so multi-pipeline installs stay scannable.
  const grouped = useMemo(() => {
    const groups = new Map();
    for (const l of items) {
      const k = typeof l.model === 'string' && l.model ? l.model : 'other';
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(l);
    }
    return [...groups.entries()]
      .sort(([a], [b]) => String(a).localeCompare(String(b)))
      .map(([model, list]) => ({ model, list: list.sort((a, b) => String(a.name).localeCompare(String(b.name))) }));
  }, [items]);

  const activeSet = useMemo(() => new Set(active), [active]);

  const toggle = useCallback(async (key) => {
    const next = activeSet.has(key)
      ? active.filter(k => k !== key)
      : [...active, key];
    setBusy(true);
    setErr(null);
    try {
      await sb.updateActiveLoras(next);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }, [active, activeSet, sb]);

  const clearAll = useCallback(async () => {
    if (!active.length) return;
    setBusy(true);
    setErr(null);
    try {
      await sb.updateActiveLoras([]);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }, [active, sb]);

  return (
    <div className="storyboard-loras">
      <div className="storyboard-subsection-header">
        <h3>Active LoRAs</h3>
        <span className="storyboard-count">{active.length}</span>
        {active.length > 0 && (
          <button
            type="button"
            className="storyboard-loras-clear"
            onClick={clearAll}
            disabled={busy}
            title="Deactivate all"
          >clear</button>
        )}
      </div>
      <p className="storyboard-section-hint">
        Click to toggle. Activating a LoRA surfaces its trigger marker (e.g. <code>#italian_horror</code>) and any caption phrases the LoRA shipped with — they appear in <code>#</code> autocomplete everywhere.
      </p>
      {items.length === 0 ? (
        <div className="storyboard-empty-hint">No LoRAs found in your install.</div>
      ) : (
        <div className="storyboard-loras-groups">
          {grouped.map(g => (
            <div key={g.model} className="storyboard-loras-group">
              <div className="storyboard-loras-group-header">{g.model}</div>
              <ul className="storyboard-loras-chip-list">
                {g.list.map(l => {
                  const isActive = activeSet.has(l.key);
                  return (
                    <li key={l.key}>
                      <button
                        type="button"
                        className={`storyboard-lora-chip ${isActive ? 'storyboard-lora-chip--on' : ''}`}
                        onClick={() => toggle(l.key)}
                        disabled={busy}
                        title={l.trigger_word ? `trigger: ${l.trigger_word}` : l.key}
                      >
                        {l.name}
                        {l.trigger_word && (
                          <span className="storyboard-lora-chip-trigger">#{l.trigger_word}</span>
                        )}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      )}
      {err && <div className="storyboard-error">{err}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tags: project-local `#marker` shorthand. Lives in the storyboard manifest,
// shadows workspace tags on marker collision.
// ---------------------------------------------------------------------------

const TAG_KINDS = ['character', 'prop', 'location', 'other'];

function TagsEditor({ sb, tags }) {
  const [draft, setDraft] = useState({ name: '', value: '', category: 'character' });
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState(null);

  const handleAdd = useCallback(async (e) => {
    e.preventDefault();
    if (!draft.name.trim() || !draft.value.trim()) return;
    setSubmitting(true);
    setErr(null);
    try {
      await sb.createTag({
        name: draft.name,
        value: draft.value,
        category: draft.category || null,
      });
      setDraft({ name: '', value: '', category: 'character' });
    } catch (e) {
      setErr(e.message);
    } finally {
      setSubmitting(false);
    }
  }, [draft, sb]);

  return (
    <div className="storyboard-tags">
      <div className="storyboard-subsection-header">
        <h3>Project tags</h3>
        <span className="storyboard-count">{tags.length}</span>
      </div>
      <p className="storyboard-section-hint">
        Project-local <code>#markers</code> — characters, props, locations, or anything else. Lives with the storyboard, not the global tag library. Shadows workspace tags on marker collision.
      </p>
      <ul className="storyboard-subject-list">
        {tags.map(t => (
          <TagRow key={t.id} tag={t} sb={sb} />
        ))}
        {tags.length === 0 && (
          <li className="storyboard-empty-hint">No project tags yet. Add one below to create a <code>#marker</code> scoped to this storyboard.</li>
        )}
      </ul>
      <form className="storyboard-subject-form" onSubmit={handleAdd}>
        <input
          className="fuk-input"
          placeholder="name (e.g. sarah, studio_kettle)"
          value={draft.name}
          onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
          disabled={submitting}
        />
        <select
          className="fuk-input"
          value={draft.category}
          onChange={e => setDraft(d => ({ ...d, category: e.target.value }))}
          disabled={submitting}
        >
          {TAG_KINDS.map(k => <option key={k} value={k}>{k}</option>)}
        </select>
        <MarkerTextarea
          className="fuk-textarea"
          placeholder="value — what #marker expands to at generation time"
          rows={2}
          value={draft.value}
          onChange={(v) => setDraft(d => ({ ...d, value: v }))}
          disabled={submitting}
        />
        <button
          type="submit"
          className="fuk-btn fuk-btn-primary"
          disabled={submitting || !draft.name.trim() || !draft.value.trim()}
        >
          {submitting ? <Loader2 className="animate-spin" /> : <Plus />} Add tag
        </button>
        {err && <div className="storyboard-error">{err}</div>}
      </form>
    </div>
  );
}

function TagRow({ tag, sb }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ ...tag, category: tag.category || 'other' });
  const [status, setStatus] = useState('idle');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!editing) setDraft({ ...tag, category: tag.category || 'other' });
  }, [tag, editing]);

  useDebouncedEffect(draft, AUTOSAVE_MS, async (current) => {
    if (!editing) return;
    if (
      current.name === tag.name &&
      current.value === tag.value &&
      (current.category || '') === (tag.category || '')
    ) return;
    if (!current.name.trim() || !current.value.trim()) return;
    setStatus('saving');
    try {
      await sb.updateTag(tag.id, {
        name: current.name,
        value: current.value,
        category: current.category || null,
      });
      setStatus('saved');
    } catch (e) {
      setStatus('error');
    }
  });

  const handleDelete = useCallback(async () => {
    if (!confirm(`Delete tag "${tag.name}"?`)) return;
    setBusy(true);
    try {
      await sb.deleteTag(tag.id);
    } catch (e) {
      alert(e.message);
    } finally {
      setBusy(false);
    }
  }, [sb, tag.id, tag.name]);

  if (editing) {
    return (
      <li className="storyboard-subject-row storyboard-subject-row--editing">
        <input
          className="fuk-input"
          value={draft.name}
          onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
        />
        <select
          className="fuk-input"
          value={draft.category}
          onChange={e => setDraft(d => ({ ...d, category: e.target.value }))}
        >
          {TAG_KINDS.map(k => <option key={k} value={k}>{k}</option>)}
        </select>
        <MarkerTextarea
          className="fuk-textarea"
          rows={2}
          value={draft.value}
          onChange={(v) => setDraft(d => ({ ...d, value: v }))}
        />
        <div className="storyboard-subject-row-actions">
          <SaveStatus state={status} />
          <button className="fuk-btn fuk-btn-secondary" onClick={() => { setDraft({ ...tag, category: tag.category || 'other' }); setEditing(false); setStatus('idle'); }}>Done</button>
        </div>
      </li>
    );
  }

  return (
    <li className="storyboard-subject-row">
      <div className="storyboard-subject-summary">
        <code className="storyboard-subject-marker">{tag.marker}</code>
        <span className="storyboard-subject-kind">{tag.category || 'other'}</span>
        <span className="storyboard-subject-desc">{tag.value}</span>
      </div>
      <div className="storyboard-subject-row-actions">
        <button className="fuk-btn fuk-btn-secondary" onClick={() => setEditing(true)} disabled={busy}>Edit</button>
        <button className="fuk-btn fuk-btn-secondary" onClick={handleDelete} disabled={busy}><Trash2 /></button>
      </div>
    </li>
  );
}

function MoodEditor({ sb, mood }) {
  const [draft, setDraft] = useState(mood);
  const [status, setStatus] = useState('idle');
  const localDirtyRef = useRef(false);

  // External update (e.g. project switch): pull the new mood in. Skip if the
  // user is mid-edit so we don't wipe their unsaved changes.
  useEffect(() => {
    if (!localDirtyRef.current) setDraft(mood);
  }, [mood]);

  useDebouncedEffect(draft, AUTOSAVE_MS, async (current) => {
    if (current === mood) return;
    setStatus('saving');
    try {
      await sb.updateMood(current);
      localDirtyRef.current = false;
      setStatus('saved');
    } catch (e) {
      setStatus('error');
    }
  });

  return (
    <div className="storyboard-mood">
      <div className="storyboard-subsection-header">
        <h3>Mood / Setting</h3>
        <SaveStatus state={status} />
      </div>
      <p className="storyboard-section-hint">
        One sentence. Style + environment merged. Appended to every shot's prompt at generation as <code>Mood: …</code>.
      </p>
      <MarkerTextarea
        className="fuk-textarea"
        rows={4}
        value={draft}
        placeholder="e.g. neo-noir, low-key lighting, rain-soaked Tokyo back alleys at night"
        onChange={(v) => { setDraft(v); localDirtyRef.current = true; }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Global seeds — uint32 baked into a new shot's image/video tab on first send.
// Per-shot edits in the Image/Video tab persist independently afterwards.
// ---------------------------------------------------------------------------

function GlobalSeedsEditor({ sb, imageSeed, videoSeed }) {
  return (
    <div className="storyboard-seeds">
      <div className="storyboard-subsection-header">
        <h3>Default seeds</h3>
      </div>
      <p className="storyboard-section-hint">
        Baked into a new shot's seed slot the first time you send it to Image/Video. The shot's seed then lives independently — changing the global later does NOT touch existing shots. Leave blank to let each shot stay on random.
      </p>
      <div className="storyboard-seeds-row">
        <SeedField
          label="Image"
          value={imageSeed}
          onSave={(v) => sb.updateImageSeed(v)}
        />
        <SeedField
          label="Video"
          value={videoSeed}
          onSave={(v) => sb.updateVideoSeed(v)}
        />
      </div>
    </div>
  );
}

function SeedField({ label, value, onSave }) {
  const [draft, setDraft] = useState(value == null ? '' : String(value));
  const [status, setStatus] = useState('idle');
  const localDirtyRef = useRef(false);

  useEffect(() => {
    if (!localDirtyRef.current) setDraft(value == null ? '' : String(value));
  }, [value]);

  useDebouncedEffect(draft, AUTOSAVE_MS, async (current) => {
    const parsed = current === '' ? null : Number(current);
    if (parsed === value) return;
    if (parsed !== null && (!Number.isInteger(parsed) || parsed < 0 || parsed > 4294967295)) {
      setStatus('error');
      return;
    }
    setStatus('saving');
    try {
      await onSave(parsed);
      localDirtyRef.current = false;
      setStatus('saved');
    } catch (e) {
      setStatus('error');
    }
  });

  const handleRandomize = () => {
    setDraft(String(generateRandomSeed()));
    localDirtyRef.current = true;
  };

  const handleClear = () => {
    setDraft('');
    localDirtyRef.current = true;
  };

  return (
    <label className="storyboard-seed-field">
      <span className="storyboard-seed-field-label">{label}</span>
      <div className="storyboard-seed-field-row">
        <input
          type="number"
          className="fuk-input"
          value={draft}
          min="0"
          max="4294967295"
          step="1"
          placeholder="(random per shot)"
          onChange={e => { setDraft(e.target.value); localDirtyRef.current = true; }}
        />
        <button
          type="button"
          className="fuk-btn fuk-btn-secondary"
          onClick={handleRandomize}
          title="Generate a random seed"
        >Random</button>
        <button
          type="button"
          className="fuk-btn fuk-btn-secondary"
          onClick={handleClear}
          disabled={draft === ''}
          title="Clear — new shots stay on random"
        ><X /></button>
        <SaveStatus state={status} />
      </div>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Panel grid: one card per shot
// ---------------------------------------------------------------------------

function PanelGrid({ sb, manifest, panelOrder, project, setActiveTab, config }) {
  return (
    <section className="storyboard-panels">
      <h2 className="storyboard-section-title">Shots</h2>
      <p className="storyboard-section-hint">
        Each card is one shot. Imagery describes the still; Action describes motion. Use <code>#markers</code> from subjects above — they resolve live at generation.
      </p>
      <div className="storyboard-panel-grid">
        {panelOrder.map(shotId => (
          <PanelCard
            key={shotId}
            shotId={shotId}
            panel={manifest.panels?.[shotId]}
            sb={sb}
            project={project}
            setActiveTab={setActiveTab}
            config={config}
          />
        ))}
        <AddPanelCard sb={sb} existingShots={panelOrder} project={project} />
      </div>
    </section>
  );
}

function PanelCard({ shotId, panel, sb, project, setActiveTab, config }) {
  const imageModels = config?.models?.image_models || [];
  const videoModels = config?.models?.video_models || [];
  const [draft, setDraft] = useState({
    imagery_prompt: panel?.imagery_prompt || '',
    action_prompt: panel?.action_prompt || '',
    duration_seconds: panel?.duration_seconds ?? '',
    notes: panel?.notes || '',
    image_model: panel?.image_model || '',
    video_model: panel?.video_model || '',
  });
  const [status, setStatus] = useState('idle');
  const [sending, setSending] = useState(null);
  const [sendResult, setSendResult] = useState(null); // { target, kind: 'ok' | 'err', message? }
  const localDirtyRef = useRef(false);

  const isProjectShot = project?.shots?.includes(shotId);

  // External update: pull in panel state if it changes (e.g. reorder, refresh).
  // Skip when the user is actively editing so we don't overwrite their typing.
  useEffect(() => {
    if (localDirtyRef.current) return;
    setDraft({
      imagery_prompt: panel?.imagery_prompt || '',
      action_prompt: panel?.action_prompt || '',
      duration_seconds: panel?.duration_seconds ?? '',
      notes: panel?.notes || '',
      image_model: panel?.image_model || '',
      video_model: panel?.video_model || '',
    });
  }, [
    panel?.imagery_prompt,
    panel?.action_prompt,
    panel?.duration_seconds,
    panel?.notes,
    panel?.image_model,
    panel?.video_model,
  ]);

  const update = (field, value) => {
    setDraft(d => ({ ...d, [field]: value }));
    localDirtyRef.current = true;
  };

  // Autosave on debounce. Coalesces the four fields into a single PUT.
  useDebouncedEffect(draft, AUTOSAVE_MS, async (current) => {
    // No-op if nothing actually changed vs. the manifest state.
    if (
      current.imagery_prompt === (panel?.imagery_prompt || '') &&
      current.action_prompt === (panel?.action_prompt || '') &&
      String(current.duration_seconds ?? '') === String(panel?.duration_seconds ?? '') &&
      current.notes === (panel?.notes || '') &&
      (current.image_model || '') === (panel?.image_model || '') &&
      (current.video_model || '') === (panel?.video_model || '')
    ) return;
    setStatus('saving');
    try {
      await sb.upsertPanel(shotId, {
        ...current,
        duration_seconds: current.duration_seconds === '' ? null : Number(current.duration_seconds),
        image_model: current.image_model || null,
        video_model: current.video_model || null,
      });
      localDirtyRef.current = false;
      setStatus('saved');
    } catch (e) {
      setStatus('error');
    }
  });

  const handleSend = useCallback(async (target) => {
    setSending(target);
    setSendResult(null);
    try {
      // Flush any unsaved draft before the backend reads the panel from disk.
      if (localDirtyRef.current) {
        await sb.upsertPanel(shotId, {
          ...draft,
          duration_seconds: draft.duration_seconds === '' ? null : Number(draft.duration_seconds),
          image_model: draft.image_model || null,
          video_model: draft.video_model || null,
        });
        localDirtyRef.current = false;
      }
      if (target === 'image') await sendPanelToImage(shotId);
      else await sendPanelToVideo(shotId);

      // Refresh the in-memory shot state if that shot happens to be the one
      // currently loaded — otherwise the next visit to Image/Video would
      // momentarily show stale state before useProject re-reads on switch.
      // We do NOT switch tabs or shots here: the user stays on storyboard.
      if (project.currentFileInfo?.shotNumber === shotId && project.currentFilename) {
        await project.loadProjectFile(project.currentFilename);
      }
      setSendResult({ target, kind: 'ok' });
    } catch (e) {
      setSendResult({ target, kind: 'err', message: e.message });
    } finally {
      setSending(null);
    }
  }, [draft, project, sb, shotId]);

  // Auto-dismiss the "sent" pip after a beat so the card doesn't carry stale state.
  useEffect(() => {
    if (!sendResult || sendResult.kind !== 'ok') return;
    const t = setTimeout(() => setSendResult(null), 2500);
    return () => clearTimeout(t);
  }, [sendResult]);

  return (
    <article className={`storyboard-panel-card ${!panel ? 'storyboard-panel-card--ghost' : ''}`}>
      <header className="storyboard-panel-header">
        <span className="storyboard-panel-id">{shotId}</span>
        <div className="storyboard-panel-header-meta">
          {!isProjectShot && (
            <span className="storyboard-panel-warn" title="No shot file exists yet — create the shot before sending">no shot file</span>
          )}
          <SaveStatus state={status} />
        </div>
      </header>

      <div className="storyboard-panel-body">
        <div className="storyboard-panel-inputs">
          <label className="storyboard-panel-label">Imagery (still)</label>
          <MarkerTextarea
            className="fuk-textarea storyboard-panel-textarea"
            value={draft.imagery_prompt}
            onChange={(v) => update('imagery_prompt', v)}
            placeholder="#sarah looks down the alley, framed wide, cool moonlight"
            rows={4}
          />

          <label className="storyboard-panel-label">Action (motion)</label>
          <MarkerTextarea
            className="fuk-textarea storyboard-panel-textarea"
            value={draft.action_prompt}
            onChange={(v) => update('action_prompt', v)}
            placeholder="slow dolly in as she turns toward camera"
            rows={3}
          />

          <div className="storyboard-panel-meta">
            <label>
              <span>Duration (s)</span>
              <input
                type="number"
                className="fuk-input"
                value={draft.duration_seconds ?? ''}
                min="0"
                step="0.1"
                onChange={e => update('duration_seconds', e.target.value)}
              />
            </label>
          </div>

          <label className="storyboard-panel-label">Notes</label>
          <textarea
            className="fuk-textarea storyboard-panel-textarea"
            rows={2}
            value={draft.notes}
            placeholder="director note, alt take, reference, etc."
            onChange={e => update('notes', e.target.value)}
          />

          <div className="storyboard-panel-actions">
            <div className="storyboard-panel-send-row">
              <button
                className="fuk-btn fuk-btn-secondary"
                onClick={() => handleSend('image')}
                disabled={!isProjectShot || sending !== null || !draft.imagery_prompt.trim()}
                title={!isProjectShot ? 'Create shot file first' : 'Push to Image tab — you stay here'}
              >
                {sending === 'image' ? <Loader2 className="animate-spin" /> : <Camera />} → Image
              </button>
              <select
                className="fuk-input storyboard-panel-model-select"
                value={draft.image_model}
                onChange={e => update('image_model', e.target.value)}
                title="Image model the prompt will land on"
              >
                <option value="">(tab default)</option>
                {imageModels.map(m => (
                  <option key={m.key} value={m.key}>{m.description || m.key}</option>
                ))}
              </select>
            </div>
            <div className="storyboard-panel-send-row">
              <button
                className="fuk-btn fuk-btn-secondary"
                onClick={() => handleSend('video')}
                disabled={!isProjectShot || sending !== null || !draft.action_prompt.trim()}
                title={!isProjectShot ? 'Create shot file first' : 'Push to Video tab — you stay here'}
              >
                {sending === 'video' ? <Loader2 className="animate-spin" /> : <Film />} → Video
              </button>
              <select
                className="fuk-input storyboard-panel-model-select"
                value={draft.video_model}
                onChange={e => update('video_model', e.target.value)}
                title="Video model the prompt will land on"
              >
                <option value="">(tab default)</option>
                {videoModels.map(m => (
                  <option key={m.key} value={m.key}>{m.description || m.key}</option>
                ))}
              </select>
            </div>
          </div>
          {sendResult?.kind === 'ok' && (
            <div className="storyboard-send-ok">
              ✓ Sent to {sendResult.target === 'image' ? 'Image' : 'Video'} tab
            </div>
          )}
          {sendResult?.kind === 'err' && (
            <div className="storyboard-error">
              Send to {sendResult.target} failed: {sendResult.message}
            </div>
          )}
        </div>

        <div className="storyboard-panel-previews">
          <PreviewSlot
            kind="image"
            shotId={shotId}
            path={panel?.image_preview}
            sb={sb}
          />
          <PreviewSlot
            kind="video"
            shotId={shotId}
            path={panel?.video_preview}
            sb={sb}
          />
        </div>
      </div>
    </article>
  );
}

// ---------------------------------------------------------------------------
// Preview slot — pinned generation thumbnail, accepts drag-from-history.
// ---------------------------------------------------------------------------

function PreviewSlot({ kind, shotId, path, sb }) {
  const [hover, setHover] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const url = path ? buildImageUrl(path) : null;
  const label = kind === 'image' ? 'Image' : 'Video';
  const Icon = kind === 'image' ? ImageIcon : Film;

  const setPath = useCallback(async (nextPath, prompt) => {
    setBusy(true);
    setErr(null);
    try {
      await sb.setPanelPreview(shotId, { kind, path: nextPath });
      // When the drop carried a prompt (a history generation), mirror it into
      // the matching panel field — image → imagery, video → action — so the
      // panel reflects what produced the pinned media.
      if (nextPath && prompt) {
        const field = kind === 'image' ? 'imagery_prompt' : 'action_prompt';
        await sb.upsertPanel(shotId, { [field]: prompt });
      }
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }, [sb, shotId, kind]);

  const handleDragOver = (e) => {
    // Accept drops that carry our custom MIME, or any text/plain path.
    if (e.dataTransfer.types.includes('application/x-fuk-generation') ||
        e.dataTransfer.types.includes('text/plain')) {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      setHover(true);
    }
  };
  const handleDragLeave = () => setHover(false);
  const handleDrop = (e) => {
    e.preventDefault();
    setHover(false);
    let dropped = null;
    const raw = e.dataTransfer.getData('application/x-fuk-generation');
    if (raw) {
      try { dropped = JSON.parse(raw); } catch { /* fall through */ }
    }
    const droppedPath = dropped?.path || dropped?.preview || e.dataTransfer.getData('text/plain');
    if (!droppedPath) return;
    const droppedPrompt = dropped?.promptSource || dropped?.prompt || '';
    setPath(droppedPath, droppedPrompt);
  };

  const handleClear = (e) => {
    e.stopPropagation();
    setPath(null);
  };

  return (
    <div className={`storyboard-preview-slot ${hover ? 'storyboard-preview-slot--hover' : ''} ${url ? 'storyboard-preview-slot--filled' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="storyboard-preview-slot-label">
        <Icon /> {label}
        {url && (
          <button
            type="button"
            className="storyboard-preview-slot-clear"
            onClick={handleClear}
            disabled={busy}
            title="Clear preview"
          ><X /></button>
        )}
      </div>
      <div className="storyboard-preview-slot-media">
        {url ? (
          kind === 'video' ? (
            <video src={url} muted loop playsInline preload="metadata" />
          ) : (
            <img src={url} alt={`${label} preview`} />
          )
        ) : (
          <div className="storyboard-preview-slot-empty">
            <Icon />
            <span>Drop a {label.toLowerCase()} here<br/>or use → Storyboard in History</span>
          </div>
        )}
        {busy && <div className="storyboard-preview-slot-busy"><Loader2 className="animate-spin" /></div>}
      </div>
      {err && <div className="storyboard-error">{err}</div>}
    </div>
  );
}

function AddPanelCard({ sb, existingShots, project }) {
  const [shotInput, setShotInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const handleAdd = useCallback(async () => {
    const id = shotInput.trim();
    if (!id) return;
    if (!/^[A-Za-z0-9][A-Za-z0-9-]{0,31}$/.test(id)) {
      setErr('Alphanumeric or hyphens only, must start alphanumeric.');
      return;
    }
    if (existingShots.includes(id)) {
      setErr(`Shot "${id}" already exists.`);
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await sb.upsertPanel(id, {});
      await project.createShotFile(id);
      setShotInput('');
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }, [existingShots, sb, shotInput, project]);

  return (
    <article className="storyboard-panel-card storyboard-panel-card--add">
      <h3>Add panel</h3>
      <input
        type="text"
        className="fuk-input"
        placeholder="shot id (e.g. 01, intro, alt-2)"
        pattern="[A-Za-z0-9][A-Za-z0-9-]*"
        maxLength={32}
        value={shotInput}
        onChange={e => { setShotInput(e.target.value); setErr(null); }}
        onKeyDown={e => { if (e.key === 'Enter') handleAdd(); }}
      />
      <button className="fuk-btn fuk-btn-primary" onClick={handleAdd} disabled={busy || !shotInput.trim()}>
        {busy ? <Loader2 className="animate-spin" /> : <Plus />} Create
      </button>
      {err && <div className="storyboard-error">{err}</div>}
    </article>
  );
}
