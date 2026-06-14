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
import { Plus, Trash2, Camera, Film, Loader2, AlertCircle, CheckCircle } from '../components/Icons';
import { useStoryboard } from '../hooks/useStoryboard';
import { sendPanelToImage, sendPanelToVideo } from '../utils/storyboardApi';
import MarkerTextarea from '../components/MarkerTextarea';
import {
  fetchTags,
  fetchTagCategories,
  createTag,
  updateTag,
  deleteTag,
  PROMPT_TOKENS_CHANGED_EVENT,
} from '../utils/promptApi';

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

export default function StoryboardTab({ activeTab, setActiveTab, project }) {
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
            <GlobalsSection sb={sb} manifest={manifest} />
            <PanelGrid
              sb={sb}
              manifest={manifest}
              panelOrder={panelOrder}
              project={project}
              setActiveTab={setActiveTab}
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
// Globals: subjects + mood + specs
// ---------------------------------------------------------------------------

function GlobalsSection({ sb, manifest }) {
  return (
    <section className="storyboard-globals">
      <h2 className="storyboard-section-title">Globals</h2>
      <p className="storyboard-section-hint">
        <strong>Subjects</strong> are project-scoped — they live in this storyboard. <strong>Tags</strong> are workspace-scoped — they ship with the install and are available in every project. Both surface as <code>#markers</code> in every shot; subjects win on collision. <strong>Mood</strong> is appended to every prompt at generation.
      </p>
      <div className="storyboard-globals-grid">
        <SubjectsEditor sb={sb} subjects={manifest.globals?.subjects || []} />
        <TagsEditor />
      </div>
      <div className="storyboard-globals-mood-row">
        <MoodEditor sb={sb} mood={manifest.globals?.mood || ''} />
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Tags — workspace-scoped #marker vocabulary (prompt_tags.json)
// ---------------------------------------------------------------------------

function TagsEditor() {
  const [tags, setTags] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const [t, c] = await Promise.all([fetchTags(), fetchTagCategories()]);
      setTags(t?.tags || []);
      setCategories(c?.categories || []);
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // Pick up tag CRUD that happens elsewhere (PromptPanel could add tags too
  // in the future, and multiple storyboard tabs would race otherwise).
  useEffect(() => {
    const handler = () => refresh();
    window.addEventListener(PROMPT_TOKENS_CHANGED_EVENT, handler);
    return () => window.removeEventListener(PROMPT_TOKENS_CHANGED_EVENT, handler);
  }, [refresh]);

  // Group for display. Sort categories alphabetically, with "uncategorized" last.
  const grouped = useMemo(() => {
    const groups = new Map();
    for (const t of tags) {
      const key = t.category || '~uncategorized';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(t);
    }
    return [...groups.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([cat, items]) => ({
        category: cat === '~uncategorized' ? 'uncategorized' : cat,
        items: items.sort((a, b) => (a.name || '').localeCompare(b.name || '')),
      }));
  }, [tags]);

  const handleCreate = useCallback(async (payload) => {
    const created = await createTag(payload);
    setTags(ts => [...ts, created]);
    return created;
  }, []);

  const handleUpdate = useCallback(async (id, payload) => {
    const updated = await updateTag(id, payload);
    setTags(ts => ts.map(t => (t.id === id ? updated : t)));
    return updated;
  }, []);

  const handleDelete = useCallback(async (id) => {
    await deleteTag(id);
    setTags(ts => ts.filter(t => t.id !== id));
  }, []);

  return (
    <div className="storyboard-tags">
      <div className="storyboard-subsection-header">
        <h3>Tags</h3>
        <span className="storyboard-count">{tags.length}</span>
      </div>
      {err && <div className="storyboard-error">{err}</div>}
      {loading ? (
        <div className="storyboard-loading"><Loader2 className="animate-spin" /> Loading tags…</div>
      ) : (
        <>
          <ul className="storyboard-tag-list">
            {grouped.length === 0 && (
              <li className="storyboard-empty-hint">No tags yet. Add one below to surface a <code>#marker</code> in every project.</li>
            )}
            {grouped.map(group => (
              <li key={group.category} className="storyboard-tag-group">
                <div className="storyboard-tag-group-header">{group.category}</div>
                <ul className="storyboard-tag-group-list">
                  {group.items.map(t => (
                    <TagRow
                      key={t.id}
                      tag={t}
                      categories={categories}
                      onUpdate={handleUpdate}
                      onDelete={handleDelete}
                    />
                  ))}
                </ul>
              </li>
            ))}
          </ul>
          <NewTagForm categories={categories} onCreate={handleCreate} />
        </>
      )}
    </div>
  );
}

function NewTagForm({ categories, onCreate }) {
  const [draft, setDraft] = useState({ name: '', value: '', category: '' });
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState(null);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!draft.name.trim() || !draft.value.trim()) return;
    setSubmitting(true);
    setErr(null);
    try {
      await onCreate({
        name: draft.name,
        value: draft.value,
        category: draft.category || null,
      });
      setDraft({ name: '', value: '', category: draft.category || '' });
    } catch (e) {
      setErr(e.message);
    } finally {
      setSubmitting(false);
    }
  }, [draft, onCreate]);

  return (
    <form className="storyboard-tag-form" onSubmit={handleSubmit}>
      <input
        className="fuk-input"
        placeholder="name (e.g. golden_hour)"
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
        <option value="">— category —</option>
        {categories.map(c => <option key={c} value={c}>{c}</option>)}
      </select>
      <textarea
        className="fuk-textarea"
        placeholder="expansion — what #marker stands in for at generation time"
        rows={2}
        value={draft.value}
        onChange={e => setDraft(d => ({ ...d, value: e.target.value }))}
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
  );
}

function TagRow({ tag, categories, onUpdate, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ name: tag.name, value: tag.value, category: tag.category || '' });
  const [status, setStatus] = useState('idle');
  const [busy, setBusy] = useState(false);

  // Snap back to server state when the tag changes externally — unless we're
  // mid-edit, in which case the local draft is the authority.
  useEffect(() => {
    if (!editing) setDraft({ name: tag.name, value: tag.value, category: tag.category || '' });
  }, [tag.name, tag.value, tag.category, editing]);

  useDebouncedEffect(draft, AUTOSAVE_MS, async (current) => {
    if (!editing) return;
    if (
      current.name === tag.name &&
      current.value === tag.value &&
      (current.category || null) === (tag.category || null)
    ) return;
    if (!current.name.trim() || !current.value.trim()) return;
    setStatus('saving');
    try {
      await onUpdate(tag.id, {
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
      await onDelete(tag.id);
    } catch (e) {
      alert(e.message);
    } finally {
      setBusy(false);
    }
  }, [onDelete, tag.id, tag.name]);

  // Build a `#marker` preview from the name (matches backend slug rules
  // closely enough for display — exact value comes back from the server).
  const markerPreview = `#${(tag.name || '').replace(/[^A-Za-z0-9_-]+/g, '_')}`;

  if (editing) {
    return (
      <li className="storyboard-tag-row storyboard-tag-row--editing">
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
          <option value="">— category —</option>
          {categories.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <textarea
          className="fuk-textarea"
          rows={2}
          value={draft.value}
          onChange={e => setDraft(d => ({ ...d, value: e.target.value }))}
        />
        <div className="storyboard-tag-row-actions">
          <SaveStatus state={status} />
          <button className="fuk-btn fuk-btn-secondary" onClick={() => { setEditing(false); setStatus('idle'); }}>Done</button>
        </div>
      </li>
    );
  }

  return (
    <li className="storyboard-tag-row">
      <div className="storyboard-tag-summary">
        <code className="storyboard-tag-marker">{markerPreview}</code>
        <span className="storyboard-tag-value">{tag.value}</span>
      </div>
      <div className="storyboard-tag-row-actions">
        <button className="fuk-btn fuk-btn-secondary" onClick={() => setEditing(true)} disabled={busy}>Edit</button>
        <button className="fuk-btn fuk-btn-secondary" onClick={handleDelete} disabled={busy}><Trash2 /></button>
      </div>
    </li>
  );
}

function SubjectsEditor({ sb, subjects }) {
  const [draft, setDraft] = useState({ name: '', description: '', kind: 'character' });
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState(null);

  const handleAdd = useCallback(async (e) => {
    e.preventDefault();
    if (!draft.name.trim() || !draft.description.trim()) return;
    setSubmitting(true);
    setErr(null);
    try {
      await sb.createSubject(draft);
      setDraft({ name: '', description: '', kind: 'character' });
    } catch (e) {
      setErr(e.message);
    } finally {
      setSubmitting(false);
    }
  }, [draft, sb]);

  return (
    <div className="storyboard-subjects">
      <div className="storyboard-subsection-header">
        <h3>Subjects</h3>
        <span className="storyboard-count">{subjects.length}</span>
      </div>
      <ul className="storyboard-subject-list">
        {subjects.map(s => (
          <SubjectRow key={s.id} subject={s} sb={sb} />
        ))}
        {subjects.length === 0 && (
          <li className="storyboard-empty-hint">No subjects yet. Add one below to surface a <code>#marker</code> across all shots.</li>
        )}
      </ul>
      <form className="storyboard-subject-form" onSubmit={handleAdd}>
        <input
          className="fuk-input"
          placeholder="name (e.g. sarah)"
          value={draft.name}
          onChange={e => setDraft(d => ({ ...d, name: e.target.value }))}
          disabled={submitting}
        />
        <select
          className="fuk-input"
          value={draft.kind}
          onChange={e => setDraft(d => ({ ...d, kind: e.target.value }))}
          disabled={submitting}
        >
          <option value="character">character</option>
          <option value="prop">prop</option>
          <option value="location">location</option>
          <option value="other">other</option>
        </select>
        <textarea
          className="fuk-textarea"
          placeholder="description — what #marker expands to at generation time"
          rows={2}
          value={draft.description}
          onChange={e => setDraft(d => ({ ...d, description: e.target.value }))}
          disabled={submitting}
        />
        <button
          type="submit"
          className="fuk-btn fuk-btn-primary"
          disabled={submitting || !draft.name.trim() || !draft.description.trim()}
        >
          {submitting ? <Loader2 className="animate-spin" /> : <Plus />} Add subject
        </button>
        {err && <div className="storyboard-error">{err}</div>}
      </form>
    </div>
  );
}

function SubjectRow({ subject, sb }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ ...subject });
  const [status, setStatus] = useState('idle'); // idle | saving | saved | error
  const [busy, setBusy] = useState(false);

  // Sync draft if the underlying subject changes externally (refresh, etc.)
  // — only when we're not currently editing to avoid overwriting typed input.
  useEffect(() => {
    if (!editing) setDraft({ ...subject });
  }, [subject, editing]);

  // Autosave the edit row on debounce. We only push when there's a real
  // change against the server-side value to avoid a write loop.
  useDebouncedEffect(draft, AUTOSAVE_MS, async (current) => {
    if (!editing) return;
    if (
      current.name === subject.name &&
      current.description === subject.description &&
      current.kind === subject.kind
    ) return;
    if (!current.name.trim() || !current.description.trim()) return;
    setStatus('saving');
    try {
      await sb.updateSubject(subject.id, {
        name: current.name,
        description: current.description,
        kind: current.kind,
      });
      setStatus('saved');
    } catch (e) {
      setStatus('error');
    }
  });

  const handleDelete = useCallback(async () => {
    if (!confirm(`Delete subject "${subject.name}"?`)) return;
    setBusy(true);
    try {
      await sb.deleteSubject(subject.id);
    } catch (e) {
      alert(e.message);
    } finally {
      setBusy(false);
    }
  }, [sb, subject.id, subject.name]);

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
          value={draft.kind}
          onChange={e => setDraft(d => ({ ...d, kind: e.target.value }))}
        >
          <option value="character">character</option>
          <option value="prop">prop</option>
          <option value="location">location</option>
          <option value="other">other</option>
        </select>
        <textarea
          className="fuk-textarea"
          rows={2}
          value={draft.description}
          onChange={e => setDraft(d => ({ ...d, description: e.target.value }))}
        />
        <div className="storyboard-subject-row-actions">
          <SaveStatus state={status} />
          <button className="fuk-btn fuk-btn-secondary" onClick={() => { setDraft(subject); setEditing(false); setStatus('idle'); }}>Done</button>
        </div>
      </li>
    );
  }

  return (
    <li className="storyboard-subject-row">
      <div className="storyboard-subject-summary">
        <code className="storyboard-subject-marker">{subject.marker}</code>
        <span className="storyboard-subject-kind">{subject.kind}</span>
        <span className="storyboard-subject-desc">{subject.description}</span>
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
      <textarea
        className="fuk-textarea"
        rows={4}
        value={draft}
        placeholder="e.g. neo-noir, low-key lighting, rain-soaked Tokyo back alleys at night"
        onChange={e => { setDraft(e.target.value); localDirtyRef.current = true; }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel grid: one card per shot
// ---------------------------------------------------------------------------

function PanelGrid({ sb, manifest, panelOrder, project, setActiveTab }) {
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
          />
        ))}
        <AddPanelCard sb={sb} existingShots={panelOrder} />
      </div>
    </section>
  );
}

function PanelCard({ shotId, panel, sb, project, setActiveTab }) {
  const [draft, setDraft] = useState({
    imagery_prompt: panel?.imagery_prompt || '',
    action_prompt: panel?.action_prompt || '',
    duration_seconds: panel?.duration_seconds ?? '',
    notes: panel?.notes || '',
  });
  const [status, setStatus] = useState('idle');
  const [sending, setSending] = useState(null);
  const [sendErr, setSendErr] = useState(null);
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
    });
  }, [panel?.imagery_prompt, panel?.action_prompt, panel?.duration_seconds, panel?.notes]);

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
      current.notes === (panel?.notes || '')
    ) return;
    setStatus('saving');
    try {
      await sb.upsertPanel(shotId, {
        ...current,
        duration_seconds: current.duration_seconds === '' ? null : Number(current.duration_seconds),
      });
      localDirtyRef.current = false;
      setStatus('saved');
    } catch (e) {
      setStatus('error');
    }
  });

  const handleSend = useCallback(async (target) => {
    setSending(target);
    setSendErr(null);
    try {
      // Make sure the latest draft is persisted before the backend reads it.
      if (localDirtyRef.current) {
        await sb.upsertPanel(shotId, {
          ...draft,
          duration_seconds: draft.duration_seconds === '' ? null : Number(draft.duration_seconds),
        });
        localDirtyRef.current = false;
      }
      if (target === 'image') await sendPanelToImage(shotId);
      else await sendPanelToVideo(shotId);
      if (project.currentFileInfo?.shotNumber !== shotId) {
        await project.switchShot(shotId);
      } else {
        await project.loadProjectFile(project.currentFilename);
      }
      setActiveTab(target);
    } catch (e) {
      setSendErr(e.message);
    } finally {
      setSending(null);
    }
  }, [draft, project, sb, setActiveTab, shotId]);

  return (
    <article className={`storyboard-panel-card ${!panel ? 'storyboard-panel-card--ghost' : ''}`}>
      <header className="storyboard-panel-header">
        <span className="storyboard-panel-id">Shot {shotId}</span>
        <div className="storyboard-panel-header-meta">
          {!isProjectShot && (
            <span className="storyboard-panel-warn" title="No shot file exists yet — create the shot before sending">no shot file</span>
          )}
          <SaveStatus state={status} />
        </div>
      </header>

      <label className="storyboard-panel-label">Imagery (still)</label>
      <MarkerTextarea
        className="fuk-textarea storyboard-panel-textarea"
        value={draft.imagery_prompt}
        onChange={(v) => update('imagery_prompt', v)}
        placeholder="#sarah looks down the alley, framed wide, cool moonlight"
        rows={3}
      />

      <label className="storyboard-panel-label">Action (motion)</label>
      <MarkerTextarea
        className="fuk-textarea storyboard-panel-textarea"
        value={draft.action_prompt}
        onChange={(v) => update('action_prompt', v)}
        placeholder="slow dolly in as she turns toward camera"
        rows={2}
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
        <button
          className="fuk-btn fuk-btn-secondary"
          onClick={() => handleSend('image')}
          disabled={!isProjectShot || sending !== null || !draft.imagery_prompt.trim()}
          title={!isProjectShot ? 'Create shot file first' : 'Push to Image tab'}
        >
          {sending === 'image' ? <Loader2 className="animate-spin" /> : <Camera />} → Image
        </button>
        <button
          className="fuk-btn fuk-btn-secondary"
          onClick={() => handleSend('video')}
          disabled={!isProjectShot || sending !== null || !draft.action_prompt.trim()}
          title={!isProjectShot ? 'Create shot file first' : 'Push to Video tab'}
        >
          {sending === 'video' ? <Loader2 className="animate-spin" /> : <Film />} → Video
        </button>
      </div>
      {sendErr && <div className="storyboard-error">{sendErr}</div>}
    </article>
  );
}

function AddPanelCard({ sb, existingShots }) {
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
      setShotInput('');
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }, [existingShots, sb, shotInput]);

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
