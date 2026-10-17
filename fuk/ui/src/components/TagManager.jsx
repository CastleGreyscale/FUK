/**
 * TagManager — browse, edit, version and move `#marker` tags.
 *
 * Two scopes share one UI:
 *   - Library: workspace tags (prompt_tags.json), shared across projects.
 *   - Project: the open project's storyboard tags, which shadow library tags
 *     on marker collision.
 *
 * Used inline by Describe & Tag and inside the global TagManagerModal, so the
 * editor is reachable from any tab. Layout is form across the top, then the
 * search/filter bar, then the list.
 *
 * Every save, delete and move keeps the superseded version in the tag's
 * server-side history, so the History drawer and the Deleted view can bring
 * any of them back. Library tags can be ticked and moved (or copied) into the
 * project to keep project-specific shorthand out of the shared library.
 */

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { Trash2, Clock } from './Icons';
import MarkerTextarea from './MarkerTextarea';
import {
  fetchTags,
  fetchTagCategories,
  createTag,
  updateTag,
  deleteTag,
  fetchTagHistory,
  fetchDeletedTags,
  restoreTagVersion,
  pushTagsToStoryboard,
  notifyPromptTokensChanged,
  PROMPT_TOKENS_CHANGED_EVENT,
} from '../utils/promptApi';
import {
  fetchStoryboardGlobals,
  createStoryboardTag,
  updateStoryboardTag,
  deleteStoryboardTag,
  fetchStoryboardTagHistory,
  fetchDeletedStoryboardTags,
  restoreStoryboardTagVersion,
} from '../utils/storyboardApi';
import { PROJECT_TAG_KINDS } from '../utils/constants';

// Hard cap on a library tag value — mirrors llm_endpoints._validate_payload.
// Project tags have no cap.
const TAG_VALUE_MAX = 4000;

const STORYBOARD_CHANGED_EVENT = 'fuk-storyboard-changed';

// Project writes go through storyboardApi, which leaves broadcasting to its
// caller (useStoryboard). Broadcast here so the Storyboard tab and every open
// autocomplete pick the change up.
function broadcastProject(fn) {
  return async (...args) => {
    const out = await fn(...args);
    notifyPromptTokensChanged();
    window.dispatchEvent(new CustomEvent(STORYBOARD_CHANGED_EVENT));
    return out;
  };
}

const SCOPES = {
  library: {
    label: 'Library',
    list: async () => (await fetchTags()).tags || [],
    create: createTag,
    update: updateTag,
    remove: deleteTag,
    history: async (id) => (await fetchTagHistory(id)).versions || [],
    deleted: async () => (await fetchDeletedTags()).tags || [],
    restore: restoreTagVersion,
  },
  project: {
    label: 'Project',
    list: async () => (await fetchStoryboardGlobals()).tags || [],
    create: broadcastProject(createStoryboardTag),
    update: broadcastProject(updateStoryboardTag),
    remove: broadcastProject(deleteStoryboardTag),
    history: async (id) => (await fetchStoryboardTagHistory(id)).versions || [],
    deleted: async () => (await fetchDeletedStoryboardTags()).tags || [],
    restore: broadcastProject(restoreStoryboardTagVersion),
  },
};

// What superseded a history version — see tag_history.py.
const REASON_LABEL = {
  edit: 'before edit',
  restore: 'before restore',
  replaced: 'before library push',
  snapshot: 'before snapshot load',
  delete: 'deleted',
  moved: 'moved to project',
};

// Mirrors storyboard_manager._slug_marker / llm_endpoints._name_to_marker.
// Project tags carry their marker; library tags only have a name.
export function tagMarker(name) {
  const slug = (name || '').trim().replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '');
  return slug ? `#${slug}` : '';
}

const markerOf = (tag) => tag.marker || tagMarker(tag.name);

function formatWhen(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

/**
 * @param project       useProject() state; enables the Project scope and pushes.
 * @param focusRequest  `{marker, history}` — open on the tag with that marker
 *                      (project first, since it shadows), optionally with its
 *                      history drawer open. Pass a new object to re-trigger.
 * @param ref           `{startDraft(text), isDirty()}`
 */
const TagManager = forwardRef(function TagManager({ project, focusRequest, className = '' }, ref) {
  const hasProject = !!project?.projectFolder;
  const [scope, setScope] = useState('library');
  const api = SCOPES[scope];

  const [tags, setTags] = useState([]);
  const [deleted, setDeleted] = useState([]);
  const [libraryCategories, setLibraryCategories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState(null);
  const [notice, setNotice] = useState(null);

  // Form
  const [editingId, setEditingId] = useState(null);
  const [name, setName] = useState('');
  const [value, setValue] = useState('');
  const [category, setCategory] = useState('');
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState(null);

  // List
  const [filter, setFilter] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [view, setView] = useState('live'); // 'live' | 'deleted'
  const [historyFor, setHistoryFor] = useState(null);
  const [versions, setVersions] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [selected, setSelected] = useState(() => new Set());
  const [pushing, setPushing] = useState(false);

  const nameRef = useRef(null);
  const listRef = useRef(null);
  const loadSeq = useRef(0);

  // ---- Loading -------------------------------------------------------------

  const load = useCallback(async () => {
    const seq = ++loadSeq.current;
    setLoading(true);
    try {
      // The Deleted view is secondary; don't let it take the list down.
      const [list, gone] = await Promise.all([api.list(), api.deleted().catch(() => [])]);
      if (seq !== loadSeq.current) return; // a newer load (e.g. scope switch) won
      setTags(list);
      setDeleted(gone);
      setSelected(prev => new Set([...prev].filter(id => list.some(t => t.id === id))));
      setListError(null);
    } catch (e) {
      if (seq === loadSeq.current) setListError(e.message);
    } finally {
      if (seq === loadSeq.current) setLoading(false);
    }
  }, [api]);

  useEffect(() => { load(); }, [load, project?.projectFolder]);

  // Tags change from other places too — the Describe panel under the modal,
  // the Storyboard tab, a push — so follow the same events everything uses.
  useEffect(() => {
    const handler = () => load();
    window.addEventListener(PROMPT_TOKENS_CHANGED_EVENT, handler);
    window.addEventListener(STORYBOARD_CHANGED_EVENT, handler);
    window.addEventListener('fuk-project-changed', handler);
    return () => {
      window.removeEventListener(PROMPT_TOKENS_CHANGED_EVENT, handler);
      window.removeEventListener(STORYBOARD_CHANGED_EVENT, handler);
      window.removeEventListener('fuk-project-changed', handler);
    };
  }, [load]);

  useEffect(() => {
    fetchTagCategories()
      .then(data => setLibraryCategories(data.categories || []))
      .catch(e => console.warn('Could not load tag categories:', e));
  }, []);

  useEffect(() => {
    if (!hasProject && scope === 'project') setScope('library');
  }, [hasProject, scope]);

  // History drawer follows the tag list, so a save or restore shows up in it.
  useEffect(() => {
    if (!historyFor) return;
    let cancelled = false;
    setHistoryLoading(true);
    api.history(historyFor)
      .then(v => { if (!cancelled) setVersions(v); })
      .catch(e => { if (!cancelled) setListError(e.message); })
      .finally(() => { if (!cancelled) setHistoryLoading(false); });
    return () => { cancelled = true; };
  }, [historyFor, tags, deleted, api]);

  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => setNotice(null), 5000);
    return () => clearTimeout(t);
  }, [notice]);

  // ---- Form ------------------------------------------------------------------

  const resetForm = useCallback(() => {
    setEditingId(null);
    setName('');
    setValue('');
    setCategory('');
    setFormError(null);
  }, []);

  const fillForm = useCallback((tag, id = tag.id) => {
    setEditingId(id);
    setName(tag.name || '');
    setValue(tag.value || '');
    setCategory(tag.category || '');
    setFormError(null);
  }, []);

  const beginEdit = useCallback((tag) => {
    fillForm(tag);
    requestAnimationFrame(() => nameRef.current?.focus());
  }, [fillForm]);

  const editingTag = tags.find(t => t.id === editingId) || null;
  const dirty = editingTag
    ? name !== editingTag.name || value !== editingTag.value || (category || '') !== (editingTag.category || '')
    : !!(name || value);
  const dirtyRef = useRef(dirty);
  dirtyRef.current = dirty;

  useImperativeHandle(ref, () => ({
    // Describe & Tag's "Tag selection" / "Tag all": new tag in the current
    // scope, keeping whatever name and category are already typed.
    startDraft(text) {
      setEditingId(null);
      setValue(text);
      setFormError(null);
      setView('live');
      requestAnimationFrame(() => nameRef.current?.focus());
    },
    isDirty: () => dirtyRef.current,
  }), []);

  const switchScope = useCallback((next) => {
    if (next === scope) return;
    if (dirtyRef.current && !window.confirm('Discard the unsaved tag edit?')) return;
    resetForm();
    setScope(next);
    setCategoryFilter('all');
    setHistoryFor(null);
    setSelected(new Set());
    setView('live');
  }, [scope, resetForm]);

  const handleSave = useCallback(async () => {
    if (!name.trim() || !value.trim()) {
      setFormError('Name and value are required.');
      return;
    }
    setSaving(true);
    setFormError(null);
    try {
      const payload = { name: name.trim(), value: value.trim(), category: category.trim() || null };
      if (editingId) await api.update(editingId, payload);
      else await api.create(payload);
      resetForm();
      await load();
    } catch (e) {
      setFormError(e.message);
    } finally {
      setSaving(false);
    }
  }, [api, editingId, name, value, category, resetForm, load]);

  const handleFormKeyDown = useCallback((e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    }
  }, [handleSave]);

  // ---- Focus request (modal opened on a #marker) ----------------------------

  useEffect(() => {
    if (!focusRequest?.marker) return;
    if (dirtyRef.current && !window.confirm('Discard the unsaved tag edit?')) return;
    let cancelled = false;
    const want = focusRequest.marker.toLowerCase();
    (async () => {
      for (const s of hasProject ? ['project', 'library'] : ['library']) {
        const list = await SCOPES[s].list().catch(() => []);
        if (cancelled) return;
        const hit = list.find(t => markerOf(t).toLowerCase() === want);
        if (hit) {
          setScope(s);
          setView('live');
          setFilter('');
          setCategoryFilter('all');
          fillForm(hit);
          setHistoryFor(focusRequest.history ? hit.id : null);
          return;
        }
      }
      // Not a tag (a LoRA trigger, or a typo) — leave it in the search box.
      setFilter(focusRequest.marker.replace(/^#/, ''));
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusRequest]);

  // Keep the tag being edited in view once the list has rendered it.
  useEffect(() => {
    if (!editingId) return;
    listRef.current
      ?.querySelector(`[data-tag-id="${editingId}"]`)
      ?.scrollIntoView({ block: 'nearest' });
  }, [editingId, tags]);

  // ---- List actions ------------------------------------------------------------

  const handleDelete = useCallback(async (tag) => {
    if (!window.confirm(`Delete "${tag.name}"? It stays restorable under Deleted.`)) return;
    try {
      await api.remove(tag.id);
      if (editingId === tag.id) resetForm();
      if (historyFor === tag.id) setHistoryFor(null);
      await load();
    } catch (e) {
      setListError(e.message);
    }
  }, [api, editingId, historyFor, resetForm, load]);

  const handleRestore = useCallback(async (tagId, vid) => {
    try {
      const entry = await api.restore(tagId, vid);
      // A clean form on this tag would now show stale text; resync it.
      if (editingId === tagId && !dirtyRef.current) fillForm(entry);
      await load();
      setNotice(`Restored ${markerOf(entry)}`);
    } catch (e) {
      setListError(e.message);
    }
  }, [api, editingId, fillForm, load]);

  const handleLoadVersion = useCallback((tagId, version) => {
    if (dirtyRef.current && editingId !== tagId && !window.confirm('Discard the unsaved tag edit?')) return;
    fillForm(version, tagId);
    requestAnimationFrame(() => nameRef.current?.focus());
  }, [editingId, fillForm]);

  const toggleHistory = useCallback((id) => {
    setVersions([]);
    setHistoryFor(cur => (cur === id ? null : id));
  }, []);

  const toggleSelected = useCallback((id) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const handlePush = useCallback(async (move) => {
    const ids = [...selected];
    if (!ids.length) return;
    setPushing(true);
    setListError(null);
    try {
      const first = await pushTagsToStoryboard(ids, { move });
      let replaced = first.replaced;
      let moved = first.moved.length;
      let skipped = first.conflicts;
      if (skipped.length) {
        const markers = skipped.map(c => c.marker).join(', ');
        const ok = window.confirm(
          `The project already has ${markers} with different text.\n\n`
          + `Replace the project version${skipped.length > 1 ? 's' : ''} with the library text? `
          + `The project text is kept in its history.`,
        );
        if (ok) {
          const again = await pushTagsToStoryboard(skipped.map(c => c.id), { move, overwrite: true });
          replaced = [...replaced, ...again.replaced];
          moved += again.moved.length;
          skipped = again.conflicts;
        }
      }
      const landed = first.added.length + first.unchanged.length + replaced.length;
      const parts = [`${move ? 'Moved' : 'Copied'} ${plural(landed, 'tag')} to the project`];
      if (replaced.length) parts.push(`replaced ${replaced.map(r => r.marker).join(', ')}`);
      if (first.unchanged.length) parts.push(`${first.unchanged.length} already there`);
      if (skipped.length) parts.push(`skipped ${skipped.map(r => r.marker).join(', ')}`);
      if (move && moved) parts.push('restorable under Deleted');
      setNotice(parts.join(' · '));
      setSelected(new Set());
      if (move && editingId && ids.includes(editingId) && !dirtyRef.current) resetForm();
      await load();
    } catch (e) {
      setListError(e.message);
    } finally {
      setPushing(false);
    }
  }, [selected, editingId, resetForm, load]);

  // ---- Derived ---------------------------------------------------------------

  const categoryOptions = useMemo(() => {
    // Library categories are validated server-side; project ones are free, so
    // also offer whatever the project's tags already use (e.g. pushed-in
    // library categories).
    const base = scope === 'library' ? libraryCategories : PROJECT_TAG_KINDS;
    const extra = scope === 'project' ? tags.map(t => t.category).filter(Boolean) : [];
    return [...new Set([...base, ...extra])];
  }, [scope, libraryCategories, tags]);

  const formCategoryOptions = category && !categoryOptions.includes(category)
    ? [...categoryOptions, category]
    : categoryOptions;

  const q = filter.trim().toLowerCase();
  const matches = (t) => !q
    || (t.name || '').toLowerCase().includes(q)
    || (t.category || '').toLowerCase().includes(q)
    || (t.value || '').toLowerCase().includes(q);

  const filteredTags = tags.filter(t => {
    if (categoryFilter !== 'all') {
      const tc = (t.category || '').toLowerCase();
      if (categoryFilter === '__none__' ? tc !== '' : tc !== categoryFilter) return false;
    }
    return matches(t);
  });
  const filteredDeleted = deleted.filter(matches);

  const countsByCategory = categoryOptions.reduce((acc, c) => {
    acc[c] = tags.filter(t => (t.category || '').toLowerCase() === c).length;
    return acc;
  }, {});
  const uncategorizedCount = tags.filter(t => !t.category).length;

  const canPush = scope === 'library' && hasProject && view === 'live';
  const allShownSelected = filteredTags.length > 0 && filteredTags.every(t => selected.has(t.id));

  const valueLen = value.length;
  const valueOver = scope === 'library' ? valueLen - TAG_VALUE_MAX : 0;
  const formMarker = tagMarker(name);

  const scopeHint = scope === 'project'
    ? 'This project only. Shadows a library tag with the same marker.'
    : hasProject
      ? 'Shared across all projects. Tick tags to move them into this project.'
      : 'Shared across all projects. Open a project to move tags into its storyboard.';

  // ---- Render ----------------------------------------------------------------

  return (
    <div className={`tag-manager ${className}`}>

      <div className="tag-manager-scopebar">
        <div className="tag-manager-scopes" role="tablist">
          {Object.entries(SCOPES).map(([key, s]) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={scope === key}
              className={`tag-manager-scope ${scope === key ? 'tag-manager-scope--active' : ''}`}
              onClick={() => switchScope(key)}
              disabled={key === 'project' && !hasProject}
              title={key === 'project' && !hasProject ? 'Open a project folder first' : undefined}
            >
              {s.label}
              {scope === key && <span className="tag-manager-count">{tags.length}</span>}
            </button>
          ))}
        </div>
        <span className="tag-manager-scope-hint">{scopeHint}</span>
      </div>

      {/* Form — across the top */}
      <div className={`tag-manager-form ${editingId ? 'tag-manager-form--editing' : ''}`}>
        <div className="tag-manager-form-row">
          <label className="tag-manager-field tag-manager-field--name">
            <span className="fuk-label">
              {editingId ? 'Editing' : 'New tag'}
              {formMarker && <code className="tag-manager-form-marker">{formMarker}</code>}
            </span>
            <input
              ref={nameRef}
              type="text"
              className="fuk-input"
              placeholder="name, e.g. fat man"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={handleFormKeyDown}
              maxLength={64}
            />
          </label>
          <label className="tag-manager-field tag-manager-field--category">
            <span className="fuk-label">Category</span>
            <select className="fuk-select" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="">— uncategorized —</option>
              {formCategoryOptions.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <div className="tag-manager-form-actions">
            <button
              className="fuk-btn fuk-btn-primary"
              onClick={handleSave}
              disabled={saving || !name.trim() || !value.trim() || valueOver > 0 || (editingId && !dirty)}
              title="Ctrl+Enter"
            >
              {saving ? 'Saving…' : editingId ? 'Update' : 'Save tag'}
            </button>
            {(editingId || name || value || category) && (
              <button className="fuk-btn fuk-btn-secondary" onClick={resetForm}>
                {editingId ? 'Cancel' : 'Clear'}
              </button>
            )}
          </div>
        </div>

        <div className="tag-manager-value-label">
          <span className="fuk-label">Value</span>
          <span className={`tag-manager-value-count ${valueOver > 0 ? 'tag-manager-value-count--over' : ''}`}>
            {valueLen.toLocaleString()}
            {scope === 'library' && ` / ${TAG_VALUE_MAX.toLocaleString()}`}
            {valueOver > 0 && ` — ${valueOver.toLocaleString()} over, trim to save`}
          </span>
        </div>
        <MarkerTextarea
          className="fuk-input tag-manager-value"
          placeholder="Full text the tag expands to. Type # to reference another tag."
          value={value}
          onChange={setValue}
          onKeyDown={handleFormKeyDown}
          autoResize={false}
        />

        {formError && <div className="tag-manager-error">{formError}</div>}
      </div>

      {/* Search / filter */}
      <div className="tag-manager-toolbar">
        {canPush && (
          <input
            type="checkbox"
            className="tag-manager-check"
            checked={allShownSelected}
            onChange={() => setSelected(allShownSelected ? new Set() : new Set(filteredTags.map(t => t.id)))}
            title={allShownSelected ? 'Clear selection' : 'Select all shown'}
            disabled={filteredTags.length === 0}
          />
        )}
        <input
          type="text"
          className="fuk-input tag-manager-search"
          placeholder={view === 'deleted' ? 'Search deleted…' : 'Search name, category or text…'}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        {view === 'live' && (
          <select
            className="fuk-select tag-manager-cat-filter"
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
          >
            <option value="all">All categories ({tags.length})</option>
            {categoryOptions.map(c => (
              <option key={c} value={c}>{c} ({countsByCategory[c] || 0})</option>
            ))}
            <option value="__none__">uncategorized ({uncategorizedCount})</option>
          </select>
        )}
        <button
          type="button"
          className={`fuk-btn fuk-btn-secondary tag-manager-mini tag-manager-deleted-toggle ${view === 'deleted' ? 'tag-manager-deleted-toggle--on' : ''}`}
          onClick={() => { setView(v => (v === 'live' ? 'deleted' : 'live')); setHistoryFor(null); }}
          title="Deleted and moved tags, restorable from their last version"
        >
          <Trash2 /> Deleted ({deleted.length})
        </button>
      </div>

      {canPush && selected.size > 0 && (
        <div className="tag-manager-selection-bar">
          <span>{plural(selected.size, 'tag')} selected</span>
          <button className="fuk-btn fuk-btn-primary tag-manager-mini" onClick={() => handlePush(true)} disabled={pushing}>
            {pushing ? 'Working…' : 'Move to project'}
          </button>
          <button className="fuk-btn fuk-btn-secondary tag-manager-mini" onClick={() => handlePush(false)} disabled={pushing}>
            Copy to project
          </button>
          <button className="tag-manager-link" onClick={() => setSelected(new Set())} disabled={pushing}>clear</button>
        </div>
      )}

      {notice && <div className="tag-manager-notice">{notice}</div>}
      {listError && <div className="tag-manager-error">{listError}</div>}

      <div className="tag-manager-list" ref={listRef}>
        {loading && tags.length === 0 && deleted.length === 0 ? (
          <div className="tag-manager-placeholder">Loading tags…</div>
        ) : view === 'deleted' ? (
          filteredDeleted.length === 0 ? (
            <div className="tag-manager-placeholder">
              {deleted.length === 0 ? 'Nothing deleted yet.' : 'No deleted tags match the search.'}
            </div>
          ) : filteredDeleted.map(row => (
            <div key={row.id} data-tag-id={row.id} className="tag-manager-item tag-manager-item--deleted">
              <div className="tag-manager-item-head">
                <span className="tag-manager-marker">{markerOf(row)}</span>
                {row.category && <span className="tag-manager-cat">{row.category}</span>}
                <span className="tag-manager-when">
                  {REASON_LABEL[row.reason] || row.reason} · {formatWhen(row.deleted_at)}
                </span>
                <div className="tag-manager-item-actions">
                  {row.version_count > 1 && (
                    <button className="fuk-btn fuk-btn-secondary tag-manager-mini" onClick={() => toggleHistory(row.id)}>
                      <Clock /> {row.version_count}
                    </button>
                  )}
                  <button className="fuk-btn fuk-btn-primary tag-manager-mini" onClick={() => handleRestore(row.id, row.vid)}>
                    Restore
                  </button>
                </div>
              </div>
              <div className="tag-manager-value-preview">{row.value}</div>
              {historyFor === row.id && (
                <HistoryDrawer
                  versions={versions.slice(1)}
                  loading={historyLoading}
                  current={row}
                  onRestore={(v) => handleRestore(row.id, v.vid)}
                />
              )}
            </div>
          ))
        ) : filteredTags.length === 0 ? (
          <div className="tag-manager-placeholder">
            {tags.length === 0
              ? (scope === 'project'
                ? 'No project tags yet. Save one above, or move tags in from the Library.'
                : 'No tags yet. Describe an image, select text, and save your first tag.')
              : 'No tags match the filter.'}
          </div>
        ) : filteredTags.map(tag => (
          <div
            key={tag.id}
            data-tag-id={tag.id}
            className={`tag-manager-item ${editingId === tag.id ? 'tag-manager-item--editing' : ''} ${selected.has(tag.id) ? 'tag-manager-item--selected' : ''}`}
          >
            <div className="tag-manager-item-head">
              {canPush && (
                <input
                  type="checkbox"
                  className="tag-manager-check"
                  checked={selected.has(tag.id)}
                  onChange={() => toggleSelected(tag.id)}
                />
              )}
              <span className="tag-manager-marker">{markerOf(tag)}</span>
              {tag.category && <span className="tag-manager-cat">{tag.category}</span>}
              <div className="tag-manager-item-actions">
                <button className="fuk-btn fuk-btn-secondary tag-manager-mini" onClick={() => beginEdit(tag)}>
                  Edit
                </button>
                <button
                  className={`fuk-btn fuk-btn-secondary tag-manager-mini ${historyFor === tag.id ? 'tag-manager-mini--on' : ''}`}
                  onClick={() => toggleHistory(tag.id)}
                  title="Earlier versions"
                >
                  <Clock /> History
                </button>
                <button
                  className="fuk-btn fuk-btn-secondary tag-manager-mini tag-manager-remove"
                  onClick={() => handleDelete(tag)}
                  title="Delete tag (restorable)"
                >
                  <Trash2 />
                </button>
              </div>
            </div>
            <div
              className="tag-manager-value-preview"
              onClick={() => beginEdit(tag)}
              title="Click to edit"
            >
              {tag.value}
            </div>
            {historyFor === tag.id && (
              <HistoryDrawer
                versions={versions}
                loading={historyLoading}
                current={tag}
                onRestore={(v) => handleRestore(tag.id, v.vid)}
                onLoad={(v) => handleLoadVersion(tag.id, v)}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
});

export default TagManager;

function HistoryDrawer({ versions, loading, current, onRestore, onLoad }) {
  if (loading && versions.length === 0) {
    return <div className="tag-manager-history tag-manager-placeholder">Loading history…</div>;
  }
  if (versions.length === 0) {
    return (
      <div className="tag-manager-history tag-manager-placeholder">
        No earlier versions. From now on each save keeps the text it replaces.
      </div>
    );
  }
  return (
    <ol className="tag-manager-history">
      {versions.map(v => (
        <li key={v.vid} className="tag-manager-version">
          <div className="tag-manager-version-head">
            <span className="tag-manager-when">{formatWhen(v.replaced_at)}</span>
            <span className="tag-manager-version-reason">{REASON_LABEL[v.reason] || v.reason}</span>
            {v.name !== current.name && <span className="tag-manager-version-diff">was {tagMarker(v.name)}</span>}
            {(v.category || '') !== (current.category || '') && (
              <span className="tag-manager-version-diff">{v.category || 'uncategorized'}</span>
            )}
            {v.value === current.value && v.name === current.name && (
              <span className="tag-manager-version-diff">same text as now</span>
            )}
            <div className="tag-manager-item-actions">
              {onLoad && (
                <button className="fuk-btn fuk-btn-secondary tag-manager-mini" onClick={() => onLoad(v)} title="Load into the editor without saving">
                  Edit from this
                </button>
              )}
              <button className="fuk-btn fuk-btn-secondary tag-manager-mini" onClick={() => onRestore(v)} title="Make this the current version">
                Restore
              </button>
            </div>
          </div>
          <div className="tag-manager-version-value">{v.value}</div>
        </li>
      ))}
    </ol>
  );
}
