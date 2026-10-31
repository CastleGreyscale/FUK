/**
 * FileBrowser — in-app file and folder picker.
 *
 * Replaces the native tkinter dialogs. Opened through utils/fileBrowser.js
 * (browseFiles / browseDirectory); FileBrowserHost is mounted once in App.
 *
 * - Path bar: type or paste a folder, file, or sequence (####, %04d, Nuke's
 *   trailing "1001-1100"). Enter on a pasted file picks it straight away.
 * - Places, favorites (☆ in the toolbar) and recent folders in the sidebar.
 * - Thumbnail grid or list; frame sequences collapse to one entry.
 * - Folders load a page at a time as you scroll.
 *
 * Keys: Enter opens/picks, Backspace goes up, Esc cancels, Ctrl+A selects
 * all; Ctrl/Shift+click multi-selects when the caller allows it.
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { X, Folder, Film, Camera, Layers, SequenceIcon, ArrowUp, Loader2 } from './Icons';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl, sizedImageUrl, API_URL } from '../utils/constants';
import { registerFileBrowser, CANCELLED } from '../utils/fileBrowser';

const PAGE_SIZE = 200;
const MAX_RECENT = 8;
const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

const FILTER_OPTIONS = [
  { value: 'all', label: 'Images & video' },
  { value: 'images', label: 'Images' },
  { value: 'videos', label: 'Video' },
  { value: 'exr', label: 'EXR / DPX' },
  { value: 'layered', label: 'PSD / ORA' },
  { value: 'any', label: 'All files' },
];

const KIND_ICONS = { dir: Folder, image: Camera, video: Film, sequence: SequenceIcon, file: Layers };

const baseName = (path) => path.replace(/\/+$/, '').split('/').pop() || path;

function formatSize(bytes) {
  if (bytes == null) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

const formatDate = (mtime) => (mtime ? new Date(mtime * 1000).toLocaleDateString() : '');

// The file shape the old /api/browser/open endpoint returned.
function toMediaFile(entry) {
  return {
    path: entry.path,
    display_name: entry.name,
    media_type: entry.kind === 'video' || entry.kind === 'sequence' ? entry.kind : 'image',
    first_frame: entry.first_frame ?? null,
    last_frame: entry.last_frame ?? null,
    frame_count: entry.frame_count ?? null,
    frame_pattern: entry.frame_pattern ?? null,
  };
}

function EntryThumb({ entry }) {
  const [failed, setFailed] = useState(false);
  const Icon = KIND_ICONS[entry.kind] || Layers;
  if (!entry.thumb || failed) return <Icon className="fb-kind-icon" />;
  return (
    <img
      src={sizedImageUrl(buildImageUrl(entry.thumb), 'thumb')}
      alt=""
      loading="lazy"
      decoding="async"
      draggable={false}
      onError={() => setFailed(true)}
    />
  );
}

function FileBrowser({ request, projectFolder, onDone }) {
  const { mode, title, multiple, detectSequences } = request;
  const pickingFiles = mode === 'files';

  const [favorites, setFavorites] = useLocalStorage('fuk_browser_favorites', []);
  const [recent, setRecent] = useLocalStorage('fuk_browser_recent', []);
  const [view, setView] = useLocalStorage('fuk_browser_view', 'grid');
  const [filter, setFilter] = useState(request.filter || 'all');
  const [listing, setListing] = useState(null);
  const [pathInput, setPathInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(() => new Set());

  const anchorRef = useRef(null);       // shift-click range start
  const loadSeq = useRef(0);            // drops responses from superseded loads
  const listRef = useRef(null);
  const sentinelRef = useRef(null);
  const pathInputRef = useRef(null);
  const currentPathRef = useRef(request.initialDir || recent[0] || projectFolder || null);

  const entries = listing?.entries ?? [];
  const isSelectable = (entry) => pickingFiles || entry.kind === 'dir';

  // Selected entries, including a pasted pick that sits beyond the loaded pages
  const chosenEntries = useMemo(() => {
    const byPath = new Map(entries.map(e => [e.path, e]));
    if (listing?.selected_entry) byPath.set(listing.selected_entry.path, listing.selected_entry);
    return [...selected].map(p => byPath.get(p)).filter(Boolean);
  }, [entries, listing, selected]);

  const fetchListing = useCallback(async (path, offset = 0) => {
    const params = new URLSearchParams({
      filter,
      sequences: String(detectSequences !== false),
      offset: String(offset),
      limit: String(PAGE_SIZE),
    });
    if (path) params.set('path', path);
    const res = await fetch(`${API_URL}/browser/list?${params}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    return data;
  }, [filter, detectSequences]);

  // Load a folder (or a pasted file/sequence path). Resolves with the listing, or null.
  const navigate = useCallback(async (path) => {
    const seq = ++loadSeq.current;
    setLoading(true);
    try {
      const data = await fetchListing(path);
      if (seq !== loadSeq.current) return null;
      currentPathRef.current = data.path;
      setListing(data);
      setPathInput(data.path);
      setSelected(data.selected ? new Set([data.selected]) : new Set());
      anchorRef.current = null;
      setError(data.missing ? `Not found: ${path}. Showing the nearest folder.` : null);
      listRef.current?.scrollTo(0, 0);
      return data;
    } catch (e) {
      if (seq === loadSeq.current) setError(e.message);
      return null;
    } finally {
      if (seq === loadSeq.current) setLoading(false);
    }
  }, [fetchListing]);

  const loadMore = useCallback(async () => {
    if (!listing?.has_more || loading) return;
    const seq = loadSeq.current;
    setLoading(true);
    try {
      const data = await fetchListing(listing.path, listing.entries.length);
      if (seq !== loadSeq.current) return;
      setListing(prev => ({
        ...prev,
        entries: [...prev.entries, ...data.entries],
        has_more: data.has_more,
        total: data.total,
      }));
    } catch (e) {
      if (seq === loadSeq.current) setError(e.message);
    } finally {
      if (seq === loadSeq.current) setLoading(false);
    }
  }, [listing, loading, fetchListing]);

  // First load, and reload in place when the filter changes
  useEffect(() => { navigate(currentPathRef.current); }, [navigate]);

  // Next page when the end of the list scrolls into view
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !listing?.has_more) return;
    const io = new IntersectionObserver(
      ([hit]) => { if (hit.isIntersecting) loadMore(); },
      { root: listRef.current, rootMargin: '400px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [listing, loadMore]);

  useEffect(() => {
    pathInputRef.current?.focus();
    pathInputRef.current?.select();
  }, []);

  const rememberRecent = useCallback((dir) => {
    setRecent(prev => [dir, ...prev.filter(d => d !== dir)].slice(0, MAX_RECENT));
  }, [setRecent]);

  const pickFiles = useCallback((picked, dir) => {
    const files = picked.filter(e => e.kind !== 'dir');
    if (!files.length) return;
    rememberRecent(dir);
    onDone({ success: true, files: (multiple ? files : files.slice(0, 1)).map(toMediaFile) });
  }, [multiple, onDone, rememberRecent]);

  const confirm = useCallback(() => {
    if (!listing) return;
    if (pickingFiles) {
      if (chosenEntries.length === 1 && chosenEntries[0].kind === 'dir') {
        navigate(chosenEntries[0].path);
      } else {
        pickFiles(chosenEntries, listing.path);
      }
      return;
    }
    const dir = chosenEntries.length === 1 && chosenEntries[0].kind === 'dir'
      ? chosenEntries[0].path
      : listing.path;
    rememberRecent(dir);
    onDone({ success: true, directory: dir });
  }, [listing, pickingFiles, chosenEntries, navigate, pickFiles, rememberRecent, onDone]);

  const cancel = useCallback(() => onDone(CANCELLED), [onDone]);

  const goUp = useCallback(() => {
    if (listing?.parent) navigate(listing.parent);
  }, [listing, navigate]);

  const selectAllFiles = useCallback(() => {
    setSelected(new Set(entries.filter(e => e.kind !== 'dir').map(e => e.path)));
  }, [entries]);

  const handlePathSubmit = async (e) => {
    e.preventDefault();
    const data = await navigate(pathInput);
    // A pasted file or sequence: pick it, no second click needed
    const hit = data?.selected_entry;
    if (pickingFiles && hit && hit.kind !== 'dir') pickFiles([hit], data.path);
  };

  const handleEntryClick = (entry, index, e) => {
    if (!isSelectable(entry)) return;
    const multi = multiple && pickingFiles;
    if (multi && e.shiftKey && anchorRef.current != null) {
      const [a, b] = [anchorRef.current, index].sort((x, y) => x - y);
      setSelected(new Set(entries.slice(a, b + 1).filter(x => x.kind !== 'dir').map(x => x.path)));
      return;
    }
    if (multi && (e.ctrlKey || e.metaKey)) {
      setSelected(prev => {
        const next = new Set(prev);
        if (next.has(entry.path)) next.delete(entry.path); else next.add(entry.path);
        return next;
      });
    } else {
      setSelected(new Set([entry.path]));
    }
    anchorRef.current = index;
  };

  const handleEntryDoubleClick = (entry) => {
    if (entry.kind === 'dir') navigate(entry.path);
    else if (pickingFiles) pickFiles([entry], listing.path);
  };

  // Keys run in the capture phase and stop there, so global shortcuts don't
  // fire behind the modal. Typing in the path bar or filter is left alone.
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        cancel();
        return;
      }
      if (INPUT_TAGS.has(document.activeElement?.tagName)) return;
      e.stopPropagation();
      // Enter/Space on a focused button presses that button
      if (document.activeElement?.tagName === 'BUTTON' && (e.key === 'Enter' || e.key === ' ')) return;
      if (e.key === 'Enter') {
        e.preventDefault();
        confirm();
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        goUp();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a' && multiple && pickingFiles) {
        e.preventDefault();
        selectAllFiles();
      }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [cancel, confirm, goUp, selectAllFiles, multiple, pickingFiles]);

  const isFavorite = listing && favorites.includes(listing.path);
  const toggleFavorite = () => {
    if (!listing) return;
    setFavorites(prev => (prev.includes(listing.path)
      ? prev.filter(d => d !== listing.path)
      : [...prev, listing.path]));
  };

  const places = [
    projectFolder && { label: 'Project', path: projectFolder },
    listing?.home && { label: 'Home', path: listing.home },
  ].filter(Boolean);

  const renderPlace = (path, label, onRemove) => (
    <div key={path} className={`fb-place ${listing?.path === path ? 'fb-place--active' : ''}`}>
      <button className="fb-place-btn" onClick={() => navigate(path)} title={path}>
        <Folder className="fb-place-icon" />
        <span>{label || baseName(path)}</span>
      </button>
      {onRemove && (
        <button className="fb-place-remove" onClick={onRemove} title="Remove from favorites"><X /></button>
      )}
    </div>
  );

  const fileCount = chosenEntries.filter(e => e.kind !== 'dir').length;
  const primaryLabel = pickingFiles
    ? (chosenEntries.length === 1 && chosenEntries[0].kind === 'dir' ? 'Open Folder' : 'Open')
    : (chosenEntries.length === 1 && chosenEntries[0].kind === 'dir' ? 'Select Folder' : 'Select This Folder');
  const primaryDisabled = !listing || (pickingFiles && fileCount === 0 && !(chosenEntries.length === 1 && chosenEntries[0].kind === 'dir'));
  const selectionSummary = pickingFiles
    ? (fileCount > 1 ? `${fileCount} selected` : chosenEntries[0]?.name || (multiple ? 'Ctrl/Shift+click to select several' : ''))
    : (chosenEntries[0]?.kind === 'dir' ? chosenEntries[0].path : listing?.path || '');

  return createPortal(
    <div className="fb-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) cancel(); }}>
      <div className="fb-modal" role="dialog" aria-modal="true" aria-label={title}>
        <div className="fb-header">
          <h2 className="fb-title">{title}</h2>
          <button className="fb-close" onClick={cancel} title="Cancel (Esc)"><X /></button>
        </div>

        <div className="fb-toolbar">
          <button className="fb-tool-btn" onClick={goUp} disabled={!listing?.parent} title="Up (Backspace)">
            <ArrowUp />
          </button>
          <form className="fb-path-form" onSubmit={handlePathSubmit}>
            <input
              ref={pathInputRef}
              className="fuk-input fb-path-input"
              value={pathInput}
              onChange={e => setPathInput(e.target.value)}
              placeholder="Type or paste a folder, file, or sequence (shot.####.exr)"
              spellCheck={false}
            />
          </form>
          <button
            className={`fb-tool-btn fb-fav-btn ${isFavorite ? 'fb-fav-btn--on' : ''}`}
            onClick={toggleFavorite}
            disabled={!listing}
            title={isFavorite ? 'Remove folder from favorites' : 'Add folder to favorites'}
          >
            {isFavorite ? '★' : '☆'}
          </button>
          <div className="fb-view-toggle">
            <button className={view === 'grid' ? 'active' : ''} onClick={() => setView('grid')} title="Thumbnails">Grid</button>
            <button className={view === 'list' ? 'active' : ''} onClick={() => setView('list')} title="Details">List</button>
          </div>
        </div>

        <div className="fb-body">
          <aside className="fb-sidebar">
            <div className="fb-section-label">Places</div>
            {places.map(p => renderPlace(p.path, p.label))}
            {favorites.length > 0 && <div className="fb-section-label">Favorites</div>}
            {favorites.map(path => renderPlace(path, null, () => setFavorites(prev => prev.filter(d => d !== path))))}
            {recent.length > 0 && <div className="fb-section-label">Recent</div>}
            {recent.map(path => renderPlace(path))}
          </aside>

          <div className="fb-main" ref={listRef}>
            {error && <div className="fb-error">{error}</div>}
            {!listing && loading && <div className="fb-empty"><Loader2 className="animate-spin" /></div>}
            {listing && entries.length === 0 && !loading && <div className="fb-empty">Nothing here matches the filter</div>}

            <div className={view === 'grid' ? 'fb-grid' : 'fb-list'}>
              {entries.map((entry, index) => {
                const Icon = KIND_ICONS[entry.kind] || Layers;
                const cls = [
                  view === 'grid' ? 'fb-tile' : 'fb-row',
                  selected.has(entry.path) && 'fb-entry--selected',
                  !isSelectable(entry) && 'fb-entry--disabled',
                ].filter(Boolean).join(' ');
                const detail = entry.kind === 'sequence'
                  ? `${entry.frame_count} frames`
                  : formatSize(entry.size);
                return view === 'grid' ? (
                  <div
                    key={entry.path}
                    className={cls}
                    title={entry.name}
                    onClick={(e) => handleEntryClick(entry, index, e)}
                    onDoubleClick={() => handleEntryDoubleClick(entry)}
                  >
                    <div className="fb-tile-media"><EntryThumb entry={entry} /></div>
                    {entry.kind === 'sequence' && <span className="fb-tile-badge">{entry.frame_count}f</span>}
                    <div className="fb-tile-name">{entry.name}</div>
                  </div>
                ) : (
                  <div
                    key={entry.path}
                    className={cls}
                    title={entry.path}
                    onClick={(e) => handleEntryClick(entry, index, e)}
                    onDoubleClick={() => handleEntryDoubleClick(entry)}
                  >
                    <Icon className="fb-row-icon" />
                    <span className="fb-row-name">{entry.name}</span>
                    <span className="fb-row-detail">{detail}</span>
                    <span className="fb-row-date">{formatDate(entry.mtime)}</span>
                  </div>
                );
              })}
            </div>
            {listing?.has_more && (
              <div ref={sentinelRef} className="fb-more">
                {loading ? 'Loading…' : `${entries.length} of ${listing.total}`}
              </div>
            )}
          </div>
        </div>

        <div className="fb-footer">
          <span className="fb-selection" title={selectionSummary}>{selectionSummary}</span>
          <select className="fuk-select fb-filter" value={filter} onChange={e => setFilter(e.target.value)}>
            {FILTER_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
          <button className="fuk-btn fuk-btn-secondary" onClick={cancel}>Cancel</button>
          <button className="fuk-btn fuk-btn-primary" onClick={confirm} disabled={primaryDisabled}>
            {primaryLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}

/** Mounted once in App; shows the browser whenever browseFiles/browseDirectory is called. */
export default function FileBrowserHost({ projectFolder }) {
  const [request, setRequest] = useState(null);
  const requestRef = useRef(null);
  const counterRef = useRef(0);

  useEffect(() => registerFileBrowser((req) => {
    requestRef.current?.resolve(CANCELLED);  // a new request supersedes an open one
    const next = { ...req, id: ++counterRef.current };
    requestRef.current = next;
    setRequest(next);
  }), []);

  const finish = useCallback((result) => {
    requestRef.current?.resolve(result);
    requestRef.current = null;
    setRequest(null);
  }, []);

  if (!request) return null;
  return <FileBrowser key={request.id} request={request} projectFolder={projectFolder} onDone={finish} />;
}
