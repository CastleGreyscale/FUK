import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { compilePrompt, expandPrompt, fetchPromptTokens } from '../utils/promptApi';

function useAutoResize(ref, value) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
  }, [ref, value]);
}

// Match a `#` immediately before the caret followed by an in-progress word.
// Anchored at end-of-string so we only see the active token, not earlier ones.
const MARKER_TYPING_RE = /(?:^|[\s,;()])#([A-Za-z0-9_\-]*)$/;

const SOURCE_BADGE = {
  tag: { label: 'tag', className: 'prompt-token-badge--tag' },
  lora: { label: 'lora', className: 'prompt-token-badge--lora' },
  caption: { label: 'phrase', className: 'prompt-token-badge--caption' },
};

function scoreToken(token, query) {
  if (!query) return 0;
  const q = query.toLowerCase();
  const name = (token.name || '').toLowerCase();
  if (name === q) return 100;
  if (name.startsWith(q)) return 80;
  if (name.includes(q)) return 50;
  const cat = (token.category || '').toLowerCase();
  if (cat.startsWith(q)) return 30;
  return -1;
}

export default function PromptPanel({
  prompt,
  negativePrompt,
  onChange,
  disabled,
  model,
  loras,
  mode = 'image',
}) {
  const promptRef = useRef(null);
  const negRef = useRef(null);

  useAutoResize(promptRef, prompt);
  useAutoResize(negRef, negativePrompt);

  // Stable, comparable representation of active LoRA keys for the deps array.
  const activeLoraKeys = useMemo(
    () => (loras || []).map(l => l?.key).filter(Boolean),
    [loras],
  );
  const activeLoraKey = activeLoraKeys.join(',');

  const [tokens, setTokens] = useState([]);
  const [categories, setCategories] = useState([]);
  const [autocomplete, setAutocomplete] = useState(null); // { query, selected, anchor: 'prompt' | 'neg' }
  const [compiling, setCompiling] = useState(false);
  const [compileError, setCompileError] = useState(null);
  const [expanding, setExpanding] = useState(false);
  const [expandNotes, setExpandNotes] = useState(null);
  const [slotsOpen, setSlotsOpen] = useState(false);
  const [slotCategory, setSlotCategory] = useState(null); // null = grid view, string = drilled-in
  // Style chips: slot-picker insertions accumulate here and bake into a
  // trailing "Style: …" sentence on Compile. Inline `#` autocomplete still
  // writes to the textarea so the user can place markers explicitly mid-sentence.
  const [styleChips, setStyleChips] = useState([]);
  const lastCaretRef = useRef(null); // remembers caret in the prompt textarea for slot inserts

  // Fetch tokens whenever the prompt context changes.
  useEffect(() => {
    let cancelled = false;
    fetchPromptTokens({ model, activeLoras: activeLoraKeys })
      .then(data => {
        if (cancelled) return;
        setTokens(data?.tokens || []);
        setCategories(data?.categories || []);
      })
      .catch(() => {
        if (cancelled) return;
        setTokens([]);
        setCategories([]);
      });
    return () => { cancelled = true; };
  }, [model, activeLoraKey]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const focusPrompt = () => promptRef.current?.focus();
    const focusNeg = () => negRef.current?.focus();
    window.addEventListener('fuk-shortcut-focus-prompt', focusPrompt);
    window.addEventListener('fuk-shortcut-focus-neg-prompt', focusNeg);
    return () => {
      window.removeEventListener('fuk-shortcut-focus-prompt', focusPrompt);
      window.removeEventListener('fuk-shortcut-focus-neg-prompt', focusNeg);
    };
  }, []);

  const handleResize = useCallback((e) => {
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
  }, []);

  // Detect that the user is mid-typing a `#marker` and pop the dropdown.
  const detectAutocomplete = useCallback((textarea, anchor) => {
    const value = textarea.value;
    const caret = textarea.selectionStart ?? value.length;
    if (caret !== (textarea.selectionEnd ?? caret)) {
      setAutocomplete(null);
      return;
    }
    const upTo = value.slice(0, caret);
    const m = MARKER_TYPING_RE.exec(upTo);
    if (!m) {
      setAutocomplete(null);
      return;
    }
    setAutocomplete(prev => ({
      query: m[1] || '',
      selected: 0,
      anchor,
      // Caret offset of the `#` itself (so we know where to splice).
      markerStart: caret - m[1].length - 1,
      caret,
    }));
  }, []);

  const handlePromptChange = (e) => {
    onChange('prompt', e.target.value);
    handleResize(e);
    detectAutocomplete(e.target, 'prompt');
    lastCaretRef.current = e.target.selectionStart ?? null;
  };

  const handlePromptSelect = (e) => {
    lastCaretRef.current = e.target.selectionStart ?? null;
    // Keep dropdown in sync with caret moves (arrow keys without typing).
    if (autocomplete?.anchor === 'prompt') {
      detectAutocomplete(e.target, 'prompt');
    }
  };

  const filteredTokens = useMemo(() => {
    if (!autocomplete) return [];
    const q = autocomplete.query;
    // Include only tokens that have a marker — caption phrases (marker=null)
    // don't show up in the `#` flow. They surface elsewhere (slot picker, future).
    const candidates = tokens.filter(t => t.marker);
    if (!q) {
      return candidates.slice(0, 30);
    }
    return candidates
      .map(t => ({ t, score: scoreToken(t, q) }))
      .filter(x => x.score >= 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 30)
      .map(x => x.t);
  }, [tokens, autocomplete]);

  const insertToken = useCallback((token) => {
    if (!autocomplete) return;
    const targetRef = autocomplete.anchor === 'neg' ? negRef : promptRef;
    const textarea = targetRef.current;
    if (!textarea) return;
    const field = autocomplete.anchor === 'neg' ? 'negative_prompt' : 'prompt';
    const value = autocomplete.anchor === 'neg' ? negativePrompt : prompt;
    const insert = token.marker; // tags + loras have markers
    const before = value.slice(0, autocomplete.markerStart);
    const after = value.slice(autocomplete.caret);
    const next = `${before}${insert}${after.startsWith(' ') ? '' : ' '}${after}`;
    onChange(field, next);
    setAutocomplete(null);

    // Restore caret just after the inserted marker (+1 if we added a space).
    const newCaret = before.length + insert.length + (after.startsWith(' ') ? 0 : 1);
    requestAnimationFrame(() => {
      if (textarea) {
        textarea.focus();
        textarea.setSelectionRange(newCaret, newCaret);
      }
    });
  }, [autocomplete, prompt, negativePrompt, onChange]);

  const handleKeyDown = (e) => {
    if (!autocomplete || filteredTokens.length === 0) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setAutocomplete(a => ({ ...a, selected: Math.min(a.selected + 1, filteredTokens.length - 1) }));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setAutocomplete(a => ({ ...a, selected: Math.max(a.selected - 1, 0) }));
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault();
      insertToken(filteredTokens[autocomplete.selected]);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setAutocomplete(null);
    }
  };

  const handleCompile = async () => {
    if (compiling) return;
    if (!prompt && styleChips.length === 0) return;
    setCompileError(null);
    setCompiling(true);
    try {
      const res = await compilePrompt({
        text: prompt,
        model,
        activeLoras: activeLoraKeys,
        styleChips: styleChips.map(c => c.chip),
      });
      if (res?.compiled != null) {
        onChange('prompt', res.compiled);
      }
      if (res?.unknown_markers?.length) {
        setCompileError(`Unknown markers: ${res.unknown_markers.join(', ')}`);
      }
      // Bake-in succeeded — chips have been folded into the textarea.
      setStyleChips([]);
    } catch (err) {
      setCompileError(err.message || 'Compile failed');
    } finally {
      setCompiling(false);
    }
  };

  const handleExpand = async () => {
    if (expanding) return;
    if (!prompt && styleChips.length === 0) return;
    setCompileError(null);
    setExpandNotes(null);
    setExpanding(true);
    try {
      const res = await expandPrompt({
        text: prompt,
        model,
        activeLoras: activeLoraKeys,
        styleChips: styleChips.map(c => c.chip),
        mode,
      });
      // Build a single patch so the parent's setFormData applies both
      // updates in one shot. The tabs' setFormData snapshots formData from
      // a ref, not React state, so back-to-back onChange calls in the same
      // tick would otherwise overwrite each other.
      const patch = {};
      if (res?.positive) patch.prompt = res.positive;
      // Only overwrite negative if the LLM provided something substantive —
      // the author may have hand-tuned the negative and we don't want an
      // empty string from the model wiping it out.
      if (res?.negative) patch.negative_prompt = res.negative;
      if (Object.keys(patch).length > 0) {
        onChange(patch);
      }
      if (res?.notes) {
        setExpandNotes(res.notes);
      }
      setStyleChips([]);
    } catch (err) {
      setCompileError(err.message || 'Expand failed');
    } finally {
      setExpanding(false);
    }
  };

  const hasMarkers = useMemo(() => /(^|\s)#[A-Za-z0-9]/.test(prompt || ''), [prompt]);
  const canCompile = hasMarkers || styleChips.length > 0;
  const canExpand = (prompt && prompt.trim().length > 0) || styleChips.length > 0;

  // Group tokens by category for the slot picker. The category list comes
  // from prompt_tag_categories.json so the user can edit / extend it from
  // disk; anything outside that list (including tokens with no category)
  // collects into an "other" bucket at the end.
  const tokensByCategory = useMemo(() => {
    const groups = new Map();
    const ordered = [...categories, 'other'];
    ordered.forEach(c => groups.set(c, []));
    for (const t of tokens) {
      const c = t.category && groups.has(t.category) ? t.category : 'other';
      groups.get(c).push(t);
    }
    return ordered.map(c => ({ category: c, items: groups.get(c) || [] }));
  }, [tokens, categories]);

  // Slot-picker insertions become style chips, not inline edits — so the
  // textarea stays the user's subject/action description and Compile can
  // append a structured "Style: …" sentence built from these chips.
  const addStyleChip = useCallback((token) => {
    const chipText = token.marker || token.name || token.expansion;
    if (!chipText) return;
    setStyleChips(prev => {
      // Dedupe by id when available, otherwise by chip text.
      const key = token.id || chipText;
      if (prev.some(c => (c.id || c.chip) === key)) return prev;
      return [...prev, {
        id: token.id,
        source: token.source,
        name: token.name,
        chip: chipText,           // what we send to the backend
        expansion: token.expansion,
        category: token.category,
      }];
    });
  }, []);

  const removeStyleChip = useCallback((idx) => {
    setStyleChips(prev => prev.filter((_, i) => i !== idx));
  }, []);

  return (
    <div className="prompt-panel">
      <div className="prompt-panel-header">
        <span className="prompt-panel-title">Prompt</span>
        <div className="prompt-panel-header-actions">
          <button
            type="button"
            className={`prompt-panel-headerbtn ${slotsOpen ? 'prompt-panel-headerbtn--active' : ''}`}
            onClick={() => { setSlotsOpen(o => !o); setSlotCategory(null); }}
            disabled={disabled}
            title="Browse tags, LoRA triggers and caption phrases by category"
          >
            Slots
          </button>
          <button
            type="button"
            className="prompt-panel-headerbtn"
            onClick={handleExpand}
            disabled={disabled || expanding || !canExpand}
            title={canExpand
              ? 'LLM refines the draft, integrating style chips and LoRA vocabulary'
              : 'Write something to expand'}
          >
            {expanding ? '…' : 'Expand'}
          </button>
          <button
            type="button"
            className="prompt-panel-headerbtn"
            onClick={handleCompile}
            disabled={disabled || compiling || !canCompile}
            title={canCompile ? 'Deterministic: expand #markers and append style chips' : 'Add #markers or style chips to compile'}
          >
            {compiling ? '…' : 'Compile'}
          </button>
        </div>
      </div>
      {slotsOpen && (
        <SlotPicker
          groups={tokensByCategory}
          category={slotCategory}
          onPickCategory={setSlotCategory}
          onBack={() => setSlotCategory(null)}
          onInsert={addStyleChip}
          activeChipIds={new Set(styleChips.map(c => c.id).filter(Boolean))}
        />
      )}
      <div className="prompt-panel-body">
        <div className="prompt-panel-field">
          <textarea
            ref={promptRef}
            className="fuk-textarea prompt-panel-textarea"
            value={prompt}
            onChange={handlePromptChange}
            onKeyDown={handleKeyDown}
            onSelect={handlePromptSelect}
            onBlur={() => setTimeout(() => setAutocomplete(null), 120)}
            placeholder="A cinematic video of... (type # for tags & LoRA triggers)"
            rows={3}
            disabled={disabled}
          />
          {autocomplete?.anchor === 'prompt' && filteredTokens.length > 0 && (
            <TokenDropdown
              tokens={filteredTokens}
              selected={autocomplete.selected}
              onPick={insertToken}
              onHover={(idx) => setAutocomplete(a => ({ ...a, selected: idx }))}
            />
          )}
        </div>
        {styleChips.length > 0 && (
          <StyleChipTray chips={styleChips} onRemove={removeStyleChip} onClear={() => setStyleChips([])} />
        )}
        {compileError && (
          <div className="prompt-panel-warning">{compileError}</div>
        )}
        {expandNotes && (
          <div className="prompt-panel-note">
            <span>{expandNotes}</span>
            <button
              type="button"
              className="prompt-panel-note-dismiss"
              onClick={() => setExpandNotes(null)}
              aria-label="Dismiss note"
            >×</button>
          </div>
        )}
        <label className="prompt-panel-neg-label">Negative</label>
        <textarea
          ref={negRef}
          className="fuk-textarea prompt-panel-textarea"
          value={negativePrompt}
          onChange={(e) => { onChange('negative_prompt', e.target.value); handleResize(e); }}
          placeholder="blurry, low quality..."
          rows={2}
          disabled={disabled}
        />
      </div>
    </div>
  );
}

function TokenDropdown({ tokens, selected, onPick, onHover }) {
  return (
    <ul className="prompt-token-dropdown" role="listbox">
      {tokens.map((t, idx) => {
        const badge = SOURCE_BADGE[t.source] || SOURCE_BADGE.tag;
        const isSelected = idx === selected;
        return (
          <li
            key={t.id || `${t.source}:${t.name}`}
            role="option"
            aria-selected={isSelected}
            className={`prompt-token-item ${isSelected ? 'prompt-token-item--selected' : ''}`}
            onMouseDown={(e) => { e.preventDefault(); onPick(t); }}
            onMouseEnter={() => onHover(idx)}
          >
            <span className={`prompt-token-badge ${badge.className}`}>{badge.label}</span>
            <span className="prompt-token-name">{t.marker || t.name}</span>
            {t.category && (
              <span className="prompt-token-category">{t.category}</span>
            )}
            <span className="prompt-token-expansion">{t.expansion}</span>
          </li>
        );
      })}
    </ul>
  );
}

function StyleChipTray({ chips, onRemove, onClear }) {
  return (
    <div className="prompt-style-tray">
      <div className="prompt-style-tray-header">
        <span className="prompt-style-tray-label">Style</span>
        <button type="button" className="prompt-style-tray-clear" onClick={onClear} title="Remove all style chips">
          clear
        </button>
      </div>
      <ul className="prompt-style-tray-chips">
        {chips.map((c, idx) => (
          <li key={c.id || `${c.source}:${c.chip}:${idx}`} className="prompt-style-chip" title={c.expansion || c.chip}>
            <span className="prompt-style-chip-name">{c.name || c.chip}</span>
            <button
              type="button"
              className="prompt-style-chip-x"
              onClick={() => onRemove(idx)}
              aria-label={`Remove ${c.name || c.chip}`}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function SlotPicker({ groups, category, onPickCategory, onBack, onInsert, activeChipIds }) {
  // Grid view: chips per category showing how many tokens are available.
  if (!category) {
    const nonEmpty = groups.filter(g => g.items.length > 0);
    if (nonEmpty.length === 0) {
      return (
        <div className="prompt-slot-empty">
          No tokens yet. Add tags or activate a LoRA with caption phrases.
        </div>
      );
    }
    return (
      <div className="prompt-slot-grid">
        {nonEmpty.map(g => (
          <button
            key={g.category}
            type="button"
            className="prompt-slot-chip"
            onClick={() => onPickCategory(g.category)}
          >
            <span className="prompt-slot-chip-label">{g.category}</span>
            <span className="prompt-slot-chip-count">{g.items.length}</span>
          </button>
        ))}
      </div>
    );
  }

  // Drilled-in view: token list for one category.
  const group = groups.find(g => g.category === category);
  const items = group?.items || [];
  return (
    <div className="prompt-slot-list">
      <div className="prompt-slot-list-header">
        <button type="button" className="prompt-slot-back" onClick={onBack}>← {category}</button>
        <span className="prompt-slot-list-count">{items.length}</span>
      </div>
      <ul className="prompt-slot-items">
        {items.map(t => {
          const badge = SOURCE_BADGE[t.source] || SOURCE_BADGE.tag;
          const isActive = activeChipIds?.has(t.id);
          return (
            <li
              key={t.id || `${t.source}:${t.name}`}
              className={`prompt-slot-item ${isActive ? 'prompt-slot-item--active' : ''}`}
              onMouseDown={(e) => { e.preventDefault(); if (!isActive) onInsert(t); }}
              title={isActive ? 'Already in style' : 'Add to style'}
            >
              <span className={`prompt-token-badge ${badge.className}`}>{badge.label}</span>
              <span className="prompt-token-name">{t.marker || t.name}</span>
              {isActive && <span className="prompt-slot-item-check">✓</span>}
              {t.expansion && t.expansion !== t.name && (
                <span className="prompt-token-expansion">{t.expansion}</span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
