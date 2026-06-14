import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { compilePrompt, expandPrompt, fetchPromptTokens, PROMPT_TOKENS_CHANGED_EVENT } from '../utils/promptApi';
import MarkerTextarea from './MarkerTextarea';

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

  // Stable, comparable representation of active LoRA keys for the deps array.
  const activeLoraKeys = useMemo(
    () => (loras || []).map(l => l?.key).filter(Boolean),
    [loras],
  );
  const activeLoraKey = activeLoraKeys.join(',');

  const [tokens, setTokens] = useState([]);
  const [categories, setCategories] = useState([]);
  const [compiling, setCompiling] = useState(false);
  const [compileError, setCompileError] = useState(null);
  const [expanding, setExpanding] = useState(false);
  const [expandNotes, setExpandNotes] = useState(null);
  const [slotsOpen, setSlotsOpen] = useState(false);
  const [slotCategory, setSlotCategory] = useState(null); // null = grid view, string = drilled-in
  // Style chips: slot-picker insertions accumulate here and bake into a
  // trailing "Style: …" sentence on Compile. Inline `#` autocomplete writes
  // markers directly to the textarea via MarkerTextarea.
  const [styleChips, setStyleChips] = useState([]);

  // Tokens drive both the autocomplete (passed down to MarkerTextarea) and
  // the slot picker / category groupings. One fetch shared across both.
  // Refetches on the prompt-tokens-changed event so tag/subject CRUD
  // elsewhere updates the Slot Picker without a reload.
  const [refetchKey, setRefetchKey] = useState(0);
  useEffect(() => {
    const handler = () => setRefetchKey(k => k + 1);
    window.addEventListener(PROMPT_TOKENS_CHANGED_EVENT, handler);
    return () => window.removeEventListener(PROMPT_TOKENS_CHANGED_EVENT, handler);
  }, []);
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
  }, [model, activeLoraKey, refetchKey]); // eslint-disable-line react-hooks/exhaustive-deps

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

  // `#markers` resolve at generation time (live globals + tags + LoRA triggers),
  // so Compile only matters when there are style chips to drain into a sentence.
  const canCompile = styleChips.length > 0;
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
            title={canCompile
              ? 'Drain style chips into a trailing "Style:" sentence (#markers resolve at generation)'
              : 'Add style chips to compile'}
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
          <MarkerTextarea
            ref={promptRef}
            className="fuk-textarea prompt-panel-textarea"
            value={prompt}
            onChange={(next) => onChange('prompt', next)}
            placeholder="A cinematic video of... (type # for tags & LoRA triggers)"
            rows={3}
            disabled={disabled}
            tokens={tokens}
          />
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
        <MarkerTextarea
          ref={negRef}
          className="fuk-textarea prompt-panel-textarea"
          value={negativePrompt}
          onChange={(next) => onChange('negative_prompt', next)}
          placeholder="blurry, low quality..."
          rows={2}
          disabled={disabled}
          tokens={tokens}
        />
      </div>
    </div>
  );
}

const SOURCE_BADGE = {
  global: { label: 'global', className: 'prompt-token-badge--global' },
  tag: { label: 'tag', className: 'prompt-token-badge--tag' },
  lora: { label: 'lora', className: 'prompt-token-badge--lora' },
  caption: { label: 'phrase', className: 'prompt-token-badge--caption' },
};

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
