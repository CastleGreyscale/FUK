/**
 * MarkerTextarea — textarea with `#marker` autocomplete.
 *
 * Shared between PromptPanel and StoryboardTab so subjects, tags, and LoRA
 * triggers all surface through the same dropdown. Markers stay raw in the
 * saved value — the generation pipeline resolves them server-side.
 *
 * Behavior matches the original PromptPanel autocomplete:
 *   - Typing `#word` opens a filtered dropdown
 *   - ↑/↓ select, Enter/Tab insert, Esc dismiss
 *   - Click an item to insert
 *   - Caret moves (arrows without typing) keep the dropdown in sync
 *
 * Tokens are fetched once per (model, activeLoras) pair. Pass `tokens` /
 * `tokensError` directly when the parent already has them (PromptPanel does
 * — to feed the Slot Picker — so it avoids a duplicate fetch).
 */

import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { fetchPromptTokens, PROMPT_TOKENS_CHANGED_EVENT } from '../utils/promptApi';

// Match a `#` immediately before the caret followed by an in-progress word.
const MARKER_TYPING_RE = /(?:^|[\s,;()])#([A-Za-z0-9_\-]*)$/;

const SOURCE_BADGE = {
  global: { label: 'global', className: 'prompt-token-badge--global' },
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

function useAutoResize(ref, value, autoResize) {
  useEffect(() => {
    if (!autoResize) return;
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
  }, [ref, value, autoResize]);
}

const MarkerTextarea = forwardRef(function MarkerTextarea({
  value,
  onChange,
  placeholder,
  rows = 3,
  disabled,
  className = 'fuk-textarea',
  model,
  activeLoras,
  tokens: tokensProp,            // optional: pre-fetched tokens (PromptPanel passes its own)
  autoResize = true,
  onKeyDown: onKeyDownExtra,     // pass-through for parent shortcuts
}, externalRef) {
  const innerRef = useRef(null);
  useImperativeHandle(externalRef, () => innerRef.current, []);

  useAutoResize(innerRef, value, autoResize);

  // Stable, comparable representation of active LoRA keys for the deps array.
  const activeLoraKeys = useMemo(
    () => (activeLoras || []).map(l => (typeof l === 'string' ? l : l?.key)).filter(Boolean),
    [activeLoras],
  );
  const activeLoraKey = activeLoraKeys.join(',');

  // Local tokens fetch — only runs when caller didn't supply tokensProp.
  // Refetches on (model, activeLoras) change AND on the global
  // `fuk-prompt-tokens-changed` event so subject/tag CRUD elsewhere flows
  // through to every open autocomplete.
  const [localTokens, setLocalTokens] = useState([]);
  const [refetchKey, setRefetchKey] = useState(0);
  useEffect(() => {
    if (tokensProp) return;
    const handler = () => setRefetchKey(k => k + 1);
    window.addEventListener(PROMPT_TOKENS_CHANGED_EVENT, handler);
    return () => window.removeEventListener(PROMPT_TOKENS_CHANGED_EVENT, handler);
  }, [tokensProp]);
  useEffect(() => {
    if (tokensProp) return; // parent owns the fetch
    let cancelled = false;
    fetchPromptTokens({ model, activeLoras: activeLoraKeys })
      .then(data => {
        if (cancelled) return;
        setLocalTokens(data?.tokens || []);
      })
      .catch(() => {
        if (cancelled) return;
        setLocalTokens([]);
      });
    return () => { cancelled = true; };
  }, [model, activeLoraKey, tokensProp, refetchKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const tokens = tokensProp || localTokens;

  const [autocomplete, setAutocomplete] = useState(null); // { query, selected, markerStart, caret }
  // Computed JS positioning so the dropdown can flip up/down based on
  // available space and escape ancestor `overflow: hidden/auto` clipping.
  const [dropdownStyle, setDropdownStyle] = useState(null);

  // Re-compute placement on open and on scroll/resize while open. Capture
  // the scroll listener so an ancestor scroll container (e.g. the storyboard
  // tab's overflow-y: auto wrapper) still triggers a reposition.
  useEffect(() => {
    if (!autocomplete) {
      setDropdownStyle(null);
      return;
    }
    const compute = () => {
      const ta = innerRef.current;
      if (!ta) return;
      const r = ta.getBoundingClientRect();
      const margin = 8;            // gap from the textarea edge
      const cap = 260;             // visual ceiling for the dropdown
      const viewportH = window.innerHeight;
      const spaceBelow = viewportH - r.bottom - margin;
      const spaceAbove = r.top - margin;
      // Open in whichever direction has more room. Prefer downward when tied
      // since users read top-to-bottom.
      const openDown = spaceBelow >= spaceAbove;
      const maxH = Math.max(120, Math.min(cap, openDown ? spaceBelow : spaceAbove));
      const base = {
        position: 'fixed',
        left: r.left + 'px',
        width: r.width + 'px',
        maxHeight: maxH + 'px',
        zIndex: 1000,
      };
      setDropdownStyle(openDown
        ? { ...base, top: (r.bottom + margin) + 'px' }
        : { ...base, top: (r.top - margin - maxH) + 'px' });
    };
    compute();
    window.addEventListener('scroll', compute, true);
    window.addEventListener('resize', compute);
    return () => {
      window.removeEventListener('scroll', compute, true);
      window.removeEventListener('resize', compute);
    };
  }, [autocomplete]);

  const handleResize = useCallback((e) => {
    if (!autoResize) return;
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
  }, [autoResize]);

  const detectAutocomplete = useCallback((textarea) => {
    const v = textarea.value;
    const caret = textarea.selectionStart ?? v.length;
    if (caret !== (textarea.selectionEnd ?? caret)) {
      setAutocomplete(null);
      return;
    }
    const upTo = v.slice(0, caret);
    const m = MARKER_TYPING_RE.exec(upTo);
    if (!m) {
      setAutocomplete(null);
      return;
    }
    setAutocomplete({
      query: m[1] || '',
      selected: 0,
      markerStart: caret - m[1].length - 1, // position of the `#`
      caret,
    });
  }, []);

  const handleChange = (e) => {
    onChange(e.target.value);
    handleResize(e);
    detectAutocomplete(e.target);
  };

  const handleSelect = (e) => {
    if (autocomplete) detectAutocomplete(e.target);
  };

  const filteredTokens = useMemo(() => {
    if (!autocomplete) return [];
    const q = autocomplete.query;
    // Include markered tokens (tags, LoRA triggers, project subjects) AND
    // caption phrases (markerless) — the backend only ships caption tokens
    // when the corresponding LoRA is active, so they're already scoped.
    // On pick, captions insert as raw text (see insertToken).
    const candidates = tokens.filter(t => t.marker || t.source === 'caption');
    if (!q) return candidates.slice(0, 30);
    return candidates
      .map(t => ({ t, score: scoreToken(t, q) }))
      .filter(x => x.score >= 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 30)
      .map(x => x.t);
  }, [tokens, autocomplete]);

  const insertToken = useCallback((token) => {
    if (!autocomplete) return;
    const textarea = innerRef.current;
    if (!textarea) return;
    const before = value.slice(0, autocomplete.markerStart);
    const after = value.slice(autocomplete.caret);
    // Markered tokens (tags, LoRA triggers, subjects) insert their `#marker`
    // so resolution happens at generation. Caption phrases have no marker —
    // insert the raw phrase text and let the LLM see it verbatim.
    const insert = token.marker || token.expansion || token.name || '';
    if (!insert) return;
    const next = `${before}${insert}${after.startsWith(' ') ? '' : ' '}${after}`;
    onChange(next);
    setAutocomplete(null);

    const newCaret = before.length + insert.length + (after.startsWith(' ') ? 0 : 1);
    requestAnimationFrame(() => {
      const t = innerRef.current;
      if (t) {
        t.focus();
        t.setSelectionRange(newCaret, newCaret);
      }
    });
  }, [autocomplete, value, onChange]);

  const handleKeyDown = (e) => {
    if (autocomplete && filteredTokens.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setAutocomplete(a => ({ ...a, selected: Math.min(a.selected + 1, filteredTokens.length - 1) }));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setAutocomplete(a => ({ ...a, selected: Math.max(a.selected - 1, 0) }));
        return;
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        insertToken(filteredTokens[autocomplete.selected]);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setAutocomplete(null);
        return;
      }
    }
    onKeyDownExtra?.(e);
  };

  return (
    <div className="marker-textarea-wrapper">
      <textarea
        ref={innerRef}
        className={className}
        value={value || ''}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onSelect={handleSelect}
        onBlur={() => setTimeout(() => setAutocomplete(null), 120)}
        placeholder={placeholder}
        rows={rows}
        disabled={disabled}
      />
      {autocomplete && filteredTokens.length > 0 && dropdownStyle && createPortal(
        <TokenDropdown
          tokens={filteredTokens}
          selected={autocomplete.selected}
          onPick={insertToken}
          onHover={(idx) => setAutocomplete(a => ({ ...a, selected: idx }))}
          style={dropdownStyle}
        />,
        document.body,
      )}
    </div>
  );
});

export default MarkerTextarea;

function TokenDropdown({ tokens, selected, onPick, onHover, style }) {
  return (
    <ul className="prompt-token-dropdown" role="listbox" style={style}>
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
            {t.expansion && t.expansion !== t.name && (
              <span className="prompt-token-expansion">{t.expansion}</span>
            )}
          </li>
        );
      })}
    </ul>
  );
}
