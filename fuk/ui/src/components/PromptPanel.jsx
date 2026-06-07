import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { compilePrompt, fetchPromptTokens } from '../utils/promptApi';

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
  const [autocomplete, setAutocomplete] = useState(null); // { query, selected, anchor: 'prompt' | 'neg' }
  const [compiling, setCompiling] = useState(false);
  const [compileError, setCompileError] = useState(null);

  // Fetch tokens whenever the prompt context changes.
  useEffect(() => {
    let cancelled = false;
    fetchPromptTokens({ model, activeLoras: activeLoraKeys })
      .then(data => { if (!cancelled) setTokens(data?.tokens || []); })
      .catch(() => { if (!cancelled) setTokens([]); });
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
  };

  const handlePromptSelect = (e) => {
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
    if (!prompt || compiling) return;
    setCompileError(null);
    setCompiling(true);
    try {
      const res = await compilePrompt({
        text: prompt,
        model,
        activeLoras: activeLoraKeys,
      });
      if (res?.compiled != null) {
        onChange('prompt', res.compiled);
      }
      if (res?.unknown_markers?.length) {
        setCompileError(`Unknown markers: ${res.unknown_markers.join(', ')}`);
      }
    } catch (err) {
      setCompileError(err.message || 'Compile failed');
    } finally {
      setCompiling(false);
    }
  };

  const hasMarkers = useMemo(() => /(^|\s)#[A-Za-z0-9]/.test(prompt || ''), [prompt]);

  return (
    <div className="prompt-panel">
      <div className="prompt-panel-header">
        <span className="prompt-panel-title">Prompt</span>
        <button
          type="button"
          className="prompt-panel-compile"
          onClick={handleCompile}
          disabled={disabled || compiling || !hasMarkers}
          title={hasMarkers ? 'Expand #markers using tags and LoRA triggers' : 'No #markers to expand'}
        >
          {compiling ? '…' : 'Compile'}
        </button>
      </div>
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
        {compileError && (
          <div className="prompt-panel-warning">{compileError}</div>
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
