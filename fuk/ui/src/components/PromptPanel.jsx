import { useCallback, useEffect, useRef } from 'react';

function useAutoResize(ref, value) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = el.scrollHeight + 'px';
  }, [ref, value]);
}

export default function PromptPanel({ prompt, negativePrompt, onChange, disabled }) {
  const promptRef = useRef(null);
  const negRef = useRef(null);

  useAutoResize(promptRef, prompt);
  useAutoResize(negRef, negativePrompt);

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

  return (
    <div className="prompt-panel">
      <div className="prompt-panel-header">
        <span className="prompt-panel-title">Prompt</span>
      </div>
      <div className="prompt-panel-body">
        <textarea
          ref={promptRef}
          className="fuk-textarea prompt-panel-textarea"
          value={prompt}
          onChange={(e) => { onChange('prompt', e.target.value); handleResize(e); }}
          placeholder="A cinematic video of..."
          rows={3}
          disabled={disabled}
        />
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
