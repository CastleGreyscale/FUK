import { useEffect, useRef } from 'react';

const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);
const inInput = () => INPUT_TAGS.has(document.activeElement?.tagName);
const dispatch = (name, detail) =>
  window.dispatchEvent(new CustomEvent(name, detail !== undefined ? { detail } : undefined));

// Tab order matches footer bar left-to-right
const TAB_KEYS = {
  '1': 'utilities',
  '2': 'storyboard',
  '3': 'preprocess',
  '4': 'image',
  '5': 'video',
  '6': 'postprocess',
  '7': 'layers',
  '8': 'export',
};

export function useKeyboardShortcuts({
  activeTab,
  setActiveTab,
  historyCollapsed,
  setHistoryCollapsed,
  historyFullscreen,
  setHistoryFullscreen,
}) {
  const optsRef = useRef({
    activeTab, setActiveTab,
    historyCollapsed, setHistoryCollapsed,
    historyFullscreen, setHistoryFullscreen,
  });
  useEffect(() => {
    optsRef.current = {
      activeTab, setActiveTab,
      historyCollapsed, setHistoryCollapsed,
      historyFullscreen, setHistoryFullscreen,
    };
  });

  useEffect(() => {
    const handler = (e) => {
      if (e.isComposing) return;
      const { key, code, shiftKey: shift, ctrlKey, metaKey } = e;
      const ctrl = ctrlKey || metaKey;
      const {
        activeTab, setActiveTab,
        historyCollapsed, setHistoryCollapsed,
        historyFullscreen, setHistoryFullscreen,
      } = optsRef.current;

      // 1–7: switch tabs
      if (!shift && !ctrl && !inInput() && TAB_KEYS[key]) {
        setActiveTab(TAB_KEYS[key]);
        return;
      }

      // Shift+G: generate
      if (key === 'G' && shift && !ctrl) {
        dispatch('fuk-shortcut-generate');
        return;
      }

      // Escape: blur focused input or cancel generation
      if (key === 'Escape') {
        const active = document.activeElement;
        if (active?.tagName === 'TEXTAREA' || active?.tagName === 'INPUT') {
          active.blur();
        } else {
          dispatch('fuk-shortcut-cancel');
        }
        return;
      }

      // H: cycle history panel states — closed → compact → fullscreen → closed
      if (key === 'h' && !shift && !ctrl && !inInput()) {
        if (historyCollapsed) {
          // closed → compact
          setHistoryCollapsed(false);
          setHistoryFullscreen(false);
        } else if (!historyFullscreen) {
          // compact → fullscreen
          setHistoryFullscreen(true);
        } else {
          // fullscreen → closed
          setHistoryFullscreen(false);
          setHistoryCollapsed(true);
        }
        return;
      }

      // Shift+Delete: delete selected history items (only when panel open)
      if (key === 'Delete' && shift && !ctrl && !inInput()) {
        if (!historyCollapsed) {
          dispatch('fuk-history-delete-selected');
        }
        return;
      }

      // Numpad +/- : vote
      if (!inInput()) {
        if (code === 'NumpadAdd') { dispatch('fuk-history-vote', 'up'); return; }
        if (code === 'NumpadSubtract') { dispatch('fuk-history-vote', 'down'); return; }
      }

      // Image + Video shared shortcuts
      if (activeTab === 'image' || activeTab === 'video') {
        if (!inInput()) {
          if (key === 'p' && !shift && !ctrl) { dispatch('fuk-shortcut-focus-prompt'); return; }
          if (key === 'P' && shift && !ctrl) { dispatch('fuk-shortcut-focus-neg-prompt'); return; }
          if (key === 'R' && shift && !ctrl) { dispatch('fuk-shortcut-seed-mode', 'random'); return; }
          if (key === 'F' && shift && !ctrl) { dispatch('fuk-shortcut-seed-mode', 'fixed'); return; }
          if (key === 'I' && shift && !ctrl) { dispatch('fuk-shortcut-seed-mode', 'increment'); return; }
          if (key === 's' && !shift && !ctrl) { dispatch('fuk-shortcut-seed-mode', 'cycle'); return; }
        }
      }

      // Video-only shortcuts
      if (activeTab === 'video' && !inInput()) {
        if (key === ' ') { e.preventDefault(); dispatch('fuk-shortcut-video-play-pause'); return; }
        if (key === 'ArrowRight') { dispatch('fuk-shortcut-video-step-forward'); return; }
        if (key === 'ArrowLeft') { dispatch('fuk-shortcut-video-step-back'); return; }
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []); // mounted once; reads latest state via optsRef
}
