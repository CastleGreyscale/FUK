/**
 * TagManagerModal — the tag editor as a popup, reachable from any tab.
 *
 * Opened by the header "Tags" button, `t`, or Alt+T inside a prompt (which
 * opens straight onto the `#marker` under the caret). See App.jsx.
 *
 * Esc closes it when no field is focused; the listener runs in the capture
 * phase and stops the event so the global Esc handler doesn't also cancel a
 * running generation.
 */

import { useCallback, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { X } from './Icons';
import TagManager from './TagManager';

const INPUT_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

export default function TagManagerModal({ project, focusRequest, onClose }) {
  const managerRef = useRef(null);

  const requestClose = useCallback(() => {
    if (managerRef.current?.isDirty() && !window.confirm('Close and discard the unsaved tag edit?')) return;
    onClose();
  }, [onClose]);

  useEffect(() => {
    const handler = (e) => {
      if (e.key !== 'Escape') return;
      if (INPUT_TAGS.has(document.activeElement?.tagName)) return; // first Esc just blurs
      e.stopPropagation();
      e.preventDefault();
      requestClose();
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [requestClose]);

  // A bare `t` toggles the popup shut; Alt+T on a marker retargets it (App).
  useEffect(() => {
    const handler = (e) => { if (!e.detail?.marker) requestClose(); };
    window.addEventListener('fuk-shortcut-tag-manager', handler);
    return () => window.removeEventListener('fuk-shortcut-tag-manager', handler);
  }, [requestClose]);

  return createPortal(
    <div className="tag-modal-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) requestClose(); }}>
      <div className="tag-modal" role="dialog" aria-modal="true" aria-label="Tags">
        <div className="tag-modal-header">
          <h2 className="tag-modal-title">Tags</h2>
          <span className="tag-modal-hint">
            <kbd>t</kbd> toggles · <kbd>Alt</kbd>+<kbd>T</kbd> on a <code>#marker</code> in a prompt opens it · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> saves
          </span>
          <button className="tag-modal-close" onClick={requestClose} title="Close (Esc)">
            <X />
          </button>
        </div>
        <TagManager
          ref={managerRef}
          project={project}
          focusRequest={focusRequest}
          className="tag-manager--modal"
        />
      </div>
    </div>,
    document.body,
  );
}
