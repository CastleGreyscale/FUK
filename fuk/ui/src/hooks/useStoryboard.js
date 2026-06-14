/**
 * useStoryboard — manifest state for the Storyboard tab.
 *
 * Each mutation goes through the typed API helpers (which call into the
 * backend manifest), then mirrors the returned state locally. The hook is
 * always the source of truth for what's on disk — no optimistic edits, so
 * we don't drift if a save fails.
 */

import { useCallback, useEffect, useState } from 'react';
import {
  fetchManifest,
  saveSpecs,
  createSubject as apiCreateSubject,
  updateSubject as apiUpdateSubject,
  deleteSubject as apiDeleteSubject,
  saveMood as apiSaveMood,
  upsertPanel as apiUpsertPanel,
  deletePanel as apiDeletePanel,
  reorderSequence as apiReorderSequence,
} from '../utils/storyboardApi';
import { notifyPromptTokensChanged } from '../utils/promptApi';

export function useStoryboard(project) {
  const [manifest, setManifest] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    if (!project?.projectFolder) {
      setManifest(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await fetchManifest();
      setManifest(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [project?.projectFolder]);

  useEffect(() => { refresh(); }, [refresh]);

  // Reload when the active project file changes — globals stay alive across
  // shot switches but the panel grid wants to reflect whatever's on disk.
  useEffect(() => {
    const handler = () => refresh();
    window.addEventListener('fuk-project-changed', handler);
    return () => window.removeEventListener('fuk-project-changed', handler);
  }, [refresh]);

  const updateSpecs = useCallback(async (specs) => {
    const next = await saveSpecs(specs);
    setManifest(next);
  }, []);

  const createSubject = useCallback(async (payload) => {
    const entry = await apiCreateSubject(payload);
    setManifest(m => m && ({ ...m, globals: { ...m.globals, subjects: [...(m.globals?.subjects || []), entry] } }));
    notifyPromptTokensChanged();
    return entry;
  }, []);

  const updateSubject = useCallback(async (id, payload) => {
    const entry = await apiUpdateSubject(id, payload);
    setManifest(m => m && ({
      ...m,
      globals: {
        ...m.globals,
        subjects: (m.globals?.subjects || []).map(s => s.id === id ? entry : s),
      },
    }));
    notifyPromptTokensChanged();
    return entry;
  }, []);

  const deleteSubject = useCallback(async (id) => {
    await apiDeleteSubject(id);
    setManifest(m => m && ({
      ...m,
      globals: {
        ...m.globals,
        subjects: (m.globals?.subjects || []).filter(s => s.id !== id),
      },
    }));
    notifyPromptTokensChanged();
  }, []);

  const updateMood = useCallback(async (mood) => {
    const { mood: cleaned } = await apiSaveMood(mood);
    setManifest(m => m && ({ ...m, globals: { ...m.globals, mood: cleaned } }));
  }, []);

  const upsertPanel = useCallback(async (shotId, patch) => {
    const entry = await apiUpsertPanel(shotId, patch);
    setManifest(m => {
      if (!m) return m;
      const sequence = m.sequence.includes(shotId) ? m.sequence : [...m.sequence, shotId];
      return { ...m, panels: { ...m.panels, [shotId]: entry }, sequence };
    });
    return entry;
  }, []);

  const deletePanel = useCallback(async (shotId) => {
    await apiDeletePanel(shotId);
    setManifest(m => {
      if (!m) return m;
      const panels = { ...m.panels };
      delete panels[shotId];
      return { ...m, panels, sequence: m.sequence.filter(s => s !== shotId) };
    });
  }, []);

  const reorderSequence = useCallback(async (sequence) => {
    const { sequence: cleaned } = await apiReorderSequence(sequence);
    setManifest(m => m && ({ ...m, sequence: cleaned }));
  }, []);

  return {
    manifest,
    loading,
    error,
    refresh,
    updateSpecs,
    createSubject,
    updateSubject,
    deleteSubject,
    updateMood,
    upsertPanel,
    deletePanel,
    reorderSequence,
  };
}
