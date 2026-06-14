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
  createStoryboardTag as apiCreateTag,
  updateStoryboardTag as apiUpdateTag,
  deleteStoryboardTag as apiDeleteTag,
  saveMood as apiSaveMood,
  saveGlobalImageSeed as apiSaveImageSeed,
  saveGlobalVideoSeed as apiSaveVideoSeed,
  saveActiveLoras as apiSaveActiveLoras,
  upsertPanel as apiUpsertPanel,
  deletePanel as apiDeletePanel,
  reorderSequence as apiReorderSequence,
  setPanelPreview as apiSetPanelPreview,
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
  // Also reload when the storyboard is mutated from outside the tab (e.g.
  // the History panel's "Pin to storyboard" button).
  useEffect(() => {
    const handler = () => refresh();
    window.addEventListener('fuk-project-changed', handler);
    window.addEventListener('fuk-storyboard-changed', handler);
    return () => {
      window.removeEventListener('fuk-project-changed', handler);
      window.removeEventListener('fuk-storyboard-changed', handler);
    };
  }, [refresh]);

  const updateSpecs = useCallback(async (specs) => {
    const next = await saveSpecs(specs);
    setManifest(next);
  }, []);

  const createTag = useCallback(async (payload) => {
    const entry = await apiCreateTag(payload);
    setManifest(m => m && ({ ...m, globals: { ...m.globals, tags: [...(m.globals?.tags || []), entry] } }));
    notifyPromptTokensChanged();
    return entry;
  }, []);

  const updateTag = useCallback(async (id, payload) => {
    const entry = await apiUpdateTag(id, payload);
    setManifest(m => m && ({
      ...m,
      globals: {
        ...m.globals,
        tags: (m.globals?.tags || []).map(t => t.id === id ? entry : t),
      },
    }));
    notifyPromptTokensChanged();
    return entry;
  }, []);

  const deleteTag = useCallback(async (id) => {
    await apiDeleteTag(id);
    setManifest(m => m && ({
      ...m,
      globals: {
        ...m.globals,
        tags: (m.globals?.tags || []).filter(t => t.id !== id),
      },
    }));
    notifyPromptTokensChanged();
  }, []);

  const updateMood = useCallback(async (mood) => {
    const { mood: cleaned } = await apiSaveMood(mood);
    setManifest(m => m && ({ ...m, globals: { ...m.globals, mood: cleaned } }));
  }, []);

  const updateImageSeed = useCallback(async (seed) => {
    const { image_seed } = await apiSaveImageSeed(seed);
    setManifest(m => m && ({ ...m, globals: { ...m.globals, image_seed } }));
    return image_seed;
  }, []);

  const updateVideoSeed = useCallback(async (seed) => {
    const { video_seed } = await apiSaveVideoSeed(seed);
    setManifest(m => m && ({ ...m, globals: { ...m.globals, video_seed } }));
    return video_seed;
  }, []);

  const updateActiveLoras = useCallback(async (loras) => {
    const { active_loras } = await apiSaveActiveLoras(loras);
    setManifest(m => m && ({ ...m, globals: { ...m.globals, active_loras } }));
    // Caption tokens are gated on active LoRAs server-side, so this flips
    // them on/off in every open MarkerTextarea instantly.
    notifyPromptTokensChanged();
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

  const setPanelPreview = useCallback(async (shotId, { kind, path }) => {
    const entry = await apiSetPanelPreview(shotId, { kind, path });
    setManifest(m => m && ({
      ...m,
      panels: { ...m.panels, [shotId]: entry },
    }));
    return entry;
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
    createTag,
    updateTag,
    deleteTag,
    updateMood,
    updateImageSeed,
    updateVideoSeed,
    updateActiveLoras,
    upsertPanel,
    deletePanel,
    reorderSequence,
    setPanelPreview,
  };
}
