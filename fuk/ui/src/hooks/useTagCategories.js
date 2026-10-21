/**
 * useTagCategories — the shared two-level tag category tree.
 *
 * Prime categories come from prompt_tag_categories.json and are the same for
 * workspace and project tags. Subcategories are free text; the server offers
 * the file's suggestions plus every subcategory already in use, so this
 * refetches whenever the prompt vocabulary changes.
 */

import { useCallback, useEffect, useState } from 'react';
import { fetchTagCategories, PROMPT_TOKENS_CHANGED_EVENT } from '../utils/promptApi';

export function useTagCategories() {
  const [categories, setCategories] = useState([]);
  const [subcategories, setSubcategories] = useState({});

  const load = useCallback(() => {
    fetchTagCategories()
      .then(data => {
        setCategories(data.categories || []);
        setSubcategories(data.subcategories || {});
      })
      .catch(e => console.warn('Could not load tag categories:', e));
  }, []);

  useEffect(() => {
    load();
    window.addEventListener(PROMPT_TOKENS_CHANGED_EVENT, load);
    window.addEventListener('fuk-storyboard-changed', load);
    return () => {
      window.removeEventListener(PROMPT_TOKENS_CHANGED_EVENT, load);
      window.removeEventListener('fuk-storyboard-changed', load);
    };
  }, [load]);

  return { categories, subcategories };
}

/** "lighting › rim", "lighting", or "" — how a tag's category reads in the UI. */
export function categoryLabel(category, subcategory) {
  if (!category) return '';
  return subcategory ? `${category} › ${subcategory}` : category;
}
