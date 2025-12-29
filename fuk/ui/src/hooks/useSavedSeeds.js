/**
 * useSavedSeeds Hook
 * Manages saved/favorite seeds organized by model
 * Seeds are stored server-side in data/saved_seeds.json
 * 
 * Storage structure:
 * {
 *   "qwen_image": [
 *     { seed: 123456, note: "Great lighting", timestamp: "..." },
 *     ...
 *   ],
 *   "qwen_image_2509_edit": [...],
 *   "i2v-14B": [...],
 *   ...
 * }
 */

import { useState, useEffect, useCallback } from 'react';
import { API_URL } from '../utils/constants';

export function useSavedSeeds() {
  const [allSeeds, setAllSeeds] = useState({});
  const [isLoading, setIsLoading] = useState(true);

  // Load seeds from server on mount
  useEffect(() => {
    loadSeeds();
  }, []);

  /**
   * Load all seeds from server
   */
  const loadSeeds = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/seeds`);
      if (res.ok) {
        const data = await res.json();
        setAllSeeds(data);
      }
    } catch (e) {
      console.error('Failed to load saved seeds:', e);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Get seeds for a specific model
   */
  const getSeedsForModel = useCallback((model) => {
    return allSeeds[model] || [];
  }, [allSeeds]);

  /**
   * Save a seed for a model
   * @param {string} model - Model name (e.g., 'qwen_image', 'i2v-14B')
   * @param {number} seed - The seed value
   * @param {string} note - Optional note/description
   */
  const saveSeed = useCallback(async (model, seed, note = '') => {
    if (seed === null || seed === undefined) {
      console.warn('Cannot save null/undefined seed');
      return false;
    }

    try {
      const res = await fetch(`${API_URL}/seeds`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, seed, note }),
      });
      
      if (res.ok) {
        const data = await res.json();
        setAllSeeds(prev => ({ ...prev, [model]: data.seeds }));
        return true;
      }
    } catch (e) {
      console.error('Failed to save seed:', e);
    }
    return false;
  }, []);

  /**
   * Remove a seed from a model
   */
  const removeSeed = useCallback(async (model, seed) => {
    try {
      const res = await fetch(`${API_URL}/seeds`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, seed }),
      });
      
      if (res.ok) {
        setAllSeeds(prev => {
          const modelSeeds = (prev[model] || []).filter(s => s.seed !== seed);
          if (modelSeeds.length === 0) {
            const { [model]: removed, ...rest } = prev;
            return rest;
          }
          return { ...prev, [model]: modelSeeds };
        });
      }
    } catch (e) {
      console.error('Failed to remove seed:', e);
    }
  }, []);

  /**
   * Check if a seed is already saved for a model
   */
  const isSeedSaved = useCallback((model, seed) => {
    const modelSeeds = allSeeds[model] || [];
    return modelSeeds.some(s => s.seed === seed);
  }, [allSeeds]);

  /**
   * Get all models that have saved seeds
   */
  const getModelsWithSeeds = useCallback(() => {
    return Object.keys(allSeeds).filter(model => allSeeds[model].length > 0);
  }, [allSeeds]);

  return {
    // State
    allSeeds,
    isLoading,
    
    // Getters
    getSeedsForModel,
    isSeedSaved,
    getModelsWithSeeds,
    
    // Actions
    saveSeed,
    removeSeed,
    loadSeeds,
  };
}