/**
 * useLocalStorage Hook
 * Persist state to localStorage
 */

import { useState, useEffect } from 'react';

export function useLocalStorage(key, initialValue) {
  // Get initial value from localStorage or use provided default
  const [value, setValue] = useState(() => {
    try {
      const saved = localStorage.getItem(key);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch (e) {
      console.error(`Failed to load ${key} from localStorage:`, e);
    }
    return initialValue;
  });

  // Save to localStorage when value changes
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      console.error(`Failed to save ${key} to localStorage:`, e);
    }
  }, [key, value]);

  return [value, setValue];
}
