/**
 * useActiveTab Hook
 * Manages active tab state with project persistence
 * 
 * Features:
 * - Persists active tab to project lastState
 * - Restores active tab when project loads
 * - Falls back to localStorage when no project loaded
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useLocalStorage } from './useLocalStorage';

// Default tab order for validation
const VALID_TABS = ['image', 'video', 'preprocess', 'postprocess', 'layers', 'export'];

export function useActiveTab(project, defaultTab = 'image') {
  // Fallback to localStorage when no project
  const [savedTab, setSavedTab] = useLocalStorage('fuk_active_tab', defaultTab);
  
  // Local state for current tab
  const [activeTab, setActiveTabState] = useState(defaultTab);
  
  // Track if we've restored from project state
  const hasRestoredRef = useRef(false);
  
  // Restore active tab from project state when project loads
  useEffect(() => {
    // Only restore once per project file load
    if (hasRestoredRef.current) return;
    
    const lastState = project?.projectState?.lastState;
    
    if (lastState?.activeTab && VALID_TABS.includes(lastState.activeTab)) {
      console.log('[useActiveTab] Restoring tab from project:', lastState.activeTab);
      setActiveTabState(lastState.activeTab);
      hasRestoredRef.current = true;
    } else if (savedTab && VALID_TABS.includes(savedTab)) {
      // Fall back to localStorage if no project tab saved
      setActiveTabState(savedTab);
    }
  }, [project?.projectState?.lastState, savedTab]);
  
  // Reset restoration flag when project file changes
  useEffect(() => {
    if (project?.currentFilename) {
      hasRestoredRef.current = false;
    }
  }, [project?.currentFilename]);
  
  // Set active tab with persistence
  const setActiveTab = useCallback((tab) => {
    if (!VALID_TABS.includes(tab)) {
      console.warn('[useActiveTab] Invalid tab:', tab);
      return;
    }
    
    setActiveTabState(tab);
    
    // Save to localStorage
    setSavedTab(tab);
    
    // Save to project lastState if available
    if (project?.updateLastState) {
      project.updateLastState({ activeTab: tab });
    }
  }, [project?.updateLastState, setSavedTab]);
  
  return [activeTab, setActiveTab];
}

export default useActiveTab;