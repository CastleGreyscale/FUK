/**
 * useProject Hook
 * Manages project state, loading, saving, and versioning
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useLocalStorage } from './useLocalStorage';
import {
  browseForFolder,
  listProjectFiles,
  loadProject,
  saveProject,
  setProjectFolder,
  getProjectConfig,
  createNewProject,
  getCurrentProject,
} from '../utils/projectApi';
import {
  parseProjectFilename,
  generateProjectFilename,
  generateVersion,
  getNextSequentialVersion,
  createEmptyProjectState,
  mergeWithDefaults,
  organizeProjectFiles,
} from '../utils/projectManager';

// Autosave delay in milliseconds
const AUTOSAVE_DELAY = 2000;

export function useProject() {
  // Persist last used project folder
  const [savedProjectFolder, setSavedProjectFolder] = useLocalStorage('fuk_project_folder', null);
  
  // Project folder state (actual current folder)
  const [projectFolder, setProjectFolderLocal] = useState(null);
  
  // Project file state
  const [projectFiles, setProjectFiles] = useState([]);
  const [currentFilename, setCurrentFilename] = useState(null);
  const [projectState, setProjectState] = useState(null);
  const [projectConfig, setProjectConfig] = useState({ versionFormat: 'date' });
  
  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [lastSaved, setLastSaved] = useState(null);
  
  // Autosave timer ref
  const autosaveTimerRef = useRef(null);
  
  // Organized files (grouped by shot/version)
  const organizedFiles = useMemo(() => {
    return organizeProjectFiles(projectFiles);
  }, [projectFiles]);
  
  // Current file info
  const currentFileInfo = useMemo(() => {
    if (!currentFilename) return null;
    return parseProjectFilename(currentFilename);
  }, [currentFilename]);
  
  // Available shots
  const shots = useMemo(() => {
    return Object.keys(organizedFiles).sort();
  }, [organizedFiles]);
  
  // Current shot's versions
  const currentShotVersions = useMemo(() => {
    if (!currentFileInfo) return [];
    return organizedFiles[currentFileInfo.shotNumber]?.versions || [];
  }, [organizedFiles, currentFileInfo]);

  /**
   * Open folder browser dialog
   */
  const openFolderBrowser = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('[Project] Opening folder browser...');
      const result = await browseForFolder();
      
      if (result.cancelled) {
        console.log('[Project] Folder selection cancelled');
        setIsLoading(false);
        return null;
      }
      
      if (result.path) {
        console.log('[Project] Folder selected:', result.path);
        setProjectFolderLocal(result.path);
        setSavedProjectFolder(result.path);
        
        // Load project files
        await refreshProjectFiles();
        
        return result.path;
      }
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      return null;
    } catch (err) {
      console.error('[Project] Folder browser error:', err);
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [setSavedProjectFolder]);

  /**
   * Refresh project files list
   */
  const refreshProjectFiles = useCallback(async () => {
    try {
      const result = await listProjectFiles();
      console.log('[Project] Found files:', result.files?.length || 0);
      setProjectFiles(result.files || []);
      setProjectFolderLocal(result.folder);
      
      // Auto-load most recent file if available
      if (result.files && result.files.length > 0) {
        const organized = organizeProjectFiles(result.files);
        const firstShot = Object.values(organized)[0];
        if (firstShot?.versions?.length > 0) {
          await loadProjectFile(firstShot.versions[0].filename);
        }
      }
      
      return result.files;
    } catch (err) {
      console.error('[Project] Failed to list files:', err);
      setProjectFiles([]);
      return [];
    }
  }, []);

  /**
   * Set project folder manually (by path)
   */
  const openProjectFolder = useCallback(async (folderPath) => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('[Project] Setting folder:', folderPath);
      const result = await setProjectFolder(folderPath);
      
      setProjectFolderLocal(result.folder);
      setSavedProjectFolder(result.folder);
      
      // Get config
      try {
        const config = await getProjectConfig();
        setProjectConfig(config);
      } catch (e) {
        console.warn('[Project] Could not load config, using defaults');
      }
      
      // Load project files
      await refreshProjectFiles();
      
      return result;
    } catch (err) {
      console.error('[Project] Failed to open folder:', err);
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [setSavedProjectFolder, refreshProjectFiles]);

  /**
   * Load a specific project file
   */
  const loadProjectFile = useCallback(async (filename) => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('[Project] Loading file:', filename);
      const data = await loadProject(filename);
      const merged = mergeWithDefaults(data);
      
      setProjectState(merged);
      setCurrentFilename(filename);
      setHasUnsavedChanges(false);
      
      console.log('[Project] File loaded successfully');
      return merged;
    } catch (err) {
      console.error('[Project] Failed to load file:', err);
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Save current project state
   */
  const save = useCallback(async () => {
    if (!currentFilename || !projectState) {
      throw new Error('No project to save');
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      const stateToSave = {
        ...projectState,
        meta: {
          ...projectState.meta,
          updatedAt: new Date().toISOString(),
        },
      };
      
      await saveProject(currentFilename, stateToSave);
      setProjectState(stateToSave);
      setHasUnsavedChanges(false);
      setLastSaved(new Date());
      
      console.log('[Project] Saved:', currentFilename);
      return true;
    } catch (err) {
      console.error('[Project] Save failed:', err);
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [currentFilename, projectState]);

  /**
   * Save as new version
   */
  const saveAsNewVersion = useCallback(async () => {
    if (!currentFileInfo || !projectState) {
      throw new Error('No project loaded');
    }
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Generate new version string
      let newVersion;
      if (projectConfig.versionFormat === 'date') {
        newVersion = generateVersion('date');
        
        // If same day, append a letter
        const existingVersions = currentShotVersions.map(v => v.version);
        if (existingVersions.includes(newVersion)) {
          let suffix = 'a';
          while (existingVersions.includes(newVersion + suffix)) {
            suffix = String.fromCharCode(suffix.charCodeAt(0) + 1);
          }
          newVersion = newVersion + suffix;
        }
      } else {
        const existingVersions = currentShotVersions.map(v => v.version);
        newVersion = getNextSequentialVersion(existingVersions);
      }
      
      const newFilename = generateProjectFilename(
        currentFileInfo.projectName,
        currentFileInfo.shotNumber,
        newVersion
      );
      
      const stateToSave = {
        ...projectState,
        meta: {
          ...projectState.meta,
          updatedAt: new Date().toISOString(),
          createdAt: new Date().toISOString(),
        },
        project: {
          ...projectState.project,
          version: newVersion,
        },
      };
      
      await saveProject(newFilename, stateToSave);
      
      // Refresh file list
      await refreshProjectFiles();
      
      setProjectState(stateToSave);
      setCurrentFilename(newFilename);
      setHasUnsavedChanges(false);
      
      console.log('[Project] New version created:', newFilename);
      return newFilename;
    } catch (err) {
      console.error('[Project] Version up failed:', err);
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [currentFileInfo, projectState, projectConfig, currentShotVersions, refreshProjectFiles]);

  /**
   * Create new project file
   */
  const createProject = useCallback(async (projectName, shotNumber = '01') => {
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('[Project] Creating new project:', projectName, 'shot', shotNumber);
      const result = await createNewProject(projectName, shotNumber);
      
      // Refresh and load the new file
      await refreshProjectFiles();
      await loadProjectFile(result.filename);
      
      console.log('[Project] New project created:', result.filename);
      return result.filename;
    } catch (err) {
      console.error('[Project] Create failed:', err);
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [refreshProjectFiles, loadProjectFile]);

  /**
   * Switch to different shot
   */
  const switchShot = useCallback(async (shotNumber) => {
    const shotData = organizedFiles[shotNumber];
    if (!shotData || shotData.versions.length === 0) {
      throw new Error(`No versions found for shot ${shotNumber}`);
    }
    
    await loadProjectFile(shotData.versions[0].filename);
  }, [organizedFiles, loadProjectFile]);

  /**
   * Switch to different version of current shot
   */
  const switchVersion = useCallback(async (version) => {
    if (!currentFileInfo) return;
    
    const shotData = organizedFiles[currentFileInfo.shotNumber];
    const versionData = shotData?.versions.find(v => v.version === version);
    
    if (!versionData) {
      throw new Error(`Version ${version} not found`);
    }
    
    await loadProjectFile(versionData.filename);
  }, [currentFileInfo, organizedFiles, loadProjectFile]);

  /**
   * Create new shot
   */
  const createNewShot = useCallback(async (shotNumber) => {
    if (!currentFileInfo) {
      throw new Error('No project loaded');
    }
    
    await createProject(currentFileInfo.projectName, shotNumber);
  }, [currentFileInfo, createProject]);

  /**
   * Update tab state (marks as dirty, triggers autosave)
   */
  const updateTabState = useCallback((tabName, updates) => {
    setProjectState(prev => {
      if (!prev) return prev;
      
      return {
        ...prev,
        tabs: {
          ...prev.tabs,
          [tabName]: {
            ...prev.tabs[tabName],
            ...updates,
          },
        },
      };
    });
    setHasUnsavedChanges(true);
  }, []);

  /**
   * Update last state (for restore functionality, also triggers autosave)
   */
  const updateLastState = useCallback((updates) => {
    setProjectState(prev => {
      if (!prev) return prev;
      
      return {
        ...prev,
        lastState: {
          ...prev.lastState,
          ...updates,
        },
      };
    });
    setHasUnsavedChanges(true);  // Also mark dirty so autosave triggers
  }, []);

  /**
   * Autosave effect - saves after delay when changes are made
   */
  useEffect(() => {
    // Only autosave if we have unsaved changes and a file is loaded
    if (!hasUnsavedChanges || !currentFilename || !projectState) {
      return;
    }

    // Clear any existing timer
    if (autosaveTimerRef.current) {
      clearTimeout(autosaveTimerRef.current);
    }

    // Set new autosave timer
    autosaveTimerRef.current = setTimeout(async () => {
      try {
        console.log('[Project] Autosaving...');
        const stateToSave = {
          ...projectState,
          meta: {
            ...projectState.meta,
            updatedAt: new Date().toISOString(),
          },
        };
        
        await saveProject(currentFilename, stateToSave);
        setHasUnsavedChanges(false);
        setLastSaved(new Date());
        console.log('[Project] Autosaved:', currentFilename);
      } catch (err) {
        console.error('[Project] Autosave failed:', err);
        // Don't clear hasUnsavedChanges on error - will retry
      }
    }, AUTOSAVE_DELAY);

    // Cleanup on unmount or when dependencies change
    return () => {
      if (autosaveTimerRef.current) {
        clearTimeout(autosaveTimerRef.current);
      }
    };
  }, [hasUnsavedChanges, currentFilename, projectState]);

  // Check for existing project folder on mount
  useEffect(() => {
    const checkExistingProject = async () => {
      try {
        const current = await getCurrentProject();
        if (current.isSet) {
          console.log('[Project] Found existing project folder:', current.folder);
          setProjectFolderLocal(current.folder);
          await refreshProjectFiles();
        } else if (savedProjectFolder) {
          // Try to restore saved folder
          console.log('[Project] Restoring saved folder:', savedProjectFolder);
          await openProjectFolder(savedProjectFolder);
        }
      } catch (err) {
        console.warn('[Project] Could not restore project:', err);
      }
    };
    
    checkExistingProject();
  }, []);

  return {
    // State
    projectFolder,
    projectFiles,
    organizedFiles,
    currentFilename,
    currentFileInfo,
    projectState,
    projectConfig,
    shots,
    currentShotVersions,
    
    // Status
    isLoading,
    error,
    hasUnsavedChanges,
    lastSaved,
    isProjectLoaded: !!projectState,
    hasProjectFolder: !!projectFolder,
    
    // Actions
    openFolderBrowser,
    openProjectFolder,
    loadProjectFile,
    save,
    saveAsNewVersion,
    switchShot,
    switchVersion,
    createNewShot,
    createProject,
    updateTabState,
    updateLastState,
    refreshProjectFiles,
  };
}
