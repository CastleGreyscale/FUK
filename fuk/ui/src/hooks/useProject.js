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
  getDefaults,
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

  // Persist last loaded file (for restoring shot/version on reload)
  const [savedLastFile, setSavedLastFile] = useLocalStorage('fuk_last_file', null);

  // Persist project name when no shot files exist yet
  const [savedProjectName, setSavedProjectName] = useLocalStorage('fuk_project_name', null);
  const [projectName, setProjectName] = useState(savedProjectName);
  const savedLastFileRef = useRef(savedLastFile);
  useEffect(() => { savedLastFileRef.current = savedLastFile; }, [savedLastFile]);

  // Ref so refreshProjectFiles (empty deps) can always call current loadProjectFile
  const loadProjectFileRef = useRef(null);

  // Project folder state (actual current folder)
  const [projectFolder, setProjectFolderLocal] = useState(null);
  
  // Project file state
  const [projectFiles, setProjectFiles] = useState([]);
  const [currentFilename, setCurrentFilename] = useState(null);
  const [projectState, setProjectState] = useState(null);
  const [projectConfig, setProjectConfig] = useState({ versionFormat: 'date' });
  const [defaults, setDefaults] = useState(null);  // User defaults from defaults.json
  
  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);  // NEW: actual save in progress
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
  
  // Available shots — natural sort so `1, 2, 10` orders intuitively, and
  // alphanumeric ids like `intro`, `alt-2` slot in alphabetically.
  const shots = useMemo(() => {
    return Object.keys(organizedFiles).sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
    );
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
        
        // Tell backend about the folder change (updates _project_folder and _cache_root)
        await setProjectFolder(result.path);
        
        setProjectFolderLocal(result.path);
        setSavedProjectFolder(result.path);
        
        // Get config
        try {
          const config = await getProjectConfig();
          setProjectConfig(config);
        } catch (e) {
          console.warn('[Project] Could not load config, using defaults');
        }
        
        // Get defaults from backend
        try {
          const defaultsData = await getDefaults();
          setDefaults(defaultsData);
        } catch (e) {
          console.warn('[Project] Could not load defaults.json');
        }
        
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
        // Clear any pending project name — real files are present
        setProjectName(null);
        setSavedProjectName(null);
        const organized = organizeProjectFiles(result.files);
        const fileNames = result.files.map(f => f.name);
        const lastFile = savedLastFileRef.current;
        const loader = loadProjectFileRef.current || loadProjectFile;

        if (lastFile && fileNames.includes(lastFile)) {
          // Restore the last file the user had open
          await loader(lastFile);
        } else {
          // Fall back to first shot's latest version
          const firstShot = Object.values(organized)[0];
          if (firstShot?.versions?.length > 0) {
            await loader(firstShot.versions[0].filename);
          }
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
      
      // Get defaults from backend
      try {
        const defaultsData = await getDefaults();
        setDefaults(defaultsData);
      } catch (e) {
        console.warn('[Project] Could not load defaults.json');
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
      const response = await loadProject(filename);

      // Handle new response format with comprehensive info
      const data = response.data || response;  // Backwards compatible
      const merged = mergeWithDefaults(data, defaults || {});

      setProjectState(merged);
      setCurrentFilename(filename);
      setHasUnsavedChanges(false);
      setSavedLastFile(filename);
      
      // Log cache information
      if (response.projectCache) {
        console.log('[Project] Project cache:', response.projectCache);
      }
      if (response.projectInfo) {
        console.log('[Project] Project info:', response.projectInfo);
      }
      
      console.log('[Project] File loaded successfully');
      
      // Dispatch custom event to notify other components that project changed
      window.dispatchEvent(new CustomEvent('fuk-project-changed', {
        detail: {
          filename,
          projectInfo: response.projectInfo,
          projectCache: response.projectCache,
          timestamp: Date.now(),
        }
      }));
      
      return merged;
    } catch (err) {
      console.error('[Project] Failed to load file:', err);
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [defaults, setSavedLastFile]);

  // Keep ref current so refreshProjectFiles (empty deps) calls the latest version
  useEffect(() => { loadProjectFileRef.current = loadProjectFile; }, [loadProjectFile]);

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
   * Register a project name — no shot file is created yet.
   * The first shot is created explicitly via createNewShot().
   */
  const createProject = useCallback(async (name) => {
    setProjectName(name);
    setSavedProjectName(name);
  }, [setSavedProjectName]);

  /**
   * Create a shot file and refresh the shot list WITHOUT switching to it.
   * Used by the storyboard tab so the active shot doesn't change.
   */
  const createShotFile = useCallback(async (shotId) => {
    const name = currentFileInfo?.projectName || projectName;
    if (!name) throw new Error('No project name set — open or name a project first');
    setIsLoading(true);
    setError(null);
    try {
      await createNewProject(name, shotId);
      setProjectName(null);
      setSavedProjectName(null);
      await refreshProjectFiles();
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [currentFileInfo, projectName, refreshProjectFiles, setSavedProjectName]);

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
   * Create a new shot file. Uses the current project name (from loaded file
   * or the pending name set via createProject).
   */
  const createNewShot = useCallback(async (shotId) => {
    const name = currentFileInfo?.projectName || projectName;
    if (!name) throw new Error('No project name set');
    setIsLoading(true);
    setError(null);
    try {
      console.log('[Project] Creating shot:', name, shotId);
      const result = await createNewProject(name, shotId);
      setProjectName(null);
      setSavedProjectName(null);
      await refreshProjectFiles();
      await loadProjectFile(result.filename);
      console.log('[Project] Shot created:', result.filename);
      return result.filename;
    } catch (err) {
      console.error('[Project] Shot create failed:', err);
      setError(err.message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [currentFileInfo, projectName, refreshProjectFiles, loadProjectFile, setSavedProjectName]);

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
   * Update pinned generations list (for history panel)
   */
  const updatePinnedGenerations = useCallback((pinnedIds) => {
    setProjectState(prev => {
      if (!prev) return prev;
      
      return {
        ...prev,
        pinnedGenerations: pinnedIds,
      };
    });
    setHasUnsavedChanges(true);
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
        setIsSaving(true);  // Show saving indicator
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
      } finally {
        setIsSaving(false);  // Hide saving indicator
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
        // Load defaults first
        try {
          const defaultsData = await getDefaults();
          setDefaults(defaultsData);
        } catch (e) {
          console.warn('[Project] Could not load defaults.json');
        }
        
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
    projectName,
    projectFiles,
    organizedFiles,
    currentFilename,
    currentFileInfo,
    projectState,
    projectConfig,
    defaults,  // User defaults from defaults.json
    shots,
    currentShotVersions,
    
    // Status
    isLoading,
    isSaving,  // NEW
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
    createShotFile,
    createProject,
    updateTabState,
    updateLastState,
    updatePinnedGenerations,
    refreshProjectFiles,
  };
}