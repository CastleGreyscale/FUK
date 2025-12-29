/**
 * FUK Project API
 * API calls for project file management
 */

import { API_URL } from './constants';

/**
 * Browse for folder using native dialog
 */
export async function browseForFolder() {
  const res = await fetch(`${API_URL}/project/browse-folder`, {
    method: 'POST',
  });
  
  if (!res.ok) {
    throw new Error(`Failed to open folder browser: ${res.statusText}`);
  }
  
  return res.json();
}

/**
 * Set the active project folder (manual path)
 */
export async function setProjectFolder(folderPath) {
  const res = await fetch(`${API_URL}/project/set-folder`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: folderPath }),
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Failed to set project folder`);
  }
  
  return res.json();
}

/**
 * Get current project state
 */
export async function getCurrentProject() {
  const res = await fetch(`${API_URL}/project/current`);
  
  if (!res.ok) {
    throw new Error(`Failed to get current project: ${res.statusText}`);
  }
  
  return res.json();
}

/**
 * Get list of project files in current folder
 */
export async function listProjectFiles() {
  const res = await fetch(`${API_URL}/project/list`);
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Failed to list project files`);
  }
  
  return res.json();
}

/**
 * Load a specific project file
 */
export async function loadProject(filename) {
  const res = await fetch(`${API_URL}/project/load/${encodeURIComponent(filename)}`);
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Failed to load project`);
  }
  
  return res.json();
}

/**
 * Save project state to file
 */
export async function saveProject(filename, state) {
  const res = await fetch(`${API_URL}/project/save/${encodeURIComponent(filename)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(state),
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Failed to save project`);
  }
  
  return res.json();
}

/**
 * Create a new project file
 */
export async function createNewProject(projectName, shotNumber = '01') {
  const res = await fetch(`${API_URL}/project/new`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ 
      projectName,
      shotNumber,
    }),
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `Failed to create project`);
  }
  
  return res.json();
}

/**
 * Get project config (version format, cache location, etc.)
 */
export async function getProjectConfig() {
  const res = await fetch(`${API_URL}/project/config`);
  
  if (!res.ok) {
    throw new Error(`Failed to get project config: ${res.statusText}`);
  }
  
  return res.json();
}

/**
 * Get cache folder info
 */
export async function getCacheInfo() {
  const res = await fetch(`${API_URL}/project/cache-info`);
  
  if (!res.ok) {
    throw new Error(`Failed to get cache info: ${res.statusText}`);
  }
  
  return res.json();
}
