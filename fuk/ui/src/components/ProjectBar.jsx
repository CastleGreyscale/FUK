/**
 * ProjectBar Component
 * Horizontal project navigation: Project > Shot > Version
 */

import { useState } from 'react';
import { Save, Loader2, AlertCircle, CheckCircle, Folder, FilePlus, Plus, ChevronDown } from './Icons';


export default function ProjectBar({
  projectFolder,
  currentFileInfo,
  shots,
  currentShotVersions,
  hasUnsavedChanges,
  isSaving,
  isLoading,
  hasFiles,
  onBrowseFolder,
  onSwitchShot,
  onSwitchVersion,
  onSave,
  onVersionUp,
  onNewShot,
  onNewProject,
}) {
  const [showNewShotInput, setShowNewShotInput] = useState(false);
  const [newShotNumber, setNewShotNumber] = useState('');
  const [showNewProjectDialog, setShowNewProjectDialog] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');

  const handleNewShotSubmit = (e) => {
    e.preventDefault();
    if (newShotNumber) {
      onNewShot(newShotNumber);
      setNewShotNumber('');
      setShowNewShotInput(false);
    }
  };

  const handleNewProjectSubmit = (e) => {
    e.preventDefault();
    if (newProjectName) {
      onNewProject(newProjectName);
      setNewProjectName('');
      setShowNewProjectDialog(false);
    }
  };

  // Get folder display name
  const getFolderDisplayName = () => {
    if (!projectFolder) return null;
    const parts = projectFolder.split('/').filter(Boolean);
    return parts[parts.length - 1] || 'Fuk';
  };

  // No project folder state
  if (!projectFolder) {
    return (
      <div className="fuk-project-bar fuk-project-bar-empty">
        <button 
          className="fuk-btn fuk-btn-secondary"
          onClick={onBrowseFolder}
          disabled={isLoading}
        >
          {isLoading ? (
            <Loader2 className="fuk-icon--md animate-spin" />
          ) : (
            <Folder className="fuk-icon--md" />
          )}
          {isLoading ? 'Opening...' : 'Open Project Folder'}
        </button>
        <span className="fuk-project-hint">
          Navigate to: ProjectName/Projects/Fuk/
        </span>
      </div>
    );
  }

  // Folder set but no project files - show "New Project" prompt
  if (!hasFiles) {
    return (
      <div className="fuk-project-bar">
        <button 
          className="fuk-project-folder-btn"
          onClick={onBrowseFolder}
          title={projectFolder}
        >
          <Folder className="fuk-icon--md" />
          <span className="fuk-project-folder-name">{getFolderDisplayName()}</span>
          <ChevronDown className="fuk-project-chevron" />
        </button>

        <span className="fuk-project-separator">/</span>

        {showNewProjectDialog ? (
          <form onSubmit={handleNewProjectSubmit} className="fuk-new-project-form">
            <input
              type="text"
              className="fuk-new-project-input"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="project_name"
              autoFocus
            />
            <button type="submit" className="fuk-btn fuk-btn-save" disabled={!newProjectName}>
              Create
            </button>
            <button 
              type="button" 
              className="fuk-btn fuk-btn-secondary"
              onClick={() => setShowNewProjectDialog(false)}
            >
              Cancel
            </button>
          </form>
        ) : (
          <button 
            className="fuk-btn fuk-btn-new-project"
            onClick={() => setShowNewProjectDialog(true)}
          >
            <FilePlus className="fuk-icon--md" />
            New Project File
          </button>
        )}

        <span className="fuk-project-hint fuk-project-hint--spaced">
          No project files found. Create one to get started.
        </span>
      </div>
    );
  }

  return (
    <div className="fuk-project-bar">
      {/* Project Folder Button */}
      <button 
        className="fuk-project-select"
        onClick={onBrowseFolder}
        title={projectFolder}
        disabled={isLoading}
      >
        {isLoading ? (
          <Loader2 className="fuk-icon--md animate-spin" />
        ) : (
          <Folder className="fuk-icon--md" />
        )}
        <span className="fuk-project-folder-name">{getFolderDisplayName()}</span>
        <ChevronDown className="fuk-project-chevron" />
      </button>
      
      <span className="fuk-project-separator">/</span>

      {/* Shot Selector */}
      <div className="fuk-project-selector">
        <label className="fuk-project-label">Shot</label>
        <div className="fuk-project-dropdown-group">
          <select
            className="fuk-project-select"
            value={currentFileInfo?.shotNumber || ''}
            onChange={(e) => onSwitchShot(e.target.value)}
            disabled={isLoading || shots.length === 0}
          >
            {shots.length === 0 ? (
              <option value="">No shots</option>
            ) : (
              shots.map(shot => (
                <option key={shot} value={shot}>
                  {shot.padStart(2, '0')}
                </option>
              ))
            )}
          </select>
          
          {/* New Shot Button */}
          {showNewShotInput ? (
            <form onSubmit={handleNewShotSubmit} className="fuk-new-shot-form">
              <input
                type="number"
                className="fuk-new-shot-input"
                value={newShotNumber}
                onChange={(e) => setNewShotNumber(e.target.value)}
                placeholder="##"
                min="1"
                max="99"
                autoFocus
                onBlur={() => {
                  if (!newShotNumber) setShowNewShotInput(false);
                }}
              />
            </form>
          ) : (
            <button
              className="fuk-project-add-btn"
              onClick={() => setShowNewShotInput(true)}
              title="New Shot"
            >
              <Plus className="fuk-project-add-icon" />
            </button>
          )}
        </div>
      </div>

      <span className="fuk-project-separator">/</span>

      {/* Version Selector */}
      <div className="fuk-project-selector">
        <label className="fuk-project-label">Version</label>
        <select
          className="fuk-project-select"
          value={currentFileInfo?.version || ''}
          onChange={(e) => onSwitchVersion(e.target.value)}
          disabled={isLoading || currentShotVersions.length === 0}
        >
          {currentShotVersions.length === 0 ? (
            <option value="">No versions</option>
          ) : (
            currentShotVersions.map(v => (
              <option key={v.version} value={v.version}>
                {v.version}
              </option>
            ))
          )}
        </select>
      </div>

      {/* Spacer */}
      <div className="fuk-project-spacer" />

      {/* Autosave Status Indicator */}
      {isSaving ? (
        <span className="fuk-autosave-indicator fuk-autosave-pending" title="Saving...">
          <Loader2 className="fuk-autosave-icon animate-spin" />
          <span>Saving...</span>
        </span>
      ) : hasUnsavedChanges ? (
        <span className="fuk-autosave-indicator fuk-autosave-dirty" title="Unsaved changes">
          <span className="fuk-dirty-dot" />
        </span>
      ) : currentFileInfo && (
        <span className="fuk-autosave-indicator fuk-autosave-saved" title="All changes saved">
          <CheckCircle className="fuk-autosave-icon" />
          <span>Saved</span>
        </span>
      )}

      {/* Version Up Button */}
      <button
        className="fuk-btn fuk-btn-version-up"
        onClick={onVersionUp}
        disabled={isLoading || !currentFileInfo}
        title="Save as new version"
      >
        <Plus className="fuk-icon--md" />
        Version Up
      </button>

      {/* Save Button */}
      <button
        className="fuk-btn fuk-btn-save"
        onClick={onSave}
        disabled={isLoading || !currentFileInfo || !hasUnsavedChanges}
        title="Save current version"
      >
        {isLoading ? (
          <Loader2 className="fuk-icon--md animate-spin" />
        ) : (
          <Save className="fuk-icon--md" />
        )}
        Save
      </button>
    </div>
  );
}