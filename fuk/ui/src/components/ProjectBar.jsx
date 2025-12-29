/**
 * ProjectBar Component
 * Horizontal project navigation: Project > Shot > Version
 */

import { useState } from 'react';
import { Save, Loader2, AlertCircle, CheckCircle } from './Icons';

// Folder icon
const Folder = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
  </svg>
);

// Plus icon
const Plus = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
  </svg>
);

// File plus icon
const FilePlus = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

// Chevron down icon
const ChevronDown = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
  </svg>
);

export default function ProjectBar({
  projectFolder,
  currentFileInfo,
  shots,
  currentShotVersions,
  hasUnsavedChanges,
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
            <Loader2 className="animate-spin" style={{ width: '1rem', height: '1rem' }} />
          ) : (
            <Folder style={{ width: '1rem', height: '1rem' }} />
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
          <Folder style={{ width: '1rem', height: '1rem' }} />
          <span className="fuk-project-folder-name">{getFolderDisplayName()}</span>
          <ChevronDown style={{ width: '0.75rem', height: '0.75rem', opacity: 0.5 }} />
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
            <FilePlus style={{ width: '1rem', height: '1rem' }} />
            New Project File
          </button>
        )}

        <span className="fuk-project-hint" style={{ marginLeft: '1rem' }}>
          No project files found. Create one to get started.
        </span>
      </div>
    );
  }

  return (
    <div className="fuk-project-bar">
      {/* Project Folder Button */}
      <button 
        className="fuk-project-folder-btn"
        onClick={onBrowseFolder}
        title={projectFolder}
        disabled={isLoading}
      >
        {isLoading ? (
          <Loader2 className="animate-spin" style={{ width: '1rem', height: '1rem' }} />
        ) : (
          <Folder style={{ width: '1rem', height: '1rem' }} />
        )}
        <span className="fuk-project-folder-name">{getFolderDisplayName()}</span>
        <ChevronDown style={{ width: '0.75rem', height: '0.75rem', opacity: 0.5 }} />
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
              <Plus style={{ width: '0.875rem', height: '0.875rem' }} />
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
      {hasUnsavedChanges ? (
        <span className="fuk-autosave-indicator fuk-autosave-pending" title="Autosaving...">
          <Loader2 className="animate-spin" style={{ width: '0.875rem', height: '0.875rem' }} />
          <span>Saving...</span>
        </span>
      ) : currentFileInfo && (
        <span className="fuk-autosave-indicator fuk-autosave-saved" title="All changes saved">
          <CheckCircle style={{ width: '0.875rem', height: '0.875rem' }} />
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
        <Plus style={{ width: '1rem', height: '1rem' }} />
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
          <Loader2 className="animate-spin" style={{ width: '1rem', height: '1rem' }} />
        ) : (
          <Save style={{ width: '1rem', height: '1rem' }} />
        )}
        Save
      </button>
    </div>
  );
}
