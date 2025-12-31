/**
 * FUK Generation Pipeline - Main App
 */

import { useState, useEffect, useCallback } from 'react';
import { Zap, Loader2, AlertCircle, Pipeline, Enhance, Save } from './components/Icons';
import ProjectBar from './components/ProjectBar';
import ImageTab from './tabs/ImageTab';
import VideoTab from './tabs/VideoTab';
import PreprocessTab from './tabs/PreprocessTab';
import PostprocessTab from './tabs/PostprocessTab';
import LayersTab from './tabs/LayersTab';
import ExportTab from './tabs/ExportTab';
import { fetchConfig } from './utils/api';
import { useProject } from './hooks/useProject';

export default function App() {
  const [activeTab, setActiveTab] = useState('image');
  const [config, setConfig] = useState(null);
  const [status, setStatus] = useState('loading');

  // Project management
  const project = useProject();

  // Load config on mount
  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const data = await fetchConfig();
      console.log('✓ Config loaded:', data);
      console.log('✓ Available LoRAs:', data.models.loras);
      setConfig(data);
      setStatus('ready');
    } catch (error) {
      console.error('Failed to load config:', error);
      setStatus('error');
    }
  };

  // Handle folder browser
  const handleBrowseFolder = useCallback(async () => {
    try {
      await project.openFolderBrowser();
    } catch (err) {
      console.error('Folder browser error:', err);
      // Don't alert on cancel
      if (!err.message?.includes('cancel')) {
        alert(`Failed to open folder: ${err.message}`);
      }
    }
  }, [project]);

  // Handle save
  const handleSave = useCallback(async () => {
    try {
      await project.save();
      console.log('✓ Project saved');
    } catch (err) {
      alert(`Failed to save: ${err.message}`);
    }
  }, [project]);

  // Handle version up
  const handleVersionUp = useCallback(async () => {
    try {
      const newFilename = await project.saveAsNewVersion();
      console.log('✓ New version created:', newFilename);
    } catch (err) {
      alert(`Failed to create new version: ${err.message}`);
    }
  }, [project]);

  // Handle new shot
  const handleNewShot = useCallback(async (shotNumber) => {
    try {
      const newFilename = await project.createNewShot(shotNumber);
      console.log('✓ New shot created:', newFilename);
    } catch (err) {
      alert(`Failed to create new shot: ${err.message}`);
    }
  }, [project]);

  // Handle new project
  const handleNewProject = useCallback(async (projectName) => {
    try {
      const newFilename = await project.createProject(projectName);
      console.log('✓ New project created:', newFilename);
    } catch (err) {
      alert(`Failed to create project: ${err.message}`);
    }
  }, [project]);

  // Loading state
  if (status === 'loading') {
    return (
      <div className="fuk-loading">
        <div className="fuk-loading-content">
          <Loader2 
            className="animate-spin" 
            style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', color: '#a855f7' }} 
          />
          <p style={{ fontSize: '0.875rem', color: '#9ca3af' }}>
            Loading FUK Generation Pipeline...
          </p>
        </div>
      </div>
    );
  }

  // Error state
  if (status === 'error') {
    return (
      <div className="fuk-error">
        <div className="fuk-error-content">
          <AlertCircle style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', color: '#ef4444' }} />
          <h2 className="fuk-error-title">Cannot Connect to Server</h2>
          <p className="fuk-error-message">
            Make sure the backend server is running on port 8000.
          </p>
          <pre className="fuk-error-code">python fuk_web_server.py</pre>
          <button 
            onClick={loadConfig}
            className="fuk-btn fuk-btn-secondary"
            style={{ marginTop: '1rem' }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // Render active tab with project state
  const renderTab = () => {
    const tabProps = {
      config,
      activeTab,
      setActiveTab,
      project, // Pass project for state access
    };

    switch (activeTab) {
      case 'image':
        return <ImageTab {...tabProps} />;
      case 'video':
        return <VideoTab {...tabProps} />;
      case 'preprocess':
         return <PreprocessTab {...tabProps} />;
      case 'postprocess':
        return <PostprocessTab {...tabProps} />;
      case 'layers':
        return <LayersTab {...tabProps} />;
      case 'export':
        return <ExportTab {...tabProps} />;
      default:
        return null;
    }
  };

  return (
    <div className="fuk-app">
      {/* Header */}
      <header className="fuk-header">
        <div className="fuk-header-content">
          <div className="fuk-header-inner">
            <div className="fuk-logo">
              <Zap />
            </div>
            <div>
              <h1 className="fuk-title">FUK Generation Pipeline</h1>
              <p className="fuk-subtitle">Qwen Image × Wan Video</p>
            </div>
          </div>
        </div>
      </header>

      {/* Project Bar */}
      <ProjectBar
        projectFolder={project.projectFolder}
        currentFileInfo={project.currentFileInfo}
        shots={project.shots}
        currentShotVersions={project.currentShotVersions}
        hasUnsavedChanges={project.hasUnsavedChanges}
        isLoading={project.isLoading}
        hasFiles={project.projectFiles.length > 0}
        onBrowseFolder={handleBrowseFolder}
        onSwitchShot={project.switchShot}
        onSwitchVersion={project.switchVersion}
        onSave={handleSave}
        onVersionUp={handleVersionUp}
        onNewShot={handleNewShot}
        onNewProject={handleNewProject}
      />

      {/* Main Content */}
      <main className="fuk-main">
        {renderTab()}
      </main>
    </div>
  );
}
