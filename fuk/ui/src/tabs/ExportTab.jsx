/**
 * Export Tab
 * Export AOV layers to multi-layer EXR for compositing
 * Supports both single image and video sequence export
 */

import { useState, useMemo, useCallback } from 'react';
import { CheckCircle, Download, AlertCircle, Film , Folder} from '../components/Icons';
import InlineImageInput from '../components/InlineImageInput';
import ZoomableImage from '../components/ZoomableImage';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl, API_URL } from '../utils/constants';
import Footer from '../components/Footer';


const DEFAULT_SETTINGS = {
  exports: {
    multiLayerEXR: true,
    singleLayerEXRs: false,
  },
  exrBitDepth: 32,
  exrCompression: 'ZIP',
  exrColorSpace: 'Linear',
  exportFilename: 'export',
  exportPath: '',
  // Video sequence settings
  sequenceStartFrame: 1,
  sequencePattern: '{name}.{frame:04d}',
};

// Helper to check if a path is a video file
const isVideoPath = (path) => {
  if (!path) return false;
  const ext = path.split('.').pop()?.toLowerCase();
  return ['mp4', 'mov', 'avi', 'webm', 'mkv'].includes(ext);
};

export default function ExportTab({ config, activeTab, setActiveTab, project }) {
  // Settings
  const [localSettings, setLocalSettings] = useLocalStorage('fuk_export_settings', DEFAULT_SETTINGS);
  
  const settings = useMemo(() => {
    if (project?.projectState?.tabs?.export) {
      return { ...DEFAULT_SETTINGS, ...project.projectState.tabs.export };
    }
    return { ...DEFAULT_SETTINGS, ...localSettings };
  }, [project?.projectState?.tabs?.export, localSettings]);
  
  const updateSettings = useCallback((updates) => {
    const newSettings = { ...settings, ...updates };
    
    if (project?.isProjectLoaded && project.updateTabState) {
      project.updateTabState('export', newSettings);
    } else {
      setLocalSettings(newSettings);
    }
  }, [project, settings, setLocalSettings]);
  
  // Layer inputs
  const [layers, setLayers] = useState({
    beauty: null,
    depth: null,
    normals: null,
    crypto: null,
  });
  
  // Video mode - auto-detected from dropped layers
  const [isVideoMode, setIsVideoMode] = useState(false);
  
  // UI state
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  // Detect if any layer is a video
  const hasVideoLayers = useMemo(() => {
    return Object.values(layers).some(path => isVideoPath(path));
  }, [layers]);
  
  const updateLayer = (layerName, path) => {
    setLayers(prev => ({
      ...prev,
      [layerName]: path,
    }));
    setExportResult(null);
    setError(null);
    
    // Auto-detect video mode when a video is added
    if (isVideoPath(path)) {
      setIsVideoMode(true);
    }
  };
  
  // Handle layers pack drop - populate all layer inputs at once
  const handleLayersPackDrop = useCallback((genData) => {
    console.log('[ExportTab] Layers pack detected:', genData.metadata);
    
    // Check if this is a video layers pack
    const isVideo = genData.metadata?.is_video || false;
    
    // Populate all available layers
    const newLayers = { ...layers };
    const availableLayers = genData.metadata.available_layers || {};
    
    // Get layers from available_layers (includes beauty if present)
    if (availableLayers.beauty) {
      newLayers.beauty = availableLayers.beauty;
    }
    if (availableLayers.depth) {
      newLayers.depth = availableLayers.depth;
    }
    if (availableLayers.normals) {
      newLayers.normals = availableLayers.normals;
    }
    if (availableLayers.crypto) {
      newLayers.crypto = availableLayers.crypto;
    }
    
    // Fallback: if no beauty in available_layers, try preview path
    if (!newLayers.beauty && genData.preview) {
      if (genData.preview.includes('source.mp4') || genData.preview.includes('source.png')) {
        newLayers.beauty = genData.preview;
      }
    }
    
    setLayers(newLayers);
    setIsVideoMode(isVideo);
    setExportResult(null);
    setError(null);
    
    console.log('[ExportTab] Populated layers:', newLayers, 'video mode:', isVideo);
  }, [layers]);
  
  const updateExport = (exportName, enabled) => {
    updateSettings({
      exports: {
        ...settings.exports,
        [exportName]: enabled,
      }
    });
  };
  
  // Browse for save location
  const handleBrowseSaveLocation = async () => {
    try {
      const response = await fetch(`${API_URL}/project/browse-save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: isVideoMode ? 'Select Output Directory' : 'Save EXR Export',
          defaultName: isVideoMode ? '' : `${settings.exportFilename}.exr`,
          fileTypes: isVideoMode 
            ? [['All Files', '*.*']]
            : [['EXR Files', '*.exr'], ['All Files', '*.*']],
          directory: isVideoMode,
        }),
      });
      
      const data = await response.json();
      
      if (data.path && !data.cancelled) {
        if (isVideoMode) {
          // For video, the path is a directory
          updateSettings({ exportPath: data.path });
        } else {
          // For single image, extract directory and filename
          const lastSlash = Math.max(data.path.lastIndexOf('/'), data.path.lastIndexOf('\\'));
          const dir = data.path.substring(0, lastSlash);
          let filename = data.path.substring(lastSlash + 1);
          
          // Remove .exr extension if present
          if (filename.toLowerCase().endsWith('.exr')) {
            filename = filename.slice(0, -4);
          }
          
          updateSettings({
            exportPath: dir,
            exportFilename: filename,
          });
        }
      }
    } catch (err) {
      console.error('Browse failed:', err);
    }
  };
  
  const handleExport = async () => {
    if (!layers.beauty && !layers.depth && !layers.normals && !layers.crypto) {
      setError('Please add at least one layer to export');
      return;
    }
    
    if (!settings.exports.multiLayerEXR && !settings.exports.singleLayerEXRs) {
      setError('Please select at least one export type');
      return;
    }
    
    setExporting(true);
    setExportResult(null);
    setError(null);
    setElapsedTime(0);
    
    const startTime = Date.now();
    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    
    try {
      // Choose endpoint based on video mode
      const endpoint = isVideoMode ? `${API_URL}/export/exr/sequence` : `${API_URL}/export/exr`;
      
      const payload = {
        layers: {
          beauty: layers.beauty,
          depth: layers.depth,
          normals: layers.normals,
          crypto: layers.crypto,
        },
        bit_depth: settings.exrBitDepth,
        compression: settings.exrCompression,
        color_space: settings.exrColorSpace,
        multi_layer: settings.exports.multiLayerEXR,
        single_files: settings.exports.singleLayerEXRs,
        filename: settings.exportFilename,
        export_path: settings.exportPath,
      };
      
      // Add video-specific options
      if (isVideoMode) {
        payload.start_frame = settings.sequenceStartFrame;
        payload.filename_pattern = `${settings.exportFilename}.{frame:04d}.exr`;
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }
      
      const result = await response.json();
      setExportResult(result);
      
    } catch (err) {
      console.error('Export failed:', err);
      setError(err.message);
    } finally {
      clearInterval(timer);
      setExporting(false);
    }
  };
  
  const availableLayersCount = Object.values(layers).filter(v => v !== null).length;
  const canExport = availableLayersCount > 0 && (settings.exports.multiLayerEXR || settings.exports.singleLayerEXRs);
  
  // Get the largest layer for main preview
  const mainPreview = layers.beauty || layers.depth || layers.normals || layers.crypto;
  const mainPreviewIsVideo = isVideoPath(mainPreview);
  
  return (
    <>
      {/* Preview Area */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-centered">
          {/* Main preview */}
          <div className="fuk-preview-main">
            {mainPreview ? (
              <div className="fuk-media-frame">
                {mainPreviewIsVideo ? (
                  <video
                    src={buildImageUrl(mainPreview)}
                    controls
                    muted
                    loop
                    className="fuk-media-preview"
                  />
                ) : (
                  <ZoomableImage
                    src={buildImageUrl(mainPreview)}
                    alt="Preview"
                    className="fuk-preview-media fuk-preview-media--constrained"
                  />
                )}
              </div>
            ) : (
              <div className="fuk-placeholder-card fuk-placeholder-card--60 fuk-placeholder-card--16x9">
                <Download className="fuk-placeholder-icon fuk-placeholder-icon--faded" />
                <p>Drag layers from History to export</p>
              </div>
            )}
          </div>
          
          {/* Video Mode Indicator */}
          {isVideoMode && (
            <div className="fuk-video-mode-badge">
              <Film className="fuk-icon fuk-icon--sm" />
              <span>Video Sequence Export</span>
            </div>
          )}
          
          {/* Export Result */}
          {exportResult && (
            <div className="fuk-result-banner fuk-result-banner--success">
              <div className="fuk-result-content">
                <CheckCircle className="fuk-icon fuk-icon--lg" />
                <span className="fuk-result-text">
                  {isVideoMode 
                    ? `Exported ${exportResult.frame_count} frames`
                    : 'Export Complete'}
                </span>
              </div>
              {exportResult.output_path && (
                <p className="fuk-text-sm fuk-text-muted fuk-mt-1">
                  {exportResult.output_path}
                </p>
              )}
              {exportResult.output_dir && (
                <p className="fuk-text-sm fuk-text-muted fuk-mt-1">
                  {exportResult.output_dir}
                </p>
              )}
              {exportResult.total_size_mb && (
                <p className="fuk-text-sm fuk-text-muted">
                  Total size: {exportResult.total_size_mb.toFixed(2)} MB
                </p>
              )}
            </div>
          )}
          
          {/* Error */}
          {error && (
            <div className="fuk-result-banner fuk-result-banner--error">
              <div className="fuk-result-content fuk-result-content--error">
                <AlertCircle className="fuk-icon fuk-icon--lg" />
                <span className="fuk-result-text">{error}</span>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Sidebar */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Layer Inputs */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">
              AOV Layers
              {isVideoMode && <span className="fuk-badge fuk-badge--info fuk-ml-2">Video</span>}
            </h3>
            
            <div className="layer-inputs-grid">
              <InlineImageInput
                label="Beauty"
                required
                value={layers.beauty}
                onChange={(path) => updateLayer('beauty', path)}
                onLayersPackDrop={handleLayersPackDrop}
                disabled={exporting}
                placeholder="Beauty pass"
              />
              
              <InlineImageInput
                label="Depth"
                value={layers.depth}
                onChange={(path) => updateLayer('depth', path)}
                onLayersPackDrop={handleLayersPackDrop}
                disabled={exporting}
                placeholder="Depth map"
              />
              
              <InlineImageInput
                label="Normals"
                value={layers.normals}
                onChange={(path) => updateLayer('normals', path)}
                onLayersPackDrop={handleLayersPackDrop}
                disabled={exporting}
                placeholder="Normal map"
              />
              
              <InlineImageInput
                label="Crypto"
                value={layers.crypto}
                onChange={(path) => updateLayer('crypto', path)}
                onLayersPackDrop={handleLayersPackDrop}
                disabled={exporting}
                placeholder="Cryptomatte"
              />
            </div>
            
            <p className="fuk-help-text fuk-help-text--sm">
              Drag from History or click to upload
            </p>
          </div>

          {/* Save Location */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Save Location</h3>
            
            <div className="export-filename-group">
              <label className="fuk-label">
                {isVideoMode ? 'Sequence Name' : 'Filename'}
              </label>
              <div className="export-filename-row">
                <input
                  type="text"
                  className="fuk-input"
                  value={settings.exportFilename}
                  onChange={(e) => updateSettings({ exportFilename: e.target.value })}
                  placeholder="export"
                  disabled={exporting}
                />
                <span className="extension">
                  {isVideoMode ? '.####.exr' : '.exr'}
                </span>
              </div>
              {isVideoMode && (
                <p className="fuk-help-text fuk-help-text--sm fuk-help-text--inline">
                  Output: {settings.exportFilename}.0001.exr, {settings.exportFilename}.0002.exr, ...
                </p>
              )}
            </div>
            
            {/* Video-specific: Start Frame */}
            {isVideoMode && (
              <div className="fuk-form-section">
                <label className="fuk-label">Start Frame</label>
                <input
                  type="number"
                  className="fuk-input fuk-input--compact"
                  value={settings.sequenceStartFrame}
                  onChange={(e) => updateSettings({ sequenceStartFrame: parseInt(e.target.value) || 1 })}
                  min={0}
                  disabled={exporting}
                />
              </div>
            )}
            
            <div className="fuk-form-section">
              <label className="fuk-label">Export To</label>
              <div className="save-location-row">
                <input
                  type="text"
                  className={`fuk-input ${settings.exportPath ? 'fuk-input--readonly-filled' : 'fuk-input--readonly'}`}
                  value={settings.exportPath || '(project cache)'}
                  readOnly
                />
                <button
                  className="fuk-btn fuk-btn-secondary"
                  onClick={handleBrowseSaveLocation}
                  disabled={exporting}
                  title="Browse for save location"
                >
                  <Folder className="fuk-icon fuk-icon--md" />
                </button>
              </div>
              <p className="fuk-help-text fuk-help-text--sm fuk-help-text--inline">
                Leave empty to save in project cache
              </p>
            </div>
          </div>

          {/* Export Options - Compact */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-2">Export Format</h3>
            
            <div className="fuk-export-format-group">
              <label className="fuk-checkbox-group fuk-export-format-item">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.multiLayerEXR}
                  onChange={(e) => updateExport('multiLayerEXR', e.target.checked)}
                  disabled={exporting}
                />
                <div>
                  <span className="fuk-export-format-label">Multi-Layer</span>
                  <span className="fuk-export-format-desc">
                    {isVideoMode ? 'All AOVs per frame' : 'All AOVs in one file'}
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group fuk-export-format-item">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.singleLayerEXRs}
                  onChange={(e) => updateExport('singleLayerEXRs', e.target.checked)}
                  disabled={exporting}
                />
                <div>
                  <span className="fuk-export-format-label">Separate Files</span>
                  <span className="fuk-export-format-desc">
                    {isVideoMode ? 'One sequence per AOV' : 'One file per AOV'}
                  </span>
                </div>
              </label>
            </div>
          </div>

          {/* EXR Settings - Compact Row Layout */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-2">EXR Settings</h3>
            
            <div className="fuk-form-stack">
              <div className="compact-form-row">
                <label className="fuk-label">Bit Depth</label>
                <select
                  className="fuk-select fuk-select--compact"
                  value={settings.exrBitDepth}
                  onChange={(e) => updateSettings({ exrBitDepth: parseInt(e.target.value) })}
                  disabled={exporting}
                >
                  <option value="16">16-bit Half</option>
                  <option value="32">32-bit Float</option>
                </select>
              </div>
              
              <div className="compact-form-row">
                <label className="fuk-label">Compress</label>
                <select
                  className="fuk-select fuk-select--compact"
                  value={settings.exrCompression}
                  onChange={(e) => updateSettings({ exrCompression: e.target.value })}
                  disabled={exporting}
                >
                  <option value="ZIP">ZIP</option>
                  <option value="PIZ">PIZ</option>
                  <option value="PXR24">PXR24</option>
                  <option value="DWAA">DWAA</option>
                  <option value="NONE">None</option>
                </select>
              </div>
              
              <div className="compact-form-row">
                <label className="fuk-label">Color</label>
                <select
                  className="fuk-select fuk-select--compact"
                  value={settings.exrColorSpace}
                  onChange={(e) => updateSettings({ exrColorSpace: e.target.value })}
                  disabled={exporting}
                >
                  <option value="Linear">Linear</option>
                  <option value="sRGB">sRGB</option>
                </select>
              </div>
            </div>
          </div>

          {/* Channel Summary - Compact */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-2">Channels</h3>
            
            <div className="fuk-channel-grid">
              <div className={`fuk-channel-item ${layers.beauty ? 'fuk-channel-item--active' : ''}`}>
                <span className="fuk-channel-name">R,G,B</span>
                <span className={`fuk-channel-status ${layers.beauty ? 'fuk-channel-status--active' : ''}`}>
                  {layers.beauty ? '✓' : '—'}
                </span>
              </div>
              <div className={`fuk-channel-item ${layers.depth ? 'fuk-channel-item--active' : ''}`}>
                <span className="fuk-channel-name">Z</span>
                <span className={`fuk-channel-status ${layers.depth ? 'fuk-channel-status--active' : ''}`}>
                  {layers.depth ? '✓' : '—'}
                </span>
              </div>
              <div className={`fuk-channel-item ${layers.normals ? 'fuk-channel-item--active' : ''}`}>
                <span className="fuk-channel-name">N.XYZ</span>
                <span className={`fuk-channel-status ${layers.normals ? 'fuk-channel-status--active' : ''}`}>
                  {layers.normals ? '✓' : '—'}
                </span>
              </div>
              <div className={`fuk-channel-item ${layers.crypto ? 'fuk-channel-item--active' : ''}`}>
                <span className="fuk-channel-name">Crypto</span>
                <span className={`fuk-channel-status ${layers.crypto ? 'fuk-channel-status--active' : ''}`}>
                  {layers.crypto ? '✓' : '—'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={exporting}
        progress={exporting ? { progress: 0.5, phase: 'exporting' } : null}
        elapsedSeconds={elapsedTime}
        onGenerate={handleExport}
        onCancel={() => setExporting(false)}
        canGenerate={canExport}
        generateLabel={isVideoMode 
          ? `Export Sequence${settings.exportFilename ? ` (${settings.exportFilename})` : ''}`
          : `Export EXR${settings.exportFilename ? ` (${settings.exportFilename})` : ''}`
        }
        generatingLabel={isVideoMode ? "Exporting Sequence..." : "Exporting..."}
      />
    </>
  );
}