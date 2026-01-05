/**
 * Export Tab
 * Export AOV layers to multi-layer EXR for compositing
 */

import { useState, useMemo, useCallback } from 'react';
import { Save, Loader2, CheckCircle, Download, AlertCircle } from '../components/Icons';
import InlineImageInput from '../components/InlineImageInput';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl, API_URL } from '../utils/constants';
import Footer from '../components/Footer';

// Folder icon
const Folder = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
  </svg>
);

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
  
  // UI state
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  
  const updateLayer = (layerName, path) => {
    setLayers(prev => ({
      ...prev,
      [layerName]: path,
    }));
    setExportResult(null);
    setError(null);
  };
  
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
          title: 'Save EXR Export',
          defaultName: `${settings.exportFilename}.exr`,
          fileTypes: [['EXR Files', '*.exr'], ['All Files', '*.*']],
        }),
      });
      
      const data = await response.json();
      
      if (data.path && !data.cancelled) {
        // Extract directory and filename
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
    } catch (err) {
      console.error('Browse failed:', err);
    }
  };
  
  const handleExport = async () => {
    if (!layers.beauty) {
      setError('Please add at least a Beauty pass to export');
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
      const response = await fetch(`${API_URL}/export/exr`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
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
        }),
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
  const canExport = layers.beauty && (settings.exports.multiLayerEXR || settings.exports.singleLayerEXRs);
  
  // Get the largest layer for main preview
  const mainPreview = layers.beauty || layers.depth || layers.normals || layers.crypto;
  
  return (
    <>
      {/* Preview Area */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-centered">
          {/* Main preview */}
          <div className="fuk-preview-main">
            {mainPreview ? (
              <div className="fuk-media-frame">
                <img
                  src={buildImageUrl(mainPreview)}
                  alt="Preview"
                  className="fuk-preview-media fuk-preview-media--constrained"
                />
              </div>
            ) : (
              <div className="fuk-placeholder-card fuk-placeholder-card--60 fuk-placeholder-card--16x9">
                <div className="fuk-placeholder">
                  <Download className="fuk-placeholder-icon fuk-placeholder-icon--faded" />
                  <p className="fuk-placeholder-text">Add layers to preview</p>
                </div>
              </div>
            )}
          </div>
          
          {/* Layer thumbnails row */}
          {availableLayersCount > 0 && (
            <div className="fuk-thumb-strip">
              {Object.entries(layers).map(([name, path]) => path && (
                <div key={name} className="fuk-thumb-item">
                  <div className={`fuk-thumb-frame ${name === 'beauty' ? 'fuk-thumb-frame--primary' : ''}`}>
                    <img
                      src={buildImageUrl(path)}
                      alt={name}
                    />
                  </div>
                  <span className={`fuk-thumb-label ${name === 'beauty' ? 'fuk-thumb-label--primary' : ''}`}>
                    {name}
                  </span>
                </div>
              ))}
            </div>
          )}
          
          {/* Export result */}
          {exportResult && (
            <div className="fuk-result-banner fuk-result-banner--success">
              <div className="fuk-result-content">
                <CheckCircle className="fuk-icon fuk-icon--lg" />
                <span className="fuk-result-text">
                  Exported to: {exportResult.saved_path || exportResult.multi_layer?.url || 'cache'}
                </span>
              </div>
              {exportResult.multi_layer?.url && (
                <a
                  href={`/${exportResult.multi_layer.url}`}
                  download
                  className="fuk-download-btn"
                >
                  Download EXR
                </a>
              )}
            </div>
          )}
          
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

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Layer Inputs - Compact Grid */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Layer Inputs</h3>
            
            <div className="layer-inputs-grid">
              <InlineImageInput
                label="Beauty"
                required
                value={layers.beauty}
                onChange={(path) => updateLayer('beauty', path)}
                disabled={exporting}
                placeholder="Beauty pass"
              />
              
              <InlineImageInput
                label="Depth"
                value={layers.depth}
                onChange={(path) => updateLayer('depth', path)}
                disabled={exporting}
                placeholder="Depth map"
              />
              
              <InlineImageInput
                label="Normals"
                value={layers.normals}
                onChange={(path) => updateLayer('normals', path)}
                disabled={exporting}
                placeholder="Normal map"
              />
              
              <InlineImageInput
                label="Crypto"
                value={layers.crypto}
                onChange={(path) => updateLayer('crypto', path)}
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
              <label className="fuk-label">Filename</label>
              <div className="export-filename-row">
                <input
                  type="text"
                  className="fuk-input"
                  value={settings.exportFilename}
                  onChange={(e) => updateSettings({ exportFilename: e.target.value })}
                  placeholder="export"
                  disabled={exporting}
                />
                <span className="extension">.exr</span>
              </div>
            </div>
            
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
                  <span className="fuk-export-format-desc">All AOVs in one file</span>
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
                  <span className="fuk-export-format-desc">One file per AOV</span>
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
        generateLabel={`Export EXR${settings.exportFilename ? ` (${settings.exportFilename})` : ''}`}
        generatingLabel="Exporting..."
      />
    </>
  );
}