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
const Folder = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
        <div className="fuk-preview-container" style={{ 
          width: '100%', 
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          padding: '1rem',
        }}>
          {/* Main preview */}
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {mainPreview ? (
              <div style={{
                maxWidth: '100%',
                maxHeight: '100%',
                borderRadius: '0.5rem',
                border: '1px solid #374151',
                overflow: 'hidden',
                background: '#000',
              }}>
                <img
                  src={buildImageUrl(mainPreview)}
                  alt="Preview"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '60vh',
                    objectFit: 'contain',
                  }}
                />
              </div>
            ) : (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.5rem',
                color: '#6b7280',
              }}>
                <Download style={{ width: '3rem', height: '3rem', opacity: 0.3 }} />
                <p style={{ fontSize: '0.875rem' }}>Add layers to preview</p>
              </div>
            )}
          </div>
          
          {/* Layer thumbnails row */}
          {availableLayersCount > 0 && (
            <div style={{
              display: 'flex',
              gap: '0.5rem',
              justifyContent: 'center',
              paddingTop: '0.75rem',
              borderTop: '1px solid #374151',
              marginTop: '0.75rem',
            }}>
              {Object.entries(layers).map(([name, path]) => path && (
                <div key={name} style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '0.25rem',
                }}>
                  <div style={{
                    width: '60px',
                    height: '60px',
                    borderRadius: '0.25rem',
                    overflow: 'hidden',
                    border: name === 'beauty' ? '2px solid #a855f7' : '1px solid #374151',
                  }}>
                    <img
                      src={buildImageUrl(path)}
                      alt={name}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  </div>
                  <span style={{
                    fontSize: '0.6rem',
                    color: name === 'beauty' ? '#c084fc' : '#9ca3af',
                    textTransform: 'capitalize',
                  }}>
                    {name}
                  </span>
                </div>
              ))}
            </div>
          )}
          
          {/* Export result */}
          {exportResult && (
            <div style={{
              marginTop: '0.75rem',
              padding: '0.75rem',
              background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              borderRadius: '0.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#10b981' }}>
                <CheckCircle style={{ width: '1.25rem', height: '1.25rem' }} />
                <span style={{ fontSize: '0.875rem' }}>
                  Exported to: {exportResult.saved_path || exportResult.multi_layer?.url || 'cache'}
                </span>
              </div>
              {exportResult.multi_layer?.url && (
                <a
                  href={`/${exportResult.multi_layer.url}`}
                  download
                  style={{
                    padding: '0.375rem 0.75rem',
                    background: '#10b981',
                    color: 'white',
                    borderRadius: '0.25rem',
                    fontSize: '0.75rem',
                    textDecoration: 'none',
                  }}
                >
                  Download EXR
                </a>
              )}
            </div>
          )}
          
          {error && (
            <div style={{
              marginTop: '0.75rem',
              padding: '0.75rem',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '0.5rem',
              color: '#ef4444',
              fontSize: '0.875rem',
            }}>
              {error}
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
            
            <p style={{ fontSize: '0.65rem', color: '#6b7280', marginTop: '0.5rem' }}>
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
            
            <div style={{ marginTop: '0.75rem' }}>
              <label className="fuk-label">Export To</label>
              <div className="save-location-row">
                <input
                  type="text"
                  className="fuk-input"
                  value={settings.exportPath || '(project cache)'}
                  readOnly
                  style={{ 
                    color: settings.exportPath ? '#d1d5db' : '#6b7280',
                    fontStyle: settings.exportPath ? 'normal' : 'italic',
                  }}
                />
                <button
                  className="fuk-btn fuk-btn-secondary"
                  onClick={handleBrowseSaveLocation}
                  disabled={exporting}
                  title="Browse for save location"
                >
                  <Folder style={{ width: '1rem', height: '1rem' }} />
                </button>
              </div>
              <p style={{ fontSize: '0.65rem', color: '#6b7280', marginTop: '0.25rem' }}>
                Leave empty to save in project cache
              </p>
            </div>
          </div>

          {/* Export Options - Compact */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-2">Export Format</h3>
            
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <label className="fuk-checkbox-group" style={{ flex: 1, minWidth: '140px' }}>
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.multiLayerEXR}
                  onChange={(e) => updateExport('multiLayerEXR', e.target.checked)}
                  disabled={exporting}
                />
                <div>
                  <span className="fuk-label" style={{ marginBottom: 0, fontSize: '0.75rem' }}>Multi-Layer</span>
                  <span style={{ fontSize: '0.6rem', color: '#6b7280', display: 'block' }}>
                    All AOVs in one file
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group" style={{ flex: 1, minWidth: '140px' }}>
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.singleLayerEXRs}
                  onChange={(e) => updateExport('singleLayerEXRs', e.target.checked)}
                  disabled={exporting}
                />
                <div>
                  <span className="fuk-label" style={{ marginBottom: 0, fontSize: '0.75rem' }}>Separate Files</span>
                  <span style={{ fontSize: '0.6rem', color: '#6b7280', display: 'block' }}>
                    One file per AOV
                  </span>
                </div>
              </label>
            </div>
          </div>

          {/* EXR Settings - Compact Row Layout */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-2">EXR Settings</h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div className="compact-form-row">
                <label className="fuk-label" style={{ marginBottom: 0, minWidth: '70px' }}>Bit Depth</label>
                <select
                  className="fuk-select"
                  value={settings.exrBitDepth}
                  onChange={(e) => updateSettings({ exrBitDepth: parseInt(e.target.value) })}
                  disabled={exporting}
                  style={{ fontSize: '0.75rem', padding: '0.375rem' }}
                >
                  <option value="16">16-bit Half</option>
                  <option value="32">32-bit Float</option>
                </select>
              </div>
              
              <div className="compact-form-row">
                <label className="fuk-label" style={{ marginBottom: 0, minWidth: '70px' }}>Compress</label>
                <select
                  className="fuk-select"
                  value={settings.exrCompression}
                  onChange={(e) => updateSettings({ exrCompression: e.target.value })}
                  disabled={exporting}
                  style={{ fontSize: '0.75rem', padding: '0.375rem' }}
                >
                  <option value="ZIP">ZIP</option>
                  <option value="PIZ">PIZ</option>
                  <option value="PXR24">PXR24</option>
                  <option value="DWAA">DWAA</option>
                  <option value="NONE">None</option>
                </select>
              </div>
              
              <div className="compact-form-row">
                <label className="fuk-label" style={{ marginBottom: 0, minWidth: '70px' }}>Color</label>
                <select
                  className="fuk-select"
                  value={settings.exrColorSpace}
                  onChange={(e) => updateSettings({ exrColorSpace: e.target.value })}
                  disabled={exporting}
                  style={{ fontSize: '0.75rem', padding: '0.375rem' }}
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
            
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(2, 1fr)', 
              gap: '0.25rem',
              fontSize: '0.7rem',
            }}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '0.25rem 0.375rem',
                background: layers.beauty ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
                borderRadius: '0.25rem',
              }}>
                <span style={{ color: '#9ca3af' }}>R,G,B</span>
                <span style={{ color: layers.beauty ? '#10b981' : '#4b5563' }}>
                  {layers.beauty ? '✓' : '—'}
                </span>
              </div>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '0.25rem 0.375rem',
                background: layers.depth ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
                borderRadius: '0.25rem',
              }}>
                <span style={{ color: '#9ca3af' }}>Z</span>
                <span style={{ color: layers.depth ? '#10b981' : '#4b5563' }}>
                  {layers.depth ? '✓' : '—'}
                </span>
              </div>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '0.25rem 0.375rem',
                background: layers.normals ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
                borderRadius: '0.25rem',
              }}>
                <span style={{ color: '#9ca3af' }}>N.XYZ</span>
                <span style={{ color: layers.normals ? '#10b981' : '#4b5563' }}>
                  {layers.normals ? '✓' : '—'}
                </span>
              </div>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between',
                padding: '0.25rem 0.375rem',
                background: layers.crypto ? 'rgba(16, 185, 129, 0.1)' : 'transparent',
                borderRadius: '0.25rem',
              }}>
                <span style={{ color: '#9ca3af' }}>Crypto</span>
                <span style={{ color: layers.crypto ? '#10b981' : '#4b5563' }}>
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