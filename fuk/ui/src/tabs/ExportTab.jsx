/**
 * Export Tab
 * Final export options with visual layer preview
 * Shows Beauty pass large + smaller thumbnails for all other layers
 */

import { useState, useMemo, useCallback } from 'react';
import { Save, Loader2, CheckCircle, Download, Pipeline } from '../components/Icons';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
import Footer from '../components/Footer';

// Folder icon
const Folder = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
  </svg>
);

const DEFAULT_SETTINGS = {
  // What to export
  exports: {
    multiLayerEXR: true,
    singleLayerEXRs: false,
    previewRender: true,
    saveTensors: false,
    collectProject: false,
  },
  
  // EXR settings
  exrBitDepth: 32,
  exrCompression: 'ZIP',
  exrColorSpace: 'Linear',
  
  // Preview settings
  previewFormat: 'PNG',
  previewQuality: 95,
  previewColorSpace: 'sRGB',
  
  // Tensor settings
  tensorFormat: 'SafeTensors',
  includeLatents: true,
  includeVAE: false,
  
  // Collection settings
  includeInputs: true,
  includeIntermediates: false,
  includeConfig: true,
  packageFormat: 'zip',
};

export default function ExportTab({ config, activeTab, setActiveTab, project }) {
  // Settings (localStorage fallback)
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
  
  // UI state
  const [exporting, setExporting] = useState(false);
  const [exportPath, setExportPath] = useState('');
  const [exportComplete, setExportComplete] = useState(false);
  
  // Layer data - in reality, populated from project state or generation results
  // For now, empty to show placeholders
  const [availableLayers] = useState({
    // beauty: { url: '/path/to/beauty.png', name: 'Beauty Pass' },
    // depth: { url: '/path/to/depth.png', name: 'Depth (Z)' },
    // normals: { url: '/path/to/normals.png', name: 'Normals' },
    // cryptomatte: { url: '/path/to/crypto.png', name: 'Cryptomatte' },
  });
  
  const updateExport = (exportName, enabled) => {
    updateSettings({
      exports: {
        ...settings.exports,
        [exportName]: enabled,
      }
    });
  };
  
  const handleBrowsePath = async () => {
    // TODO: Implement native folder browser via backend
    // For now, just prompt
    const path = prompt('Enter export path:', exportPath || '/outputs/exports/');
    if (path) {
      setExportPath(path);
    }
  };
  
  const handleExport = async () => {
    if (!exportPath) {
      alert('Please select an export path');
      return;
    }
    
    setExporting(true);
    setExportComplete(false);
    
    // TODO: Implement backend API call for export
    // Simulate for now
    setTimeout(() => {
      setExporting(false);
      setExportComplete(true);
    }, 3000);
  };
  
  // Count enabled exports
  const enabledExportsCount = Object.values(settings.exports).filter(v => v).length;
  
  return (
    <>
      {/* Preview Area - Large Beauty + Small Layer Thumbnails */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-container" style={{ 
          width: '100%', 
          height: '100%',
          display: 'flex',
          gap: '1.5rem',
          padding: '1.5rem',
        }}>
          {/* Large Beauty Pass */}
          <div style={{ flex: 2, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: '0.5rem',
            }}>
              <h3 style={{ 
                fontSize: '0.875rem', 
                fontWeight: 600, 
                color: '#c084fc',
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}>
                Beauty Pass
              </h3>
              {exportComplete && (
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.25rem',
                  color: '#10b981',
                  fontSize: '0.75rem',
                }}>
                  <CheckCircle style={{ width: '1rem', height: '1rem' }} />
                  Exported
                </div>
              )}
            </div>
            
            {availableLayers.beauty ? (
              <div style={{ 
                flex: 1,
                borderRadius: '0.5rem',
                border: '1px solid #374151',
                overflow: 'hidden',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: '#000',
              }}>
                <img
                  src={buildImageUrl(availableLayers.beauty.url)}
                  alt="Beauty Pass"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                  }}
                />
              </div>
            ) : (
              <div className="fuk-card-dashed" style={{ flex: 1 }}>
                <div className="fuk-placeholder">
                  <Save style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', opacity: 0.3 }} />
                  <p style={{ color: '#6b7280' }}>No render available</p>
                </div>
              </div>
            )}
          </div>
          
          {/* Layer Thumbnails Grid */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <h3 style={{ 
              fontSize: '0.75rem', 
              fontWeight: 600, 
              color: '#9ca3af',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              paddingBottom: '0.5rem',
            }}>
              AOV Layers
            </h3>
            
            <div style={{ 
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '0.75rem',
              flex: 1,
              alignContent: 'start',
            }}>
              {Object.entries(availableLayers).filter(([name]) => name !== 'beauty').length > 0 ? (
                // Show actual layers
                Object.entries(availableLayers)
                  .filter(([name]) => name !== 'beauty')
                  .map(([layerName, layerData]) => (
                    <div 
                      key={layerName}
                      style={{
                        background: 'rgba(31, 41, 55, 0.5)',
                        borderRadius: '0.375rem',
                        border: '1px solid #374151',
                        overflow: 'hidden',
                      }}
                    >
                      <div style={{ aspectRatio: '16/9', overflow: 'hidden', background: '#000' }}>
                        <img
                          src={buildImageUrl(layerData.url)}
                          alt={layerData.name}
                          style={{
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                          }}
                        />
                      </div>
                      <div style={{ 
                        padding: '0.375rem 0.5rem',
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        color: '#9ca3af',
                        letterSpacing: '0.025em',
                        textAlign: 'center',
                      }}>
                        {layerData.name}
                      </div>
                    </div>
                  ))
              ) : (
                // Show placeholders for each AOV type
                ['Depth', 'Normals', 'Cryptomatte'].map((layerName) => (
                  <div 
                    key={layerName}
                    style={{
                      background: 'rgba(31, 41, 55, 0.3)',
                      borderRadius: '0.375rem',
                      border: '1px dashed #374151',
                      overflow: 'hidden',
                    }}
                  >
                    <div style={{ 
                      aspectRatio: '16/9', 
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: 'rgba(0, 0, 0, 0.2)',
                    }}>
                      <Pipeline style={{ width: '1.5rem', height: '1.5rem', opacity: 0.2 }} />
                    </div>
                    <div style={{ 
                      padding: '0.375rem 0.5rem',
                      fontSize: '0.65rem',
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      color: '#4b5563',
                      letterSpacing: '0.025em',
                      textAlign: 'center',
                    }}>
                      {layerName}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Export Options Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Export Options</h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.multiLayerEXR}
                  onChange={(e) => updateExport('multiLayerEXR', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Multi-Layer EXR</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    All AOVs in single file (Nuke/Fusion)
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.singleLayerEXRs}
                  onChange={(e) => updateExport('singleLayerEXRs', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Single-Layer EXRs</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Separate file per AOV
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.previewRender}
                  onChange={(e) => updateExport('previewRender', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Preview Render</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    PNG/JPG for review
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.saveTensors}
                  onChange={(e) => updateExport('saveTensors', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Save Tensors</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Raw latents for re-processing
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.exports.collectProject}
                  onChange={(e) => updateExport('collectProject', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Collect Project</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Package everything for archival
                  </span>
                </div>
              </label>
            </div>
            
            <div style={{ 
              marginTop: '1rem', 
              padding: '0.5rem 0.75rem', 
              background: 'rgba(168, 85, 247, 0.1)', 
              borderRadius: '0.375rem',
              fontSize: '0.7rem',
              color: '#c084fc',
            }}>
              {enabledExportsCount} export{enabledExportsCount !== 1 ? 's' : ''} selected
            </div>
          </div>

          {/* Export Path Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Export Destination</h3>
            
            <div className="fuk-form-group-compact">
              <label className="fuk-label">Output Path</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <input
                  type="text"
                  className="fuk-input"
                  value={exportPath}
                  onChange={(e) => setExportPath(e.target.value)}
                  placeholder="/outputs/exports/"
                  style={{ flex: 1 }}
                />
                <button
                  onClick={handleBrowsePath}
                  className="fuk-btn fuk-btn-secondary"
                  style={{ paddingLeft: '1rem', paddingRight: '1rem' }}
                >
                  <Folder style={{ width: '1rem', height: '1rem' }} />
                </button>
              </div>
            </div>
            
            <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
              Files will be organized by export type
            </p>
          </div>

          {/* EXR Settings */}
          {(settings.exports.multiLayerEXR || settings.exports.singleLayerEXRs) && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">EXR Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Bit Depth</label>
                <select
                  className="fuk-select"
                  value={settings.exrBitDepth}
                  onChange={(e) => updateSettings({ exrBitDepth: parseInt(e.target.value) })}
                >
                  <option value="16">16-bit Half Float</option>
                  <option value="32">32-bit Float (Recommended)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Compression</label>
                <select
                  className="fuk-select"
                  value={settings.exrCompression}
                  onChange={(e) => updateSettings({ exrCompression: e.target.value })}
                >
                  <option value="ZIP">ZIP (Lossless, Recommended)</option>
                  <option value="PIZ">PIZ (Lossless, Wavelet)</option>
                  <option value="PXR24">PXR24 (Lossy)</option>
                  <option value="B44">B44 (Lossy, Fast)</option>
                  <option value="DWAA">DWAA (Lossy, Small)</option>
                  <option value="None">None (Uncompressed)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Color Space</label>
                <select
                  className="fuk-select"
                  value={settings.exrColorSpace}
                  onChange={(e) => updateSettings({ exrColorSpace: e.target.value })}
                >
                  <option value="Linear">Linear (Recommended)</option>
                  <option value="sRGB">sRGB</option>
                  <option value="ACES">ACES</option>
                </select>
              </div>
            </div>
          )}

          {/* Preview Settings */}
          {settings.exports.previewRender && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Preview Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Format</label>
                <select
                  className="fuk-select"
                  value={settings.previewFormat}
                  onChange={(e) => updateSettings({ previewFormat: e.target.value })}
                >
                  <option value="PNG">PNG (Lossless)</option>
                  <option value="JPG">JPG (Compressed)</option>
                  <option value="MP4">MP4 (Video)</option>
                  <option value="MOV">MOV (ProRes)</option>
                </select>
              </div>
              
              {settings.previewFormat === 'JPG' && (
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">Quality</label>
                  <div className="fuk-input-inline">
                    <input
                      type="range"
                      className="fuk-input"
                      style={{ flex: 2 }}
                      value={settings.previewQuality}
                      onChange={(e) => updateSettings({ previewQuality: parseInt(e.target.value) })}
                      min={50}
                      max={100}
                    />
                    <input
                      type="number"
                      className="fuk-input"
                      style={{ width: '80px' }}
                      value={settings.previewQuality}
                      onChange={(e) => updateSettings({ previewQuality: parseInt(e.target.value) })}
                      min={50}
                      max={100}
                    />
                  </div>
                </div>
              )}
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Color Space</label>
                <select
                  className="fuk-select"
                  value={settings.previewColorSpace}
                  onChange={(e) => updateSettings({ previewColorSpace: e.target.value })}
                >
                  <option value="sRGB">sRGB (Recommended)</option>
                  <option value="Linear">Linear</option>
                  <option value="ACES">ACES</option>
                </select>
              </div>
            </div>
          )}

          {/* Tensor Settings */}
          {settings.exports.saveTensors && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Tensor Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Format</label>
                <select
                  className="fuk-select"
                  value={settings.tensorFormat}
                  onChange={(e) => updateSettings({ tensorFormat: e.target.value })}
                >
                  <option value="SafeTensors">SafeTensors (Recommended)</option>
                  <option value="PyTorch">PyTorch (.pt)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.includeLatents}
                    onChange={(e) => updateSettings({ includeLatents: e.target.checked })}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Include Latent Tensors</span>
                </label>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.includeVAE}
                    onChange={(e) => updateSettings({ includeVAE: e.target.checked })}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Include VAE Outputs</span>
                </label>
              </div>
              
              <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
                Raw tensors can be re-decoded with different VAE settings
              </p>
            </div>
          )}

          {/* Collection Settings */}
          {settings.exports.collectProject && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Project Collection</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.includeInputs}
                    onChange={(e) => updateSettings({ includeInputs: e.target.checked })}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Include Source Files</span>
                </label>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.includeIntermediates}
                    onChange={(e) => updateSettings({ includeIntermediates: e.target.checked })}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Include Intermediate Files</span>
                </label>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.includeConfig}
                    onChange={(e) => updateSettings({ includeConfig: e.target.checked })}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Include Project Config</span>
                </label>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Package Format</label>
                <select
                  className="fuk-select"
                  value={settings.packageFormat}
                  onChange={(e) => updateSettings({ packageFormat: e.target.value })}
                >
                  <option value="zip">ZIP Archive</option>
                  <option value="folder">Folder Structure</option>
                </select>
              </div>
              
              <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
                Creates complete project package for archival or transfer
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={exporting}
        progress={exporting ? { progress: 0.5, phase: 'exporting' } : null}
        elapsedSeconds={0}
        onGenerate={handleExport}
        onCancel={() => setExporting(false)}
        canGenerate={!!exportPath && enabledExportsCount > 0}
        generateLabel="Export"
        generatingLabel="Exporting..."
      />
    </>
  );
}