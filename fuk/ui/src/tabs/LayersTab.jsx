/**
 * Layers Tab
 * Generate and manage AOV layers (Arbitrary Output Variables)
 * Layers: Beauty, Depth, Normals, Cryptomatte
 */

import { useState, useMemo, useCallback } from 'react';
import { Loader2, CheckCircle, Upload } from '../components/Icons';
import ImageUploader from '../components/ImageUploader';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
import Footer from '../components/Footer';

// Layers icon
const Layers = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
  </svg>
);

const DEFAULT_SETTINGS = {
  // Active layers to generate
  layers: {
    beauty: true,
    depth: true,
    normals: false,
    cryptomatte: false,
  },
  
  // Depth settings
  depthModel: 'depth_anything_v2',
  depthInvert: false,
  depthNormalize: true,
  
  // Normals settings
  normalsSpace: 'tangent', // 'tangent', 'object', 'world'
  normalsFlipY: false,
  
  // Cryptomatte settings
  cryptoLayers: 3,
  cryptoAccuracy: 'high',
};

export default function LayersTab({ config, activeTab, setActiveTab, project }) {
  // Settings (localStorage fallback)
  const [localSettings, setLocalSettings] = useLocalStorage('fuk_layers_settings', DEFAULT_SETTINGS);
  
  const settings = useMemo(() => {
    if (project?.projectState?.tabs?.layers) {
      return { ...DEFAULT_SETTINGS, ...project.projectState.tabs.layers };
    }
    return { ...DEFAULT_SETTINGS, ...localSettings };
  }, [project?.projectState?.tabs?.layers, localSettings]);
  
  const updateSettings = useCallback((updates) => {
    const newSettings = { ...settings, ...updates };
    
    if (project?.isProjectLoaded && project.updateTabState) {
      project.updateTabState('layers', newSettings);
    } else {
      setLocalSettings(newSettings);
    }
  }, [project, settings, setLocalSettings]);
  
  // UI state
  const [sourceInput, setSourceInput] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [generatedLayers, setGeneratedLayers] = useState({});
  
  const handleSourceChange = (paths) => {
    setSourceInput(paths[0] || null);
    setGeneratedLayers({});
  };
  
  const updateLayer = (layerName, enabled) => {
    updateSettings({
      layers: {
        ...settings.layers,
        [layerName]: enabled,
      }
    });
  };
  
  const handleGenerate = async () => {
    if (!sourceInput) {
      alert('Please select a source image or video');
      return;
    }
    
    setProcessing(true);
    // TODO: Implement backend API call
    // For now, simulate with timeout
    setTimeout(() => {
      setProcessing(false);
      // Placeholder results
      const layers = {};
      Object.entries(settings.layers).forEach(([layer, enabled]) => {
        if (enabled) {
          layers[layer] = { url: sourceInput }; // In reality, different URLs per layer
        }
      });
      setGeneratedLayers(layers);
    }, 3000);
  };
  
  // Count enabled layers
  const enabledLayersCount = Object.values(settings.layers).filter(v => v).length;
  
  return (
    <>
      {/* Preview Area - Grid Layout for Layers */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-container" style={{ width: '100%', height: '100%', padding: '1rem' }}>
          {Object.keys(generatedLayers).length > 0 ? (
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '1rem',
              width: '100%',
              height: '100%',
              alignContent: 'start',
            }}>
              {Object.entries(generatedLayers).map(([layerName, layerData]) => (
                <div 
                  key={layerName}
                  style={{
                    background: 'rgba(31, 41, 55, 0.5)',
                    borderRadius: '0.5rem',
                    border: '1px solid #374151',
                    overflow: 'hidden',
                  }}
                >
                  <div style={{ 
                    padding: '0.5rem 0.75rem', 
                    borderBottom: '1px solid #374151',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}>
                    <span style={{ 
                      fontSize: '0.75rem', 
                      fontWeight: 600, 
                      textTransform: 'uppercase',
                      color: '#c084fc',
                      letterSpacing: '0.05em',
                    }}>
                      {layerName}
                    </span>
                    <CheckCircle style={{ width: '1rem', height: '1rem', color: '#10b981' }} />
                  </div>
                  <div style={{ aspectRatio: '16/9', overflow: 'hidden' }}>
                    <img
                      src={buildImageUrl(layerData.url)}
                      alt={layerName}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="fuk-card-dashed" style={{ width: '60%', aspectRatio: '16/9' }}>
              <div className="fuk-placeholder">
                <Layers style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', opacity: 0.3 }} />
                <p style={{ color: '#6b7280' }}>
                  {processing ? 'Generating layers...' : 'Select layers and generate'}
                </p>
                <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
                  {enabledLayersCount} layer{enabledLayersCount !== 1 ? 's' : ''} selected
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Source Input Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Source Input</h3>
            
            <ImageUploader
              images={sourceInput ? [sourceInput] : []}
              onImagesChange={handleSourceChange}
              disabled={processing}
            />
            
            <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
              Upload image or video to generate AOV layers from
            </p>
          </div>

          {/* Layer Selection Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Active Layers</h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.layers.beauty}
                  onChange={(e) => updateLayer('beauty', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Beauty Pass</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Final RGB render with all lighting
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.layers.depth}
                  onChange={(e) => updateLayer('depth', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Depth (Z-Depth)</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Distance from camera for DOF/fog
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.layers.normals}
                  onChange={(e) => updateLayer('normals', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Normals</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Surface orientation for relighting
                  </span>
                </div>
              </label>
              
              <label className="fuk-checkbox-group">
                <input
                  type="checkbox"
                  className="fuk-checkbox"
                  checked={settings.layers.cryptomatte}
                  onChange={(e) => updateLayer('cryptomatte', e.target.checked)}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Cryptomatte</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Per-object ID masks for selection
                  </span>
                </div>
              </label>
            </div>
            
            <div style={{ 
              marginTop: '1rem', 
              padding: '0.75rem', 
              background: 'rgba(168, 85, 247, 0.1)', 
              borderRadius: '0.375rem',
              fontSize: '0.7rem',
              color: '#c084fc',
            }}>
              ℹ️ AOVs export to multi-layer EXR for compositing in Nuke/Fusion
            </div>
          </div>

          {/* Depth Settings */}
          {settings.layers.depth && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Depth Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Depth Model</label>
                <select
                  className="fuk-select"
                  value={settings.depthModel}
                  onChange={(e) => updateSettings({ depthModel: e.target.value })}
                >
                  <option value="depth_anything_v2">Depth Anything V2 (Recommended)</option>
                  <option value="depth_anything_v3">Depth Anything V3 (Latest)</option>
                  <option value="midas_large">MiDaS Large</option>
                  <option value="zoedepth">ZoeDepth (Metric)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.depthNormalize}
                    onChange={(e) => updateSettings({ depthNormalize: e.target.checked })}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Normalize to 0-1 range</span>
                </label>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.depthInvert}
                    onChange={(e) => updateSettings({ depthInvert: e.target.checked })}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Invert (near = black)</span>
                </label>
              </div>
            </div>
          )}

          {/* Normals Settings */}
          {settings.layers.normals && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Normals Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Normal Space</label>
                <select
                  className="fuk-select"
                  value={settings.normalsSpace}
                  onChange={(e) => updateSettings({ normalsSpace: e.target.value })}
                >
                  <option value="tangent">Tangent Space (UV-based)</option>
                  <option value="object">Object Space</option>
                  <option value="world">World Space</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.normalsFlipY}
                    onChange={(e) => updateSettings({ normalsFlipY: e.target.checked })}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Flip Y (OpenGL → DirectX)</span>
                </label>
              </div>
              
              <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
                Normals are encoded as RGB colors (XYZ → RGB)
              </p>
            </div>
          )}

          {/* Cryptomatte Settings */}
          {settings.layers.cryptomatte && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Cryptomatte Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Number of Layers</label>
                <select
                  className="fuk-select"
                  value={settings.cryptoLayers}
                  onChange={(e) => updateSettings({ cryptoLayers: parseInt(e.target.value) })}
                >
                  <option value="1">1 Layer (Basic)</option>
                  <option value="3">3 Layers (Recommended)</option>
                  <option value="6">6 Layers (Maximum)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Segmentation Accuracy</label>
                <select
                  className="fuk-select"
                  value={settings.cryptoAccuracy}
                  onChange={(e) => updateSettings({ cryptoAccuracy: e.target.value })}
                >
                  <option value="fast">Fast (Lower quality)</option>
                  <option value="balanced">Balanced</option>
                  <option value="high">High (Recommended)</option>
                </select>
              </div>
              
              <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
                Cryptomatte creates per-object mattes using AI segmentation
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={processing}
        progress={processing ? { progress: 0.5, phase: 'generating_layers' } : null}
        elapsedSeconds={0}
        onGenerate={handleGenerate}
        onCancel={() => setProcessing(false)}
        canGenerate={!!sourceInput && enabledLayersCount > 0}
        generateLabel="Generate Layers"
        generatingLabel="Generating..."
      />
    </>
  );
}