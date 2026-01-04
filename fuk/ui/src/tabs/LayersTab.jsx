/**
 * Layers Tab
 * Generate and manage AOV layers (Arbitrary Output Variables)
 * Layers: Depth, Normals, Cryptomatte
 */

import { useState, useMemo, useCallback } from 'react';
import { Loader2, CheckCircle, Layers, AlertCircle } from '../components/Icons';
import ImageUploader from '../components/ImageUploader';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
import Footer from '../components/Footer';


const DEFAULT_SETTINGS = {
  // Active layers to generate
  layers: {
    depth: true,
    normals: true,
    crypto: false,
  },
  
  // Depth settings
  depthModel: 'depth_anything_v2',
  depthInvert: false,
  depthNormalize: true,
  depthColormap: 'inferno',
  
  // Normals settings
  normalsMethod: 'from_depth',  // 'from_depth' or 'dsine'
  normalsDepthModel: 'depth_anything_v2',
  normalsSpace: 'tangent',
  normalsFlipY: false,
  normalsIntensity: 1.0,
  
  // Cryptomatte settings
  cryptoModel: 'sam2_hiera_large',
  cryptoMaxObjects: 50,
  cryptoMinArea: 500,
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
  const [errors, setErrors] = useState({});
  const [elapsedTime, setElapsedTime] = useState(0);
  
  const handleSourceChange = (paths) => {
    setSourceInput(paths[0] || null);
    setGeneratedLayers({});
    setErrors({});
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
      alert('Please select a source image');
      return;
    }
    
    setProcessing(true);
    setGeneratedLayers({});
    setErrors({});
    setElapsedTime(0);
    
    const startTime = Date.now();
    const timer = setInterval(() => {
      setElapsedTime(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    
    try {
      const response = await fetch('/api/layers/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_path: sourceInput,
          layers: settings.layers,
          
          // Depth settings
          depth_model: settings.depthModel,
          depth_invert: settings.depthInvert,
          depth_normalize: settings.depthNormalize,
          depth_colormap: settings.depthColormap,
          
          // Normals settings
          normals_method: settings.normalsMethod,
          normals_depth_model: settings.normalsDepthModel,
          normals_space: settings.normalsSpace,
          normals_flip_y: settings.normalsFlipY,
          normals_intensity: settings.normalsIntensity,
          
          // Crypto settings
          crypto_model: settings.cryptoModel,
          crypto_max_objects: settings.cryptoMaxObjects,
          crypto_min_area: settings.cryptoMinArea,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Transform layers result into our display format
      const layers = {};
      if (result.layers) {
        Object.entries(result.layers).forEach(([name, data]) => {
          layers[name] = {
            url: data.url,
            name: name.charAt(0).toUpperCase() + name.slice(1),
            info: data.model || data.method || (data.num_objects ? `${data.num_objects} objects` : ''),
          };
        });
      }
      
      // Add source/beauty as first layer
      if (result.source) {
        layers.beauty = {
          url: result.source,
          name: 'Beauty (Source)',
          info: 'Original',
        };
      }
      
      setGeneratedLayers(layers);
      setErrors(result.errors || {});
      
    } catch (err) {
      console.error('Layer generation failed:', err);
      setErrors({ _general: err.message });
    } finally {
      clearInterval(timer);
      setProcessing(false);
    }
  };
  
  const handleCancel = () => {
    // TODO: Implement actual cancel via AbortController
    setProcessing(false);
  };
  
  // Count enabled layers
  const enabledLayersCount = Object.values(settings.layers).filter(v => v).length;
  
  // Check if we have any errors
  const hasErrors = Object.keys(errors).length > 0;
  
  return (
    <>
      {/* Preview Area - Grid Layout for Layers */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-container" style={{ width: '100%', height: '100%', padding: '1rem' }}>
          {Object.keys(generatedLayers).length > 0 ? (
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
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
                    <div>
                      <span style={{ 
                        fontSize: '0.75rem', 
                        fontWeight: 600, 
                        textTransform: 'uppercase',
                        color: '#c084fc',
                        letterSpacing: '0.05em',
                      }}>
                        {layerData.name || layerName}
                      </span>
                      {layerData.info && (
                        <span style={{ 
                          fontSize: '0.65rem', 
                          color: '#6b7280',
                          marginLeft: '0.5rem',
                        }}>
                          {layerData.info}
                        </span>
                      )}
                    </div>
                    <CheckCircle style={{ width: '1rem', height: '1rem', color: '#10b981' }} />
                  </div>
                  <div style={{overflow: 'hidden' }}>
                    <img
                      src={buildImageUrl(layerData.url)}
                      alt={layerName}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                      }}
                    />
                  </div>
                </div>
              ))}
              
              {/* Show errors if any */}
              {hasErrors && (
                <div style={{
                  background: 'rgba(239, 68, 68, 0.1)',
                  borderRadius: '0.5rem',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  padding: '1rem',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <AlertCircle style={{ width: '1rem', height: '1rem', color: '#ef4444' }} />
                    <span style={{ color: '#ef4444', fontWeight: 600, fontSize: '0.75rem' }}>
                      Some layers failed
                    </span>
                  </div>
                  {Object.entries(errors).map(([layer, error]) => (
                    <p key={layer} style={{ fontSize: '0.7rem', color: '#f87171', margin: '0.25rem 0' }}>
                      {layer}: {error}
                    </p>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="fuk-card-dashed" style={{ width: '60%', aspectRatio: '16/9' }}>
              <div className="fuk-placeholder">
                {processing ? (
                  <>
                    <Loader2 style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', opacity: 0.5 }} className="fuk-spin" />
                    <p style={{ color: '#c084fc' }}>Generating {enabledLayersCount} layer{enabledLayersCount !== 1 ? 's' : ''}...</p>
                    <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
                      {elapsedTime}s elapsed
                    </p>
                  </>
                ) : (
                  <>
                    <Layers style={{ width: '3rem', height: '3rem', margin: '0 auto 1rem', opacity: 0.3 }} />
                    <p style={{ color: '#6b7280' }}>Select layers and generate</p>
                    <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.5rem' }}>
                      {enabledLayersCount} layer{enabledLayersCount !== 1 ? 's' : ''} selected
                    </p>
                  </>
                )}
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
              Upload image to generate AOV layers from
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
                  checked={settings.layers.depth}
                  onChange={(e) => updateLayer('depth', e.target.checked)}
                  disabled={processing}
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
                  disabled={processing}
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
                  checked={settings.layers.crypto}
                  onChange={(e) => updateLayer('crypto', e.target.checked)}
                  disabled={processing}
                />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Cryptomatte</span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Per-object ID masks via SAM2
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
                  disabled={processing}
                >
                  <option value="depth_anything_v2">Depth Anything V2 (Recommended)</option>
                  <option value="depth_anything_v3">Depth Anything V3 (Latest)</option>
                  <option value="midas_large">MiDaS Large</option>
                  <option value="midas_small">MiDaS Small (Fast)</option>
                  <option value="zoedepth">ZoeDepth (Metric)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Colormap</label>
                <select
                  className="fuk-select"
                  value={settings.depthColormap || ''}
                  onChange={(e) => updateSettings({ depthColormap: e.target.value || null })}
                  disabled={processing}
                >
                  <option value="">Grayscale</option>
                  <option value="inferno">Inferno (Heat)</option>
                  <option value="viridis">Viridis (Green-Blue)</option>
                  <option value="magma">Magma (Purple-Orange)</option>
                  <option value="plasma">Plasma (Purple-Yellow)</option>
                  <option value="turbo">Turbo (Rainbow)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-checkbox-group">
                  <input
                    type="checkbox"
                    className="fuk-checkbox"
                    checked={settings.depthNormalize}
                    onChange={(e) => updateSettings({ depthNormalize: e.target.checked })}
                    disabled={processing}
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
                    disabled={processing}
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
                <label className="fuk-label">Estimation Method</label>
                <select
                  className="fuk-select"
                  value={settings.normalsMethod}
                  onChange={(e) => updateSettings({ normalsMethod: e.target.value })}
                  disabled={processing}
                >
                  <option value="from_depth">From Depth (Fast)</option>
                  <option value="dsine">DSINE (Better Quality)</option>
                </select>
              </div>
              
              {settings.normalsMethod === 'from_depth' && (
                <>
                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">Depth Model</label>
                    <select
                      className="fuk-select"
                      value={settings.normalsDepthModel}
                      onChange={(e) => updateSettings({ normalsDepthModel: e.target.value })}
                      disabled={processing}
                    >
                      <option value="depth_anything_v2">Depth Anything V2</option>
                      <option value="depth_anything_v3">Depth Anything V3</option>
                      <option value="midas_large">MiDaS Large</option>
                      <option value="zoedepth">ZoeDepth</option>
                    </select>
                  </div>
                  
                  <div className="fuk-form-group-compact">
                    <label className="fuk-label">
                      Intensity: {settings.normalsIntensity.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      className="fuk-slider"
                      min="0.1"
                      max="3.0"
                      step="0.1"
                      value={settings.normalsIntensity}
                      onChange={(e) => updateSettings({ normalsIntensity: parseFloat(e.target.value) })}
                      disabled={processing}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#6b7280' }}>
                      <span>Subtle</span>
                      <span>Strong</span>
                    </div>
                  </div>
                </>
              )}
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Normal Space</label>
                <select
                  className="fuk-select"
                  value={settings.normalsSpace}
                  onChange={(e) => updateSettings({ normalsSpace: e.target.value })}
                  disabled={processing}
                >
                  <option value="tangent">Tangent Space</option>
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
                    disabled={processing}
                  />
                  <span className="fuk-label" style={{ marginBottom: 0 }}>Flip Y (OpenGL → DirectX)</span>
                </label>
              </div>
              
              <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
                Normals encoded as RGB (XYZ → RGB)
              </p>
            </div>
          )}

          {/* Cryptomatte Settings */}
          {settings.layers.crypto && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Cryptomatte Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">SAM2 Model</label>
                <select
                  className="fuk-select"
                  value={settings.cryptoModel}
                  onChange={(e) => updateSettings({ cryptoModel: e.target.value })}
                  disabled={processing}
                >
                  <option value="sam2_hiera_large">Large (Best Quality)</option>
                  <option value="sam2_hiera_base_plus">Base+ (Good)</option>
                  <option value="sam2_hiera_small">Small (Balanced)</option>
                  <option value="sam2_hiera_tiny">Tiny (Fast)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">
                  Max Objects: {settings.cryptoMaxObjects}
                </label>
                <input
                  type="range"
                  className="fuk-slider"
                  min="10"
                  max="100"
                  step="5"
                  value={settings.cryptoMaxObjects}
                  onChange={(e) => updateSettings({ cryptoMaxObjects: parseInt(e.target.value) })}
                  disabled={processing}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#6b7280' }}>
                  <span>10</span>
                  <span>100</span>
                </div>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">
                  Min Area: {settings.cryptoMinArea}px
                </label>
                <input
                  type="range"
                  className="fuk-slider"
                  min="100"
                  max="2000"
                  step="100"
                  value={settings.cryptoMinArea}
                  onChange={(e) => updateSettings({ cryptoMinArea: parseInt(e.target.value) })}
                  disabled={processing}
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#6b7280' }}>
                  <span>Small</span>
                  <span>Large only</span>
                </div>
              </div>
              
              <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
                Creates per-object mattes using Segment Anything 2
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
        elapsedSeconds={elapsedTime}
        onGenerate={handleGenerate}
        onCancel={handleCancel}
        canGenerate={!!sourceInput && enabledLayersCount > 0}
        generateLabel="Generate Layers"
        generatingLabel="Generating..."
      />
    </>
  );
}