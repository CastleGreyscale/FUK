/**
 * Post-Process Tab
 * Upscaling and frame interpolation for final polish
 * Before/After comparison layout
 */

import { useState, useMemo, useCallback } from 'react';
import { Enhance, Loader2, CheckCircle, Camera } from '../components/Icons';
import ImageUploader from '../components/ImageUploader';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
import Footer from '../components/Footer';

const DEFAULT_SETTINGS = {
  // Upscaling
  upscaleMethod: 'realesrgan',
  upscaleFactor: 2,
  denoise: 0.5,
  
  // Frame Interpolation
  interpolationMethod: 'rife',
  targetFramerate: 24,
  sourceFramerate: 16,
  interpolationStrength: 1.0,
};

export default function PostprocessTab({ config, activeTab, setActiveTab, project }) {
  // Settings (localStorage fallback)
  const [localSettings, setLocalSettings] = useLocalStorage('fuk_postprocess_settings', DEFAULT_SETTINGS);
  
  const settings = useMemo(() => {
    if (project?.projectState?.tabs?.postprocess) {
      return { ...DEFAULT_SETTINGS, ...project.projectState.tabs.postprocess };
    }
    return { ...DEFAULT_SETTINGS, ...localSettings };
  }, [project?.projectState?.tabs?.postprocess, localSettings]);
  
  const updateSettings = useCallback((updates) => {
    const newSettings = { ...settings, ...updates };
    
    if (project?.isProjectLoaded && project.updateTabState) {
      project.updateTabState('postprocess', newSettings);
    } else {
      setLocalSettings(newSettings);
    }
  }, [project, settings, setLocalSettings]);
  
  // UI state
  const [activeProcess, setActiveProcess] = useState('upscale'); // 'upscale' or 'interpolate'
  const [sourceInput, setSourceInput] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  
  const handleSourceChange = (paths) => {
    setSourceInput(paths[0] || null);
    setResult(null);
  };
  
  const handleProcess = async () => {
    if (!sourceInput) {
      alert('Please select a source file');
      return;
    }
    
    setProcessing(true);
    // TODO: Implement backend API call
    // Simulate for now
    setTimeout(() => {
      setProcessing(false);
      setResult({ url: sourceInput }); // In reality, processed result
    }, 3000);
  };
  
  return (
    <>
      {/* Preview Area - Side by Side Comparison */}
      <div className="fuk-preview-area">
        <div className="fuk-preview-container" style={{ 
          display: 'flex', 
          gap: '1.5rem', 
          alignItems: 'center', 
          justifyContent: 'center',
          width: '100%',
          padding: '1.5rem',
        }}>
          {/* Input */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: '0.5rem',
            }}>
              <h3 style={{ 
                fontSize: '0.75rem', 
                textTransform: 'uppercase', 
                color: '#9ca3af', 
                letterSpacing: '0.05em',
                fontWeight: 600,
              }}>
                Input
              </h3>
              {sourceInput && (
                <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                  {activeProcess === 'upscale' ? 'Original Resolution' : `${settings.sourceFramerate}fps`}
                </span>
              )}
            </div>
            
            {sourceInput ? (
              <div style={{
                width: '100%',
                aspectRatio: '16/9',
                borderRadius: '0.5rem',
                border: '1px solid #374151',
                overflow: 'hidden',
                background: '#000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <img
                  src={buildImageUrl(sourceInput)}
                  alt="Input"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                  }}
                />
              </div>
            ) : (
              <div className="fuk-card-dashed" style={{ width: '100%', aspectRatio: '16/9' }}>
                <div className="fuk-placeholder">
                  <Camera style={{ width: '2rem', height: '2rem', margin: '0 auto 0.5rem', opacity: 0.3 }} />
                  <p style={{ color: '#6b7280', fontSize: '0.75rem' }}>
                    {activeProcess === 'upscale' ? 'Upload image' : 'Upload video'}
                  </p>
                </div>
              </div>
            )}
          </div>
          
          {/* Arrow */}
          <div style={{ 
            fontSize: '2.5rem', 
            color: processing ? '#a855f7' : '#4b5563',
            animation: processing ? 'pulse 2s infinite' : 'none',
          }}>
            →
          </div>
          
          {/* Output */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: '0.5rem',
            }}>
              <h3 style={{ 
                fontSize: '0.75rem', 
                textTransform: 'uppercase', 
                color: '#9ca3af', 
                letterSpacing: '0.05em',
                fontWeight: 600,
              }}>
                {activeProcess === 'upscale' ? 'Upscaled' : 'Interpolated'}
              </h3>
              {result && (
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '0.25rem',
                  fontSize: '0.7rem',
                }}>
                  <CheckCircle style={{ width: '0.875rem', height: '0.875rem', color: '#10b981' }} />
                  <span style={{ color: '#10b981' }}>
                    {activeProcess === 'upscale' ? `${settings.upscaleFactor}x` : `${settings.targetFramerate}fps`}
                  </span>
                </div>
              )}
            </div>
            
            {result ? (
              <div style={{
                width: '100%',
                aspectRatio: '16/9',
                borderRadius: '0.5rem',
                border: '2px solid #10b981',
                overflow: 'hidden',
                background: '#000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
              }}>
                <img
                  src={buildImageUrl(result.url)}
                  alt="Processed"
                  style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                  }}
                />
              </div>
            ) : (
              <div 
                className="fuk-card-dashed" 
                style={{ 
                  width: '100%', 
                  aspectRatio: '16/9',
                  borderColor: processing ? '#a855f7' : '#374151',
                }}
              >
                <div className="fuk-placeholder">
                  {processing ? (
                    <>
                      <Loader2 className="animate-spin" style={{ width: '2rem', height: '2rem', margin: '0 auto 0.5rem', color: '#a855f7' }} />
                      <p style={{ color: '#a855f7', fontSize: '0.75rem' }}>Processing...</p>
                    </>
                  ) : (
                    <>
                      <Enhance style={{ width: '2rem', height: '2rem', margin: '0 auto 0.5rem', opacity: 0.3 }} />
                      <p style={{ color: '#6b7280', fontSize: '0.75rem' }}>
                        {activeProcess === 'upscale' ? 'Enhanced resolution' : 'Smooth motion'}
                      </p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Process Type Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Process Type</h3>
            
            <div className="fuk-form-group-compact">
              <div className="fuk-radio-group" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '1rem' }}>
                <label className="fuk-radio-option" style={{ cursor: 'pointer' }}>
                  <input
                    type="radio"
                    className="fuk-radio"
                    checked={activeProcess === 'upscale'}
                    onChange={() => {
                      setActiveProcess('upscale');
                      setResult(null);
                    }}
                  />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <span className="fuk-radio-label" style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                      Upscaling
                    </span>
                    <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                      Increase image resolution with AI enhancement
                    </span>
                  </div>
                </label>
                
                <label className="fuk-radio-option" style={{ cursor: 'pointer' }}>
                  <input
                    type="radio"
                    className="fuk-radio"
                    checked={activeProcess === 'interpolate'}
                    onChange={() => {
                      setActiveProcess('interpolate');
                      setResult(null);
                    }}
                  />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <span className="fuk-radio-label" style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                      Frame Interpolation
                    </span>
                    <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                      Generate intermediate frames for smooth motion
                    </span>
                  </div>
                </label>
              </div>
            </div>
          </div>

          {/* Input Source Card */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">
              {activeProcess === 'upscale' ? 'Image Input' : 'Video Input'}
            </h3>
            
            <ImageUploader
              images={sourceInput ? [sourceInput] : []}
              onImagesChange={handleSourceChange}
              disabled={processing}
            />
            
            <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.5rem' }}>
              {activeProcess === 'upscale' 
                ? 'Upload low-resolution image to enhance'
                : 'Upload video sequence or MP4 to interpolate frames'
              }
            </p>
          </div>

          {/* Upscaling Settings */}
          {activeProcess === 'upscale' && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Upscaling Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">AI Model</label>
                <select
                  className="fuk-select"
                  value={settings.upscaleMethod}
                  onChange={(e) => updateSettings({ upscaleMethod: e.target.value })}
                >
                  <option value="esrgan">ESRGAN (Photo-realistic)</option>
                  <option value="realesrgan">Real-ESRGAN (General, Recommended)</option>
                  <option value="swinir">SwinIR (Fine Detail)</option>
                  <option value="scunet">SCUNet (Denoise + Upscale)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Scale Factor</label>
                <div className="fuk-radio-group">
                  {[2, 4, 8].map(factor => (
                    <label key={factor} className="fuk-radio-option">
                      <input
                        type="radio"
                        className="fuk-radio"
                        checked={settings.upscaleFactor === factor}
                        onChange={() => updateSettings({ upscaleFactor: factor })}
                      />
                      <span className="fuk-radio-label">{factor}x</span>
                    </label>
                  ))}
                </div>
                <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                  {settings.upscaleFactor}x = {settings.upscaleFactor * settings.upscaleFactor}x more pixels
                </p>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Denoise Strength</label>
                <div className="fuk-input-inline">
                  <input
                    type="range"
                    className="fuk-input"
                    style={{ flex: 2 }}
                    value={settings.denoise}
                    onChange={(e) => updateSettings({ denoise: parseFloat(e.target.value) })}
                    min={0}
                    max={1}
                    step={0.05}
                  />
                  <input
                    type="number"
                    className="fuk-input"
                    style={{ width: '80px' }}
                    value={settings.denoise}
                    onChange={(e) => updateSettings({ denoise: parseFloat(e.target.value) })}
                    step={0.05}
                    min={0}
                    max={1}
                  />
                </div>
                <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                  Higher = more noise reduction (may soften details)
                </p>
              </div>
            </div>
          )}

          {/* Frame Interpolation Settings */}
          {activeProcess === 'interpolate' && (
            <div className="fuk-card">
              <h3 className="fuk-card-title fuk-mb-3">Interpolation Settings</h3>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">AI Model</label>
                <select
                  className="fuk-select"
                  value={settings.interpolationMethod}
                  onChange={(e) => updateSettings({ interpolationMethod: e.target.value })}
                >
                  <option value="rife">RIFE (Recommended, Fast + Quality)</option>
                  <option value="dain">DAIN (Highest Quality, Slower)</option>
                  <option value="flavr">FLAVR (Fastest)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Source Framerate</label>
                <select
                  className="fuk-select"
                  value={settings.sourceFramerate}
                  onChange={(e) => updateSettings({ sourceFramerate: parseInt(e.target.value) })}
                >
                  <option value="8">8 fps</option>
                  <option value="12">12 fps</option>
                  <option value="16">16 fps (AI Video Default)</option>
                  <option value="24">24 fps</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Target Framerate</label>
                <select
                  className="fuk-select"
                  value={settings.targetFramerate}
                  onChange={(e) => updateSettings({ targetFramerate: parseInt(e.target.value) })}
                >
                  <option value="24">24 fps (Film)</option>
                  <option value="30">30 fps (Video)</option>
                  <option value="60">60 fps (Smooth)</option>
                </select>
                <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                  {settings.targetFramerate / settings.sourceFramerate}x frame count
                </p>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Interpolation Strength</label>
                <div className="fuk-input-inline">
                  <input
                    type="range"
                    className="fuk-input"
                    style={{ flex: 2 }}
                    value={settings.interpolationStrength}
                    onChange={(e) => updateSettings({ interpolationStrength: parseFloat(e.target.value) })}
                    min={0.5}
                    max={1}
                    step={0.05}
                  />
                  <input
                    type="number"
                    className="fuk-input"
                    style={{ width: '80px' }}
                    value={settings.interpolationStrength}
                    onChange={(e) => updateSettings({ interpolationStrength: parseFloat(e.target.value) })}
                    step={0.05}
                    min={0.5}
                    max={1}
                  />
                </div>
                <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                  Lower = blend existing frames, Higher = generate new motion
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <Footer
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        generating={processing}
        progress={processing ? { progress: 0.5, phase: 'processing' } : null}
        elapsedSeconds={0}
        onGenerate={handleProcess}
        onCancel={() => setProcessing(false)}
        canGenerate={!!sourceInput}
        generateLabel="Process"
        generatingLabel="Processing..."
      />
    </>
  );
}