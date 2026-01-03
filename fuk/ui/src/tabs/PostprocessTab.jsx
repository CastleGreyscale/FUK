/**
 * Post-Process Tab
 * Upscaling and frame interpolation for final polish
 * Before/After comparison layout
 */

import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { Enhance, Loader2, CheckCircle, Camera, Film, AlertCircle } from '../components/Icons';
import ImageUploader from '../components/ImageUploader';
import { useLocalStorage } from '../hooks/useLocalStorage';
import { buildImageUrl } from '../utils/constants';
import Footer from '../components/Footer';

const API_URL = '/api';

const DEFAULT_SETTINGS = {
  // Upscaling
  upscaleMethod: 'realesrgan',
  upscaleFactor: 4,
  denoise: 0.5,
  
  // Frame Interpolation
  interpolationMethod: 'rife',
  targetFramerate: 24,
  sourceFramerate: 16,
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
  
  // Ref to avoid infinite loop
  const settingsRef = useRef(settings);
  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);
  
  const updateSettings = useCallback((updates) => {
    const newSettings = { ...settingsRef.current, ...updates };
    
    if (project?.isProjectLoaded && project?.updateTabState) {
      project.updateTabState('postprocess', newSettings);
    } else {
      setLocalSettings(newSettings);
    }
  }, [project?.isProjectLoaded, project?.updateTabState, setLocalSettings]);
  
  // UI state
  const [activeProcess, setActiveProcess] = useState('upscale'); // 'upscale' or 'interpolate'
  const [sourceInput, setSourceInput] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  
  // Fetch capabilities on mount
  useEffect(() => {
    fetch(`${API_URL}/postprocess/capabilities`)
      .then(res => res.json())
      .then(data => {
        console.log('[PostProcess] Capabilities:', data);
        setCapabilities(data);
      })
      .catch(err => {
        console.warn('[PostProcess] Could not fetch capabilities:', err);
      });
  }, []);
  
  const handleSourceChange = (paths) => {
    setSourceInput(paths[0] || null);
    setResult(null);
    setError(null);
  };
  
  const handleUpscale = async () => {
    if (!sourceInput) {
      setError('Please select a source image');
      return;
    }
    
    setProcessing(true);
    setProgress({ phase: 'Starting upscale...', progress: 0.1 });
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/postprocess/upscale`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path: sourceInput,
          scale: settings.upscaleFactor,
          model: settings.upscaleMethod,
          denoise: settings.denoise,
        }),
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Upscale failed');
      }
      
      const data = await response.json();
      console.log('[PostProcess] Upscale result:', data);
      
      setResult({
        type: 'image',
        url: data.output_url,
        path: data.output_path,
        inputSize: data.input_size,
        outputSize: data.output_size,
        scale: data.scale,
        method: data.method,
      });
      
    } catch (err) {
      console.error('[PostProcess] Upscale error:', err);
      setError(err.message);
    } finally {
      setProcessing(false);
      setProgress(null);
    }
  };
  
  const handleInterpolate = async () => {
    if (!sourceInput) {
      setError('Please select a source video');
      return;
    }
    
    setProcessing(true);
    setProgress({ phase: 'Starting interpolation...', progress: 0.1 });
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/postprocess/interpolate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path: sourceInput,
          source_fps: settings.sourceFramerate,
          target_fps: settings.targetFramerate,
          model: settings.interpolationMethod,
        }),
      });
      
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Interpolation failed');
      }
      
      const data = await response.json();
      console.log('[PostProcess] Interpolate result:', data);
      
      setResult({
        type: 'video',
        url: data.output_url,
        path: data.output_path,
        sourceFps: data.source_fps,
        targetFps: data.target_fps,
        multiplier: data.multiplier,
        method: data.method,
      });
      
    } catch (err) {
      console.error('[PostProcess] Interpolate error:', err);
      setError(err.message);
    } finally {
      setProcessing(false);
      setProgress(null);
    }
  };
  
  const handleProcess = async () => {
    if (activeProcess === 'upscale') {
      await handleUpscale();
    } else {
      await handleInterpolate();
    }
  };
  
  const canProcess = sourceInput && !processing;
  
  // Calculate preview info
  const getInputInfo = () => {
    if (!sourceInput) return null;
    if (activeProcess === 'upscale') {
      return 'Original Resolution';
    }
    return `${settings.sourceFramerate} fps`;
  };
  
  const getOutputInfo = () => {
    if (!result) return null;
    if (result.type === 'image') {
      return `${result.outputSize.width}x${result.outputSize.height} (${result.scale}x)`;
    }
    return `${result.targetFps} fps (${result.multiplier}x frames)`;
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
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', maxWidth: '45%' }}>
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
                  {getInputInfo()}
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
                {activeProcess === 'upscale' || sourceInput.match(/\.(png|jpg|jpeg|webp)$/i) ? (
                  <img
                    src={buildImageUrl(sourceInput)}
                    alt="Input"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain',
                    }}
                  />
                ) : (
                  <video
                    src={buildImageUrl(sourceInput)}
                    controls
                    loop
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain',
                    }}
                  />
                )}
              </div>
            ) : (
              <div className="fuk-card-dashed" style={{ width: '100%', aspectRatio: '16/9' }}>
                <div className="fuk-placeholder">
                  {activeProcess === 'upscale' ? (
                    <Camera style={{ width: '2rem', height: '2rem', margin: '0 auto 0.5rem', opacity: 0.3 }} />
                  ) : (
                    <Film style={{ width: '2rem', height: '2rem', margin: '0 auto 0.5rem', opacity: 0.3 }} />
                  )}
                  <p style={{ color: '#6b7280', fontSize: '0.75rem' }}>
                    {activeProcess === 'upscale' ? 'Select image to upscale' : 'Select video to interpolate'}
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
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.5rem',
          }}>
            {processing ? (
              <Loader2 style={{ width: '2rem', height: '2rem', animation: 'spin 1s linear infinite' }} />
            ) : (
              <span>→</span>
            )}
            {processing && progress && (
              <span style={{ fontSize: '0.7rem', color: '#a855f7' }}>
                {progress.phase}
              </span>
            )}
          </div>
          
          {/* Output */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', maxWidth: '45%' }}>
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
                    {getOutputInfo()}
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
                {result.type === 'image' ? (
                  <img
                    src={buildImageUrl(result.url)}
                    alt="Processed"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain',
                    }}
                  />
                ) : (
                  <video
                    src={buildImageUrl(result.url)}
                    controls
                    autoPlay
                    loop
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain',
                    }}
                  />
                )}
                
                {/* Method badge */}
                <div style={{
                  position: 'absolute',
                  bottom: '0.5rem',
                  right: '0.5rem',
                  padding: '0.25rem 0.5rem',
                  background: 'rgba(0,0,0,0.7)',
                  borderRadius: '0.25rem',
                  fontSize: '0.65rem',
                  color: '#9ca3af',
                }}>
                  {result.method}
                </div>
              </div>
            ) : (
              <div className="fuk-card-dashed" style={{ width: '100%', aspectRatio: '16/9' }}>
                <div className="fuk-placeholder">
                  <Enhance style={{ width: '2rem', height: '2rem', margin: '0 auto 0.5rem', opacity: 0.3 }} />
                  <p style={{ color: '#6b7280', fontSize: '0.75rem' }}>
                    {processing ? 'Processing...' : 'Result will appear here'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Settings Area */}
      <div className="fuk-settings-area">
        <div className="fuk-settings-grid">
          {/* Process Type Selection */}
          <div className="fuk-card">
            <h3 className="fuk-card-title fuk-mb-3">Process Type</h3>
            
            <div className="fuk-radio-group" style={{ flexDirection: 'column', gap: '0.75rem' }}>
              <label className="fuk-radio-option" style={{ 
                padding: '0.75rem',
                background: activeProcess === 'upscale' ? 'rgba(168, 85, 247, 0.1)' : 'transparent',
                border: activeProcess === 'upscale' ? '1px solid #a855f7' : '1px solid #374151',
                borderRadius: '0.5rem',
                cursor: 'pointer',
              }}>
                <input
                  type="radio"
                  className="fuk-radio"
                  checked={activeProcess === 'upscale'}
                  onChange={() => {
                    setActiveProcess('upscale');
                    setResult(null);
                    setError(null);
                  }}
                />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                  <span className="fuk-radio-label" style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                    <Camera style={{ width: '1rem', height: '1rem', display: 'inline', marginRight: '0.5rem' }} />
                    Image Upscaling
                  </span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Increase resolution using AI (Real-ESRGAN)
                  </span>
                </div>
              </label>
              
              <label className="fuk-radio-option" style={{ 
                padding: '0.75rem',
                background: activeProcess === 'interpolate' ? 'rgba(168, 85, 247, 0.1)' : 'transparent',
                border: activeProcess === 'interpolate' ? '1px solid #a855f7' : '1px solid #374151',
                borderRadius: '0.5rem',
                cursor: 'pointer',
              }}>
                <input
                  type="radio"
                  className="fuk-radio"
                  checked={activeProcess === 'interpolate'}
                  onChange={() => {
                    setActiveProcess('interpolate');
                    setResult(null);
                    setError(null);
                  }}
                />
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                  <span className="fuk-radio-label" style={{ fontSize: '0.875rem', fontWeight: 600 }}>
                    <Film style={{ width: '1rem', height: '1rem', display: 'inline', marginRight: '0.5rem' }} />
                    Frame Interpolation
                  </span>
                  <span style={{ fontSize: '0.7rem', color: '#6b7280' }}>
                    Generate intermediate frames (RIFE)
                  </span>
                </div>
              </label>
            </div>
            
            {/* Capability indicators */}
            {capabilities && (
              <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: '#1f2937', borderRadius: '0.375rem' }}>
                <div style={{ fontSize: '0.65rem', color: '#6b7280', marginBottom: '0.25rem' }}>Available backends:</div>
                <div style={{ display: 'flex', gap: '0.5rem', fontSize: '0.7rem' }}>
                  <span style={{ color: capabilities.upscaling?.ncnn_available ? '#10b981' : '#6b7280' }}>
                    {capabilities.upscaling?.ncnn_available ? '✓' : '○'} ESRGAN-NCNN
                  </span>
                  <span style={{ color: capabilities.interpolation?.ncnn_available ? '#10b981' : '#6b7280' }}>
                    {capabilities.interpolation?.ncnn_available ? '✓' : '○'} RIFE-NCNN
                  </span>
                </div>
              </div>
            )}
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
                ? 'Drag from History or upload an image to enhance'
                : 'Drag from History or upload a video to interpolate'
              }
            </p>
            
            {error && (
              <div style={{ 
                marginTop: '0.75rem', 
                padding: '0.5rem 0.75rem',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '0.375rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
              }}>
                <AlertCircle style={{ width: '1rem', height: '1rem', color: '#ef4444' }} />
                <span style={{ fontSize: '0.75rem', color: '#ef4444' }}>{error}</span>
              </div>
            )}
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
                  disabled={processing}
                >
                  <option value="realesrgan">Real-ESRGAN (Recommended)</option>
                  <option value="lanczos">Lanczos (Fast, No AI)</option>
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
                        disabled={processing}
                      />
                      <span className="fuk-radio-label">{factor}x</span>
                    </label>
                  ))}
                </div>
                <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                  {settings.upscaleFactor}x scale = {settings.upscaleFactor * settings.upscaleFactor}x more pixels
                </p>
              </div>
              
              {settings.upscaleMethod === 'realesrgan' && (
                <div className="fuk-form-group-compact">
                  <label className="fuk-label">Denoise Strength</label>
                  <div className="fuk-input-inline">
                    <input
                      type="range"
                      className="fuk-range"
                      style={{ flex: 2 }}
                      value={settings.denoise}
                      onChange={(e) => updateSettings({ denoise: parseFloat(e.target.value) })}
                      min={0}
                      max={1}
                      step={0.05}
                      disabled={processing}
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
                      disabled={processing}
                    />
                  </div>
                  <p style={{ fontSize: '0.7rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                    Higher = more noise reduction (may soften details)
                  </p>
                </div>
              )}
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
                  disabled={processing}
                >
                  <option value="rife">RIFE (Recommended)</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Source Framerate</label>
                <select
                  className="fuk-select"
                  value={settings.sourceFramerate}
                  onChange={(e) => updateSettings({ sourceFramerate: parseInt(e.target.value) })}
                  disabled={processing}
                >
                  <option value="8">8 fps</option>
                  <option value="12">12 fps</option>
                  <option value="16">16 fps (Wan default)</option>
                  <option value="24">24 fps</option>
                </select>
              </div>
              
              <div className="fuk-form-group-compact">
                <label className="fuk-label">Target Framerate</label>
                <select
                  className="fuk-select"
                  value={settings.targetFramerate}
                  onChange={(e) => updateSettings({ targetFramerate: parseInt(e.target.value) })}
                  disabled={processing}
                >
                  <option value="24">24 fps (Film)</option>
                  <option value="30">30 fps (Video)</option>
                  <option value="60">60 fps (Smooth)</option>
                </select>
              </div>
              
              <div style={{ 
                marginTop: '0.75rem',
                padding: '0.5rem 0.75rem',
                background: '#1f2937',
                borderRadius: '0.375rem',
              }}>
                <div style={{ fontSize: '0.75rem', color: '#d1d5db' }}>
                  Frame multiplier: <strong style={{ color: '#a855f7' }}>
                    {Math.round(settings.targetFramerate / settings.sourceFramerate)}x
                  </strong>
                </div>
                <div style={{ fontSize: '0.65rem', color: '#6b7280', marginTop: '0.25rem' }}>
                  {settings.sourceFramerate} fps → {settings.targetFramerate} fps
                </div>
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
        progress={progress}
        elapsedSeconds={0}
        onGenerate={handleProcess}
        onCancel={() => setProcessing(false)}
        canGenerate={canProcess}
        generateLabel={activeProcess === 'upscale' ? 'Upscale' : 'Interpolate'}
        generatingLabel={activeProcess === 'upscale' ? 'Upscaling...' : 'Interpolating...'}
      />
    </>
  );
}