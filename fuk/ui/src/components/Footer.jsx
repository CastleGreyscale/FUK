/**
 * Footer Component
 * Contains progress bar, tabs, and action buttons
 */

import { useState, useEffect, useCallback } from 'react';
import TabButton from './TabButton';
import ProgressBar from './ProgressBar';
import { Camera, Film, Pipeline, Enhance, Save, FukMonogram, Loader2, X, Layers, Wrench, Trash2, RefreshCw } from './Icons';

export default function Footer({
  activeTab,
  setActiveTab,
  generating,
  progress,
  elapsedSeconds,
  onGenerate,
  onCancel,
  canGenerate,
  generateLabel = 'Generate',
  generatingLabel = 'Generating...',
  batchCount,
  onBatchCountChange,
}) {
  const [batchInput, setBatchInput] = useState(String(batchCount ?? 1));
  useEffect(() => { setBatchInput(String(batchCount ?? 1)); }, [batchCount]);

  const [evicting, setEvicting] = useState(false);
  const [restarting, setRestarting] = useState(false);

  const handleEvict = useCallback(async () => {
    setEvicting(true);
    try {
      await fetch('/api/system/evict', { method: 'POST' });
    } finally {
      setEvicting(false);
    }
  }, []);

  const handleRestart = useCallback(async () => {
    setRestarting(true);
    try {
      await fetch('/api/system/restart', { method: 'POST' });
    } catch (_) {}
    // Poll until server is back, then reload
    const poll = async () => {
      try {
        const r = await fetch('/api/config/models');
        if (r.ok) { window.location.reload(); return; }
      } catch (_) {}
      setTimeout(poll, 1000);
    };
    setTimeout(poll, 1500);
  }, []);

  const commitBatch = (raw) => {
    const parsed = Math.max(1, Math.min(50, parseInt(raw) || 1));
    setBatchInput(String(parsed));
    onBatchCountChange?.(parsed);
  };

  return (
    <div className="fuk-footer" style={{ position: 'relative' }}>

      {/* System buttons - Left */}
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <button
          onClick={handleEvict}
          disabled={evicting || restarting}
          className="fuk-btn fuk-btn-secondary"
          title="Evict all models from VRAM"
          style={{ padding: '0.35rem 0.65rem' }}
        >
          {evicting
            ? <Loader2 style={{ width: '1rem', height: '1rem', animation: 'spin 1s linear infinite' }} />
            : <Trash2 style={{ width: '1rem', height: '1rem' }} />}
          Evict
        </button>
        <button
          onClick={handleRestart}
          disabled={evicting || restarting}
          className="fuk-btn fuk-btn-secondary"
          title="Restart the server"
          style={{ padding: '0.35rem 0.65rem' }}
        >
          {restarting
            ? <Loader2 style={{ width: '1rem', height: '1rem', animation: 'spin 1s linear infinite' }} />
            : <RefreshCw style={{ width: '1rem', height: '1rem' }} />}
          {restarting ? 'Restarting...' : 'Restart'}
        </button>
      </div>

      {/* Tabs - Center */}
      <div className="fuk-footer-center">
        <div className="fuk-tabs">
          <TabButton
            active={activeTab === 'utilities'}
            onClick={() => setActiveTab('utilities')}
            icon={<Wrench style={{ width: '1rem', height: '1rem' }} />}
            label="Utilities"
          />
          <TabButton
            active={activeTab === 'preprocess'}
            onClick={() => setActiveTab('preprocess')}
            icon={<Pipeline style={{ width: '1rem', height: '1rem' }} />}
            label="Pre-Processors"
          />
          <TabButton
            active={activeTab === 'image'}
            onClick={() => setActiveTab('image')}
            icon={<Camera style={{ width: '1rem', height: '1rem' }} />}
            label="Image"
          />
          <TabButton
            active={activeTab === 'video'}
            onClick={() => setActiveTab('video')}
            icon={<Film style={{ width: '1rem', height: '1rem' }} />}
            label="Video"
          />
          <TabButton
            active={activeTab === 'postprocess'}
            onClick={() => setActiveTab('postprocess')}
            icon={<Enhance style={{ width: '1rem', height: '1rem' }} />}
            label="Post-Processors"
          />
          <TabButton
            active={activeTab === 'layers'}
            onClick={() => setActiveTab('layers')}
            icon={<Layers style={{ width: '1rem', height: '1rem' }} />}
            label="Layers"
          />
          <TabButton
            active={activeTab === 'export'}
            onClick={() => setActiveTab('export')}
            icon={<Save style={{ width: '1rem', height: '1rem' }} />}
            label="Exports"
          />
        </div>
      </div>

      {/* Buttons - Right */}
      <div className="fuk-footer-right" style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
        {/* Batch Count */}
        {onBatchCountChange && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', userSelect: 'none' }}>×</span>
            <input
              type="number"
              min="1"
              max="50"
              value={batchInput}
              onChange={e => setBatchInput(e.target.value)}
              onBlur={e => commitBatch(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && commitBatch(e.target.value)}
              disabled={generating}
              className="fuk-input"
              style={{ width: '3.5rem', textAlign: 'center', padding: '0.35rem 0.4rem' }}
              title="Batch count"
            />
          </div>
        )}
        <button
          onClick={onGenerate}
          disabled={generating || !canGenerate}
          className={`fuk-btn fuk-btn-primary ${generating ? 'fuk-btn-generating' : ''}`}
        >
          {generating ? (
            <>
              <Loader2 
                style={{ 
                  width: '1.25rem', 
                  height: '1.25rem',
                  animation: 'spin 1s linear infinite'
                }} 
              />
              {generatingLabel}
            </>
          ) : (
            <>
              <FukMonogram style={{ width: '1.25rem', height: '1.25rem' }} />
              {generateLabel}
            </>
          )}
        </button>
        
        {generating && (
          <button
            onClick={onCancel}
            className="fuk-btn fuk-btn-secondary"
            style={{ paddingLeft: '1.5rem', paddingRight: '1.5rem' }}
          >
            <X style={{ width: '1rem', height: '1rem' }} />
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}