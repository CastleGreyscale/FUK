/**
 * Footer Component
 * Contains progress bar, tabs, and action buttons
 */

import { useState, useEffect, useCallback } from 'react';
import TabButton from './TabButton';
import ProgressBar from './ProgressBar';
import { Camera, Film, Pipeline, Enhance, Save, FukMonogram, Loader2, X, Layers, Wrench, Trash2, RefreshCw, SequenceIcon } from './Icons';

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
    <div className="fuk-footer">

      {/* System buttons - Left */}
      <div className="fuk-footer-system">
        <button
          onClick={handleEvict}
          disabled={evicting || restarting}
          className="fuk-btn fuk-btn-secondary fuk-btn-compact"
          title="Evict all models from VRAM"
        >
          {evicting
            ? <Loader2 className="animate-spin" />
            : <Trash2 />}
          Evict
        </button>
        <button
          onClick={handleRestart}
          disabled={evicting || restarting}
          className="fuk-btn fuk-btn-secondary fuk-btn-compact"
          title="Restart the server"
        >
          {restarting
            ? <Loader2 className="animate-spin" />
            : <RefreshCw />}
          {restarting ? 'Restarting...' : 'Restart'}
        </button>
      </div>

      {/* Tabs - Center */}
      <div className="fuk-footer-center">
        <div className="fuk-tabs">
          <TabButton active={activeTab === 'utilities'} onClick={() => setActiveTab('utilities')} icon={<Wrench />} label="Utilities" />
          <TabButton active={activeTab === 'storyboard'} onClick={() => setActiveTab('storyboard')} icon={<SequenceIcon />} label="Storyboard" />
          <TabButton active={activeTab === 'preprocess'} onClick={() => setActiveTab('preprocess')} icon={<Pipeline />} label="Pre-Processors" />
          <TabButton active={activeTab === 'image'} onClick={() => setActiveTab('image')} icon={<Camera />} label="Image" />
          <TabButton active={activeTab === 'video'} onClick={() => setActiveTab('video')} icon={<Film />} label="Video" />
          <TabButton active={activeTab === 'postprocess'} onClick={() => setActiveTab('postprocess')} icon={<Enhance />} label="Post-Processors" />
          <TabButton active={activeTab === 'layers'} onClick={() => setActiveTab('layers')} icon={<Layers />} label="Layers" />
          <TabButton active={activeTab === 'export'} onClick={() => setActiveTab('export')} icon={<Save />} label="Exports" />
        </div>
      </div>

      {/* Buttons - Right */}
      <div className="fuk-footer-right">
        {/* Batch Count */}
        {onBatchCountChange && (
          <div className="fuk-batch-counter">
            <span className="fuk-batch-multiplier">×</span>
            <input
              type="number"
              min="1"
              max="50"
              value={batchInput}
              onChange={e => setBatchInput(e.target.value)}
              onBlur={e => commitBatch(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && commitBatch(e.target.value)}
              disabled={generating}
              className="fuk-input fuk-batch-input"
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
              <Loader2 className="animate-spin" />
              {generatingLabel}
            </>
          ) : (
            <>
              <FukMonogram />
              {generateLabel}
            </>
          )}
        </button>

        {generating && (
          <button onClick={onCancel} className="fuk-btn fuk-btn-secondary fuk-btn-cancel">
            <X />
            Cancel
          </button>
        )}
      </div>
    </div>
  );
}