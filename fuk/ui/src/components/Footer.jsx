/**
 * Footer Component
 * Contains progress bar, tabs, and action buttons
 */

import TabButton from './TabButton';
import ProgressBar from './ProgressBar';
import { Camera, Film, Pipeline, Enhance, Save, FukMonogram, Loader2, X, Layers } from './Icons';

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
  return (
    <div className="fuk-footer" style={{ position: 'relative' }}>

      {/* Tabs - Center */}
      <div className="fuk-footer-center">
        <div className="fuk-tabs">
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
              value={batchCount ?? 1}
              onChange={e => onBatchCountChange(Math.max(1, Math.min(50, parseInt(e.target.value) || 1)))}
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