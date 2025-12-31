/**
 * Footer Component
 * Contains progress bar, tabs, and action buttons
 */

import TabButton from './TabButton';
import ProgressBar from './ProgressBar';
import { Camera, Film, Pipeline, Enhance, Save, Zap, Loader2, X } from './Icons';

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
}) {
  return (
    <div className="fuk-footer" style={{ position: 'relative' }}>
      {/* Progress Bar - Left */}
      <div style={{ flex: 1, maxWidth: '400px' }}>
        <ProgressBar
          progress={progress}
          elapsedSeconds={elapsedSeconds}
          generating={generating}
        />
      </div>

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
            icon={<Enhance style={{ width: '1rem', height: '1rem' }} />}
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
      <div className="fuk-footer-right" style={{ display: 'flex', gap: '0.75rem' }}>
        <button
          onClick={onGenerate}
          disabled={generating || !canGenerate}
          className="fuk-btn fuk-btn-primary"
        >
          {generating ? (
            <>
              <Loader2 className="animate-spin" style={{ width: '1.25rem', height: '1.25rem' }} />
              {generatingLabel}
            </>
          ) : (
            <>
              <Zap style={{ width: '1.25rem', height: '1.25rem' }} />
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
