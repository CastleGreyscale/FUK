/**
 * Progress Bar Component
 */

import { formatTime } from '../utils/helpers';

export default function ProgressBar({ progress, elapsedSeconds, generating }) {
  if (!generating && !progress) {
    return (
      <div style={{ color: '#6b7280', fontSize: '0.75rem', padding: '0.5rem 0' }}>
        Ready to generate
      </div>
    );
  }

  return (
    <div className="fuk-progress-compact">
      <div className="fuk-progress-header" style={{ marginBottom: '0.375rem' }}>
        <span className="fuk-progress-title" style={{ fontSize: '0.75rem' }}>
          {progress?.phase || 'Starting...'}
        </span>
        <span className="fuk-progress-time" style={{ fontSize: '0.75rem' }}>
          {formatTime(elapsedSeconds)}
        </span>
      </div>
      
      <div className="fuk-progress-bar-container" style={{ height: '0.25rem', marginBottom: '0.25rem' }}>
        <div
          className="fuk-progress-bar"
          style={{ width: `${(progress?.progress || 0) * 100}%` }}
        />
      </div>
      
      <div className="fuk-progress-text" style={{ fontSize: '0.65rem' }}>
        {progress?.current_step && progress?.total_steps ? (
          <span className="fuk-progress-steps">
            Step {progress.current_step}/{progress.total_steps}
          </span>
        ) : (
          <span>Initializing...</span>
        )}
        <span>{Math.round((progress?.progress || 0) * 100)}%</span>
      </div>
    </div>
  );
}
