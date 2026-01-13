/**
 * Progress Bar Component
 */

import { formatTime } from '../utils/helpers';

export default function ProgressBar({ progress, elapsedSeconds, generating }) {
  if (!generating && !progress) {
    return (
      <div className="fuk-progress-ready">
        Ready to generate
      </div>
    );
  }

  return (
    <div className="fuk-progress-compact">
      <div className="fuk-progress-header">
        <span className="fuk-progress-title">
          {progress?.phase || 'Starting...'}
        </span>
        <span className="fuk-progress-time">
          {formatTime(elapsedSeconds)}
        </span>
      </div>
      
      <div className="fuk-progress-bar-container">
        <div
          className="fuk-progress-bar"
          style={{ width: `${(progress?.progress || 0) * 100}%` }}
        />
      </div>
      
      <div className="fuk-progress-text">
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
