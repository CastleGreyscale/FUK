/**
 * GenerationModal
 * Full-screen modal that displays during generation with live console output
 * - Shows terminal-style console output streaming in real-time
 * - Progress bar and elapsed time at top
 * - Cancel button during generation
 * - Auto-closes on success, stays open with error styling on failure
 */

import { useEffect, useRef } from 'react';
import { X, Square } from '../../src/components/Icons';
import ProgressBar from './ProgressBar';
import { formatTime } from '../utils/helpers';

export default function GenerationModal({
  isOpen,
  type = 'image', // 'image' or 'video'
  generating,
  progress,
  elapsedSeconds,
  consoleLog = [],
  error,
  onCancel,
  onClose,
}) {
  const consoleEndRef = useRef(null);
  const consoleContainerRef = useRef(null);

  // Auto-scroll to bottom when new logs come in
  useEffect(() => {
    if (consoleEndRef.current && consoleContainerRef.current) {
      const container = consoleContainerRef.current;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;
      
      // Only auto-scroll if user is near the bottom
      if (isNearBottom) {
        consoleEndRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }, [consoleLog]);

  if (!isOpen) return null;

  const phase = progress?.phase || 'initializing';
  const progressValue = progress?.progress || 0;
  const currentStep = progress?.current_step || 0;
  const totalSteps = progress?.total_steps || 0;

  // Determine status
  const isComplete = !generating && !error;
  const isFailed = !generating && error;

  // Get level-based styling for log lines
  const getLevelClass = (level) => {
    switch (level) {
      case 'error': return 'gen-modal-log-error';
      case 'warning': return 'gen-modal-log-warning';
      case 'success': return 'gen-modal-log-success';
      case 'timing': return 'gen-modal-log-timing';
      case 'header': return 'gen-modal-log-header';
      case 'section': return 'gen-modal-log-section';
      case 'params':
      case 'paths': return 'gen-modal-log-dim';
      default: return '';
    }
  };

  return (
    <div className={`gen-modal-overlay ${isFailed ? 'gen-modal-overlay--error' : ''}`}>
      <div className={`gen-modal ${isFailed ? 'gen-modal--error' : ''}`}>
        {/* Header */}
        <div className="gen-modal-header">
          <div className="gen-modal-title">
            <span className="gen-modal-title-icon">
              {generating && <span className="gen-modal-spinner" />}
              {isComplete && <span className="gen-modal-success-icon">✓</span>}
              {isFailed && <span className="gen-modal-error-icon">✗</span>}
            </span>
            <span>
              {generating && `Generating ${type}...`}
              {isComplete && `${type.charAt(0).toUpperCase() + type.slice(1)} Complete`}
              {isFailed && `${type.charAt(0).toUpperCase() + type.slice(1)} Failed`}
            </span>
          </div>
          
          {/* Close button - only when not generating */}
          {!generating && (
            <button 
              className="gen-modal-close"
              onClick={onClose}
              title="Close"
            >
              <X size={20} />
            </button>
          )}
        </div>

        {/* Progress section */}
        <div className="gen-modal-progress">
          <div className="gen-modal-progress-info">
            <span className="gen-modal-phase">{phase}</span>
            <span className="gen-modal-time">{formatTime(elapsedSeconds)}</span>
          </div>
          <div className="gen-modal-progress-bar">
            <div 
              className={`gen-modal-progress-fill ${isFailed ? 'gen-modal-progress-fill--error' : ''}`}
              style={{ width: `${Math.round(progressValue * 100)}%` }}
            />
          </div>
          {totalSteps > 0 && (
            <div className="gen-modal-steps">
              Step {currentStep} / {totalSteps}
            </div>
          )}
        </div>

        {/* Console output */}
        <div className="gen-modal-console" ref={consoleContainerRef}>
          {consoleLog.length === 0 ? (
            <div className="gen-modal-console-empty">
              Waiting for output...
            </div>
          ) : (
            consoleLog.map((entry, idx) => (
              <div 
                key={idx} 
                className={`gen-modal-log-line ${getLevelClass(entry.level)}`}
              >
                {entry.line}
              </div>
            ))
          )}
          <div ref={consoleEndRef} />
        </div>

        {/* Error message */}
        {error && (
          <div className="gen-modal-error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Footer with actions */}
        <div className="gen-modal-footer">
          {generating ? (
            <button 
              className="gen-modal-btn gen-modal-btn--cancel"
              onClick={onCancel}
            >
              <Square size={16} />
              Cancel
            </button>
          ) : (
            <button 
              className="gen-modal-btn gen-modal-btn--close"
              onClick={onClose}
            >
              {isFailed ? 'Close' : 'Done'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
