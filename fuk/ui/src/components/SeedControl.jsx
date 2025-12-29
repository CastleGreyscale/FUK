/**
 * SeedControl Component
 * Manages seed mode (fixed/random/increment), input, and favorites
 */

import { useState, useMemo } from 'react';
import { SEED_MODES, generateRandomSeed } from '../utils/constants';

// Star icon for favorites
const Star = ({ filled, className, style, ...props }) => (
  <svg 
    className={className} 
    style={style} 
    fill={filled ? 'currentColor' : 'none'} 
    stroke="currentColor" 
    viewBox="0 0 24 24"
    {...props}
  >
    <path 
      strokeLinecap="round" 
      strokeLinejoin="round" 
      strokeWidth={2} 
      d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" 
    />
  </svg>
);

// Dice icon for random
const Dice = ({ className, style }) => (
  <svg className={className} style={style} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth={2} />
    <circle cx="8" cy="8" r="1.5" fill="currentColor" />
    <circle cx="16" cy="16" r="1.5" fill="currentColor" />
    <circle cx="12" cy="12" r="1.5" fill="currentColor" />
  </svg>
);

export default function SeedControl({
  seed,
  seedMode,
  lastUsedSeed,
  model,
  savedSeeds,
  isSeedSaved,
  onSeedChange,
  onSeedModeChange,
  onSaveSeed,
  onRemoveSeed,
  onSelectSavedSeed,
  disabled = false,
}) {
  const [showNoteInput, setShowNoteInput] = useState(false);
  const [noteText, setNoteText] = useState('');

  // Get seeds for current model
  const modelSeeds = useMemo(() => {
    return savedSeeds || [];
  }, [savedSeeds]);

  // Current seed to display
  const displaySeed = useMemo(() => {
    if (seedMode === SEED_MODES.RANDOM) {
      return lastUsedSeed !== null ? lastUsedSeed : '(random)';
    }
    return seed !== null ? seed : '';
  }, [seed, seedMode, lastUsedSeed]);

  // Get seed value for saving/checking
  const currentSeedValue = useMemo(() => {
    return seedMode === SEED_MODES.RANDOM ? lastUsedSeed : seed;
  }, [seed, seedMode, lastUsedSeed]);

  // Check if current seed is already saved
  const currentSeedSaved = useMemo(() => {
    return currentSeedValue !== null && isSeedSaved(model, currentSeedValue);
  }, [currentSeedValue, model, isSeedSaved]);

  // Handle save with optional note
  const handleSave = () => {
    if (currentSeedValue !== null) {
      onSaveSeed(currentSeedValue, noteText);
      setNoteText('');
      setShowNoteInput(false);
    }
  };

  // Generate new random seed
  const handleRandomize = () => {
    const newSeed = generateRandomSeed();
    onSeedChange(newSeed);
  };

  // Handle dropdown selection - set seed AND switch to fixed mode
  const handleSelectSaved = (seedValue) => {
    onSeedChange(seedValue);
    onSeedModeChange(SEED_MODES.FIXED);
  };

  return (
    <div className="fuk-seed-control">
      {/* Seed Mode Selector */}
      <div className="fuk-form-group-compact">
        <label className="fuk-label">Seed Mode</label>
        <div className="fuk-seed-mode-group">
          <label className="fuk-seed-mode-option">
            <input
              type="radio"
              className="fuk-radio"
              checked={seedMode === SEED_MODES.RANDOM}
              onChange={() => onSeedModeChange(SEED_MODES.RANDOM)}
              disabled={disabled}
            />
            <span className="fuk-seed-mode-label">Random</span>
          </label>
          
          <label className="fuk-seed-mode-option">
            <input
              type="radio"
              className="fuk-radio"
              checked={seedMode === SEED_MODES.FIXED}
              onChange={() => onSeedModeChange(SEED_MODES.FIXED)}
              disabled={disabled}
            />
            <span className="fuk-seed-mode-label">Fixed</span>
          </label>
          
          <label className="fuk-seed-mode-option">
            <input
              type="radio"
              className="fuk-radio"
              checked={seedMode === SEED_MODES.INCREMENT}
              onChange={() => onSeedModeChange(SEED_MODES.INCREMENT)}
              disabled={disabled}
            />
            <span className="fuk-seed-mode-label">Increment</span>
          </label>
        </div>
      </div>

      {/* Seed Value */}
      <div className="fuk-form-group-compact">
        <label className="fuk-label">
          {seedMode === SEED_MODES.RANDOM ? 'Last Used Seed' : 'Seed Value'}
        </label>
        <div className="fuk-seed-input-row">
          <input
            type="number"
            className="fuk-input"
            style={{ flex: 1 }}
            value={typeof displaySeed === 'number' ? displaySeed : ''}
            onChange={(e) => {
              const val = e.target.value ? parseInt(e.target.value) : null;
              onSeedChange(val);
            }}
            placeholder={seedMode === SEED_MODES.RANDOM ? 'Auto-generated' : 'Enter seed'}
            disabled={disabled || seedMode === SEED_MODES.RANDOM}
          />
          
          {/* Randomize button (for fixed/increment modes) */}
          {seedMode !== SEED_MODES.RANDOM && (
            <button
              type="button"
              className="fuk-seed-btn fuk-seed-btn-dice"
              onClick={handleRandomize}
              disabled={disabled}
              title="Generate random seed"
            >
              <Dice style={{ width: '1rem', height: '1rem' }} />
            </button>
          )}
          
          {/* Save to favorites button */}
          <button
            type="button"
            className={`fuk-seed-btn ${currentSeedSaved ? 'fuk-seed-btn-saved' : 'fuk-seed-btn-save'}`}
            onClick={() => {
              if (currentSeedSaved) {
                onRemoveSeed(model, currentSeedValue);
              } else {
                setShowNoteInput(!showNoteInput);
              }
            }}
            disabled={disabled || currentSeedValue === null}
            title={currentSeedSaved ? 'Remove from favorites' : 'Save to favorites'}
          >
            <Star 
              filled={currentSeedSaved} 
              style={{ width: '1rem', height: '1rem' }} 
            />
          </button>
        </div>
      </div>

      {/* Note input for saving */}
      {showNoteInput && !currentSeedSaved && (
        <div className="fuk-seed-note-input">
          <input
            type="text"
            className="fuk-input"
            value={noteText}
            onChange={(e) => setNoteText(e.target.value)}
            placeholder="Add a note (optional)"
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSave();
              if (e.key === 'Escape') {
                setShowNoteInput(false);
                setNoteText('');
              }
            }}
            autoFocus
          />
          <button
            type="button"
            className="fuk-btn fuk-btn-secondary fuk-btn-sm"
            onClick={handleSave}
          >
            Save
          </button>
          <button
            type="button"
            className="fuk-btn fuk-btn-secondary fuk-btn-sm"
            onClick={() => {
              setShowNoteInput(false);
              setNoteText('');
            }}
          >
            Cancel
          </button>
        </div>
      )}

      {/* Saved Seeds Dropdown */}
      {modelSeeds.length > 0 && (
        <div className="fuk-form-group-compact">
          <label className="fuk-label">
            Saved Seeds 
            <span className="fuk-label-description">({modelSeeds.length})</span>
          </label>
          <div className="fuk-seed-saved-container">
            <select
              className="fuk-select"
              value=""
              onChange={(e) => {
                if (e.target.value) {
                  handleSelectSaved(parseInt(e.target.value));
                }
              }}
              disabled={disabled}
            >
              <option value="">Select a saved seed...</option>
              {modelSeeds.map((entry) => (
                <option key={entry.seed} value={entry.seed}>
                  {entry.seed}{entry.note ? ` - ${entry.note}` : ''}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Mode description */}
      <p className="fuk-seed-description">
        {seedMode === SEED_MODES.RANDOM && 'New random seed each generation'}
        {seedMode === SEED_MODES.FIXED && 'Use the same seed every time'}
        {seedMode === SEED_MODES.INCREMENT && 'Seed +1 after each generation'}
      </p>
    </div>
  );
}