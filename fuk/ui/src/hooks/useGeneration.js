/**
 * useGeneration Hook
 * Shared logic for image/video generation with progress tracking
 * Includes console log capture for the generation modal
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { createProgressStream, cancelGeneration } from '../utils/api';

export function useGeneration() {
  const [generating, setGenerating] = useState(false);
  const [generationId, setGenerationId] = useState(null);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [consoleLog, setConsoleLog] = useState([]);
  const [showModal, setShowModal] = useState(false);
  
  // Ref to track if component is mounted
  const mountedRef = useRef(true);
  // Ref to track start time (using ref to avoid dependency issues)
  const startTimeRef = useRef(null);
  // Ref to track timer interval
  const timerRef = useRef(null);
  // Ref to track last console log length (to avoid duplicate processing)
  const lastLogLengthRef = useRef(0);
  
  useEffect(() => {
    mountedRef.current = true;
    return () => { 
      mountedRef.current = false;
      // Clean up timer on unmount
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  // Start/stop timer based on generating state
  useEffect(() => {
    if (generating && startTimeRef.current) {
      // Start the timer
      console.log('[useGeneration] Starting timer');
      timerRef.current = setInterval(() => {
        if (mountedRef.current && startTimeRef.current) {
          const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
          setElapsedSeconds(elapsed);
        }
      }, 1000);
      
      return () => {
        console.log('[useGeneration] Stopping timer');
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      };
    }
  }, [generating]);

  // SSE progress stream
  useEffect(() => {
    if (!generationId) return;
    
    console.log(`[useGeneration] Starting progress stream for ${generationId}`);
    
    let eventSource = null;
    
    try {
      eventSource = createProgressStream(
        generationId,
        // onProgress
        (data) => {
          if (!mountedRef.current) return;
          console.log('[useGeneration] Progress:', data.phase, `${Math.round((data.progress || 0) * 100)}%`);
          console.log('[useGeneration] console_log length:', data.console_log?.length || 0);  // ADD THIS
          setProgress(data);
          
          // Update console log if new entries
          if (data.console_log && data.console_log.length > lastLogLengthRef.current) {
            console.log('[useGeneration] Updating consoleLog, new entries:', data.console_log.length - lastLogLengthRef.current);  // ADD THIS
            setConsoleLog(data.console_log);
            lastLogLengthRef.current = data.console_log.length;
          }
        },
        // onComplete
        (data) => {
          if (!mountedRef.current) return;
          const finalElapsed = startTimeRef.current 
            ? Math.floor((Date.now() - startTimeRef.current) / 1000)
            : 0;
          console.log('[useGeneration] Complete:', data, `(${finalElapsed}s)`);
          setElapsedSeconds(finalElapsed);  // Set final elapsed time
          setResult(data);
          setGenerating(false);
          
          // Update final console log
          if (data.console_log) {
            setConsoleLog(data.console_log);
          }
          
          // Auto-close modal on success after a brief delay
          setTimeout(() => {
            if (mountedRef.current && !data.error) {
              setShowModal(false);
            }
          }, 1500);
          
          // Notify history to refresh
          window.dispatchEvent(new CustomEvent('fuk-generation-complete', {
            detail: { generationId, result: data, elapsed: finalElapsed }
          }));
        },
        // onError
        (errorMsg) => {
          if (!mountedRef.current) return;
          console.log('[useGeneration] Error:', errorMsg);
          setError(errorMsg);
          setGenerating(false);
          // Keep modal open on error
        }
      );
    } catch (err) {
      console.error('[useGeneration] Failed to create SSE stream:', err);
      setError('Failed to connect to server');
      setGenerating(false);
    }
    
    return () => {
      console.log('[useGeneration] Closing connection');
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [generationId]);

  const startGeneration = useCallback((id) => {
    console.log('[useGeneration] Starting generation:', id);
    startTimeRef.current = Date.now();  // Set start time using ref
    lastLogLengthRef.current = 0;  // Reset log length tracker
    setGenerating(true);
    setProgress(null);
    setResult(null);
    setError(null);
    setElapsedSeconds(0);
    setConsoleLog([]);
    setShowModal(true);  // Open modal when generation starts
    setGenerationId(id);
  }, []);

  const cancel = useCallback(async () => {
    if (!generationId) return;
    
    console.log('[useGeneration] Cancelling:', generationId);
    try {
      await cancelGeneration(generationId);
      setGenerating(false);
      setProgress(null);
      setGenerationId(null);
      setError('Generation cancelled');
      // Keep modal open briefly to show cancellation
    } catch (err) {
      console.error('Cancel failed:', err);
    }
  }, [generationId]);

  const closeModal = useCallback(() => {
    setShowModal(false);
  }, []);

  const reset = useCallback(() => {
    setGenerating(false);
    setGenerationId(null);
    setProgress(null);
    setResult(null);
    setError(null);
    setConsoleLog([]);
    setShowModal(false);
    startTimeRef.current = null;
    lastLogLengthRef.current = 0;
    setElapsedSeconds(0);
  }, []);

  return {
    generating,
    generationId,
    progress,
    result,
    error,
    elapsedSeconds,
    consoleLog,
    showModal,
    startGeneration,
    cancel,
    closeModal,
    reset,
  };
}