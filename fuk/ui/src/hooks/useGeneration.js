/**
 * useGeneration Hook
 * Shared logic for image/video generation with progress tracking
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { createProgressStream, cancelGeneration } from '../utils/api';

export function useGeneration() {
  const [generating, setGenerating] = useState(false);
  const [generationId, setGenerationId] = useState(null);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [startTime, setStartTime] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  
  // Ref to track if component is mounted
  const mountedRef = useRef(true);
  
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Timer for elapsed time
  useEffect(() => {
    if (!generating || !startTime) return;
    
    const interval = setInterval(() => {
      if (mountedRef.current) {
        setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [generating, startTime]);

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
          setProgress(data);
        },
        // onComplete
        (data) => {
          if (!mountedRef.current) return;
          console.log('[useGeneration] Complete:', data);
          setResult(data);
          setGenerating(false);
        },
        // onError
        (errorMsg) => {
          if (!mountedRef.current) return;
          console.log('[useGeneration] Error:', errorMsg);
          setError(errorMsg);
          setGenerating(false);
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
    setGenerating(true);
    setProgress(null);
    setResult(null);
    setError(null);
    setStartTime(Date.now());
    setElapsedSeconds(0);
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
    } catch (err) {
      console.error('Cancel failed:', err);
    }
  }, [generationId]);

  const reset = useCallback(() => {
    setGenerating(false);
    setGenerationId(null);
    setProgress(null);
    setResult(null);
    setError(null);
    setStartTime(null);
    setElapsedSeconds(0);
  }, []);

  return {
    generating,
    generationId,
    progress,
    result,
    error,
    elapsedSeconds,
    startGeneration,
    cancel,
    reset,
  };
}
