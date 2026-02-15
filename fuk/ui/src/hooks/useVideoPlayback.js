/**
 * Hook to sync video playback speed across all video elements
 * Usage: const videoRef = useVideoPlayback(playbackSpeed);
 */

import { useEffect, useRef } from 'react';

export function useVideoPlayback(playbackSpeed = 1.0) {
  const videoRef = useRef(null);
  
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.playbackRate = playbackSpeed;
    }
  }, [playbackSpeed]);
  
  return videoRef;
}
