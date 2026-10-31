/**
 * useProgressiveSrc Hook
 * Show the cached 1920px preview of a project image, and swap in the original
 * only when `wantFull` is set (the user is zooming in). The original is
 * fetched in the background and swapped in once loaded, so the view never
 * blanks. URLs the backend can't downscale are returned unchanged.
 */

import { useState, useEffect } from 'react';
import { sizedImageUrl } from '../utils/constants';

export function useProgressiveSrc(src, wantFull) {
  const previewSrc = sizedImageUrl(src, 'preview');
  const [fullLoadedFor, setFullLoadedFor] = useState(null);
  const fullReady = fullLoadedFor === src;

  useEffect(() => {
    if (!wantFull || fullReady || previewSrc === src) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => { if (!cancelled) setFullLoadedFor(src); };
    img.src = src;
    return () => { cancelled = true; };
  }, [wantFull, fullReady, src, previewSrc]);

  return fullReady ? src : previewSrc;
}

export default useProgressiveSrc;
