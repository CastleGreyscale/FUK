/**
 * VideoSyncController Component
 * Unified playback controls for syncing multiple videos
 * Used in side-by-side comparison views (Preprocess, Postprocess tabs)
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, SkipBack, SkipForward } from './Icons';

export default function VideoSyncController({ 
  videoRefs = [],  // Array of refs to video elements
  className = '',
}) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isSeeking, setIsSeeking] = useState(false);
  const animationRef = useRef(null);

  // Get all valid video elements from refs
  const getVideos = useCallback(() => {
    return videoRefs
      .map(ref => ref?.current)
      .filter(video => video && video.readyState >= 1);
  }, [videoRefs]);

  // Update duration when videos load
  useEffect(() => {
    const videos = getVideos();
    if (videos.length === 0) return;

    const handleLoadedMetadata = () => {
      // Use the shortest duration among all videos
      const durations = videos.map(v => v.duration).filter(d => !isNaN(d) && d > 0);
      if (durations.length > 0) {
        setDuration(Math.min(...durations));
      }
    };

    // Check if already loaded
    handleLoadedMetadata();

    // Listen for metadata load
    videos.forEach(video => {
      video.addEventListener('loadedmetadata', handleLoadedMetadata);
    });

    return () => {
      videos.forEach(video => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      });
    };
  }, [getVideos]);

  // Sync time updates during playback
  useEffect(() => {
    if (!isPlaying) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    const updateTime = () => {
      const videos = getVideos();
      if (videos.length > 0) {
        setCurrentTime(videos[0].currentTime);
      }
      animationRef.current = requestAnimationFrame(updateTime);
    };

    animationRef.current = requestAnimationFrame(updateTime);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, getVideos]);

  // Handle play/pause
  const togglePlay = useCallback(() => {
    const videos = getVideos();
    if (videos.length === 0) return;

    if (isPlaying) {
      videos.forEach(video => video.pause());
      setIsPlaying(false);
    } else {
      // Sync all videos to the same time before playing
      const syncTime = videos[0].currentTime;
      videos.forEach(video => {
        video.currentTime = syncTime;
        video.play().catch(() => {});
      });
      setIsPlaying(true);
    }
  }, [isPlaying, getVideos]);

  // Handle seeking via scrubber
  const handleSeek = useCallback((e) => {
    const videos = getVideos();
    if (videos.length === 0 || duration === 0) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const newTime = percent * duration;

    setCurrentTime(newTime);
    videos.forEach(video => {
      video.currentTime = newTime;
    });
  }, [getVideos, duration]);

  const handleSeekStart = useCallback((e) => {
    setIsSeeking(true);
    const videos = getVideos();
    videos.forEach(video => video.pause());
    setIsPlaying(false);
    handleSeek(e);
  }, [getVideos, handleSeek]);

  const handleSeekMove = useCallback((e) => {
    if (!isSeeking) return;
    handleSeek(e);
  }, [isSeeking, handleSeek]);

  const handleSeekEnd = useCallback(() => {
    setIsSeeking(false);
  }, []);

  // Skip forward/backward
  const skip = useCallback((seconds) => {
    const videos = getVideos();
    if (videos.length === 0) return;

    const newTime = Math.max(0, Math.min(duration, currentTime + seconds));
    setCurrentTime(newTime);
    videos.forEach(video => {
      video.currentTime = newTime;
    });
  }, [getVideos, currentTime, duration]);

  // Handle video end - loop
  useEffect(() => {
    const videos = getVideos();
    if (videos.length === 0) return;

    const handleEnded = () => {
      // Loop back to start
      videos.forEach(video => {
        video.currentTime = 0;
      });
      setCurrentTime(0);
      // Continue playing
      if (isPlaying) {
        videos.forEach(video => video.play().catch(() => {}));
      }
    };

    videos.forEach(video => {
      video.addEventListener('ended', handleEnded);
    });

    return () => {
      videos.forEach(video => {
        video.removeEventListener('ended', handleEnded);
      });
    };
  }, [getVideos, isPlaying]);

  // Format time as MM:SS
  const formatTime = (time) => {
    if (isNaN(time)) return '0:00';
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className={`video-sync-controller ${className}`}>
      <div className="video-sync-controls">
        <button 
          className="video-sync-btn"
          onClick={() => skip(-2)}
          title="Back 2s"
        >
          <SkipBack className="video-sync-icon" />
        </button>
        
        <button 
          className="video-sync-btn video-sync-btn--primary"
          onClick={togglePlay}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <Pause className="video-sync-icon" />
          ) : (
            <Play className="video-sync-icon" />
          )}
        </button>
        
        <button 
          className="video-sync-btn"
          onClick={() => skip(2)}
          title="Forward 2s"
        >
          <SkipForward className="video-sync-icon" />
        </button>
      </div>
      
      <div 
        className="video-sync-scrubber"
        onMouseDown={handleSeekStart}
        onMouseMove={handleSeekMove}
        onMouseUp={handleSeekEnd}
        onMouseLeave={handleSeekEnd}
      >
        <div className="video-sync-track">
          <div 
            className="video-sync-progress"
            style={{ width: `${progress}%` }}
          />
          <div 
            className="video-sync-handle"
            style={{ left: `${progress}%` }}
          />
        </div>
      </div>
      
      <div className="video-sync-time">
        <span>{formatTime(currentTime)}</span>
        <span className="video-sync-time-sep">/</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}
