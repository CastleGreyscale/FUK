/**
 * ZoomableImage Component
 * Hover to zoom - mouse position pans around zoomed image
 * Scroll wheel adjusts zoom level
 * 
 * IMPORTANT: This component wraps the image in a container that sizes
 * based on the image's natural dimensions, just like a regular <img>.
 * Pass className to apply the same styles you would to an <img>.
 */

import { useState, useRef, useCallback, useEffect } from 'react';

export default function ZoomableImage({ 
  src, 
  alt = '', 
  className = '',
  minZoom = 2,
  maxZoom = 6,
  defaultZoom = 3,
}) {
  const [isHovering, setIsHovering] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(defaultZoom);
  const [origin, setOrigin] = useState({ x: 50, y: 50 });
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  const handleMouseMove = useCallback((e) => {
    if (!containerRef.current || !imageRef.current) return;
    
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();
    
    // Calculate mouse position as percentage of container
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    
    // Clamp to 0-100
    setOrigin({
      x: Math.max(0, Math.min(100, x)),
      y: Math.max(0, Math.min(100, y)),
    });
  }, []);

  const handleMouseEnter = useCallback(() => {
    setIsHovering(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsHovering(false);
    setOrigin({ x: 50, y: 50 });
  }, []);

  const handleWheel = useCallback((e) => {
    if (!isHovering) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    const delta = e.deltaY > 0 ? -0.5 : 0.5;
    setZoomLevel(prev => Math.max(minZoom, Math.min(maxZoom, prev + delta)));
  }, [isHovering, minZoom, maxZoom]);

  const imageStyle = isHovering ? {
    transform: `scale(${zoomLevel})`,
    transformOrigin: `${origin.x}% ${origin.y}%`,
  } : {};

  return (
    <div 
      ref={containerRef}
      className={`zoomable-image-container ${isHovering ? 'zooming' : ''} ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onMouseMove={handleMouseMove}
      onWheel={handleWheel}
    >
      <img
        ref={imageRef}
        src={src}
        alt={alt}
        className="zoomable-image"
        style={imageStyle}
        draggable={false}
      />
      {isHovering && (
        <div className="zoomable-zoom-indicator">
          {zoomLevel.toFixed(1)}x
        </div>
      )}
    </div>
  );
}