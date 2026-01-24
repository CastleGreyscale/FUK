/**
 * ZoomableImage Component
 * 
 * SIMPLE CONCEPT:
 * - Normal state: Image behaves exactly like a regular <img> with your className
 * - Hover state: Lock the container to current image size, then zoom within it
 * 
 * EVENT HANDLING:
 * - Normal: mouseEnter on image (container is display:contents, no box)
 * - Zooming: mouseLeave/mouseMove on container (has clipped bounds)
 * This prevents the flicker when cursor moves over the clipped overflow.
 */

import { useState, useRef, useCallback } from 'react';

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
  const [lockedSize, setLockedSize] = useState(null);
  const containerRef = useRef(null);
  const imageRef = useRef(null);

  // When zooming, track mouse on the CONTAINER (clipped bounds)
  const handleContainerMouseMove = useCallback((e) => {
    const container = containerRef.current;
    if (!container) return;
    
    const rect = container.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    
    setOrigin({
      x: Math.max(0, Math.min(100, x)),
      y: Math.max(0, Math.min(100, y)),
    });
  }, []);

  // Enter via the image (only element with a box in normal state)
  const handleMouseEnter = useCallback(() => {
    const img = imageRef.current;
    if (img) {
      setLockedSize({
        width: img.offsetWidth,
        height: img.offsetHeight,
      });
    }
    setIsHovering(true);
  }, []);

  // Leave via the container when zooming (respects clipped bounds)
  const handleMouseLeave = useCallback(() => {
    setIsHovering(false);
    setLockedSize(null);
    setOrigin({ x: 50, y: 50 });
  }, []);

  const handleWheel = useCallback((e) => {
    if (!isHovering) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    const delta = e.deltaY > 0 ? -0.5 : 0.5;
    setZoomLevel(prev => Math.max(minZoom, Math.min(maxZoom, prev + delta)));
  }, [isHovering, minZoom, maxZoom]);

  // Container style: invisible normally, sized box when zooming
  const containerStyle = isHovering && lockedSize ? {
    width: lockedSize.width,
    height: lockedSize.height,
  } : {};

  // Image style: normal sizing, or transformed when zooming
  const imageStyle = isHovering ? {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
    transform: `scale(${zoomLevel})`,
    transformOrigin: `${origin.x}% ${origin.y}%`,
  } : {};

  return (
    <div 
      ref={containerRef}
      className={`zoomable-container ${isHovering ? 'zoomable-container--zooming' : ''}`}
      style={containerStyle}
      // When zooming, container handles leave/move (has clipped bounds)
      onMouseLeave={isHovering ? handleMouseLeave : undefined}
      onMouseMove={isHovering ? handleContainerMouseMove : undefined}
      onWheel={isHovering ? handleWheel : undefined}
    >
      <img
        ref={imageRef}
        src={src}
        alt={alt}
        className={`zoomable-img ${className}`}
        style={imageStyle}
        draggable={false}
        // Image only handles enter (it's the only box in normal state)
        onMouseEnter={!isHovering ? handleMouseEnter : undefined}
      />
      {isHovering && (
        <div className="zoomable-indicator">
          {zoomLevel.toFixed(1)}x
        </div>
      )}
    </div>
  );
}