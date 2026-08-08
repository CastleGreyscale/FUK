/**
 * MeshViewer
 * Three.js viewport for reviewing reconstructed proxy geometry.
 *
 * Deliberately minimal — this is a proxy review tool, not a material editor.
 * Matcap shading reads form better than any quick PBR setup would, and needs
 * no lights or environment maps to configure.
 *
 * Controls: left-drag orbit, scroll zoom, right-drag pan.
 */

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';

// ============================================================================
// Matcap — generated rather than shipped, so there's no binary asset to host
// ============================================================================

function createMatcapTexture() {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');

  // Offset highlight reads as a light above and to the left, which is what
  // makes surface curvature legible on untextured geometry.
  const gradient = ctx.createRadialGradient(
    size * 0.35, size * 0.3, size * 0.05,
    size * 0.5, size * 0.5, size * 0.55,
  );
  gradient.addColorStop(0.0, '#ffffff');
  gradient.addColorStop(0.25, '#d8dde3');
  gradient.addColorStop(0.6, '#8a9099');
  gradient.addColorStop(0.85, '#4a4f57');
  gradient.addColorStop(1.0, '#23262b');

  ctx.fillStyle = '#1a1c20';
  ctx.fillRect(0, 0, size, size);
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2);
  ctx.fillStyle = gradient;
  ctx.fill();

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

// ============================================================================
// Gaussian splat PLY
// ============================================================================

// Zeroth-order spherical harmonic, the constant 3DGS uses to turn its DC
// coefficients back into linear colour.
const SH_C0 = 0.28209479177387814;

/**
 * Give a 3D Gaussian Splatting PLY a usable `color` attribute.
 *
 * TRELLIS writes the standard 3DGS layout: colour lives in `f_dc_0..2` as
 * spherical-harmonic DC terms and opacity is stored pre-sigmoid. PLYLoader
 * looks for `red`/`green`/`blue`, finds nothing, and hands back a colourless
 * cloud — so pull the real properties across and decode them.
 */
function decodeSplatColor(geometry) {
  const sh = geometry.getAttribute('splatSH');
  if (!sh) return false;

  const opacity = geometry.getAttribute('splatOpacity');
  const colors = new Float32Array(sh.count * 3);

  for (let i = 0; i < sh.count; i += 1) {
    // Near-transparent gaussians barely register in a real splat render.
    // Scaling colour by alpha sinks them toward the dark background rather
    // than letting them read as solid surface points they aren't.
    const alpha = opacity ? 1 / (1 + Math.exp(-opacity.getX(i))) : 1;
    for (let c = 0; c < 3; c += 1) {
      const value = 0.5 + SH_C0 * sh.getComponent(i, c);
      colors[i * 3 + c] = Math.min(1, Math.max(0, value)) * alpha;
    }
  }

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return true;
}

// ============================================================================
// Baked-appearance materials
// ============================================================================

// Light rig at brightness 1.0, tuned so a white albedo lands just under clip
// facing the key. The previous 1.6/1.8 pair was compensating for the metalness
// bug below and blows out once the material is corrected. Only textured meshes
// see these — matcap and point clouds are unlit.
const BASE_AMBIENT = 0.35;
const BASE_KEY = 0.8;
const BASE_FILL = 0.3;

/**
 * Make a reconstructed GLB's material renderable without an environment map.
 *
 * TRELLIS bakes appearance into a `baseColorTexture` but never writes
 * `metallicFactor`, and the glTF default is 1.0 — fully metallic. A fully
 * rough metal has no diffuse response and draws its specular entirely from
 * the environment, so in a scene with no env map it renders near-black
 * however hard the lights are driven. The texture is plain baked albedo, so
 * the mesh is never metal and the fix is to say so.
 */
function normaliseBakedMaterial(root) {
  root.traverse((child) => {
    if (!child.isMesh || !child.material) return;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => {
      if (material.metalness === undefined) return;
      material.metalness = 0;
      material.needsUpdate = true;
    });
  });
}

// ============================================================================
// Scene helpers
// ============================================================================

/** Fit the camera to the object and re-centre the orbit target on it. */
function frameObject(object, camera, controls) {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.5 || 1;

  const distance = (radius / Math.sin((camera.fov * Math.PI) / 360)) * 1.6;
  camera.position.set(
    center.x + distance * 0.6,
    center.y + distance * 0.45,
    center.z + distance * 0.8,
  );
  camera.near = Math.max(distance / 1000, 0.001);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}

/** Free every GPU resource an object owns. */
function disposeObject(object) {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => {
        Object.values(material).forEach((value) => {
          if (value && value.isTexture) value.dispose();
        });
        material.dispose();
      });
    }
  });
}

export default function MeshViewer({
  url,
  displayMode = 'solid',   // solid | wireframe | points
  plyKind = 'points',      // points | splat — what a .ply url actually holds
  brightness = 1,          // preview exposure, cosmetic only
  pointSize = 0.004,
  background = '#15171a',
  className = '',
}) {
  const mountRef = useRef(null);
  const stateRef = useRef({});
  const [status, setStatus] = useState({ phase: 'idle', message: '' });
  const [stats, setStats] = useState(null);

  // --- One-time scene setup -------------------------------------------------
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(background);

    const camera = new THREE.PerspectiveCamera(
      45,
      mount.clientWidth / Math.max(mount.clientHeight, 1),
      0.01,
      1000,
    );
    camera.position.set(2, 1.5, 2.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;
    controls.panSpeed = 0.8;
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };

    // Matcap needs no lights, but GLBs carrying their own PBR materials
    // (TRELLIS bakes a texture) still want something to light them. The fill
    // keeps the side facing away from the key off pure black.
    const ambient = new THREE.AmbientLight(0xffffff, BASE_AMBIENT);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, BASE_KEY);
    key.position.set(3, 5, 4);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, BASE_FILL);
    fill.position.set(-4, 1, -3);
    scene.add(fill);

    const grid = new THREE.GridHelper(4, 16, 0x3a3f47, 0x24272c);
    grid.material.transparent = true;
    grid.material.opacity = 0.35;
    scene.add(grid);

    const matcap = createMatcapTexture();

    let frameId;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const resize = () => {
      if (!mount.clientWidth || !mount.clientHeight) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    stateRef.current = {
      scene, camera, renderer, controls, matcap, grid,
      ambient, key, fill, model: null,
    };

    return () => {
      cancelAnimationFrame(frameId);
      observer.disconnect();
      if (stateRef.current.model) disposeObject(stateRef.current.model);
      grid.geometry.dispose();
      grid.material.dispose();
      matcap.dispose();
      renderer.dispose();
      // dispose() frees GPU resources but keeps the WebGL context alive.
      // This component remounts every time the Utilities sub-tab changes, and
      // browsers hard-cap live contexts (~16) — without this, enough tab
      // switching silently kills the oldest contexts and the viewport goes
      // black or the tab stalls.
      renderer.forceContextLoss?.();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      stateRef.current = {};
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Brightness -----------------------------------------------------------
  // Textured meshes are lit, so they scale with the rig. Matcap is unlit, so
  // it scales through its material tint instead — otherwise the control would
  // do nothing on VGGT's untextured geometry. Purely a preview setting;
  // nothing here touches the exported file.
  useEffect(() => {
    const { ambient, key, fill, model } = stateRef.current;
    if (!ambient) return;

    ambient.intensity = BASE_AMBIENT * brightness;
    key.intensity = BASE_KEY * brightness;
    fill.intensity = BASE_FILL * brightness;

    model?.traverse((child) => {
      const material = child.userData?.matcapMaterial;
      if (material) material.color.setScalar(brightness);
    });
  }, [brightness, stats]);

  // --- Background can change without rebuilding the scene -------------------
  useEffect(() => {
    const { scene } = stateRef.current;
    if (scene) scene.background = new THREE.Color(background);
  }, [background]);

  // --- Load the asset -------------------------------------------------------
  useEffect(() => {
    const { scene, camera, controls, matcap } = stateRef.current;
    if (!scene || !url) {
      if (scene && stateRef.current.model) {
        scene.remove(stateRef.current.model);
        disposeObject(stateRef.current.model);
        stateRef.current.model = null;
        setStats(null);
      }
      return;
    }

    let cancelled = false;
    setStatus({ phase: 'loading', message: 'Loading geometry…' });

    const isPly = url.toLowerCase().split('?')[0].endsWith('.ply');
    const isSplat = isPly && plyKind === 'splat';
    const loader = isPly ? new PLYLoader() : new GLTFLoader();

    if (isSplat) {
      loader.setCustomPropertyNameMapping({
        splatSH: ['f_dc_0', 'f_dc_1', 'f_dc_2'],
        splatOpacity: ['opacity'],
      });
    }

    const install = (object) => {
      if (cancelled) {
        disposeObject(object);
        return;
      }
      if (stateRef.current.model) {
        scene.remove(stateRef.current.model);
        disposeObject(stateRef.current.model);
      }

      let vertices = 0;
      let faces = 0;
      object.traverse((child) => {
        if (!child.geometry) return;
        const position = child.geometry.attributes.position;
        if (position) vertices += position.count;
        if (child.geometry.index) faces += child.geometry.index.count / 3;
      });

      object.userData.matcap = matcap;
      scene.add(object);
      stateRef.current.model = object;
      frameObject(object, camera, controls);

      setStats({ vertices, faces: Math.round(faces) });
      setStatus({ phase: 'ready', message: '' });
    };

    loader.load(
      url,
      (loaded) => {
        if (isPly) {
          // PLYLoader hands back raw geometry. VGGT's PLY is a genuine point
          // cloud; TRELLIS's is a Gaussian splat, and all we can show of it
          // without a splat rasteriser is the gaussian centres — scales,
          // rotations and view-dependent SH are not rendered.
          const geometry = loaded;
          if (isSplat) decodeSplatColor(geometry);
          geometry.computeBoundingBox();
          const material = new THREE.PointsMaterial({
            size: pointSize,
            sizeAttenuation: true,
            vertexColors: !!geometry.attributes.color,
            color: geometry.attributes.color ? 0xffffff : 0x9aa4b0,
          });
          install(new THREE.Points(geometry, material));
        } else {
          const root = loaded.scene || loaded.scenes?.[0];
          if (root) normaliseBakedMaterial(root);
          install(root);
        }
      },
      (event) => {
        if (event.lengthComputable && event.total) {
          const pct = Math.round((event.loaded / event.total) * 100);
          setStatus({ phase: 'loading', message: `Loading geometry… ${pct}%` });
        }
      },
      (error) => {
        if (cancelled) return;
        console.error('[MeshViewer] Load failed:', error);
        setStatus({ phase: 'error', message: error?.message || 'Failed to load geometry' });
      },
    );

    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, plyKind]);

  // --- Display mode ---------------------------------------------------------
  useEffect(() => {
    const { model, matcap } = stateRef.current;
    if (!model) return;

    model.traverse((child) => {
      if (child.isPoints) {
        // A point cloud has no solid or wireframe form to switch to.
        child.visible = true;
        if (child.material) child.material.size = pointSize;
        return;
      }
      if (!child.isMesh) return;

      // Stash whatever the file shipped with so 'solid' can restore a real
      // baked texture rather than replacing it with matcap forever.
      if (!child.userData.originalMaterial) {
        child.userData.originalMaterial = child.material;
      }
      const original = child.userData.originalMaterial;
      const hasTexture = !!(original && (original.map || original.emissiveMap));

      if (displayMode === 'wireframe') {
        if (!child.userData.wireMaterial) {
          child.userData.wireMaterial = new THREE.MeshBasicMaterial({
            color: 0x7fd4ff, wireframe: true,
          });
        }
        child.material = child.userData.wireMaterial;
        child.visible = true;
      } else if (displayMode === 'points') {
        child.visible = false;
      } else if (hasTexture) {
        child.material = original;
        child.visible = true;
      } else {
        if (!child.userData.matcapMaterial) {
          child.userData.matcapMaterial = new THREE.MeshMatcapMaterial({
            matcap,
            color: new THREE.Color().setScalar(brightness),
            vertexColors: !!child.geometry.attributes.color,
            flatShading: false,
          });
        }
        child.material = child.userData.matcapMaterial;
        child.visible = true;
      }
    });

    // 'points' on a mesh-only asset renders its vertices, so the toggle still
    // does something useful when only a GLB mesh was exported.
    if (displayMode === 'points') {
      if (!stateRef.current.pointsProxy) {
        const merged = [];
        model.traverse((child) => {
          if (child.isMesh && child.geometry?.attributes.position) merged.push(child);
        });
        if (merged.length) {
          const group = new THREE.Group();
          merged.forEach((mesh) => {
            const points = new THREE.Points(
              mesh.geometry,
              new THREE.PointsMaterial({
                size: pointSize,
                sizeAttenuation: true,
                vertexColors: !!mesh.geometry.attributes.color,
                color: mesh.geometry.attributes.color ? 0xffffff : 0x9aa4b0,
              }),
            );
            mesh.updateWorldMatrix(true, false);
            points.applyMatrix4(mesh.matrixWorld);
            group.add(points);
          });
          stateRef.current.pointsProxy = group;
          stateRef.current.scene.add(group);
        }
      }
      if (stateRef.current.pointsProxy) {
        stateRef.current.pointsProxy.visible = true;
        stateRef.current.pointsProxy.traverse((child) => {
          if (child.material) child.material.size = pointSize;
        });
      }
    } else if (stateRef.current.pointsProxy) {
      stateRef.current.pointsProxy.visible = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayMode, pointSize, stats]);

  // Clear the cached points proxy whenever a different asset loads.
  useEffect(() => {
    const { scene, pointsProxy } = stateRef.current;
    if (pointsProxy && scene) {
      scene.remove(pointsProxy);
      // Geometry is shared with the mesh — dispose only the materials.
      pointsProxy.traverse((child) => child.material?.dispose());
      stateRef.current.pointsProxy = null;
    }
  }, [url]);

  const handleReframe = () => {
    const { model, camera, controls } = stateRef.current;
    if (model) frameObject(model, camera, controls);
  };

  // A splat rendered as bare centres looks sparser and noisier than the file
  // really is. Say so, or it reads as a failed reconstruction.
  const showingSplat =
    plyKind === 'splat'
    && !!url
    && url.toLowerCase().split('?')[0].endsWith('.ply');

  return (
    <div className={`mesh-viewer ${className}`}>
      <div ref={mountRef} className="mesh-viewer-canvas" />

      {status.phase === 'loading' && (
        <div className="mesh-viewer-overlay">
          <span className="mesh-viewer-spinner" />
          {status.message}
        </div>
      )}
      {status.phase === 'error' && (
        <div className="mesh-viewer-overlay mesh-viewer-overlay--error">
          {status.message}
        </div>
      )}
      {!url && (
        <div className="mesh-viewer-overlay mesh-viewer-overlay--empty">
          No geometry loaded
        </div>
      )}

      {showingSplat && status.phase === 'ready' && (
        <div className="mesh-viewer-notice">
          <strong>Gaussian splat</strong> — showing centre points only. Scale,
          rotation and view-dependent colour need a splat viewer.
        </div>
      )}

      {stats && status.phase === 'ready' && (
        <div className="mesh-viewer-stats">
          {stats.vertices.toLocaleString()} {showingSplat ? 'gaussians' : 'verts'}
          {stats.faces > 0 && <> · {stats.faces.toLocaleString()} faces</>}
        </div>
      )}

      {url && status.phase === 'ready' && (
        <button className="mesh-viewer-reframe" onClick={handleReframe} title="Reframe view">
          Reframe
        </button>
      )}
    </div>
  );
}
