# core/preprocessors/normals.py
"""
Surface Normal Map Preprocessor

Methods:
1. Depth-derived - Fast, computed from depth gradient (no extra model)
2. DSINE - Dedicated normal estimation model (better quality)

Good for:
- Relighting in comp
- Material/surface detail
- 3D reconstruction
- ControlNet normal conditioning
"""

from pathlib import Path
from typing import Dict, Any, Optional, Literal
from enum import Enum
import cv2
import numpy as np
from PIL import Image
import torch

from .base import BasePreprocessor
from .depth import DepthPreprocessor, DepthModel


class NormalsMethod(str, Enum):
    """Available normal estimation methods"""
    FROM_DEPTH = "from_depth"    # Derive from depth map (fast, no extra model)
    DSINE = "dsine"              # Dedicated normal estimator (better quality)


class NormalsPreprocessor(BasePreprocessor):
    """
    Surface normal map estimation
    
    Can derive normals from depth (fast) or use dedicated DSINE model (quality).
    """
    
    def __init__(
        self,
        method: NormalsMethod = NormalsMethod.FROM_DEPTH,
        depth_model: DepthModel = DepthModel.DA3_MONO_LARGE,
        config_path: Optional[Path] = None
    ):
        super().__init__(config_path)
        self.method = method
        self.depth_model = depth_model
        self.model = None
        self._depth_processor = None
        print(f"Normals preprocessor initialized: {method.value} on {self.device}")
    
    def _initialize(self):
        """Load model based on method"""
        if self.method == NormalsMethod.FROM_DEPTH:
            # Use depth processor for depth-derived normals
            self._depth_processor = DepthPreprocessor(
                model_type=self.depth_model,
                config_path=self.config_path
            )
        elif self.method == NormalsMethod.DSINE:
            self._load_dsine()
    
    def _load_dsine(self):
        """
        Load DSINE normal estimator (v02, dsine.pt checkpoint).
        Manually manages sys.path so DSINE's local 'utils/' is found before
        any installed PyPI 'utils' package (pip install utils installs the
        wrong thing — uninstall it if present).
        """
        try:
            vendor_path = self._get_vendor_path("DSINE")

            if not vendor_path.exists():
                raise ImportError(
                    f"DSINE not found at {vendor_path}\n"
                    f"Install: git clone https://github.com/baegwangbin/DSINE {vendor_path}"
                )

            checkpoint_path = vendor_path / "checkpoints" / "dsine.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"DSINE checkpoint not found: {checkpoint_path}\n"
                    f"Download: https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
                )

            print("Loading DSINE normal estimator...")

            import sys
            vendor_str = str(vendor_path)

            # Ensure DSINE's local dirs come first — beats any installed PyPI 'utils'
            if vendor_str in sys.path:
                sys.path.remove(vendor_str)
            sys.path.insert(0, vendor_str)

            # Evict any cached PyPI 'utils' and any stale DSINE model imports so
            # everything re-resolves fresh from vendor_str at the front of sys.path.
            stale = [
                k for k in list(sys.modules)
                if k == 'utils' or k.startswith('utils.')
                or k == 'models' or k.startswith('models.')
            ]
            for k in stale:
                del sys.modules[k]

            from models.dsine import DSINE as DSINEFactory

            ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
            model = DSINEFactory()
            model.load_state_dict(ckpt['model'], strict=True)
            model.eval()
            model = model.to(self.device)
            model.pixel_coords = model.pixel_coords.to(self.device)
            self.model = model

            print(f"✓ DSINE loaded from {checkpoint_path.name}")

        except Exception as e:
            print(f"⚠ Could not load DSINE: {e}")
            print("  Falling back to depth-derived normals...")
            self.method = NormalsMethod.FROM_DEPTH
            self._depth_processor = DepthPreprocessor(
                model_type=self.depth_model,
                config_path=self.config_path
            )
    
    def process(
        self,
        image_path: Path,
        output_path: Path,
        space: Literal["tangent", "world", "object"] = "tangent",
        flip_y: bool = False,
        flip_x: bool = False,
        intensity: float = 1.0,
        exact_output: bool = False,
        # Shared: both methods need an assumed camera
        fov_deg: float = 60.0,
        # DSINE-specific quality settings
        num_iter: int = 5,
        # Depth-derived settings
        near_ratio: float = 0.5,
        edge_aware: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate surface normals

        Args:
            image_path: Input image
            output_path: Where to save result
            space: Normal space ('tangent', 'world', 'object')
            flip_y: Flip Y component (for different engine conventions)
            flip_x: Flip X component
            intensity: Relief strength for depth-derived normals. 1.0 is
                       physically correct; higher exaggerates.
            exact_output: If True, write to exact output_path (for video frames)
            fov_deg: Assumed camera field-of-view in degrees (default 60). Used
                     by both methods — it sets the focal length that converts
                     pixel gradients to surface slope for depth-derived normals,
                     and DSINE's intrinsics, where a wrong FOV causes blob
                     artifacts at depth discontinuities.
            num_iter: DSINE GRU refinement iterations (default 5, trained value).
                      The checkpoint was trained with exactly 5 steps — going
                      beyond 7 causes GRU divergence and severe artifacts.
            near_ratio: Depth-derived only. Near distance / scene depth span;
                        smaller = deeper scene = stronger relief.
            edge_aware: Depth-derived only. Suppress silhouette bevels.

        Returns:
            Dict with output_path and metadata
        """
        self._ensure_initialized()

        # Load image (handles EXR via ANYDEPTH + gamma correction)
        image = self.load_image_bgr(image_path)

        # Generate normals based on method
        if self.method == NormalsMethod.FROM_DEPTH:
            normals = self._normals_from_depth(
                image_path,
                intensity=intensity,
                fov_deg=fov_deg,
                near_ratio=near_ratio,
                edge_aware=edge_aware,
            )
        else:
            normals = self._normals_from_dsine(image, num_iter=num_iter, fov_deg=fov_deg)

        # Apply flips if needed
        if flip_x:
            normals[:, :, 0] = -normals[:, :, 0]
        if flip_y:
            normals[:, :, 1] = -normals[:, :, 1]

        # Convert from [-1, 1] to [0, 255] for standard normal map encoding
        # Normal map convention: R=X, G=Y, B=Z
        normals_uint8 = ((normals + 1) * 0.5 * 255).astype(np.uint8)

        # OpenCV uses BGR, so swap R and B
        normals_bgr = cv2.cvtColor(normals_uint8, cv2.COLOR_RGB2BGR)

        params = {
            'method': 'normals',
            'estimation': self.method.value,
            'space': space,
            'flip_y': flip_y,
            'flip_x': flip_x,
            'intensity': intensity,
            'num_iter': num_iter,
            'fov_deg': fov_deg,
            'near_ratio': near_ratio,
            'edge_aware': edge_aware,
        }
        final_output = self._make_unique_path(output_path, params, exact_output=exact_output)
        cv2.imwrite(str(final_output), normals_bgr)

        return {
            "output_path": str(final_output),
            "method": "normals",
            "estimation": self.method.value,
            "parameters": params,
            "raw_normals": normals,  # float32 [-1, 1] for lossless EXR export
        }
    
    @staticmethod
    def _one_sided_gradients(z: np.ndarray, radius: int = 2):
        """
        Per-pixel dZ/du and dZ/dv, taking whichever one-sided difference has the
        smaller magnitude.

        At an object silhouette one side of the pixel spans the depth jump while
        the other stays on the surface, so picking the smaller keeps the normal
        on the surface instead of smearing a rim along the boundary. This is the
        cheap stand-in for DSINE's gamma test, which rejects neighbour pixels
        whose depth differs from the centre by more than 5%.

        The differences span `radius` pixels rather than one. A one-pixel stencil
        is the noisiest available, and the focal-length factor in the normal
        formula amplifies that noise into visible streaking on flat surfaces;
        radius=2 gives the same 5px support as DSINE's default k=5 while
        staying one-sided, so it smooths without rounding off silhouettes.
        """
        r = max(1, int(radius))

        def axis(z, ax):
            fwd = np.empty_like(z)
            bwd = np.empty_like(z)
            if ax == 1:
                fwd[:, :-r] = (z[:, r:] - z[:, :-r]) / r
                fwd[:, -r:] = fwd[:, -r - 1:-r]
                bwd[:, r:] = fwd[:, :-r]
                bwd[:, :r] = fwd[:, r:r + 1]
            else:
                fwd[:-r, :] = (z[r:, :] - z[:-r, :]) / r
                fwd[-r:, :] = fwd[-r - 1:-r, :]
                bwd[r:, :] = fwd[:-r, :]
                bwd[:r, :] = fwd[r:r + 1, :]
            return np.where(np.abs(fwd) < np.abs(bwd), fwd, bwd)

        return axis(z, 1), axis(z, 0)

    def _normals_from_depth(
        self,
        image_path: Path,
        intensity: float = 1.0,
        fov_deg: float = 60.0,
        near_ratio: float = 0.5,
        edge_aware: bool = True,
        radius: int = 2,
    ) -> np.ndarray:
        """
        Compute normals by unprojecting depth into camera space.

        Depth gradients live in pixel units; surface slope lives in world units,
        and the focal length is what converts between them. Unprojecting
        P = Z * K^-1 [u, v, 1] and taking cross(dP/du, dP/dv) reduces to a
        closed form, so no explicit point cloud is needed:

            n  ~  ( f * dZ/du,  f * dZ/dv,  (u-cx)*dZ/du + (v-cy)*dZ/dv + Z )

        The f factor is the whole ballgame: at 1280px and 60 deg FOV it is
        ~1109, so dropping it (as a plain height-map Sobel does) shrinks the
        X/Y components by three orders of magnitude, n collapses to (0, 0, 1),
        and the result is flat lavender with detail only on silhouette edges
        where depth jumps far enough in one pixel to survive.

        The third term is the perspective correction: away from the principal
        point, a surface at constant depth is still slanted relative to the view
        ray. It vanishes at the image centre.

        Output frame is X right, Y down, Z toward camera (DirectX-style green);
        pass flip_y for OpenGL-style.

        Args:
            image_path: Input image path
            intensity: Strength multiplier on X/Y, applied after normalisation.
                       1.0 is physically correct; higher exaggerates relief.
            fov_deg: Assumed camera field-of-view in degrees, sets focal length
            near_ratio: Near-plane distance divided by scene depth span. Relative
                        depth models fix geometry only up to this one ratio —
                        smaller = deeper scene = stronger relief. Ignored for
                        metric models, which supply a real Z.
            edge_aware: Use min-magnitude one-sided differences instead of Sobel,
                        which stops silhouettes smearing into false bevels.
            radius: Gradient stencil radius in pixels (edge_aware only). Larger
                    suppresses depth-model noise at the cost of fine detail.

        Returns:
            Normal map as float32 array in [-1, 1] range, shape (H, W, 3)
        """
        import math

        depth, is_metric = self._depth_processor.get_depth_z(image_path)
        h, w = depth.shape[:2]

        f = (max(h, w) / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        cx, cy = w / 2.0 - 0.5, h / 2.0 - 0.5

        if edge_aware:
            z_u, z_v = self._one_sided_gradients(depth, radius=radius)
        else:
            # Sobel is an 8x-scaled central difference; divide it back out so
            # these are true per-pixel derivatives and f means what it should.
            z_u = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            z_v = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0

        # Relative depth is defined up to Z = z0 + s*d. Dividing the closed form
        # through by s leaves a single free parameter, z0/s = near_ratio, so the
        # normalised depth can be used directly with that offset added.
        z = depth if is_metric else depth + near_ratio

        u = np.arange(w, dtype=np.float32)[None, :] - cx
        v = np.arange(h, dtype=np.float32)[:, None] - cy

        normals = np.empty((h, w, 3), dtype=np.float32)
        normals[:, :, 0] = f * z_u
        normals[:, :, 1] = f * z_v
        normals[:, :, 2] = u * z_u + v * z_v + z

        normals /= np.sqrt((normals ** 2).sum(axis=2, keepdims=True)) + 1e-8

        if intensity != 1.0:
            normals[:, :, :2] *= intensity
            normals /= np.sqrt((normals ** 2).sum(axis=2, keepdims=True)) + 1e-8

        return normals

    @staticmethod
    def _dsine_padding(H: int, W: int):
        """Compute (l, r, t, b) padding to make H and W multiples of 32."""
        def _side(n):
            if n % 32 == 0:
                return 0, 0
            new = 32 * ((n // 32) + 1)
            a = (new - n) // 2
            return a, (new - n) - a
        l, r = _side(W)
        t, b = _side(H)
        return l, r, t, b

    def _dsine_intrins(self, H: int, W: int, fov_deg: float = 60.0):
        """Pinhole intrinsic matrix from FOV, principal point at image centre."""
        import math
        f = (max(H, W) / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        intrins = torch.tensor(
            [[f, 0, W / 2.0 - 0.5],
             [0, f, H / 2.0 - 0.5],
             [0, 0, 1.0]],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0)  # (1, 3, 3)
        return intrins

    def _normals_from_dsine(
        self,
        image: np.ndarray,
        num_iter: int = 10,
        fov_deg: float = 60.0,
    ) -> np.ndarray:
        """
        Compute normals using the DSINE v02 model.

        Args:
            image: BGR uint8 numpy array
            num_iter: GRU refinement iterations (higher = sharper fine detail)
            fov_deg: Assumed camera FOV in degrees

        Returns:
            float32 normal map in [-1, 1], shape (H, W, 3)
        """
        import torch.nn.functional as F
        from torchvision.transforms.functional import normalize as tvf_normalize

        h, w = image.shape[:2]

        # Clamp to safe range — checkpoint trained at 5, diverges past ~7
        self.model.num_iter_train = max(1, min(num_iter, 7))

        # BGR -> RGB float32 tensor [0, 1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Pad to multiples of 32
        l, r, t, b = self._dsine_padding(h, w)
        img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)

        # ImageNet normalisation
        img = tvf_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Pinhole intrinsics; shift principal point to account for padding
        intrins = self._dsine_intrins(h, w, fov_deg=fov_deg)
        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t

        with torch.no_grad():
            pred = self.model(img, intrins=intrins)[-1]  # (1, 3, H_pad, W_pad)
            pred = pred[:, :, t:t + h, l:l + w]                      # unpad

        normals = pred[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        return normals.astype(np.float32)
    
    def get_raw_normals(
        self,
        image_path: Path,
        intensity: float = 1.0,
        num_iter: int = 5,
        fov_deg: float = 60.0,
        near_ratio: float = 0.5,
        edge_aware: bool = True,
    ) -> np.ndarray:
        """
        Get raw normal vectors (for EXR export, etc.)

        Returns float32 normal array in [-1, 1] range
        """
        self._ensure_initialized()

        image = self.load_image_bgr(image_path)

        if self.method == NormalsMethod.FROM_DEPTH:
            return self._normals_from_depth(
                image_path,
                intensity=intensity,
                fov_deg=fov_deg,
                near_ratio=near_ratio,
                edge_aware=edge_aware,
            )
        else:
            return self._normals_from_dsine(image, num_iter=num_iter, fov_deg=fov_deg)
    
    def unload(self):
        """
        Unload normals models from VRAM
        
        Handles both DSINE model and depth processor.
        """
        if self.model is not None:
            del self.model
            self.model = None
        
        if self._depth_processor is not None:
            self._depth_processor.unload()
            self._depth_processor = None
        
        # Call base cleanup
        super().unload()