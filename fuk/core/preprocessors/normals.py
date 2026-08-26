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
        near_ratio: float = 0.10,
        edge_aware: bool = True,
        denoise: float = 0.05,
        edge_gamma: float = 0.05,
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
            denoise: Depth-derived only. Bilateral range sigma as a fraction of
                     the depth span (default 0.05), removing the depth ringing
                     that speckles subject outlines. 0 disables.

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
                denoise=denoise,
                edge_gamma=edge_gamma,
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
            'denoise': denoise,
            'edge_gamma': edge_gamma,
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
    def _denoise_depth(z: np.ndarray, sigma: float, radius: int = 2) -> np.ndarray:
        """
        Bilateral filter on depth, to strip ringing without rounding silhouettes.

        The normal formula multiplies depth gradients by the focal length, so at
        1280px/60deg a slope of just 1/f = 0.0009 per pixel already tilts the
        normal 45 degrees. Depth models are nowhere near that clean at an object
        boundary: measured on a DA3 frame, depth deviates from its local median
        by 0.00012 on open surfaces but 0.0079 within a few pixels of a
        silhouette — sixty times noisier, and nine times past the point of
        saturating the normal. That ringing, not the depth step itself, is what
        renders as the speckled outline traced around every subject. It flips
        sign pixel to pixel as the sub-pixel edge position wanders along the
        contour, so it reads as dithered noise rather than as shading.

        Ringing and real geometry are far enough apart in amplitude that
        separating them is easy: the ringing sits near 0.008 while the depth
        step across the silhouette is ~0.5, so a range sigma anywhere in the
        middle keeps the cliff intact and flattens the wobble around it. The
        step comes out sharper than before, not softer, because the overshoot
        that used to straddle it is gone.

        Args:
            z:      depth, near = small
            sigma:  range sigma as a fraction of the scene's depth span.
                    Percentile span, not min/max — for a metric model one sky
                    pixel at 1000m would otherwise set the scale for everything.
            radius: spatial sigma in pixels; matches the gradient stencil so the
                    filter cleans exactly the support the gradient will read.
        """
        span = float(np.percentile(z, 99) - np.percentile(z, 1))
        if span <= 0 or sigma <= 0:
            return z
        return cv2.bilateralFilter(
            np.ascontiguousarray(z, dtype=np.float32), -1, sigma * span, float(radius)
        )

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
        near_ratio: float = 0.10,
        edge_aware: bool = True,
        radius: int = 2,
        denoise: float = 0.05,
        edge_gamma: float = 0.05,
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
            near_ratio: Near-plane distance divided by scene depth span. Only
                        used by scale-and-shift ambiguous models (MiDaS, DAv2),
                        which fix geometry up to this one ratio — smaller =
                        deeper scene = stronger relief. Ignored by the DA3
                        family, whose depth is shift-free and needs no guess.
            edge_aware: Use min-magnitude one-sided differences instead of Sobel,
                        which stops silhouettes smearing into false bevels.
            radius: Gradient stencil radius in pixels (edge_aware only). Larger
                    suppresses depth-model noise at the cost of fine detail.
            denoise: Bilateral range sigma as a fraction of the scene depth
                     span, applied before differentiating. Removes the depth
                     ringing that draws speckled outlines around subjects.
                     0 disables it; above ~0.1 genuine shallow relief starts
                     flattening into the silhouette.

        Returns:
            Normal map as float32 array in [-1, 1] range, shape (H, W, 3)
        """
        import math

        depth, is_metric = self._depth_processor.get_depth_z(image_path)
        h, w = depth.shape[:2]

        if denoise > 0:
            depth = self._denoise_depth(depth, sigma=denoise, radius=radius)

        f = (max(h, w) / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        cx, cy = w / 2.0 - 0.5, h / 2.0 - 0.5

        if edge_aware:
            z_u, z_v = self._one_sided_gradients(depth, radius=radius)
        else:
            # Sobel is an 8x-scaled central difference; divide it back out so
            # these are true per-pixel derivatives and f means what it should.
            z_u = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            z_v = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3) / 8.0

        # Width of the corrupted band either side of an occlusion boundary.
        # Measured on a 960x576 DA3 frame, depth ringing runs 9-12x the slope
        # that fully saturates a normal at 0-2px from the silhouette, 3.9x at
        # 3px, and reaches the open-surface noise floor by 5px. Tied to the
        # gradient stencil so it scales together if radius is raised for
        # larger frames.
        edge_band = max(3, int(round(2.5 * radius)))

        # Relative depth arrives as [0, 1] where 0 is the NEAREST surface in
        # frame, not the camera. Used as Z that puts the closest subject at
        # zero distance, which is where the formula degenerates: the +Z term
        # that keeps a normal pointing at the camera vanishes, the perspective
        # term takes over unopposed, and the foreground shatters into saturated
        # noise while the background — far from zero — still looks right.
        #
        # near_ratio restores the near plane. Formally it is z0/s for
        # Z = z0 + s*d, which divides out of the formula leaving one free
        # parameter; in practice it reads as a relief control, since a model's
        # relative depth is not linear enough in true depth for a physically
        # derived value to be worth computing.
        z = depth if is_metric else depth + near_ratio

        # Reach the gradient stencil past the ringing where a boundary was
        # detected. The min-magnitude rule already prefers whichever side stays
        # on the surface; the only problem is that within the band BOTH sides
        # are corrupted, so the choice is between two wrong answers. A stencil
        # that lands outside the band gets a clean surface slope instead, and
        # picking per-pixel keeps the tight stencil — and its fine detail —
        # everywhere else.
        band = None
        if edge_gamma > 0 and edge_aware:
            band = (np.abs(z_u) > edge_gamma * np.abs(z)) | \
                   (np.abs(z_v) > edge_gamma * np.abs(z))
            band = cv2.dilate(
                band.astype(np.uint8),
                np.ones((2 * edge_band + 1, 2 * edge_band + 1), np.uint8),
            ) > 0
            if band.any():
                z_u_far, z_v_far = self._one_sided_gradients(
                    depth, radius=radius + 2 * edge_band
                )
                z_u = np.where(band, z_u_far, z_u)
                z_v = np.where(band, z_v_far, z_v)

        u = np.arange(w, dtype=np.float32)[None, :] - cx
        v = np.arange(h, dtype=np.float32)[:, None] - cy

        normals = np.empty((h, w, 3), dtype=np.float32)
        normals[:, :, 0] = f * z_u
        normals[:, :, 1] = f * z_v
        normals[:, :, 2] = u * z_u + v * z_v + z

        normals /= np.sqrt((normals ** 2).sum(axis=2, keepdims=True)) + 1e-8

        if band is not None and band.any():
            # What remains inside the band is sign-flipping pixel to pixel as
            # the sub-pixel edge position wanders along the contour, which reads
            # as a dithered chain of beads rather than as shading. Averaging the
            # normals along the contour turns it back into a coherent rounded
            # edge, which is what a silhouette should look like anyway.
            #
            # Averaging DIRECTIONS, though — blurring each channel independently
            # and renormalising is the spherical mean, which stays on the unit
            # sphere. A per-channel median does not: it takes x, y and z from
            # three different pixels and can return a direction present nowhere
            # in the window. It is also capped at a 5x5 kernel for float32,
            # which cannot reach past a band this wide.
            smoothed = cv2.GaussianBlur(normals, (0, 0), edge_band / 2.0)
            smoothed /= np.sqrt((smoothed ** 2).sum(axis=2, keepdims=True)) + 1e-8
            normals = np.where(band[:, :, None], smoothed, normals)

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

        # DSINE emits the AWAY-from-camera normal in camera coords (X right,
        # Y down, Z into scene). Its own training targets are built that way:
        # utils/d2n/plane_svd.py ends with
        #     flip = sign(sum(normal * points)); normal = normal * flip
        # which forces normal . P > 0, and RayReLU likewise holds the component
        # along the view ray positive.
        #
        # A normal map wants the toward-camera normal with Z out of the screen,
        # so T = -A and D = (T_x, T_y, -T_z) = (-A_x, -A_y, A_z). Without this
        # the DSINE output is rotated 180 degrees in the image plane against
        # both the depth-derived path and every other normal map in a comp:
        # measured over 5 frames of real footage, applying it drops the
        # disagreement between the two methods from 115 deg to 45 deg.
        normals[:, :, 0] = -normals[:, :, 0]
        normals[:, :, 1] = -normals[:, :, 1]
        return normals.astype(np.float32)
    
    def get_raw_normals(
        self,
        image_path: Path,
        intensity: float = 1.0,
        num_iter: int = 5,
        fov_deg: float = 60.0,
        near_ratio: float = 0.10,
        edge_aware: bool = True,
        denoise: float = 0.05,
        edge_gamma: float = 0.05,
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
                denoise=denoise,
                edge_gamma=edge_gamma,
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