from pathlib import Path

import numpy as np
import skeliner as sk

from pywarper.arbor import get_xyprofile, get_zprofile, warp_arbor
from pywarper.surface import build_mapping, fit_sac_surface

__all__ = [
    "Warper"
]

class Warper:
    """High‑level interface around *pywarper* for IPL flattening."""

    def __init__(
        self,
        off_sac: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        on_sac: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        swc_path: str | None = None,
        *,
        voxel_resolution: list[float] = [1.0, 1.0, 1.0],
        verbose: bool = False,
    ) -> None:

        self.voxel_resolution = voxel_resolution
        self.verbose = verbose
        self.swc_path = swc_path

        if off_sac is not None:
            self.off_sac = self._as_xyz(off_sac)
        if on_sac is not None:
            self.on_sac  = self._as_xyz(on_sac)

        if swc_path is not None:
            self.swc_path = swc_path
            self.load_swc(swc_path)          # raw SWC → self.nodes / edges / radii
        else:
            self.swc_path = None

    # ---------------------------- IO -------------------------------------
    def load_swc(self, swc_path: str | None = None) -> "Warper":
        """Load the arbor from *swc_path*."""

        if self.verbose:
            print(f"[pywarper] Loading arbor → {self.swc_path}")

        if swc_path is None:
            swc_path = self.swc_path

        if swc_path is not None:
            self.skel = sk.io.load_swc(swc_path)
        else:
            raise ValueError("SWC path must be provided to load the arbor.")

        return self

    @staticmethod
    def _as_xyz(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # for load_sac()
        """Accept *dict* or tuple and return *(x, y, z)* numpy arrays."""
        if isinstance(data, dict):
            return np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])
        if isinstance(data, (tuple, list)) and len(data) == 3:
            return map(np.asarray, data)  # type: ignore[arg-type]
        raise TypeError("SAC data must be a mapping with keys x/y/z or a 3‑tuple of arrays.")


    def load_sac(self, off_sac, on_sac) -> "Warper":
        """Load the SAC meshes from *off_sac* and *on_sac*."""
        if self.verbose:
            print("[pywarper] Loading SAC meshes …")
        self.off_sac = self._as_xyz(off_sac)
        self.on_sac  = self._as_xyz(on_sac)
        return self

    def load_warped_arbor(self, 
            filepath: str,
            med_z_on: float | None = None,
            med_z_off: float | None = None,
    ) -> None:
        """Load a warped arbor from *swc_path*."""
        path = Path(filepath)

        if path.suffix.lower() == ".swc":
            self.warped_arbor = sk.io.load_swc(path)

            if (med_z_on is not None) and (med_z_off is not None):
                self.warped_arbor.extra["med_z_on"] = float(med_z_on)
                self.warped_arbor.extra["med_z_off"] = float(med_z_off)
            else:
                self.warped_arbor.extra["med_z_on"] = None
                self.warped_arbor.extra["med_z_off"] = None
        elif path.suffix.lower() == ".npz":
            self.warped_arbor = sk.io.load_npz(path)
            
        if self.verbose:
            print(f"[pywarper] Loaded warped arbor → {path}")

    # ---------------------------- Core -----------------------------------

    def fit_surfaces(self, xmax=None, ymax=None, smoothness: int = 15) -> "Warper":
        """Fit ON / OFF SAC meshes with *pygridfit*."""
        if self.verbose:
            print("[pywarper] Fitting OFF‑SAC surface …")

        self.off_sac_surface, *_ = fit_sac_surface(
            x=self.off_sac[0], 
            y=self.off_sac[1],
            z=self.off_sac[2], 
            smoothness=smoothness,
            xmax=xmax, ymax=ymax,
        )
        if self.verbose:
            print("[pywarper] Fitting ON‑SAC surface …")
        self.on_sac_surface, *_ = fit_sac_surface(
            x=self.on_sac[0], 
            y=self.on_sac[1], 
            z=self.on_sac[2], 
            smoothness=smoothness,
            xmax=xmax, ymax=ymax,
        )
        return self

    def build_mapping(self, bounds:np.ndarray | tuple | None = None, conformal_jump: int = 2) -> "Warper":
        """Create the quasi‑conformal surface mapping."""
        if self.off_sac_surface is None or self.on_sac_surface is None:
            raise RuntimeError("Surfaces not fitted. Call fit_surfaces() first.")

        if bounds is None:
            bounds = np.array([
                self.skel.nodes[:, 0].min(), self.skel.nodes[:, 0].max(),
                self.skel.nodes[:, 1].min(), self.skel.nodes[:, 1].max(),
            ])
        else:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (4,):
                raise ValueError("Bounds must be a 4‑element array or tuple (x_min, x_max, y_min, y_max).")
        
        if self.verbose:
            print("[pywarper] Building mapping …")
        self.mapping: dict = build_mapping(
            self.on_sac_surface,
            self.off_sac_surface,
            bounds,
            conformal_jump=conformal_jump,
            verbose=self.verbose,
        )
        return self

    def warp_arbor(self, voxel_resolution: list[float] | None = None, conformal_jump: int = 2) -> "Warper":
        """Apply the mapping to the arbor."""
        if self.mapping is None:
            raise RuntimeError("Mapping missing. Call build_mapping() first.")
        
        if voxel_resolution is None:
            voxel_resolution = self.voxel_resolution

        if self.verbose:
            print("[pywarper] Warping arbor …")
        self.warped_arbor = warp_arbor(
            self.skel,
            self.mapping,
            voxel_resolution=voxel_resolution,
            conformal_jump=conformal_jump,
            verbose=self.verbose,
        )
        return self

    # ---------------------------- Stats ----------------------------------
    def get_arbor_density(
            self, 
            z_res: float = 1, 
            z_window: list[float] | None = None,
            z_nbins: int = 120,
            xy_window: list[float] | None = None,
            xy_nbins: int = 20,
            xy_sigma_bins: float = 1.
    ) -> "Warper":
        """Return depth profile as in *get_zprofile*."""
        if self.warped_arbor is None:
            raise RuntimeError("Arbor not warped yet.")
        z_x, z_dist, z_hist, normed_arbor = get_zprofile(self.warped_arbor, z_res=z_res, z_window=z_window, nbins=z_nbins)
        self.z_x: np.ndarray = z_x
        self.z_dist: np.ndarray = z_dist
        self.z_hist: np.ndarray = z_hist
        self.normed_arbor = normed_arbor

        xy_x, xy_y, xy_dist, xy_hist = get_xyprofile(
            self.warped_arbor, xy_window=xy_window, nbins=xy_nbins, sigma_bins=xy_sigma_bins
        )
        self.xy_x: np.ndarray = xy_x
        self.xy_y: np.ndarray = xy_y
        self.xy_dist: np.ndarray = xy_dist
        self.xy_hist: np.ndarray = xy_hist

        return self

    def stats(self):

        """Return the statistics of the warped arbor."""
        if self.warped_arbor is None:
            raise RuntimeError("Arbor not warped yet. Call warp().")
        
        # Calculate the statistics
        ## 