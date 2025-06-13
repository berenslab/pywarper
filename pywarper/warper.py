import time
from pathlib import Path

import numpy as np
import skeliner as sk

from pywarper.arbor import warp_arbor
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

    def fit_surfaces(self, xmax=None, ymax=None, stride:int = 3, smoothness: int = 15, backward_compatible:bool=False) -> "Warper":
        """Fit ON / OFF SAC meshes with *pygridfit*."""
        if self.verbose:
            print("[pywarper] Fitting SAC surfaces …")

        _t0 = time.time()
        self.off_sac_surface, *_ = fit_sac_surface(
            x=self.off_sac[0], 
            y=self.off_sac[1],
            z=self.off_sac[2], 
            stride=stride,
            smoothness=smoothness,
            xmax=xmax, ymax=ymax,
            backward_compatible=backward_compatible,
        )
        if self.verbose:
            print(f"↳ fitting OFF (max) surface\n    done in {time.time() - _t0:.2f} seconds.")
        
        _t0 = time.time()
        self.on_sac_surface, *_ = fit_sac_surface(
            x=self.on_sac[0], 
            y=self.on_sac[1], 
            z=self.on_sac[2], 
            smoothness=smoothness,
            xmax=xmax, ymax=ymax,
            backward_compatible=backward_compatible,
        )
        if self.verbose:
            print(f"↳ fitting ON (min) surface\n    done in {time.time() - _t0:.2f} seconds.")
        return self

    def build_mapping(self, 
                      bounds:np.ndarray | tuple | None = None, 
                      conformal_jump: int = 2, 
                      n_anchors: int = 16,
                      backward_compatible: bool = False,
    ) -> "Warper":
        """Create the quasi‑conformal surface mapping."""
        if self.off_sac_surface is None or self.on_sac_surface is None:
            raise RuntimeError("Surfaces not fitted. Call fit_surfaces() first.")

        if bounds is None:
            bounds = np.array([
                self.skel.nodes[:, 0].min(), self.skel.nodes[:, 0].max(),
                self.skel.nodes[:, 1].min(), self.skel.nodes[:, 1].max(),
            ])
        elif bounds == "global":
            bounds = (0, self.on_sac_surface.shape[0], 0, self.on_sac_surface.shape[1])
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
            n_anchors=n_anchors,
            backward_compatible=backward_compatible,
            verbose=self.verbose,
        )
        return self

    def warp_arbor(self, 
                zprofile_extends: list[float] | None = None,
                zprofile_nbins: int = 120,
                xyprofile_extends: list[float] | None = None,
                xyprofile_nbins: int = 20,
                xyprofile_smooth: float = 1.,
                voxel_resolution: list[float] | None = None, 
                conformal_jump: int = 2,
                backward_compatible: bool = False,
    ) -> "Warper":
        """Apply the mapping to the arbor."""
        if self.mapping is None:
            raise RuntimeError("Mapping missing. Call build_mapping() first.")
        
        if voxel_resolution is None:
            voxel_resolution = self.voxel_resolution

        self.warped_arbor = warp_arbor(
            self.skel,
            self.mapping,
            voxel_resolution=voxel_resolution,
            conformal_jump=conformal_jump,
            zprofile_extends=zprofile_extends,
            zprofile_nbins=zprofile_nbins,
            xyprofile_extends=xyprofile_extends,
            xyprofile_nbins=xyprofile_nbins,
            xyprofile_smooth=xyprofile_smooth,
            backward_compatible=backward_compatible,
            verbose=self.verbose,
        )
        return self