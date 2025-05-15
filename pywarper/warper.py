from typing import Optional, Union

import numpy as np
import pandas as pd

from pywarper.arbor import get_xyprofile, get_zprofile, warp_arbor
from pywarper.surface import fit_surface, warp_surface
from pywarper.utils import read_arbor_trace

__all__ = [
    "Warper"
]

class Warper:
    """High‑level interface around *pywarper* for IPL flattening.

    Typical usage
    -------------
    >>> off = read_chat("off_sac.txt")
    >>> on  = read_chat("on_sac.txt")
    >>> w = Warper(off, on, "cell.swc")
    >>> w.fit_surfaces()
    >>> w.build_mapping()
    >>> w.warp()
    >>> w.save("cell_flat.swc")
    """

    def __init__(
        self,
        off_sac: Union[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], None] = None,
        on_sac: Union[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], None] = None,
        swc_path: Optional[str] = None,
        *,
        voxel_resolution: list[float] = [0.4, 0.4, 0.5],
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

    # ---------------------------------------------------------------------
    # public pipeline ------------------------------------------------------
    # ---------------------------------------------------------------------
    def load_swc(self, swc_path: Optional[str] = None) -> "Warper":
        """Load the arbor from *swc_path*."""

        if self.verbose:
            print(f"[Warper] Loading arbor → {self.swc_path}")

        if swc_path is None:
            swc_path = self.swc_path

        arbor, nodes, edges, radii = read_arbor_trace(swc_path)

        # +1 to emulate MATLAB indexing used in original scripts
        self.arbor: pd.DataFrame = arbor
        self.nodes: np.ndarray = nodes
        self.edges: np.ndarray = edges
        self.radii: np.ndarray = radii

        return self

    def load_sac(self, off_sac, on_sac) -> "Warper":
        """Load the SAC meshes from *off_sac* and *on_sac*."""
        if self.verbose:
            print("[Warper] Loading SAC meshes …")
        self.off_sac = self._as_xyz(off_sac)
        self.on_sac  = self._as_xyz(on_sac)
        return self

    def load_warped_arbor(self, 
            swc_path: str,
            medVZmin: Optional[float] = None,
            medVZmax: Optional[float] = None,
    ) -> None:
        """Load a warped arbor from *swc_path*."""
        arbor, nodes, edges, radii = read_arbor_trace(swc_path)
        self.warped_arbor = {
            "nodes": nodes,
            "edges": edges,
            "radii": radii,
        }

        if (medVZmin is not None) and (medVZmax is not None):
            self.warped_arbor["medVZmin"] = float(medVZmin)
            self.warped_arbor["medVZmax"] = float(medVZmax)
        else:
            self.warped_arbor["medVZmin"] = None
            self.warped_arbor["medVZmax"] = None
        
        if self.verbose:
            print(f"[Warper] Loaded warped arbor → {swc_path}")


    def fit_surfaces(self, smoothness: int = 15) -> "Warper":
        """Fit ON / OFF SAC meshes with *pygridfit*."""
        if self.verbose:
            print("[Warper] Fitting OFF‑SAC surface …")
        self.vz_off, *_ = fit_surface(
            x=self.off_sac[0], y=self.off_sac[1], z=self.off_sac[2], smoothness=smoothness
        )
        if self.verbose:
            print("[Warper] Fitting ON‑SAC surface …")
        self.vz_on, *_ = fit_surface(
            x=self.on_sac[0], y=self.on_sac[1], z=self.on_sac[2], smoothness=smoothness
        )
        return self

    def build_mapping(self, conformal_jump: int = 2) -> "Warper":
        """Create the quasi‑conformal surface mapping."""
        if self.vz_off is None or self.vz_on is None:
            raise RuntimeError("Surfaces not fitted. Call fit_surfaces() first.")

        bounds = np.array([
            self.nodes[:, 0].min(), self.nodes[:, 0].max(),
            self.nodes[:, 1].min(), self.nodes[:, 1].max(),
        ])
        if self.verbose:
            print("[Warper] Building mapping …")
        self.mapping: dict = warp_surface(
            self.vz_on,
            self.vz_off,
            bounds,
            conformal_jump=conformal_jump,
            verbose=self.verbose,
        )
        return self

    def warp_arbor(self, conformal_jump: int = 2) -> "Warper":
        """Apply the mapping to the arbor."""
        if self.mapping is None:
            raise RuntimeError("Mapping missing. Call build_mapping() first.")
        if self.verbose:
            print("[Warper] Warping arbor …")
        self.warped_arbor: dict = warp_arbor(
            self.nodes,
            self.edges,
            self.radii,
            self.mapping,
            voxel_resolution=self.voxel_resolution,
            conformal_jump=conformal_jump,
            verbose=self.verbose,
        )
        return self

    # convenience helpers --------------------------------------------------
    def get_arbor_denstiy(
            self, 
            z_res: float = 1, 
            z_window: Optional[list[float]] = None,
            z_nbins: int = 120,
            xy_window: Optional[list[float]] = None,
            xy_nbins: int = 20,
            xy_sigma_bins: float = 1.
    ) -> "Warper":
        """Return depth profile as in *get_zprofile*."""
        if self.warped_arbor is None:
            raise RuntimeError("Arbor not warped yet. Call warp().")
        z_x, z_dist, z_hist, normed_arbor = get_zprofile(self.warped_arbor, z_res=z_res, z_window=z_window, nbins=z_nbins)
        self.z_x: np.ndarray = z_x
        self.z_dist: np.ndarray = z_dist
        self.z_hist: np.ndarray = z_hist
        self.normed_arbor: dict = normed_arbor

        xy_x, xy_y, xy_dist, xy_hist = get_xyprofile(
            self.warped_arbor, xy_window=xy_window, nbins=xy_nbins, sigma_bins=xy_sigma_bins
        )
        self.xy_x: np.ndarray = xy_x
        self.xy_y: np.ndarray = xy_y
        self.xy_dist: np.ndarray = xy_dist
        self.xy_hist: np.ndarray = xy_hist

        return self

    def to_swc(self, out_path: str) -> None:
        """Save the warped arbor to *out_path* in SWC format."""
        if self.warped_arbor is None:
            raise RuntimeError("Arbor not warped yet. Call warp().")

        arr = np.hstack([
            self.warped_arbor["edges"][:, 0][:, None].astype(int),          # n
            np.zeros_like(self.warped_arbor["edges"][:, 1][:, None]),       # t = 0
            self.warped_arbor["nodes"],                                      # xyz
            self.warped_arbor["radii"][:, None],                            # radius
            self.warped_arbor["edges"][:, 1][:, None],                      # parent
        ])
        pd.DataFrame(arr).to_csv(out_path, sep="\t", index=False, header=False)
        if self.verbose:
            print(f"[Warper] Saved warped arbor → {out_path}")


    @staticmethod
    def _as_xyz(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Accept *dict* or tuple and return *(x, y, z)* numpy arrays."""
        if isinstance(data, dict):
            return np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])
        if isinstance(data, (tuple, list)) and len(data) == 3:
            return map(np.asarray, data)  # type: ignore[arg-type]
        raise TypeError("SAC data must be a mapping with keys x/y/z or a 3‑tuple of arrays.")


    def stats(self):

        """Return the statistics of the warped arbor."""
        if self.warped_arbor is None:
            raise RuntimeError("Arbor not warped yet. Call warp().")
        
        # Calculate the statistics
        ## 