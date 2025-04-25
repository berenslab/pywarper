from typing import Sequence

import numpy as np
import pandas as pd

from pywarper.arbor import get_zprofile, warp_arbor
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
        off_sac: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray],
        on_sac: dict[str, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray],
        swc_path: str,
        *,
        smoothness: int = 15,
        conformal_jump: int = 2,
        voxel_resolution: list[float] = [0.4, 0.4, 0.5],
        verbose: bool = False,
    ) -> None:
        self.smoothness = smoothness
        self.conformal_jump = conformal_jump
        self.voxel_resolution = voxel_resolution
        self.verbose = verbose
        self.swc_path = swc_path

        # parse SAC point clouds
        self.off_sac = self._as_xyz(off_sac)
        self.on_sac = self._as_xyz(on_sac)

        # read arbor to warp
        self._load_swc()

    # ---------------------------------------------------------------------
    # public pipeline ------------------------------------------------------
    # ---------------------------------------------------------------------
    def fit_surfaces(self) -> "Warper":
        """Fit ON / OFF SAC meshes with *pygridfit*."""
        if self.verbose:
            print("[Warper] Fitting OFF‑SAC surface …")
        self.vz_off, *_ = fit_surface(
            x=self.off_sac[0], y=self.off_sac[1], z=self.off_sac[2], smoothness=self.smoothness
        )
        if self.verbose:
            print("[Warper] Fitting ON‑SAC surface …")
        self.vz_on, *_ = fit_surface(
            x=self.on_sac[0], y=self.on_sac[1], z=self.on_sac[2], smoothness=self.smoothness
        )
        return self

    def build_mapping(self) -> "Warper":
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
            conformal_jump=self.conformal_jump,
            verbose=self.verbose,
        )
        return self

    def warp_arbor(self) -> "Warper":
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
            conformal_jump=self.conformal_jump,
            verbose=self.verbose,
        )
        return self

    # convenience helpers --------------------------------------------------
    def get_arbor_denstiy(self, z_res: float = 0.5, z_window: list[float] = [-30, 30]) -> "Warper":
        """Return depth profile as in *get_zprofile*."""
        if self.warped_arbor is None:
            raise RuntimeError("Arbor not warped yet. Call warp().")
        x, z_dist, z_hist, normed_arbor = get_zprofile(self.warped_arbor, z_res=z_res, z_window=z_window)
        self.x: np.ndarray = x
        self.z_dist: np.ndarray = z_dist
        self.z_hist: np.ndarray = z_hist
        self.normed_arbor: dict = normed_arbor

        return self

    def save(self, out_path: str) -> None:
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

    # ------------------------------------------------------------------
    # internal helpers -------------------------------------------------
    # ------------------------------------------------------------------
    def _load_swc(self):
        arbor, nodes, edges, radii = read_arbor_trace(self.swc_path)
        # +1 to emulate MATLAB indexing used in original scripts
        self.arbor: pd.DataFrame = arbor
        self.nodes: np.ndarray = nodes
        self.edges: np.ndarray = edges
        self.radii: np.ndarray = radii

    @staticmethod
    def _as_xyz(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Accept *dict* or tuple and return *(x, y, z)* numpy arrays."""
        if isinstance(data, dict):
            return np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])
        if isinstance(data, (tuple, list)) and len(data) == 3:
            return map(np.asarray, data)  # type: ignore[arg-type]
        raise TypeError("SAC data must be a mapping with keys x/y/z or a 3‑tuple of arrays.")
