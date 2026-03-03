"""pywarper.utils"""
import numpy as np


def resolve_conformal_jump(
    surface_mapping: dict,
    conformal_jump: int | None,
) -> int:
    """Resolve `conformal_jump` from argument or surface mapping metadata."""
    if conformal_jump is None:
        try:
            conformal_jump = int(surface_mapping["conformal_jump"])
        except KeyError as exc:
            raise ValueError(
                "conformal_jump must be provided or found in surface_mapping."
            ) from exc
    if conformal_jump <= 0:
        raise ValueError("conformal_jump must be a positive integer.")
    return int(conformal_jump)


def _convert_legacy_mapping(mapping: dict) -> dict:
    """
    Convert an old-format surface mapping dict (with ``mapped_on``/``mapped_off``
    keys) to the new multi-surface format.

    The new format uses:
    - ``surfaces``: dict mapping tag -> height map
    - ``mapped_surfaces``: dict mapping tag -> (N, 2) flattened coordinates
    - ``surface_order``: list of tags sorted by median depth

    Old-format keys are preserved so that downstream code that checks for them
    (e.g. cached .npz consumers) still works.
    """
    out = dict(mapping)  # shallow copy

    on_surface = np.asarray(mapping["on_sac_surface"], dtype=float)
    off_surface = np.asarray(mapping["off_sac_surface"], dtype=float)
    mapped_on = np.asarray(mapping["mapped_on"], dtype=float)
    mapped_off = np.asarray(mapping["mapped_off"], dtype=float)

    out["surfaces"] = {"on_sac": on_surface, "off_sac": off_surface}
    out["mapped_surfaces"] = {"on_sac": mapped_on, "off_sac": mapped_off}
    out["surface_order"] = ["on_sac", "off_sac"]

    return out


def _ensure_new_format(mapping: dict) -> dict:
    """Return *mapping* in the new multi-surface format, converting if needed."""
    if "mapped_surfaces" not in mapping and "mapped_on" in mapping:
        return _convert_legacy_mapping(mapping)
    return mapping


def load_surface_mapping(path: str) -> dict:
    """Load a surface mapping from .npz, auto-converting legacy format."""
    data = dict(np.load(path, allow_pickle=True))
    return _ensure_new_format(data)


def build_surface_correspondences(
    surface_mapping: dict,
    *,
    conformal_jump: int,
    backward_compatible: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    """
    Build paired control points for local LS registration.

    Returns
    -------
    input_pts_list : list of (K, 3) arrays, one per surface (depth-ordered)
    output_pts_list : list of (K, 3) arrays, one per surface (depth-ordered)
    median_depths : dict mapping tag -> median z
    """
    mapping = _ensure_new_format(surface_mapping)

    surfaces = mapping["surfaces"]
    mapped_surfaces = mapping["mapped_surfaces"]
    surface_order = mapping["surface_order"]

    if backward_compatible:
        sampled_x_idx = np.asarray(mapping["sampled_x_idx"], dtype=int) + 1
        sampled_y_idx = np.asarray(mapping["sampled_y_idx"], dtype=int) + 1
    else:
        sampled_x_idx = np.asarray(mapping["sampled_x_idx"], dtype=int)
        sampled_y_idx = np.asarray(mapping["sampled_y_idx"], dtype=int)

    x_vals = np.arange(sampled_x_idx[0], sampled_x_idx[-1] + 1, conformal_jump)
    y_vals = np.arange(sampled_y_idx[0], sampled_y_idx[-1] + 1, conformal_jump)
    xmesh, ymesh = np.meshgrid(x_vals, y_vals, indexing="ij")

    expected = xmesh.size

    input_pts_list = []
    output_pts_list = []
    median_depths = {}

    for tag in surface_order:
        surface = np.asarray(surfaces[tag], dtype=float)
        mapped = np.asarray(mapped_surfaces[tag], dtype=float)

        if mapped.shape[0] != expected:
            raise ValueError(
                f"Surface mapping size mismatch: mapped_surfaces['{tag}'] does not match sampled grid."
            )

        if backward_compatible:
            subsampled_depths = surface[x_vals[:, None] - 1, y_vals - 1]
        else:
            subsampled_depths = surface[x_vals[:, None], y_vals]

        med_z_val = float(np.median(subsampled_depths))
        median_depths[tag] = med_z_val

        input_pts = np.column_stack(
            [
                xmesh.ravel(order="F"),
                ymesh.ravel(order="F"),
                subsampled_depths.ravel(order="F"),
            ]
        )
        output_pts = np.column_stack(
            [mapped[:, 0], mapped[:, 1], np.full(mapped.shape[0], med_z_val)]
        )

        input_pts_list.append(input_pts)
        output_pts_list.append(output_pts)

    return input_pts_list, output_pts_list, median_depths


def read_sumbul_et_al_chat_bands(fname: str, unit="voxel") -> dict[str, np.ndarray]:
    """
    Read a ChAT-band point cloud exported by KNOSSOS/FiJi.

    Parameters
    ----------
    fname : str
        Plain-text file with columns Area, Mean, Min, Max, X, Y, Slice
        plus an unlabeled first column (row index).

    Returns
    -------
    dict
        Keys ``x``, ``y``, ``z`` (1-based index, float64).
    """
    # The file has eight numeric columns; we only need X (col 5),
    # Y (col 6) and Slice (col 7). 0-based indices: 5, 6, 7.
    data = np.loadtxt(
        fname,
        comments="#",
        skiprows=1,        # skip the header line
        usecols=(5, 7, 6), # X, Slice, Y in desired order
        dtype=np.float64,
    )

    x = data[:, 0] + 1          # KNOSSOS X  -> +1 for MATLAB convention
    y = data[:, 1]              # Slice (already 1-based)
    z = data[:, 2] + 1          # KNOSSOS Y  -> +1

    if unit == "voxel":
        return {"x": x, "y": y, "z": z}
    elif unit == "physical":
        # Convert to physical units (in micrometers)
        voxel_resolution = [0.4, 0.4, 0.5]
        return {
            "x": x * voxel_resolution[0],
            "y": y * voxel_resolution[1],
            "z": z * voxel_resolution[2],
        }
    else:
        raise ValueError(f"Unknown unit: {unit}. Use 'voxel' or 'physical'.")
