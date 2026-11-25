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


def surface_lattice(
    surface_mapping: dict,
    *,
    conformal_jump: int,
    backward_compatible: bool = False,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float
]:
    """
    The mapping's vertices as the lattice they lie on.

    Returns
    -------
    x_vals, y_vals : (nx,), (ny,) int
        Where the lattice's rows and columns are, in the surfaces' units.
    on_depths, off_depths : (nx, ny)
        Heights of the two bands at the vertices.
    mapped_on, mapped_off : (nx * ny, 2)
        Flattened positions of the vertices, the first index running fastest.
    med_z_on, med_z_off : float
        Median heights of the two bands over the lattice.
    """
    mapped_on = np.asarray(surface_mapping["mapped_on"], dtype=float)
    mapped_off = np.asarray(surface_mapping["mapped_off"], dtype=float)
    on_sac_surface = np.asarray(surface_mapping["on_sac_surface"], dtype=float)
    off_sac_surface = np.asarray(surface_mapping["off_sac_surface"], dtype=float)

    first = 1 if backward_compatible else 0
    sampled_x_idx = np.asarray(surface_mapping["sampled_x_idx"], dtype=int) + first
    sampled_y_idx = np.asarray(surface_mapping["sampled_y_idx"], dtype=int) + first

    x_vals = np.arange(sampled_x_idx[0], sampled_x_idx[-1] + 1, conformal_jump)
    y_vals = np.arange(sampled_y_idx[0], sampled_y_idx[-1] + 1, conformal_jump)
    on_depths = on_sac_surface[x_vals[:, None] - first, y_vals - first]
    off_depths = off_sac_surface[x_vals[:, None] - first, y_vals - first]

    expected = x_vals.size * y_vals.size
    if mapped_on.shape[0] != expected or mapped_off.shape[0] != expected:
        raise ValueError(
            "Surface mapping size mismatch: mapped_on/mapped_off do not match sampled grid."
        )
    return (
        x_vals,
        y_vals,
        on_depths,
        off_depths,
        mapped_on,
        mapped_off,
        float(np.median(on_depths)),
        float(np.median(off_depths)),
    )


def build_surface_correspondences(
    surface_mapping: dict,
    *,
    conformal_jump: int,
    backward_compatible: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Build paired control points for local LS registration.

    Returns
    -------
    on_input_pts, off_input_pts, on_output_pts, off_output_pts, med_z_on, med_z_off
    """
    (
        x_vals,
        y_vals,
        on_depths,
        off_depths,
        mapped_on,
        mapped_off,
        med_z_on,
        med_z_off,
    ) = surface_lattice(
        surface_mapping,
        conformal_jump=conformal_jump,
        backward_compatible=backward_compatible,
    )
    xmesh, ymesh = np.meshgrid(x_vals, y_vals, indexing="ij")
    on_input_pts = np.column_stack(
        [xmesh.ravel(order="F"), ymesh.ravel(order="F"), on_depths.ravel(order="F")]
    )
    off_input_pts = np.column_stack(
        [xmesh.ravel(order="F"), ymesh.ravel(order="F"), off_depths.ravel(order="F")]
    )
    on_output_pts = np.column_stack(
        [mapped_on[:, 0], mapped_on[:, 1], np.full(mapped_on.shape[0], med_z_on)]
    )
    off_output_pts = np.column_stack(
        [mapped_off[:, 0], mapped_off[:, 1], np.full(mapped_off.shape[0], med_z_off)]
    )
    return (
        on_input_pts,
        off_input_pts,
        on_output_pts,
        off_output_pts,
        med_z_on,
        med_z_off,
    )


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
        skiprows=1,  # skip the header line
        usecols=(5, 7, 6),  # X, Slice, Y in desired order
        dtype=np.float64,
    )

    x = data[:, 0] + 1  # KNOSSOS X  -> +1 for MATLAB convention
    y = data[:, 1]  # Slice (already 1-based)
    z = data[:, 2] + 1  # KNOSSOS Y  -> +1

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
