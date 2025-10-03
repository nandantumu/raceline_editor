import numpy as np
from dataclasses import replace
from scipy.interpolate import CubicSpline
from .models import (
    RecordedTrajectory,
    SplineTrajectory,
    TrajectoryPoint,
    TRAJECTORY_ATTRIBUTES,
)


_NON_S_ATTRIBUTES = tuple(attr for attr in TRAJECTORY_ATTRIBUTES if attr != "s")
_SPLINE_ATTRIBUTE_ORDER = _NON_S_ATTRIBUTES + ("s",)


def _spline_helpers_close_arrays(*arrays):
    """Appends the first element of each array to its end to ensure closure."""
    return tuple(np.concatenate([arr, arr[:1]]) for arr in arrays)


def _create_constant_spline_trajectory(
    first_point_data: TrajectoryPoint, num_spline_segments: int, original_name: str
) -> SplineTrajectory:
    """Creates a SplineTrajectory with all points being a copy of the first_point_data."""
    # Create arrays with constant values for direct use
    num_points = num_spline_segments + 1
    arrays = {
        f"{attr}_array": np.full(num_points, getattr(first_point_data, attr))
        for attr in TRAJECTORY_ATTRIBUTES
    }

    spline_traj = SplineTrajectory(
        name=f"{original_name}_spline",
        original_trajectory_name=original_name,
        spline_type="constant",
        # Store arrays directly
        **arrays,
    )

    # num_spline_segments means num_spline_segments + 1 points
    for _ in range(num_points):
        spline_traj.points.append(replace(first_point_data))
    return spline_traj


def create_spline_from_recorded(
    recorded_trajectory: RecordedTrajectory, num_spline_segments: int
) -> SplineTrajectory:
    """
    Creates a SplineTrajectory by interpolating a RecordedTrajectory using periodic cubic splines.

    Args:
        recorded_trajectory: The input RecordedTrajectory object.
        num_spline_segments: The number of segments to divide the spline into.
                             The output will have num_spline_segments + 1 points.

    Returns:
        A SplineTrajectory object.
    """
    if not recorded_trajectory.points:
        empty_arrays = np.array([])
        arrays = {f"{attr}_array": empty_arrays for attr in TRAJECTORY_ATTRIBUTES}
        return SplineTrajectory(
            name=f"{recorded_trajectory.name}_spline_empty",
            original_trajectory_name=recorded_trajectory.name,
            spline_type="empty",
            # Include empty arrays
            **arrays,
        )

    if len(recorded_trajectory.points) == 1:
        return _create_constant_spline_trajectory(
            recorded_trajectory.points[0], num_spline_segments, recorded_trajectory.name
        )

    # Extract data into numpy arrays
    attr_arrays = {
        attr: np.array([getattr(p, attr) for p in recorded_trajectory.points])
        for attr in TRAJECTORY_ATTRIBUTES
    }

    # Ensure control points are periodic for CubicSpline with bc_type='periodic'
    # The y values for CubicSpline must satisfy y[0] == y[-1] for periodic.
    ctrl_arrays = attr_arrays.copy()

    # Check if the original data is already closed for periodicity.
    # If not, append the first point's data to the end of each array.
    is_closed = True
    x_ctrl = ctrl_arrays["x"]
    y_ctrl = ctrl_arrays["y"]
    if not (np.allclose(x_ctrl[0], x_ctrl[-1]) and np.allclose(y_ctrl[0], y_ctrl[-1])):
        is_closed = False

    if not is_closed:
        ctrl_arrays = {
            attr: np.concatenate((arr, [arr[0]]))
            for attr, arr in ctrl_arrays.items()
        }
        x_ctrl = ctrl_arrays["x"]
        y_ctrl = ctrl_arrays["y"]

    # Calculate normalized path parameter 't' based on cumulative chord length of (x,y)
    dx_path = np.diff(x_ctrl)
    dy_path = np.diff(y_ctrl)
    dist_path = np.hypot(dx_path, dy_path)
    t_path = np.concatenate(([0], np.cumsum(dist_path)))
    total_path_length = t_path[-1]

    if total_path_length < 1e-9:  # All control points are co-located
        return _create_constant_spline_trajectory(
            recorded_trajectory.points[0],  # Use the first original point
            num_spline_segments,
            recorded_trajectory.name,
        )

    t_norm_path = t_path / total_path_length

    # Create a joint cubic spline for all attributes vs. t_norm_path
    # Ensure t_norm_path has unique values for CubicSpline
    # If t_norm_path has duplicate values (e.g. multiple identical points), spline creation can fail.
    # A simple way to handle this is to remove consecutive duplicates in t_norm_path and corresponding y_ctrl rows.
    unique_indices = np.concatenate(
        ([True], np.diff(t_norm_path) > 1e-9)
    )  # Keep first, then where t changes

    if not np.all(unique_indices):  # If there were duplicates
        t_norm_path_unique = t_norm_path[unique_indices]
        # Need to ensure the last point is still there if it was unique, and that it matches the first for periodicity
        if not np.allclose(t_norm_path_unique[-1], 1.0) and np.allclose(
            t_norm_path[-1], 1.0
        ):
            t_norm_path_unique = np.append(
                t_norm_path_unique, t_norm_path[-1]
            )  # ensure last point is 1.0
            # And corresponding y_ctrl row needs to be appended
            y_indices_for_unique = np.where(unique_indices)[0].tolist() + [
                -1
            ]  # take unique and last original
        else:
            y_indices_for_unique = np.where(unique_indices)[0]

    else:  # All t_norm_path points were unique enough
        t_norm_path_unique = t_norm_path
        y_indices_for_unique = slice(None)  # use all original y_ctrl rows

    y_ctrl_stacked = np.column_stack(
        tuple(
            ctrl_arrays[attr][y_indices_for_unique]
            for attr in _SPLINE_ATTRIBUTE_ORDER
        )
    )

    # Ensure the y_ctrl_stacked is also periodic for the unique t_norm_path
    if not np.allclose(y_ctrl_stacked[0], y_ctrl_stacked[-1]):
        # This can happen if duplicate removal removed the original closing point,
        # or if the original data wasn't perfectly periodic after unique_indices filtering.
        # Forcibly make the control points for the spline periodic.
        # This might be an issue if t_norm_path_unique itself isn't [0, ..., 1]
        # A robust way: if t_norm_path_unique[0] is 0 and t_norm_path_unique[-1] is 1,
        # then y_ctrl_stacked[0] must equal y_ctrl_stacked[-1].
        if np.allclose(t_norm_path_unique[0], 0.0) and np.allclose(
            t_norm_path_unique[-1], 1.0
        ):
            y_ctrl_stacked[-1] = y_ctrl_stacked[0]

    cs_all = CubicSpline(t_norm_path_unique, y_ctrl_stacked, bc_type="periodic", axis=0)

    # Sample the spline
    # num_spline_segments means num_spline_segments intervals, so num_spline_segments+1 points if endpoint=True
    # To match reference, sample num_spline_segments points (endpoint=False), then close.
    ts_sample = np.linspace(0, 1, num_spline_segments, endpoint=False)
    sampled_values_period = cs_all(ts_sample)

    # Close the sampled arrays
    closed_arrays = _spline_helpers_close_arrays(
        *(
            sampled_values_period[:, i]
            for i in range(sampled_values_period.shape[1])
        )
    )
    closed_arrays_by_attr = dict(zip(_SPLINE_ATTRIBUTE_ORDER, closed_arrays))

    # Create SplineTrajectory object
    arrays_for_spline = {
        f"{attr}_array": closed_arrays_by_attr[attr]
        for attr in TRAJECTORY_ATTRIBUTES
    }
    spline_traj = SplineTrajectory(
        name=f"{recorded_trajectory.name}_spline",
        original_trajectory_name=recorded_trajectory.name,
        spline_type="cubic_periodic",
        spline_parameters=cs_all,  # Store the spline object itself
        # Store arrays directly for easier access during visualization
        **arrays_for_spline,
    )

    num_points = len(closed_arrays_by_attr["s"])
    for i in range(num_points):  # Should be num_spline_segments + 1 points
        point_kwargs = {
            attr: closed_arrays_by_attr[attr][i] for attr in TRAJECTORY_ATTRIBUTES
        }
        point = TrajectoryPoint(**point_kwargs)
        spline_traj.points.append(point)

    return spline_traj


if __name__ == "__main__":
    # Example Usage (requires RecordedTrajectory and TrajectoryPoint to be defined)

    # Create a dummy RecordedTrajectory
    rec_traj = RecordedTrajectory(name="TestTrack")
    points_data = [
        (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
            0.0,
        ),  # s, x, y, z, psi, kappa, vx, ax, theta, phi
        (10.0, 10.0, 5.0, 0.1, 0.1, 0.01, 12.0, 0.5, 0.01, 0.0),
        (20.0, 20.0, 0.0, 0.2, 0.0, 0.02, 15.0, 0.3, 0.0, 0.0),
        (30.0, 10.0, -5.0, 0.1, -0.1, 0.01, 12.0, -0.5, -0.01, 0.0),
        # (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0) # Not explicitly closed
    ]
    for p_data in points_data:
        rec_traj.points.append(TrajectoryPoint(*p_data))

    print(f"Original trajectory '{rec_traj.name}' has {len(rec_traj.points)} points.")
    for i, p in enumerate(rec_traj.points):
        print(f"  Point {i}: s={p.s:.1f}, x={p.x:.1f}, y={p.y:.1f}, vx={p.vx:.1f}")

    num_segments = 50
    spline_trajectory = create_spline_from_recorded(rec_traj, num_segments)

    print(
        f"\nGenerated spline trajectory '{spline_trajectory.name}' with {len(spline_trajectory.points)} points ({num_segments} segments)."
    )
    if spline_trajectory.points:
        print(f"  Spline Type: {spline_trajectory.spline_type}")
        print(
            f"  First point: s={spline_trajectory.points[0].s:.2f}, x={spline_trajectory.points[0].x:.2f}, y={spline_trajectory.points[0].y:.2f}, vx={spline_trajectory.points[0].vx:.2f}"
        )
        print(
            f"  Last point:  s={spline_trajectory.points[-1].s:.2f}, x={spline_trajectory.points[-1].x:.2f}, y={spline_trajectory.points[-1].y:.2f}, vx={spline_trajectory.points[-1].vx:.2f}"
        )

        # Check closure
        if np.allclose(
            spline_trajectory.points[0].x, spline_trajectory.points[-1].x
        ) and np.allclose(
            spline_trajectory.points[0].y, spline_trajectory.points[-1].y
        ):
            print("  Path is closed (first and last x,y points match).")
        else:
            print("  Path is NOT closed (first and last x,y points differ).")

    # Test with single point
    rec_traj_single = RecordedTrajectory(name="SinglePoint")
    rec_traj_single.points.append(TrajectoryPoint(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    spline_single = create_spline_from_recorded(rec_traj_single, 5)
    print(
        f"\nSpline from single point: {len(spline_single.points)} points. First x: {spline_single.points[0].x}"
    )

    # Test with empty points
    rec_traj_empty = RecordedTrajectory(name="EmptyTrack")
    spline_empty = create_spline_from_recorded(rec_traj_empty, 5)
    print(
        f"\nSpline from empty trajectory: {len(spline_empty.points)} points, type: {spline_empty.spline_type}"
    )
