from dataclasses import dataclass, field
from typing import Any, List, Optional


TRAJECTORY_ATTRIBUTES = (
    "s",
    "x",
    "y",
    "z",
    "psi",
    "kappa",
    "vx",
    "ax",
    "theta",
    "phi",
)


@dataclass
class TrajectoryPoint:
    """Represents a single point in a trajectory."""

    s: float  # Track progress or arc length
    x: float
    y: float
    z: float
    psi: float  # Yaw
    kappa: float  # Curvature
    vx: float  # Speed
    ax: float  # Acceleration
    theta: float  # Pitch
    phi: float  # Roll
    # Add other relevant attributes like timestamp, etc. as needed
    # e.g., timestamp: Optional[float] = None


@dataclass
class RecordedTrajectory:
    """Represents a baseline recorded trajectory from a source like a CSV file."""

    name: str
    points: List[TrajectoryPoint] = field(default_factory=list)
    metadata: dict = field(
        default_factory=dict
    )  # For storing original file path, headers, etc.

    def get_arrays(self):
        """Extract trajectory data as numpy arrays for visualization."""
        import numpy as np

        if not self.points:
            return {attr: np.array([]) for attr in TRAJECTORY_ATTRIBUTES}

        return {
            attr: np.array([getattr(p, attr) for p in self.points])
            for attr in TRAJECTORY_ATTRIBUTES
        }

    def add_point(
        self,
        s: float,
        x: float,
        y: float,
        z: float,
        psi: float,
        kappa: float,
        vx: float,
        ax: float,
        theta: float,
        phi: float,
    ):
        self.points.append(
            TrajectoryPoint(
                s=s,
                x=x,
                y=y,
                z=z,
                psi=psi,
                kappa=kappa,
                vx=vx,
                ax=ax,
                theta=theta,
                phi=phi,
            )
        )


@dataclass
class SplineTrajectory:
    """Represents an interpolated spline version of a trajectory."""

    name: str
    original_trajectory_name: str  # To link back to the source RecordedTrajectory
    points: List[TrajectoryPoint] = field(
        default_factory=list
    )  # Densely sampled points from the spline
    spline_type: str = "unknown"  # e.g., "cubic_spline", "bspline"
    spline_parameters: Any = None  # To store coefficients or control points, depending on the spline library used
    metadata: dict = field(
        default_factory=dict
    )  # For storing interpolation settings, etc.

    # Arrays for direct access to spline data for visualization
    s_array: Any = None  # numpy array of s values
    x_array: Any = None  # numpy array of x values
    y_array: Any = None  # numpy array of y values
    z_array: Any = None  # numpy array of z values
    psi_array: Any = None  # numpy array of psi values
    kappa_array: Any = None  # numpy array of kappa values
    vx_array: Any = None  # numpy array of vx values
    ax_array: Any = None  # numpy array of ax values
    theta_array: Any = None  # numpy array of theta values
    phi_array: Any = None  # numpy array of phi values

    def get_arrays(self):
        """Returns a dictionary of all arrays for easy access."""
        return {
            attr: getattr(self, f"{attr}_array")
            for attr in TRAJECTORY_ATTRIBUTES
        }

    # You might add methods here to generate points from spline_parameters
    def sample_high_resolution(self, num_points: int = 1000):
        """
        Generates high-resolution arrays for visualization by sampling the spline at `num_points` points.
        This method assumes that `spline_parameters` contains callable functions for each attribute,
        or that `points` are densely sampled and can be interpolated.
        """

        if not self.points:
            # No points to sample from
            return

        # Use the min and max s values from the current points
        s_values = np.array([p.s for p in self.points])
        s_min, s_max = np.min(s_values), np.max(s_values)
        s_sampled = np.linspace(s_min, s_max, num_points)

        # If spline_parameters contains callables, use them; otherwise, interpolate from points
        def interp(attr):
            arr = np.array([getattr(p, attr) for p in self.points])
            return np.interp(s_sampled, s_values, arr)

        self.s_array = s_sampled
        for attr in TRAJECTORY_ATTRIBUTES:
            if attr == "s":
                continue
            setattr(self, f"{attr}_array", interp(attr))


if __name__ == "__main__":
    # Example Usage

    # Create a recorded trajectory
    recorded_traj = RecordedTrajectory(name="Blue_Lap_1")
    recorded_traj.add_point(
        s=0.0,
        x=0.0,
        y=0.0,
        z=0.0,
        psi=0.0,
        kappa=0.0,
        vx=10.0,
        ax=0.0,
        theta=0.0,
        phi=0.0,
    )
    recorded_traj.add_point(
        s=1.1,
        x=1.0,
        y=0.5,
        z=0.1,
        psi=0.1,
        kappa=0.01,
        vx=10.5,
        ax=0.5,
        theta=0.01,
        phi=0.0,
    )
    recorded_traj.add_point(
        s=2.3,
        x=2.0,
        y=1.5,
        z=0.2,
        psi=0.2,
        kappa=0.02,
        vx=11.0,
        ax=0.2,
        theta=0.02,
        phi=0.01,
    )
    recorded_traj.add_point(
        s=3.4,
        x=3.0,
        y=2.0,
        z=0.3,
        psi=0.3,
        kappa=0.01,
        vx=10.8,
        ax=-0.1,
        theta=0.01,
        phi=0.0,
    )
    recorded_traj.metadata["source_file"] = "blue.csv"

    print("Recorded Trajectory:")
    print(f"  Name: {recorded_traj.name}")
    print(f"  Number of points: {len(recorded_traj.points)}")
    print(f"  First point: {recorded_traj.points[0]}")
    print(f"  Metadata: {recorded_traj.metadata}")

    # Create a spline trajectory (conceptually)
    # In a real scenario, points would be generated by an interpolation function
    spline_traj = SplineTrajectory(
        name="Blue_Lap_1_Spline",
        original_trajectory_name=recorded_traj.name,
        spline_type="cubic_spline",
    )
    # Add some interpolated points (example)
    # Note: For a real spline, all fields would be calculated/interpolated
    spline_traj.points.append(
        TrajectoryPoint(
            s=0.0,
            x=0.0,
            y=0.0,
            z=0.0,
            psi=0.0,
            kappa=0.0,
            vx=10.0,
            ax=0.0,
            theta=0.0,
            phi=0.0,
        )
    )
    spline_traj.points.append(
        TrajectoryPoint(
            s=0.5,
            x=0.5,
            y=0.2,
            z=0.05,
            psi=0.05,
            kappa=0.005,
            vx=10.2,
            ax=0.2,
            theta=0.005,
            phi=0.0,
        )
    )  # Interpolated point
    spline_traj.points.append(
        TrajectoryPoint(
            s=1.1,
            x=1.0,
            y=0.5,
            z=0.1,
            psi=0.1,
            kappa=0.01,
            vx=10.5,
            ax=0.5,
            theta=0.01,
            phi=0.0,
        )
    )
    spline_traj.metadata["interpolation_resolution"] = 0.1

    print("\nSpline Trajectory:")
    print(f"  Name: {spline_traj.name}")
    print(f"  Original: {spline_traj.original_trajectory_name}")
    print(f"  Spline Type: {spline_traj.spline_type}")
    print(f"  Number of points: {len(spline_traj.points)}")
    print(f"  First point: {spline_traj.points[0]}")
    print(f"  Metadata: {spline_traj.metadata}")
