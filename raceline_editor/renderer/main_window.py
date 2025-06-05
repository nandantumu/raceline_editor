import sys
import pyqtgraph as pg
import numpy as np  # Added for array manipulation
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QLabel,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainterPath

# Imports for raceline processing
from raceline_editor.raceline.importer import import_raceline_csv
from raceline_editor.raceline.models import RecordedTrajectory, TrajectoryPoint
from raceline_editor.raceline.processing import create_spline_from_recorded


class MultiColorLine(pg.GraphicsObject):
    """A graphics object that displays a multi-colored line with colors determined by values."""

    def __init__(self, x, y, values, colormap, width=1, connect="all"):
        super().__init__()
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.values = np.asarray(values)
        self.colormap = colormap
        self.width = width

        # Determine connection points
        if connect == "all":
            self.connect_array = np.ones(len(self.x) - 1, dtype=bool)
        elif connect == "pairs":
            self.connect_array = np.zeros(len(self.x) - 1, dtype=bool)
            self.connect_array[::2] = True
        elif connect == "finite":
            self.connect_array = (
                np.isfinite(self.x[:-1])
                & np.isfinite(self.x[1:])
                & np.isfinite(self.y[:-1])
                & np.isfinite(self.y[1:])
            )
        else:
            self.connect_array = connect

        self.path = None
        self.generatePath()

    def generatePath(self):
        self.path = QPainterPath()

        # Create path
        for i in range(len(self.x) - 1):
            if self.connect_array[i]:
                # Add line segment
                self.path.moveTo(self.x[i], self.y[i])
                self.path.lineTo(self.x[i + 1], self.y[i + 1])

    def boundingRect(self):
        if self.path is None:
            return pg.QtCore.QRectF()
        return self.path.boundingRect()

    def paint(self, painter, option, widget):
        if self.path is None:
            return

        painter.setRenderHint(painter.RenderHint.Antialiasing)

        # Find min/max for color scaling
        v_min = np.min(self.values)
        v_max = np.max(self.values)
        v_range = max(0.1, v_max - v_min)  # Avoid division by zero

        # Draw each segment with a color based on its value
        for i in range(len(self.x) - 1):
            if self.connect_array[i]:
                # Calculate color
                norm_v = (self.values[i] - v_min) / v_range
                qcolor = self.colormap.mapToQColor(norm_v)

                # Set pen color
                pen = pg.mkPen(color=qcolor, width=self.width)
                painter.setPen(pen)

                # Draw line segment
                painter.drawLine(
                    QPointF(self.x[i], self.y[i]), QPointF(self.x[i + 1], self.y[i + 1])
                )

    def setData(self, x, y, values):
        """Update the data in the line."""
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.values = np.asarray(values)

        # Recompute connect_array if length changed
        if len(self.connect_array) != len(self.x) - 1:
            self.connect_array = np.ones(len(self.x) - 1, dtype=bool)

        self.generatePath()
        self.prepareGeometryChange()
        self.update()


class MainWindow(QMainWindow):
    def __init__(self, csv_path=None):
        super().__init__()
        self.csv_path = csv_path
        self.setWindowTitle("Raceline Editor")
        self.num_spline_segments = 30  # Number of segments for spline interpolation

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Plots
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        main_layout.addWidget(plots_widget, 3)  # Give more space to plots

        # Top plot: X-Y
        self.xy_plot_widget = pg.PlotWidget()
        self.xy_plot_widget.setLabel("left", "Y position (m)")
        self.xy_plot_widget.setLabel("bottom", "X position (m)")
        self.xy_plot_widget.showGrid(x=True, y=True)
        self.xy_plot_widget.getViewBox().setAspectLocked(True)
        plots_layout.addWidget(self.xy_plot_widget)

        # Bottom plot: Velocity vs. S
        self.vel_plot_widget = pg.PlotWidget()
        self.vel_plot_widget.setLabel("left", "Velocity (m/s)")
        self.vel_plot_widget.setLabel("bottom", "S (m) - Distance along path")
        self.vel_plot_widget.showGrid(x=True, y=True)
        plots_layout.addWidget(self.vel_plot_widget)

        # Right side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_widget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(controls_widget, 1)  # Less space for controls

        # Save Waypoints Button
        self.save_button = QPushButton("Save Waypoints")
        controls_layout.addWidget(self.save_button)

        # Reset Velocities Button
        self.reset_button = QPushButton("Reset Velocities")
        controls_layout.addWidget(self.reset_button)

        # Point Edits GroupBox
        point_edits_group = QGroupBox("Point Edits")
        point_edits_layout = QFormLayout(point_edits_group)

        self.point_index_label = QLabel(
            "No point selected"
        )  # To show which point is being edited
        point_edits_layout.addRow(self.point_index_label)

        self.x_edit = QLineEdit()
        point_edits_layout.addRow("X (m):", self.x_edit)

        self.y_edit = QLineEdit()
        point_edits_layout.addRow("Y (m):", self.y_edit)

        self.z_edit = QLineEdit()
        point_edits_layout.addRow("Z (m):", self.z_edit)

        self.velocity_edit = QLineEdit()
        point_edits_layout.addRow("Velocity (m/s):", self.velocity_edit)

        self.yaw_edit = QLineEdit()
        point_edits_layout.addRow("Yaw (rad):", self.yaw_edit)

        self.pitch_edit = QLineEdit()
        point_edits_layout.addRow("Pitch (rad):", self.pitch_edit)

        self.roll_edit = QLineEdit()
        point_edits_layout.addRow("Roll (rad):", self.roll_edit)

        controls_layout.addWidget(point_edits_group)

        controls_layout.addStretch()  # Pushes everything up

        # Load and plot data from CSV if path is provided, otherwise plot example
        if self.csv_path:
            self.load_and_plot_trajectory_data()
        else:
            self.plot_example_data()

    def parse_raw_data_to_recorded_trajectory(
        self, raw_data: list, trajectory_name: str
    ) -> RecordedTrajectory:
        recorded_traj = RecordedTrajectory(name=trajectory_name)
        if not raw_data or len(raw_data) < 2:  # Expect header + at least one data row
            print("Warning: CSV data is empty or has no data rows.")
            return recorded_traj

        header_line = raw_data[0]
        # If the header line is a list of strings (already split), join it first.
        # This handles cases where csv.reader might have already processed it.
        if isinstance(header_line, list):
            header_line_str = ",".join(map(str, header_line))
        else:
            header_line_str = str(header_line)

        if header_line_str.startswith("#"):
            header_line_str = header_line_str[1:]

        # Split the potentially modified header string back into a list of headers
        # and strip whitespace from each header item.
        # This assumes headers are comma-separated if they were joined.
        # If the original raw_data[0] was already a list of correctly split headers,
        # this re-splitting might be redundant but should be harmless if they were clean.
        # A more robust way would be to check if raw_data[0] was a string initially.
        # For now, we'll assume raw_data[0] from import_raceline_csv is a list of strings.

        if isinstance(raw_data[0], list):
            # If the original header was a list, process it item by item
            processed_header = []
            first_item = str(raw_data[0][0])
            if first_item.startswith("#"):
                processed_header.append(first_item[1:].strip())
                processed_header.extend([str(h).strip() for h in raw_data[0][1:]])
            else:
                processed_header = [str(h).strip() for h in raw_data[0]]
            header = processed_header
        else:  # Should not happen with current importer, but as a fallback
            header = [h.strip() for h in header_line_str.split(",")]

        # Define a mapping from TrajectoryPoint fields to possible CSV column names (case-insensitive search)
        # Order within the list defines priority.
        column_map_config = {
            "s": ["s_m", "s", "S"],
            "x": ["x_m", "x", "X"],
            "y": ["y_m", "y", "Y"],
            "z": ["z_m", "z", "Z"],
            "psi": ["psi_rad", "psi", "Psi", "yaw_rad", "yaw", "Yaw"],
            "kappa": ["kappa_radpm", "kappa", "Kappa", "curv_radpm"],
            "vx": ["vx_mps", "vx", "Vx", "v_mps"],
            "ax": ["ax_mps2", "ax", "Ax", "a_mps2"],
            "theta": ["pitch_rad", "pitch", "Pitch", "theta_rad", "theta"],
            "phi": ["roll_rad", "roll", "Roll", "phi_rad", "phi"],
        }

        found_col_indices = {}
        missing_mandatory_fields = False

        for field_name, possible_csv_names in column_map_config.items():
            found = False
            for csv_name_variant in possible_csv_names:
                try:
                    # Case-insensitive search for the column variant
                    actual_csv_header_name = next(
                        h_item
                        for h_item in header
                        if h_item.lower() == csv_name_variant.lower()
                    )
                    found_col_indices[field_name] = header.index(actual_csv_header_name)
                    found = True
                    break  # Found a match for this field_name
                except (StopIteration, ValueError):
                    continue  # This variant not found, try next
            if not found:
                # For this application, all fields in column_map_config are considered essential.
                print(
                    f"Error: Mandatory field '{field_name}' (tried: {possible_csv_names}) not found in CSV header: {header}"
                )
                missing_mandatory_fields = True

        if missing_mandatory_fields:
            print("Error: Could not parse CSV due to missing mandatory columns.")
            return recorded_traj  # Return empty trajectory

        for i, row_values in enumerate(raw_data[1:]):
            try:
                point_data_dict = {}
                valid_row = True
                for field_name, col_idx in found_col_indices.items():
                    if col_idx < len(row_values):
                        point_data_dict[field_name] = float(row_values[col_idx])
                    else:
                        print(
                            f"Warning: Row {i+1} is too short for column '{header[col_idx]}' (field '{field_name}'). Skipping row."
                        )
                        valid_row = False
                        break

                if valid_row:
                    recorded_traj.points.append(TrajectoryPoint(**point_data_dict))

            except ValueError as e:
                print(
                    f"Warning: Could not convert data to float in row {i+1}, skipping row: {row_values}. Error: {e}"
                )
            except (
                IndexError
            ):  # Should be caught by the length check above, but as a safeguard
                print(
                    f"Warning: Row {i+1} has fewer columns than expected by header, skipping row: {row_values}"
                )

        recorded_traj.metadata["source_file"] = self.csv_path
        return recorded_traj

    def load_and_plot_trajectory_data(self):
        if not self.csv_path:
            print("No CSV path provided.")
            self.plot_example_data()  # Fallback to example
            return

        print(f"Loading trajectory from: {self.csv_path}")
        raw_data = import_raceline_csv(self.csv_path)

        if not raw_data:
            print(f"Failed to import data from {self.csv_path}.")
            self.plot_example_data()  # Fallback
            return

        # Derive a name for the trajectory, e.g., from the filename
        # For "path/to/blue.csv", name becomes "blue"
        traj_name = (
            self.csv_path.split("/")[-1].split(".")[0]
            if "/" in self.csv_path
            else self.csv_path.split(".")[0]
        )

        recorded_trajectory = self.parse_raw_data_to_recorded_trajectory(
            raw_data, traj_name
        )

        if not recorded_trajectory.points:
            print(
                "No points in recorded trajectory after parsing, plotting example data."
            )
            self.plot_example_data()
            return

        print(
            f"Successfully parsed {len(recorded_trajectory.points)} points for '{recorded_trajectory.name}'."
        )

        spline_trajectory = create_spline_from_recorded(
            recorded_trajectory, self.num_spline_segments
        )

        if not spline_trajectory.points:
            print("Failed to generate spline trajectory, plotting example data.")
            self.plot_example_data()
            return

        print(f"Generated spline with {len(spline_trajectory.points)} points.")

        # Extract data for plotting
        s_spline = np.array([p.s for p in spline_trajectory.points])
        x_spline = np.array([p.x for p in spline_trajectory.points])
        y_spline = np.array([p.y for p in spline_trajectory.points])
        vx_spline = np.array([p.vx for p in spline_trajectory.points])

        # Also extract the original raw trajectory points for visualization
        x_raw = np.array([p.x for p in recorded_trajectory.points])
        y_raw = np.array([p.y for p in recorded_trajectory.points])
        s_raw = np.array([p.s for p in recorded_trajectory.points])
        vx_raw = np.array([p.vx for p in recorded_trajectory.points])

        # Clear previous plots
        self.xy_plot_widget.clear()
        self.vel_plot_widget.clear()

        # Plot new data
        # self.xy_plot_widget.plot(
        #     x_spline,
        #     y_spline,
        #     pen=pg.mkPen("c", width=2),
        #     name=f"{spline_trajectory.name} (Path)",
        # )
        # self.vel_plot_widget.plot(
        #     s_spline,
        #     vx_spline,
        #     pen=pg.mkPen("m", width=2),
        #     name=f"{spline_trajectory.name} (Velocity)",
        # )

        # --- First add the raw trajectory points as white scatter points
        # They will be rendered beneath the colored line
        scatter = pg.ScatterPlotItem(
            x=x_raw,
            y=y_raw,
            pen=pg.mkPen("w"),
            brush=pg.mkBrush("w"),
            size=5,
            name="Original points",
        )
        self.xy_plot_widget.addItem(scatter)

        # Now add the spline trajectory with velocity-based coloring ON TOP
        # Create a custom colormap for velocity: green (low) -> yellow (medium) -> red (high speed)
        custom_colors = np.array(
            [
                [0, 200, 0, 255],  # Green (low speed)
                [255, 255, 0, 255],  # Yellow (medium speed)
                [255, 0, 0, 255],  # Red (high speed)
            ]
        )
        pos = np.linspace(0, 1, len(custom_colors))
        colormap = pg.ColorMap(pos, custom_colors)

        # Plot the trajectory with velocity-based coloring
        multi_color_line = MultiColorLine(
            x_spline, y_spline, vx_spline, colormap, width=5
        )
        self.xy_plot_widget.addItem(multi_color_line)

        # Plot velocity vs. distance
        # First add the raw velocity points as a white line
        self.vel_plot_widget.plot(
            s_raw,
            vx_raw,
            pen=pg.mkPen("w", width=1.5),
            name="Original velocity points",
        )

        # Then add the spline velocity with velocity-based coloring ON TOP
        # Use only the points before the loop closure to prevent connecting the end back to the start
        # Create the velocity colored line for the velocity plot
        multi_color_vel_line = MultiColorLine(
            s_spline[:-1],  # Exclude the last point which would connect back to the start
            vx_spline[:-1],
            vx_spline[:-1],  # Use velocity as color parameter
            colormap,
            width=2,
        )
        self.vel_plot_widget.addItem(multi_color_vel_line)

        print("Trajectory plotted.")

    def plot_example_data(self):
        # Example X-Y plot
        x_coords = [0, 1, 2, 3, 2, 1, 0]
        y_coords = [0, 1, 0, -1, -2, -1, 0]
        self.xy_plot_widget.plot(x_coords, y_coords, pen=pg.mkPen("b", width=2))

        # Example Velocity plot
        s_coords = [0, 1, 2, 3, 4, 5, 6]
        velocities = [10, 12, 15, 13, 10, 8, 10]
        self.vel_plot_widget.plot(s_coords, velocities, pen=pg.mkPen("r", width=2))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # In a real scenario, you'd load data from self.csv_path
    # For this standalone test, we can pass a dummy path or None
    # Test with a known CSV file if available, e.g., blue.csv relative to project root
    # This example assumes you run this file directly from its directory for testing.
    # For the actual app, run_gui.py provides the path.
    example_csv_path = (
        "../../example_trajectories/blue.csv"  # Adjusted for direct run from renderer
    )
    window = MainWindow(csv_path=example_csv_path)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
