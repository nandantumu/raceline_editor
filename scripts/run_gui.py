#!/usr/bin/env python
# This program enables users to edit and visualize racelines for ground robots
# Copyright (C) 2025  Renukanandan Tumu

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import sys

from PyQt6.QtWidgets import QApplication
from raceline_editor.renderer.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Get CSV path from command line argument if provided
    csv_path = "../example_trajectories/blue.csv"  # Default path updated
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    w = MainWindow(csv_path=csv_path)  # Pass csv_path to MainWindow
    w.resize(1200, 800)  # Adjusted default size
    w.show()
    sys.exit(app.exec())
