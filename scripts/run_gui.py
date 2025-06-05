#!/usr/bin/env python
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
