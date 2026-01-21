#!/usr/bin/env python3
"""
Main entry point for Persson Friction Model application.
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persson_model.gui.main_window import main

if __name__ == "__main__":
    main()
