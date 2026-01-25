"""
Persson Friction Model
======================

A Python implementation of Persson's contact mechanics and friction theory
for rough surfaces in contact with viscoelastic materials.

Main components:
- G(q) calculation: Core energy density calculation
- Contact mechanics: Stress distribution and real contact area
- PSD models: Surface roughness characterization
- Viscoelastic properties: Material master curves and frequency dependence
- Friction calculation: Viscoelastic friction coefficient (mu_visc)
"""

__version__ = "1.1.0"
__author__ = "Persson Modelling Team"

from .core.g_calculator import GCalculator
from .core.psd_models import PSDModel, FractalPSD, MeasuredPSD
from .core.viscoelastic import ViscoelasticMaterial
from .core.contact import ContactMechanics
from .core.friction import FrictionCalculator, calculate_mu_visc_simple

__all__ = [
    "GCalculator",
    "PSDModel",
    "FractalPSD",
    "MeasuredPSD",
    "ViscoelasticMaterial",
    "ContactMechanics",
    "FrictionCalculator",
    "calculate_mu_visc_simple",
]
