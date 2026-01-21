"""
Power Spectral Density (PSD) Models
====================================

Implements various models for surface roughness characterization:
- Fractal (self-affine) surfaces
- Measured PSD data with interpolation
- Combined multi-scale models
"""

import numpy as np
from scipy import interpolate
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class PSDModel(ABC):
    """Abstract base class for PSD models."""

    @abstractmethod
    def __call__(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate PSD C(q) for given wavenumbers.

        Parameters
        ----------
        q : np.ndarray
            Wavenumber values (1/m)

        Returns
        -------
        np.ndarray
            PSD values C(q) (m^4)
        """
        pass

    @abstractmethod
    def get_rms_roughness(
        self,
        q_min: float,
        q_max: float,
        n_points: int = 1000
    ) -> float:
        """
        Calculate RMS roughness over a wavenumber range.

        h_rms = √(∫ C(q) d²q)

        Parameters
        ----------
        q_min : float
            Minimum wavenumber (1/m)
        q_max : float
            Maximum wavenumber (1/m)
        n_points : int, optional
            Number of integration points

        Returns
        -------
        float
            RMS roughness (m)
        """
        pass


class FractalPSD(PSDModel):
    """
    Fractal (self-affine) surface PSD model.

    C(q) = C₀ * q^(-2(1+H))

    where:
        - H: Hurst exponent (0 < H < 1)
        - C₀: amplitude prefactor
    """

    def __init__(
        self,
        hurst_exponent: float,
        amplitude: Optional[float] = None,
        rms_roughness: Optional[float] = None,
        rms_slope: Optional[float] = None,
        q_min: Optional[float] = None,
        q_max: Optional[float] = None
    ):
        """
        Initialize fractal PSD model.

        The amplitude can be specified directly or calculated from
        rms_roughness or rms_slope.

        Parameters
        ----------
        hurst_exponent : float
            Hurst exponent H (0 < H < 1)
            H ≈ 0.8 is typical for many surfaces
        amplitude : float, optional
            Amplitude prefactor C₀ (m^(4+2H))
        rms_roughness : float, optional
            RMS roughness h_rms (m). Used to calculate C₀ if amplitude not given.
        rms_slope : float, optional
            RMS slope. Used to calculate C₀ if amplitude and rms_roughness not given.
        q_min : float, optional
            Minimum wavenumber for RMS calculation (1/m)
        q_max : float, optional
            Maximum wavenumber for RMS calculation (1/m)
        """
        if not 0 < hurst_exponent < 1:
            raise ValueError("Hurst exponent must be between 0 and 1")

        self.H = hurst_exponent
        self.exponent = -2 * (1 + self.H)

        # Determine amplitude
        if amplitude is not None:
            self.C0 = amplitude
        elif rms_roughness is not None and q_min is not None and q_max is not None:
            # Calculate C₀ from h_rms
            # h_rms² = ∫∫ C(q) d²q ≈ 2π ∫ C(q) q dq
            # For fractal: h_rms² = 2π C₀ ∫ q^(1-2(1+H)) dq
            #                      = 2π C₀ [q^(2-2H) / (2-2H)]_{q_min}^{q_max}

            exp_int = 2 - 2*self.H
            if abs(exp_int) < 1e-10:
                # Special case: H = 1
                integral = np.log(q_max / q_min)
            else:
                integral = (q_max**exp_int - q_min**exp_int) / exp_int

            self.C0 = rms_roughness**2 / (2 * np.pi * integral)
        elif rms_slope is not None and q_min is not None and q_max is not None:
            # Calculate C₀ from RMS slope
            # (∇h)² = ∫∫ q² C(q) d²q ≈ 2π ∫ C(q) q³ dq

            exp_int = 4 - 2*self.H
            if abs(exp_int) < 1e-10:
                integral = np.log(q_max / q_min)
            else:
                integral = (q_max**exp_int - q_min**exp_int) / exp_int

            self.C0 = rms_slope**2 / (2 * np.pi * integral)
        else:
            raise ValueError(
                "Must specify either amplitude, or (rms_roughness with q_min/q_max), "
                "or (rms_slope with q_min/q_max)"
            )

        self.q_min = q_min
        self.q_max = q_max

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """Calculate PSD C(q) = C₀ * q^(-2(1+H))."""
        q = np.asarray(q)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.C0 * np.power(q, self.exponent)
            result[q <= 0] = 0
        return result

    def get_rms_roughness(
        self,
        q_min: Optional[float] = None,
        q_max: Optional[float] = None,
        n_points: int = 1000
    ) -> float:
        """Calculate RMS roughness over wavenumber range."""
        if q_min is None:
            q_min = self.q_min
        if q_max is None:
            q_max = self.q_max

        if q_min is None or q_max is None:
            raise ValueError("Must specify q_min and q_max")

        # Create logarithmic wavenumber array
        q = np.logspace(np.log10(q_min), np.log10(q_max), n_points)
        C_q = self(q)

        # Integrate: h_rms² = 2π ∫ C(q) q dq
        integrand = C_q * q
        integral = np.trapz(integrand, q)
        h_rms = np.sqrt(2 * np.pi * integral)

        return h_rms

    def get_rms_slope(
        self,
        q_min: Optional[float] = None,
        q_max: Optional[float] = None,
        n_points: int = 1000
    ) -> float:
        """Calculate RMS slope over wavenumber range."""
        if q_min is None:
            q_min = self.q_min
        if q_max is None:
            q_max = self.q_max

        if q_min is None or q_max is None:
            raise ValueError("Must specify q_min and q_max")

        q = np.logspace(np.log10(q_min), np.log10(q_max), n_points)
        C_q = self(q)

        # Integrate: (∇h)² = 2π ∫ C(q) q³ dq
        integrand = C_q * q**3
        integral = np.trapz(integrand, q)
        slope_rms = np.sqrt(2 * np.pi * integral)

        return slope_rms


class MeasuredPSD(PSDModel):
    """
    PSD model based on measured data with interpolation.
    """

    def __init__(
        self,
        q_data: np.ndarray,
        C_data: np.ndarray,
        interpolation_kind: str = 'linear',
        extrapolate: bool = False,
        fill_value: float = 0.0
    ):
        """
        Initialize measured PSD model.

        Parameters
        ----------
        q_data : np.ndarray
            Measured wavenumber values (1/m)
        C_data : np.ndarray
            Measured PSD values (m^4)
        interpolation_kind : str, optional
            Type of interpolation: 'linear', 'cubic', 'log-log', etc.
            (default: 'linear')
        extrapolate : bool, optional
            If True, extrapolate outside data range (default: False)
        fill_value : float, optional
            Value to use outside data range if not extrapolating (default: 0.0)
        """
        self.q_data = np.asarray(q_data)
        self.C_data = np.asarray(C_data)

        if len(self.q_data) != len(self.C_data):
            raise ValueError("q_data and C_data must have same length")

        # Sort by q
        sort_idx = np.argsort(self.q_data)
        self.q_data = self.q_data[sort_idx]
        self.C_data = self.C_data[sort_idx]

        self.q_min = self.q_data[0]
        self.q_max = self.q_data[-1]

        # Create interpolator
        if interpolation_kind == 'log-log':
            # Interpolate in log-log space
            log_q = np.log10(self.q_data[self.q_data > 0])
            log_C = np.log10(self.C_data[self.q_data > 0])
            self._interpolator = interpolate.interp1d(
                log_q,
                log_C,
                kind='linear',
                fill_value='extrapolate' if extrapolate else fill_value,
                bounds_error=False
            )
            self._log_interp = True
        else:
            self._interpolator = interpolate.interp1d(
                self.q_data,
                self.C_data,
                kind=interpolation_kind,
                fill_value='extrapolate' if extrapolate else fill_value,
                bounds_error=False
            )
            self._log_interp = False

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """Interpolate PSD at given wavenumbers."""
        q = np.asarray(q)

        if self._log_interp:
            log_q = np.log10(np.maximum(q, 1e-20))
            log_C = self._interpolator(log_q)
            result = 10**log_C
        else:
            result = self._interpolator(q)

        # Ensure positive values
        result = np.maximum(result, 0)

        return result

    def get_rms_roughness(
        self,
        q_min: Optional[float] = None,
        q_max: Optional[float] = None,
        n_points: int = 1000
    ) -> float:
        """Calculate RMS roughness from measured PSD."""
        if q_min is None:
            q_min = self.q_min
        if q_max is None:
            q_max = self.q_max

        q = np.logspace(np.log10(q_min), np.log10(q_max), n_points)
        C_q = self(q)

        integrand = C_q * q
        integral = np.trapz(integrand, q)
        h_rms = np.sqrt(2 * np.pi * integral)

        return h_rms


class CombinedPSD(PSDModel):
    """
    Combined PSD model from multiple PSD components.

    Useful for surfaces with multiple roughness scales.
    """

    def __init__(self, psd_models: list):
        """
        Initialize combined PSD.

        Parameters
        ----------
        psd_models : list of PSDModel
            List of PSD models to combine (additively)
        """
        self.models = psd_models

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """Calculate combined PSD as sum of components."""
        q = np.asarray(q)
        result = np.zeros_like(q, dtype=float)

        for model in self.models:
            result += model(q)

        return result

    def get_rms_roughness(
        self,
        q_min: float,
        q_max: float,
        n_points: int = 1000
    ) -> float:
        """Calculate combined RMS roughness."""
        q = np.logspace(np.log10(q_min), np.log10(q_max), n_points)
        C_q = self(q)

        integrand = C_q * q
        integral = np.trapz(integrand, q)
        h_rms = np.sqrt(2 * np.pi * integral)

        return h_rms


class RollOffPSD(PSDModel):
    """
    Fractal PSD with roll-off at small and large wavelengths.

    C(q) = C₀ * q^(-2(1+H)) / ((1 + (q/q_r)²)^(γ/2))

    where q_r is the roll-off wavenumber and γ controls roll-off sharpness.
    """

    def __init__(
        self,
        hurst_exponent: float,
        amplitude: float,
        q_rolloff: float,
        rolloff_gamma: float = 2.0
    ):
        """
        Initialize roll-off PSD model.

        Parameters
        ----------
        hurst_exponent : float
            Hurst exponent H
        amplitude : float
            Amplitude C₀
        q_rolloff : float
            Roll-off wavenumber (1/m)
        rolloff_gamma : float, optional
            Roll-off exponent (default: 2.0)
        """
        self.H = hurst_exponent
        self.C0 = amplitude
        self.q_r = q_rolloff
        self.gamma = rolloff_gamma

    def __call__(self, q: np.ndarray) -> np.ndarray:
        """Calculate PSD with roll-off."""
        q = np.asarray(q)

        with np.errstate(divide='ignore', invalid='ignore'):
            fractal_part = self.C0 * np.power(q, -2*(1 + self.H))
            rolloff_part = np.power(1 + (q / self.q_r)**2, -self.gamma / 2)
            result = fractal_part * rolloff_part
            result[q <= 0] = 0

        return result

    def get_rms_roughness(
        self,
        q_min: float,
        q_max: float,
        n_points: int = 1000
    ) -> float:
        """Calculate RMS roughness with roll-off."""
        q = np.logspace(np.log10(q_min), np.log10(q_max), n_points)
        C_q = self(q)

        integrand = C_q * q
        integral = np.trapz(integrand, q)
        h_rms = np.sqrt(2 * np.pi * integral)

        return h_rms
