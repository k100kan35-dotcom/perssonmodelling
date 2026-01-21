"""
G(q) Calculator for Persson Friction Model
===========================================

Implements the core G(q) calculation which represents the elastic energy
density variance in the contact stress distribution.

Mathematical Definition:
    G(q) = (1/8) ∫₀^q (1/√q') dq' q'³ C(q') ∫₀^(2π) dφ |E(q'v cosφ) / ((1-ν²)σ₀)|²

where:
    - q: wavenumber (1/m)
    - C(q): Power Spectral Density of surface roughness
    - E(ω): Complex modulus at frequency ω = qv cosφ
    - v: sliding velocity (m/s)
    - ν: Poisson's ratio
    - σ₀: nominal contact pressure (Pa)
"""

import numpy as np
from scipy import integrate
from typing import Callable, Optional, Tuple
import warnings


class GCalculator:
    """
    Calculator for G(q) in Persson friction theory.

    This class handles the double integral calculation:
    - Inner integral: integration over angle φ from 0 to 2π
    - Outer integral: integration over wavenumber q from q₀ to q
    """

    def __init__(
        self,
        psd_func: Callable[[np.ndarray], np.ndarray],
        modulus_func: Callable[[np.ndarray], complex],
        sigma_0: float,
        velocity: float,
        poisson_ratio: float = 0.5,
        n_angle_points: int = 36,
        integration_method: str = 'trapz'
    ):
        """
        Initialize G(q) calculator.

        Parameters
        ----------
        psd_func : callable
            Function C(q) that returns PSD values for given wavenumbers
        modulus_func : callable
            Function E(ω) that returns complex modulus for given frequencies
        sigma_0 : float
            Nominal contact pressure (Pa)
        velocity : float
            Sliding velocity (m/s)
        poisson_ratio : float, optional
            Poisson's ratio of the material (default: 0.5)
        n_angle_points : int, optional
            Number of points for angle integration (default: 36)
        integration_method : str, optional
            Method for numerical integration: 'trapz', 'simpson', or 'quad'
            (default: 'trapz')
        """
        self.psd_func = psd_func
        self.modulus_func = modulus_func
        self.sigma_0 = sigma_0
        self.velocity = velocity
        self.poisson_ratio = poisson_ratio
        self.n_angle_points = n_angle_points
        self.integration_method = integration_method

        # Precompute constant factor
        self.prefactor = 1.0 / ((1 - poisson_ratio**2) * sigma_0)

    def _angle_integral(self, q: float) -> float:
        """
        Compute inner integral over angle φ.

        Integrates: ∫₀^(2π) dφ |E(qv cosφ) / ((1-ν²)σ₀)|²

        Parameters
        ----------
        q : float
            Wavenumber (1/m)

        Returns
        -------
        float
            Result of angle integration
        """
        # Create angle array
        phi = np.linspace(0, 2 * np.pi, self.n_angle_points)
        dphi = phi[1] - phi[0]

        # Calculate frequencies for each angle
        # ω = q * v * cos(φ)
        omega = q * self.velocity * np.cos(phi)

        # Handle zero and negative frequencies
        # For negative frequencies, use E(-ω) = E*(ω)
        # For zero frequency, use a small offset
        omega_eval = np.abs(omega)
        omega_eval[omega_eval < 1e-10] = 1e-10

        # Get complex modulus values
        E_values = np.array([self.modulus_func(w) for w in omega_eval])

        # Calculate integrand: |E(ω) / ((1-ν²)σ₀)|²
        integrand = np.abs(E_values * self.prefactor)**2

        # Numerical integration using trapezoidal rule
        result = np.trapz(integrand, phi)

        return result

    def _integrand_q(self, q: float) -> float:
        """
        Compute integrand for q integration.

        Calculates: (1/√q) * q³ * C(q) * (angle integral)

        Parameters
        ----------
        q : float
            Wavenumber (1/m)

        Returns
        -------
        float
            Integrand value
        """
        if q <= 0:
            return 0.0

        # Get PSD value
        C_q = self.psd_func(np.array([q]))[0]

        # Compute angle integral
        angle_int = self._angle_integral(q)

        # Combine: (1/√q) * q³ * C(q) * angle_integral
        # = q^(5/2) * C(q) * angle_integral
        result = q**2.5 * C_q * angle_int

        return result

    def calculate_G(
        self,
        q_values: np.ndarray,
        q_min: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate G(q) for an array of wavenumbers.

        Parameters
        ----------
        q_values : np.ndarray
            Array of wavenumbers (1/m) in ascending order
        q_min : float, optional
            Lower integration limit (default: first value in q_values)

        Returns
        -------
        np.ndarray
            G(q) values corresponding to each q in q_values
        """
        q_values = np.asarray(q_values)

        if q_min is None:
            q_min = q_values[0]

        G_values = np.zeros_like(q_values, dtype=float)

        for i, q in enumerate(q_values):
            if q <= q_min:
                G_values[i] = 0.0
                continue

            # Create integration points from q_min to q
            # Use logarithmic spacing for better accuracy
            n_points = max(20, int(np.log10(q / q_min) * 20))
            q_int = np.logspace(np.log10(q_min), np.log10(q), n_points)

            # Calculate integrand at each point
            integrand_values = np.array([self._integrand_q(qi) for qi in q_int])

            # Numerical integration
            # Since we use log spacing, we need to integrate properly
            if self.integration_method == 'trapz':
                integral = np.trapz(integrand_values, q_int)
            elif self.integration_method == 'simpson':
                from scipy.integrate import simpson
                if len(q_int) % 2 == 0:
                    # Simpson's rule requires odd number of points
                    integral = np.trapz(integrand_values, q_int)
                else:
                    integral = simpson(integrand_values, q_int)
            else:
                # Use quad for higher accuracy (slower)
                integral, _ = integrate.quad(
                    self._integrand_q,
                    q_min,
                    q,
                    limit=100,
                    epsabs=1e-12,
                    epsrel=1e-10
                )

            # Apply prefactor 1/8
            G_values[i] = integral / 8.0

        return G_values

    def calculate_G_single(
        self,
        q: float,
        q_min: float
    ) -> float:
        """
        Calculate G(q) for a single wavenumber.

        Parameters
        ----------
        q : float
            Wavenumber (1/m)
        q_min : float
            Lower integration limit (1/m)

        Returns
        -------
        float
            G(q) value
        """
        result = self.calculate_G(np.array([q]), q_min=q_min)
        return result[0]

    def calculate_G_cumulative(
        self,
        q_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate cumulative G(q) efficiently for many q values.

        This method is more efficient than calling calculate_G repeatedly
        as it reuses integration results.

        Parameters
        ----------
        q_values : np.ndarray
            Array of wavenumbers (1/m) in ascending order

        Returns
        -------
        q_out : np.ndarray
            Wavenumber array (may be refined for accuracy)
        G_out : np.ndarray
            Cumulative G(q) values
        """
        q_values = np.asarray(q_values)
        q_values = np.sort(q_values)

        # Ensure we have enough points for accurate integration
        # Add intermediate points if needed
        q_refined = q_values.copy()

        G_cumulative = np.zeros_like(q_refined)

        # Integrate step by step
        for i in range(1, len(q_refined)):
            q_lower = q_refined[i-1]
            q_upper = q_refined[i]

            # Calculate integrand at boundaries
            integrand_lower = self._integrand_q(q_lower)
            integrand_upper = self._integrand_q(q_upper)

            # Trapezoidal rule for this interval
            delta_G = 0.5 * (integrand_lower + integrand_upper) * (q_upper - q_lower)

            # Add to cumulative sum
            G_cumulative[i] = G_cumulative[i-1] + delta_G / 8.0

        return q_refined, G_cumulative

    def update_parameters(
        self,
        sigma_0: Optional[float] = None,
        velocity: Optional[float] = None,
        poisson_ratio: Optional[float] = None
    ):
        """
        Update calculation parameters.

        Parameters
        ----------
        sigma_0 : float, optional
            New nominal contact pressure (Pa)
        velocity : float, optional
            New sliding velocity (m/s)
        poisson_ratio : float, optional
            New Poisson's ratio
        """
        if sigma_0 is not None:
            self.sigma_0 = sigma_0
            self.prefactor = 1.0 / ((1 - self.poisson_ratio**2) * sigma_0)

        if velocity is not None:
            self.velocity = velocity

        if poisson_ratio is not None:
            self.poisson_ratio = poisson_ratio
            self.prefactor = 1.0 / ((1 - self.poisson_ratio**2) * self.sigma_0)
