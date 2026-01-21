"""
Contact Mechanics Calculations
===============================

Implements Persson's contact mechanics theory:
- Stress probability distribution
- Real contact area fraction
- Contact pressure distribution
"""

import numpy as np
from scipy import integrate, special
from typing import Optional, Tuple, Callable


class ContactMechanics:
    """
    Contact mechanics calculations using Persson theory.

    Calculates stress distribution and real contact area based on
    the G(q) function and surface roughness.
    """

    def __init__(
        self,
        G_function: Callable[[np.ndarray], np.ndarray],
        sigma_0: float,
        q_values: np.ndarray,
        G_values: Optional[np.ndarray] = None
    ):
        """
        Initialize contact mechanics calculator.

        Parameters
        ----------
        G_function : callable
            Function that returns G(q) for given wavenumbers
        sigma_0 : float
            Nominal contact pressure (Pa)
        q_values : np.ndarray
            Wavenumber array for calculations
        G_values : np.ndarray, optional
            Precomputed G values. If None, will compute from G_function.
        """
        self.G_function = G_function
        self.sigma_0 = sigma_0
        self.q_values = np.asarray(q_values)

        if G_values is not None:
            self.G_values = np.asarray(G_values)
        else:
            self.G_values = G_function(self.q_values)

        # Maximum magnification (ζ_max)
        self.zeta_max = self.q_values[-1] / self.q_values[0]

    def stress_variance(self, zeta: float) -> float:
        """
        Calculate stress variance at magnification ζ.

        Var(σ) = σ₀² * G(ζ)

        Parameters
        ----------
        zeta : float
            Magnification factor ζ = q/q₀

        Returns
        -------
        float
            Stress variance (Pa²)
        """
        # Interpolate G at this zeta
        G_zeta = np.interp(
            np.log(zeta),
            np.log(self.q_values / self.q_values[0]),
            self.G_values
        )

        return self.sigma_0**2 * G_zeta

    def stress_distribution_gaussian(
        self,
        sigma_range: np.ndarray,
        zeta: float
    ) -> np.ndarray:
        """
        Calculate stress probability distribution P(σ, ζ).

        Assumes Gaussian distribution:
        P(σ, ζ) = (1/√(2πVar)) * exp(-(σ - σ₀)² / (2*Var))

        Parameters
        ----------
        sigma_range : np.ndarray
            Stress values to evaluate (Pa)
        zeta : float
            Magnification factor

        Returns
        -------
        np.ndarray
            Probability density at each stress value
        """
        variance = self.stress_variance(zeta)
        std_dev = np.sqrt(variance)

        if std_dev < 1e-10:
            # Delta function at sigma_0
            P = np.zeros_like(sigma_range)
            idx = np.argmin(np.abs(sigma_range - self.sigma_0))
            P[idx] = 1.0 / (sigma_range[1] - sigma_range[0])
            return P

        # Gaussian distribution
        P = (1.0 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
            -(sigma_range - self.sigma_0)**2 / (2 * variance)
        )

        return P

    def contact_area_fraction(
        self,
        zeta: Optional[float] = None,
        method: str = 'gaussian'
    ) -> float:
        """
        Calculate real contact area fraction A/A₀.

        The contact area is the fraction where stress > 0.

        Parameters
        ----------
        zeta : float, optional
            Magnification factor. If None, uses maximum zeta.
        method : str, optional
            Method for calculation: 'gaussian' or 'exact'
            (default: 'gaussian')

        Returns
        -------
        float
            Contact area fraction (0 to 1)
        """
        if zeta is None:
            zeta = self.zeta_max

        variance = self.stress_variance(zeta)
        std_dev = np.sqrt(variance)

        if std_dev < 1e-10:
            # No roughness, full contact
            return 1.0

        if method == 'gaussian':
            # For Gaussian distribution, contact area is:
            # A/A₀ = (1/2) * erfc(-σ₀ / (√2 * σ_std))
            # = (1/2) * (1 + erf(σ₀ / (√2 * σ_std)))

            normalized_pressure = self.sigma_0 / (np.sqrt(2) * std_dev)
            contact_fraction = 0.5 * (1 + special.erf(normalized_pressure))

            return contact_fraction
        else:
            # Numerical integration
            sigma_max = self.sigma_0 + 5 * std_dev
            sigma_range = np.linspace(0, sigma_max, 1000)

            P = self.stress_distribution_gaussian(sigma_range, zeta)
            contact_fraction = np.trapz(P, sigma_range)

            return min(contact_fraction, 1.0)

    def mean_contact_pressure(
        self,
        zeta: Optional[float] = None
    ) -> float:
        """
        Calculate mean pressure in real contact area.

        p_mean = σ₀ / (A/A₀)

        Parameters
        ----------
        zeta : float, optional
            Magnification factor

        Returns
        -------
        float
            Mean contact pressure (Pa)
        """
        area_fraction = self.contact_area_fraction(zeta)

        if area_fraction < 1e-10:
            return np.inf

        return self.sigma_0 / area_fraction

    def rms_contact_pressure(
        self,
        zeta: Optional[float] = None
    ) -> float:
        """
        Calculate RMS pressure variation.

        Parameters
        ----------
        zeta : float, optional
            Magnification factor

        Returns
        -------
        float
            RMS pressure (Pa)
        """
        if zeta is None:
            zeta = self.zeta_max

        variance = self.stress_variance(zeta)
        return np.sqrt(variance)

    def contact_statistics(
        self,
        zeta: Optional[float] = None
    ) -> dict:
        """
        Calculate comprehensive contact statistics.

        Parameters
        ----------
        zeta : float, optional
            Magnification factor

        Returns
        -------
        dict
            Dictionary containing:
            - 'area_fraction': Real contact area fraction
            - 'mean_pressure': Mean pressure in contact
            - 'rms_pressure': RMS pressure variation
            - 'stress_variance': Variance of stress distribution
            - 'magnification': Magnification factor used
        """
        if zeta is None:
            zeta = self.zeta_max

        area_frac = self.contact_area_fraction(zeta)
        mean_p = self.mean_contact_pressure(zeta)
        rms_p = self.rms_contact_pressure(zeta)
        variance = self.stress_variance(zeta)

        return {
            'area_fraction': area_frac,
            'mean_pressure': mean_p,
            'rms_pressure': rms_p,
            'stress_variance': variance,
            'magnification': zeta,
            'nominal_pressure': self.sigma_0
        }

    def plot_stress_distribution(
        self,
        zeta: Optional[float] = None,
        n_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stress distribution data for plotting.

        Parameters
        ----------
        zeta : float, optional
            Magnification factor
        n_points : int, optional
            Number of points for the distribution

        Returns
        -------
        sigma : np.ndarray
            Stress values (Pa)
        P_sigma : np.ndarray
            Probability density
        """
        if zeta is None:
            zeta = self.zeta_max

        std_dev = np.sqrt(self.stress_variance(zeta))

        # Create stress range
        sigma_min = max(0, self.sigma_0 - 4 * std_dev)
        sigma_max = self.sigma_0 + 4 * std_dev

        sigma = np.linspace(sigma_min, sigma_max, n_points)
        P_sigma = self.stress_distribution_gaussian(sigma, zeta)

        return sigma, P_sigma

    def contact_area_vs_magnification(
        self,
        zeta_range: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate contact area fraction as function of magnification.

        Parameters
        ----------
        zeta_range : np.ndarray, optional
            Range of magnification factors. If None, uses q_values.

        Returns
        -------
        zeta : np.ndarray
            Magnification factors
        area_fraction : np.ndarray
            Contact area fractions
        """
        if zeta_range is None:
            zeta_range = self.q_values / self.q_values[0]

        area_fractions = np.array([
            self.contact_area_fraction(z) for z in zeta_range
        ])

        return zeta_range, area_fractions

    def update_nominal_pressure(self, sigma_0: float):
        """
        Update nominal pressure and recalculate if needed.

        Parameters
        ----------
        sigma_0 : float
            New nominal contact pressure (Pa)
        """
        self.sigma_0 = sigma_0
