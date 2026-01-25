"""
Friction Coefficient Calculator for Persson Model
===================================================

Implements the viscoelastic friction coefficient (mu_visc) calculation
based on Persson's contact mechanics theory.

Mathematical Definition:
    mu_visc = (1/2) * integral[q0->q1] dq * q^3 * C(q) * P(q) * S(q)
              * integral[0->2pi] dphi * cos(phi) * Im[E(q*v*cos(phi)) / ((1-nu^2)*sigma0)]

where:
    - P(q) = erf(1 / (2*sqrt(G(q)))) : contact area ratio
    - S(q) = gamma + (1-gamma) * P(q)^2 : contact correction factor (gamma ~ 0.5)
    - Im[E(omega)] : loss modulus
    - omega = q * v * cos(phi) : excitation frequency

References:
    - RubberFriction Manual [Source 70, 936]
    - Hankook Tire Paper [Source 355]
    - Persson, B.N.J. (2001, 2006)
"""

import numpy as np
from scipy.special import erf
from typing import Callable, Optional, Tuple, Union
import warnings


class FrictionCalculator:
    """
    Calculator for viscoelastic friction coefficient (mu_visc).

    This class handles the double integral calculation for friction:
    - Inner integral: integration over angle phi from 0 to 2*pi
    - Outer integral: integration over wavenumber q from q0 to q1
    """

    def __init__(
        self,
        psd_func: Callable[[np.ndarray], np.ndarray],
        loss_modulus_func: Callable[[float, float], float],
        sigma_0: float,
        velocity: float,
        temperature: float = 20.0,
        poisson_ratio: float = 0.5,
        gamma: float = 0.5,
        n_angle_points: int = 72
    ):
        """
        Initialize friction calculator.

        Parameters
        ----------
        psd_func : callable
            Function C(q) that returns PSD values for given wavenumbers
        loss_modulus_func : callable
            Function that returns loss modulus Im[E(omega, T)] for given
            angular frequency and temperature
        sigma_0 : float
            Nominal contact pressure (Pa)
        velocity : float
            Sliding velocity (m/s)
        temperature : float, optional
            Temperature (Celsius), default 20.0
        poisson_ratio : float, optional
            Poisson's ratio of the material (default: 0.5)
        gamma : float, optional
            Contact correction factor (default: 0.5)
        n_angle_points : int, optional
            Number of points for angle integration (default: 72)
        """
        self.psd_func = psd_func
        self.loss_modulus_func = loss_modulus_func
        self.sigma_0 = sigma_0
        self.velocity = velocity
        self.temperature = temperature
        self.poisson_ratio = poisson_ratio
        self.gamma = gamma
        self.n_angle_points = n_angle_points

        # Precompute constant factor
        self.prefactor = 1.0 / ((1 - poisson_ratio**2) * sigma_0)

    def calculate_P_from_G(self, G: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate contact area ratio P(q) from G(q).

        P(q) = erf(1 / (2 * sqrt(G(q))))

        Parameters
        ----------
        G : float or np.ndarray
            G(q) values (dimensionless)

        Returns
        -------
        float or np.ndarray
            P(q) contact area ratio (0 to 1)
        """
        G = np.asarray(G)
        P = np.zeros_like(G)

        # Handle G close to zero (full contact)
        small_G_mask = G < 1e-10
        P[small_G_mask] = 1.0

        # Normal calculation for G > 0
        valid_mask = ~small_G_mask
        if np.any(valid_mask):
            sqrt_G = np.sqrt(G[valid_mask])
            # Prevent overflow for very large G
            arg = 1.0 / (2.0 * sqrt_G)
            arg = np.minimum(arg, 10.0)  # erf(10) ~ 1.0
            P[valid_mask] = erf(arg)

        return P

    def calculate_S_from_P(self, P: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate contact correction factor S(q) from P(q).

        S(q) = gamma + (1 - gamma) * P(q)^2

        Parameters
        ----------
        P : float or np.ndarray
            Contact area ratio P(q)

        Returns
        -------
        float or np.ndarray
            S(q) contact correction factor
        """
        P = np.asarray(P)
        return self.gamma + (1 - self.gamma) * P**2

    def _angle_integral_friction(self, q: float) -> float:
        """
        Compute inner integral over angle phi for friction.

        Integrates: integral[0->2pi] dphi * cos(phi) * Im[E(q*v*cos(phi)) / ((1-nu^2)*sigma0)]

        Due to the cos(phi) symmetry, we can use:
        integral[0->2pi] = 4 * integral[0->pi/2] for cos(phi) > 0 portion

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

        # Calculate frequencies for each angle
        # omega = q * v * cos(phi)
        omega = q * self.velocity * np.cos(phi)

        # Get cos(phi) values
        cos_phi = np.cos(phi)

        # Calculate integrand: cos(phi) * Im[E(omega)] / ((1-nu^2)*sigma0)
        integrand = np.zeros_like(phi)

        for i, (w, c) in enumerate(zip(omega, cos_phi)):
            # Handle zero and negative frequencies
            omega_abs = np.abs(w)
            if omega_abs < 1e-10:
                omega_abs = 1e-10

            # Get loss modulus at this frequency
            ImE = self.loss_modulus_func(omega_abs, self.temperature)

            # Calculate integrand term
            integrand[i] = c * ImE * self.prefactor

        # Numerical integration using trapezoidal rule
        result = np.trapezoid(integrand, phi)

        return result

    def calculate_mu_visc(
        self,
        q_array: np.ndarray,
        G_array: np.ndarray,
        C_q_array: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Tuple[float, dict]:
        """
        Calculate viscoelastic friction coefficient mu_visc.

        mu_visc = (1/2) * integral[q0->q1] dq * q^3 * C(q) * P(q) * S(q) * (angle_integral)

        Parameters
        ----------
        q_array : np.ndarray
            Array of wavenumbers (1/m) in ascending order
        G_array : np.ndarray
            Array of G(q) values (dimensionless, already divided by sigma0^2)
        C_q_array : np.ndarray, optional
            Array of PSD values C(q). If None, uses self.psd_func
        progress_callback : callable, optional
            Function to call with progress updates (0-100)

        Returns
        -------
        mu_visc : float
            Viscoelastic friction coefficient
        details : dict
            Dictionary containing intermediate values:
            - 'q': wavenumber array
            - 'P': contact area ratio P(q)
            - 'S': contact correction factor S(q)
            - 'C_q': PSD values
            - 'angle_integral': angle integration results
            - 'integrand': full integrand values
            - 'cumulative_mu': cumulative friction contribution
        """
        q_array = np.asarray(q_array)
        G_array = np.asarray(G_array)
        n = len(q_array)

        # Get PSD values
        if C_q_array is None:
            C_q_array = self.psd_func(q_array)
        else:
            C_q_array = np.asarray(C_q_array)

        # Calculate P(q) from G(q)
        P_array = self.calculate_P_from_G(G_array)

        # Calculate S(q) from P(q)
        S_array = self.calculate_S_from_P(P_array)

        # Calculate angle integral for each q
        angle_integral_array = np.zeros(n)
        integrand_array = np.zeros(n)

        for i, q in enumerate(q_array):
            # Calculate angle integral
            angle_integral_array[i] = self._angle_integral_friction(q)

            # Calculate full integrand: q^3 * C(q) * P(q) * S(q) * angle_integral
            integrand_array[i] = (
                q**3 * C_q_array[i] * P_array[i] * S_array[i] * angle_integral_array[i]
            )

            # Progress callback
            if progress_callback and i % max(1, n // 20) == 0:
                progress_callback(int((i + 1) / n * 100))

        # Numerical integration over q
        integral = np.trapezoid(integrand_array, q_array)

        # Apply prefactor 1/2
        mu_visc = 0.5 * integral

        # Calculate cumulative friction contribution
        cumulative_mu = np.zeros(n)
        for i in range(1, n):
            delta = 0.5 * (integrand_array[i-1] + integrand_array[i]) * (q_array[i] - q_array[i-1])
            cumulative_mu[i] = cumulative_mu[i-1] + 0.5 * delta

        details = {
            'q': q_array,
            'P': P_array,
            'S': S_array,
            'C_q': C_q_array,
            'angle_integral': angle_integral_array,
            'integrand': integrand_array,
            'cumulative_mu': cumulative_mu
        }

        return mu_visc, details

    def calculate_mu_visc_multi_velocity(
        self,
        q_array: np.ndarray,
        G_matrix: np.ndarray,
        v_array: np.ndarray,
        C_q_array: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Calculate mu_visc for multiple velocities.

        Parameters
        ----------
        q_array : np.ndarray
            Array of wavenumbers (1/m)
        G_matrix : np.ndarray
            2D array G(q, v) with shape (len(q), len(v))
        v_array : np.ndarray
            Array of velocities (m/s)
        C_q_array : np.ndarray, optional
            Array of PSD values C(q)
        progress_callback : callable, optional
            Function to call with progress updates (0-100)

        Returns
        -------
        mu_array : np.ndarray
            Array of mu_visc values for each velocity
        details : dict
            Dictionary containing results for each velocity
        """
        v_array = np.asarray(v_array)
        n_v = len(v_array)

        mu_array = np.zeros(n_v)
        all_details = []

        original_velocity = self.velocity

        for j, v in enumerate(v_array):
            # Update velocity
            self.velocity = v

            # Get G(q) for this velocity
            G_q = G_matrix[:, j]

            # Calculate mu_visc
            mu_visc, details = self.calculate_mu_visc(
                q_array, G_q, C_q_array
            )

            mu_array[j] = mu_visc
            details['velocity'] = v
            all_details.append(details)

            # Progress callback
            if progress_callback:
                progress_callback(int((j + 1) / n_v * 100))

        # Restore original velocity
        self.velocity = original_velocity

        return mu_array, {'velocities': v_array, 'details': all_details}

    def update_parameters(
        self,
        sigma_0: Optional[float] = None,
        velocity: Optional[float] = None,
        temperature: Optional[float] = None,
        poisson_ratio: Optional[float] = None,
        gamma: Optional[float] = None
    ):
        """
        Update calculation parameters.

        Parameters
        ----------
        sigma_0 : float, optional
            New nominal contact pressure (Pa)
        velocity : float, optional
            New sliding velocity (m/s)
        temperature : float, optional
            New temperature (Celsius)
        poisson_ratio : float, optional
            New Poisson's ratio
        gamma : float, optional
            New contact correction factor
        """
        if sigma_0 is not None:
            self.sigma_0 = sigma_0
            self.prefactor = 1.0 / ((1 - self.poisson_ratio**2) * sigma_0)

        if velocity is not None:
            self.velocity = velocity

        if temperature is not None:
            self.temperature = temperature

        if poisson_ratio is not None:
            self.poisson_ratio = poisson_ratio
            self.prefactor = 1.0 / ((1 - self.poisson_ratio**2) * self.sigma_0)

        if gamma is not None:
            self.gamma = gamma


def calculate_mu_visc_simple(
    q_array: np.ndarray,
    C_q_array: np.ndarray,
    G_array: np.ndarray,
    v: float,
    sigma0: float,
    func_ImE: Callable[[float, float], float],
    temperature: float = 20.0,
    nu: float = 0.5,
    gamma: float = 0.5,
    n_phi: int = 72
) -> Tuple[float, dict]:
    """
    Simplified function to calculate mu_visc without class instantiation.

    This is a convenience function for quick calculations.

    Parameters
    ----------
    q_array : np.ndarray
        Array of wavenumbers (1/m) from q0 to q1
    C_q_array : np.ndarray
        Array of PSD values C(q) at each wavenumber
    G_array : np.ndarray
        Array of G(q) values (dimensionless, already divided by sigma0^2)
        Note: G_array should be pre-computed G values for contact area calculation
    v : float
        Sliding velocity (m/s)
    sigma0 : float
        Nominal contact pressure (Pa)
    func_ImE : callable
        Function that returns loss modulus Im[E(omega, T)]
        Signature: func_ImE(omega, temperature) -> float
    temperature : float, optional
        Temperature (Celsius), default 20.0
    nu : float, optional
        Poisson's ratio, default 0.5
    gamma : float, optional
        Contact correction factor, default 0.5
    n_phi : int, optional
        Number of angle integration points, default 72

    Returns
    -------
    mu_visc : float
        Viscoelastic friction coefficient
    details : dict
        Dictionary with intermediate calculation values

    Example
    -------
    >>> def loss_modulus(omega, T):
    ...     # Simple example loss modulus function
    ...     return 1e7 * omega**0.2
    >>>
    >>> q = np.logspace(2, 6, 100)  # 100 to 1e6 1/m
    >>> C_q = 1e-14 * q**(-3.6)  # Fractal PSD with H=0.8
    >>> G = np.logspace(-4, 0, 100)  # Example G values
    >>>
    >>> mu, details = calculate_mu_visc_simple(
    ...     q_array=q,
    ...     C_q_array=C_q,
    ...     G_array=G,
    ...     v=0.01,
    ...     sigma0=0.3e6,
    ...     func_ImE=loss_modulus
    ... )
    >>> print(f"mu_visc = {mu:.4f}")
    """
    q_array = np.asarray(q_array)
    C_q_array = np.asarray(C_q_array)
    G_array = np.asarray(G_array)
    n = len(q_array)

    # Prefactor
    prefactor = 1.0 / ((1 - nu**2) * sigma0)

    # Calculate P(q) from G(q)
    P_array = np.zeros_like(G_array)
    small_G_mask = G_array < 1e-10
    P_array[small_G_mask] = 1.0
    valid_mask = ~small_G_mask
    if np.any(valid_mask):
        sqrt_G = np.sqrt(G_array[valid_mask])
        arg = np.minimum(1.0 / (2.0 * sqrt_G), 10.0)
        P_array[valid_mask] = erf(arg)

    # Calculate S(q) from P(q)
    S_array = gamma + (1 - gamma) * P_array**2

    # Angle array
    phi = np.linspace(0, 2 * np.pi, n_phi)

    # Calculate angle integral and full integrand for each q
    angle_integral_array = np.zeros(n)
    integrand_array = np.zeros(n)

    for i, q in enumerate(q_array):
        # Calculate omega = q * v * cos(phi)
        omega = q * v * np.cos(phi)
        cos_phi = np.cos(phi)

        # Calculate integrand: cos(phi) * Im[E(omega)] * prefactor
        integrand_phi = np.zeros_like(phi)
        for j, (w, c) in enumerate(zip(omega, cos_phi)):
            omega_abs = max(np.abs(w), 1e-10)
            ImE = func_ImE(omega_abs, temperature)
            integrand_phi[j] = c * ImE * prefactor

        # Angle integral
        angle_integral_array[i] = np.trapezoid(integrand_phi, phi)

        # Full integrand: q^3 * C(q) * P(q) * S(q) * angle_integral
        integrand_array[i] = (
            q**3 * C_q_array[i] * P_array[i] * S_array[i] * angle_integral_array[i]
        )

    # Numerical integration over q
    integral = np.trapezoid(integrand_array, q_array)
    mu_visc = 0.5 * integral

    # Cumulative friction
    cumulative_mu = np.zeros(n)
    for i in range(1, n):
        delta = 0.5 * (integrand_array[i-1] + integrand_array[i]) * (q_array[i] - q_array[i-1])
        cumulative_mu[i] = cumulative_mu[i-1] + 0.5 * delta

    details = {
        'q': q_array,
        'P': P_array,
        'S': S_array,
        'C_q': C_q_array,
        'G': G_array,
        'angle_integral': angle_integral_array,
        'integrand': integrand_array,
        'cumulative_mu': cumulative_mu,
        'velocity': v,
        'sigma0': sigma0,
        'temperature': temperature
    }

    return mu_visc, details


def apply_nonlinear_strain_correction(
    E_prime: np.ndarray,
    E_double_prime: np.ndarray,
    strain: float,
    f_curve: Callable[[float], float],
    g_curve: Callable[[float], float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply nonlinear strain correction to modulus values.

    When high local strain occurs at contact asperities, the linear
    viscoelastic moduli need to be corrected using the f,g functions
    from strain sweep experiments.

    E'_nonlinear = E' * f(strain)
    E''_nonlinear = E'' * g(strain)

    Parameters
    ----------
    E_prime : np.ndarray
        Storage modulus E' (Pa)
    E_double_prime : np.ndarray
        Loss modulus E'' (Pa)
    strain : float
        Local strain amplitude (fraction, not %)
    f_curve : callable
        Function f(strain) for storage modulus correction
        Returns value in range [0, 1]
    g_curve : callable
        Function g(strain) for loss modulus correction
        Returns value in range [0, 1]

    Returns
    -------
    E_prime_corrected : np.ndarray
        Corrected storage modulus
    E_double_prime_corrected : np.ndarray
        Corrected loss modulus
    """
    f_val = f_curve(strain)
    g_val = g_curve(strain)

    # Clip to valid range
    f_val = np.clip(f_val, 0.0, 1.0)
    g_val = np.clip(g_val, 0.0, 1.0)

    E_prime_corrected = E_prime * f_val
    E_double_prime_corrected = E_double_prime * g_val

    return E_prime_corrected, E_double_prime_corrected


def estimate_local_strain(
    G_area: float,
    C_q: float,
    q: float,
    sigma0: float,
    E_prime: float,
    method: str = 'persson'
) -> float:
    """
    Estimate local strain at contact asperities.

    The local strain depends on the contact geometry and material stiffness.
    Higher roughness and softer materials lead to higher local strains.

    Parameters
    ----------
    G_area : float
        G(q) area function value (dimensionless)
    C_q : float
        PSD value C(q) at wavenumber q
    q : float
        Wavenumber (1/m)
    sigma0 : float
        Nominal contact pressure (Pa)
    E_prime : float
        Storage modulus E' (Pa)
    method : str, optional
        Estimation method: 'persson' or 'simple'

    Returns
    -------
    strain : float
        Estimated local strain (fraction)
    """
    if method == 'persson':
        # Persson's approach: strain related to surface slope
        # Local strain ~ sqrt(C(q) * q^4) / (E'/sigma0)
        if E_prime < 1e3:
            E_prime = 1e3  # Minimum modulus

        slope_rms = np.sqrt(C_q * q**4) if C_q > 0 else 0
        strain = slope_rms * sigma0 / E_prime

    elif method == 'simple':
        # Simple estimate: strain ~ sqrt(G) * sigma0/E
        if G_area > 0:
            strain = np.sqrt(G_area) * sigma0 / max(E_prime, 1e3)
        else:
            strain = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    # Limit to physically reasonable range
    strain = np.clip(strain, 0.0, 1.0)  # 0 to 100%

    return float(strain)
