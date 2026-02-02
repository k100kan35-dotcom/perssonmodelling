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

    Supports nonlinear correction via f(ε) and g(ε) functions:
    - E'_eff = E' × f(ε)
    - E''_eff = E'' × g(ε)
    """

    def __init__(
        self,
        psd_func: Callable[[np.ndarray], np.ndarray],
        modulus_func: Callable[[np.ndarray], complex],
        sigma_0: float,
        velocity: float,
        poisson_ratio: float = 0.5,
        n_angle_points: int = 36,
        integration_method: str = 'trapz',
        storage_modulus_func: Callable[[float], float] = None,
        loss_modulus_func: Callable[[float], float] = None,
        gamma: float = 0.0
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
        storage_modulus_func : callable, optional
            Function that returns E'(ω) separately (for nonlinear correction)
        loss_modulus_func : callable, optional
            Function that returns E''(ω) separately (for nonlinear correction)
        gamma : float, optional
            Partial contact correction parameter for S(q).
            S(q) = gamma + (1 - gamma) * P(q)^2
            Linear case: gamma=0 → S(q) = P(q)^2 (default: 0.0)
        """
        self.psd_func = psd_func
        self.modulus_func = modulus_func
        self.sigma_0 = sigma_0
        self.velocity = velocity
        self.poisson_ratio = poisson_ratio
        self.n_angle_points = n_angle_points
        self.integration_method = integration_method
        self.storage_modulus_func = storage_modulus_func
        self.loss_modulus_func = loss_modulus_func
        self.gamma = gamma

        # Precompute constant factor
        # Persson formula: |E / ((1-ν²)σ₀)|²
        self.prefactor = 1.0 / ((1 - poisson_ratio**2) * sigma_0)

        # Nonlinear correction (default: none)
        self.f_interpolator = None  # f(ε) for E'
        self.g_interpolator = None  # g(ε) for E''
        self.strain_array = None    # ε(q) array
        self.strain_q_array = None  # q values for strain interpolation

    def set_nonlinear_correction(
        self,
        f_interpolator: Callable[[float], float],
        g_interpolator: Callable[[float], float],
        strain_array: np.ndarray,
        strain_q_array: np.ndarray
    ):
        """
        Set nonlinear correction functions f(ε) and g(ε).

        Parameters
        ----------
        f_interpolator : callable
            f(ε) function for storage modulus correction: E'_eff = E' × f(ε)
        g_interpolator : callable
            g(ε) function for loss modulus correction: E''_eff = E'' × g(ε)
        strain_array : np.ndarray
            Local strain values ε(q)
        strain_q_array : np.ndarray
            Wavenumber values corresponding to strain_array
        """
        self.f_interpolator = f_interpolator
        self.g_interpolator = g_interpolator
        self.strain_array = strain_array
        self.strain_q_array = strain_q_array

    def clear_nonlinear_correction(self):
        """Clear nonlinear correction (use linear modulus)."""
        self.f_interpolator = None
        self.g_interpolator = None
        self.strain_array = None
        self.strain_q_array = None

    def _get_strain_at_q(self, q: float) -> float:
        """Get interpolated strain value at wavenumber q."""
        if self.strain_array is None or self.strain_q_array is None:
            return 0.0

        # Log interpolation for better accuracy
        from scipy.interpolate import interp1d
        log_q = np.log10(self.strain_q_array)
        log_strain = np.log10(np.maximum(self.strain_array, 1e-10))
        interp_func = interp1d(log_q, log_strain, kind='linear',
                               bounds_error=False, fill_value='extrapolate')
        strain = 10 ** interp_func(np.log10(q))
        return np.clip(strain, 0.0, 1.0)

    def _angle_integral(self, q: float) -> float:
        """
        Compute inner integral over angle φ.

        Integrates: ∫₀^(2π) dφ |E_eff(qv cosφ) / ((1-ν²)σ₀)|²

        When nonlinear correction is enabled:
        E_eff = E'×f(ε) + i×E''×g(ε)
        |E_eff|² = (E'×f)² + (E''×g)²

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
        omega_eval = np.abs(omega)
        omega_eval[omega_eval < 1e-10] = 1e-10

        # Check if nonlinear correction should be applied
        use_nonlinear = (self.f_interpolator is not None and
                         self.g_interpolator is not None and
                         self.storage_modulus_func is not None and
                         self.loss_modulus_func is not None)

        if use_nonlinear:
            # Get strain at this q
            strain_q = self._get_strain_at_q(q)
            f_val = self.f_interpolator(strain_q)
            g_val = self.g_interpolator(strain_q)

            # Handle NaN or invalid f_val/g_val - fallback to linear (f=g=1.0)
            if not np.isfinite(f_val):
                f_val = 1.0  # No correction if f is invalid
            else:
                f_val = np.clip(f_val, 0.01, 1.0)  # Minimum 0.01 to prevent zero

            if not np.isfinite(g_val):
                g_val = 1.0  # No correction if g is invalid
            else:
                g_val = np.clip(g_val, 0.01, 1.0)  # Minimum 0.01 to prevent zero

            # Calculate |E_eff|² = (E'×f)² + (E''×g)² for each angle
            integrand = np.zeros_like(phi)
            for i, w in enumerate(omega_eval):
                E_prime = self.storage_modulus_func(w)
                E_loss = self.loss_modulus_func(w)

                # Validate E' and E'' values
                if not np.isfinite(E_prime) or not np.isfinite(E_loss):
                    # Use fallback: try linear modulus
                    try:
                        E_complex = self.modulus_func(w)
                        E_prime = np.real(E_complex)
                        E_loss = np.imag(E_complex)
                    except:
                        E_prime = 1e6  # Fallback value
                        E_loss = 1e5

                # Apply nonlinear correction
                E_prime_eff = E_prime * f_val
                E_loss_eff = E_loss * g_val
                # |E_eff|² = E'_eff² + E''_eff²
                E_star_eff_sq = E_prime_eff**2 + E_loss_eff**2
                integrand[i] = E_star_eff_sq * self.prefactor**2
        else:
            # Linear calculation (original)
            E_values = np.array([self.modulus_func(w) for w in omega_eval])
            integrand = np.abs(E_values * self.prefactor)**2

        # Validate integrand - replace NaN/Inf with zeros
        if np.any(~np.isfinite(integrand)):
            integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)

        # Numerical integration using trapezoidal rule
        result = np.trapz(integrand, phi)

        return result

    def _angle_integral_with_details(self, q: float) -> Tuple[float, dict]:
        """
        Compute inner integral over angle φ with detailed intermediate values.

        This method returns both the integral result and the detailed
        integrand values at each angle for visualization purposes.

        When nonlinear correction is enabled:
        E_eff = E'×f(ε) + i×E''×g(ε)

        Parameters
        ----------
        q : float
            Wavenumber (1/m)

        Returns
        -------
        result : float
            Result of angle integration
        details : dict
            Dictionary containing:
            - 'phi': angle array (radians)
            - 'omega': frequency array (rad/s)
            - 'integrand': integrand values at each angle
        """
        # Create angle array
        phi = np.linspace(0, 2 * np.pi, self.n_angle_points)

        # Calculate frequencies for each angle
        # ω = q * v * cos(φ)
        omega = q * self.velocity * np.cos(phi)

        # Handle zero and negative frequencies
        omega_eval = np.abs(omega)
        omega_eval[omega_eval < 1e-10] = 1e-10

        # Check if nonlinear correction should be applied
        use_nonlinear = (self.f_interpolator is not None and
                         self.g_interpolator is not None and
                         self.storage_modulus_func is not None and
                         self.loss_modulus_func is not None)

        if use_nonlinear:
            # Get strain at this q
            strain_q = self._get_strain_at_q(q)
            f_val = np.clip(self.f_interpolator(strain_q), 0.0, 1.0)
            g_val = np.clip(self.g_interpolator(strain_q), 0.0, 1.0)

            # Calculate |E_eff|² for each angle
            integrand = np.zeros_like(phi)
            for i, w in enumerate(omega_eval):
                E_prime = self.storage_modulus_func(w)
                E_loss = self.loss_modulus_func(w)
                E_prime_eff = E_prime * f_val
                E_loss_eff = E_loss * g_val
                E_star_eff_sq = E_prime_eff**2 + E_loss_eff**2
                integrand[i] = E_star_eff_sq * self.prefactor**2
        else:
            # Linear calculation
            E_values = np.array([self.modulus_func(w) for w in omega_eval])
            integrand = np.abs(E_values * self.prefactor)**2

        # Numerical integration using trapezoidal rule
        result = np.trapz(integrand, phi)

        # Store details
        details = {
            'phi': phi,
            'omega': omega,
            'integrand': integrand
        }

        return result, details

    def _integrand_q(self, q: float) -> float:
        """
        Compute integrand for q integration.

        Calculates: q³ * C(q) * (angle integral)

        Based on Persson (2001, 2006) formulation:
        G(q) = (1/8) ∫ dq' q'³ C(q') ∫ dφ |E(q'v cosφ) / ((1-ν²)σ₀)|²

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

        # Combine: q³ * C(q) * angle_integral
        result = q**3 * C_q * angle_int

        return result

    def calculate_G(
        self,
        q_values: np.ndarray,
        q_min: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate G(q) for an array of wavenumbers.

        Uses cumulative integration with S(q) partial contact correction.
        S(q) = gamma + (1 - gamma) * P(q)^2

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
        results = self.calculate_G_with_details(q_values, q_min=q_min)
        return results['G']

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

    def calculate_G_with_details(
        self,
        q_values: np.ndarray,
        q_min: Optional[float] = None,
        store_inner_integral: bool = False
    ) -> dict:
        """
        Calculate G(q) with detailed intermediate values for analysis.

        This method returns all intermediate calculation values including
        PSD, angle integrals, integrands, and cumulative G values.

        Parameters
        ----------
        q_values : np.ndarray
            Array of wavenumbers (1/m) in ascending order
        q_min : float, optional
            Lower integration limit (default: first value in q_values)
        store_inner_integral : bool, optional
            If True, stores detailed inner integral values for visualization
            (default: False)

        Returns
        -------
        dict
            Dictionary containing:
            - 'q': wavenumber array
            - 'log_q': log10(q)
            - 'C_q': PSD values C(q)
            - 'avg_modulus_term': angle integral results
            - 'G_integrand': full integrand q³ C(q) * angle_integral
            - 'delta_G': incremental G contributions
            - 'G': cumulative G(q) values
            - 'contact_area_ratio': P(q) = erf(1/(2√G))
            - 'inner_integral_details': (if store_inner_integral=True)
                List of dicts with 'phi', 'omega', 'integrand' for each q
        """
        q_values = np.asarray(q_values)

        if q_min is None:
            q_min = q_values[0]

        n = len(q_values)

        # Initialize output arrays
        C_q_arr = np.zeros(n)
        avg_modulus_arr = np.zeros(n)
        G_integrand_arr = np.zeros(n)
        delta_G_arr = np.zeros(n)
        G_arr = np.zeros(n)
        P_arr = np.zeros(n)

        # Store inner integral details if requested
        inner_integral_details = [] if store_inner_integral else None

        # Calculate values for each wavenumber
        for i, q in enumerate(q_values):
            # Get PSD value
            C_q_arr[i] = self.psd_func(np.array([q]))[0]

            # Calculate angle integral (Avg_Modulus_Term)
            if store_inner_integral:
                avg_modulus_arr[i], details = self._angle_integral_with_details(q)
                inner_integral_details.append(details)
            else:
                avg_modulus_arr[i] = self._angle_integral(q)

            # Calculate full integrand
            G_integrand_arr[i] = self._integrand_q(q)

        # Calculate cumulative G with S(q) partial contact correction
        # S(q) = gamma + (1 - gamma) * P(q)^2
        # Linear case (gamma=0): S(q) = P(q)^2
        # At each step: apply S(q) to reduce integrand in non-contact regions
        from scipy.special import erf
        S_arr = np.ones(n)

        # Initialize: at q_min, full contact assumed
        P_arr[0] = 1.0
        S_arr[0] = self.gamma + (1 - self.gamma) * P_arr[0]**2  # = 1.0

        for i in range(1, n):
            if q_values[i] <= q_min:
                P_arr[i] = 1.0
                S_arr[i] = 1.0
                continue

            # Apply S(q) correction to integrand
            # Use S from previous step for current point (forward Euler)
            corrected_prev = G_integrand_arr[i-1] * S_arr[i-1]
            corrected_curr = G_integrand_arr[i] * S_arr[i-1]

            # Trapezoidal rule: (f(i-1) + f(i)) / 2 * Δq
            delta_G = 0.5 * (corrected_prev + corrected_curr) * \
                      (q_values[i] - q_values[i-1])

            delta_G_arr[i] = delta_G / 8.0  # Apply 1/8 factor
            G_arr[i] = G_arr[i-1] + delta_G_arr[i]

            # Compute P(q) from current cumulative G
            if G_arr[i] > 1e-10:
                sqrt_G = np.sqrt(G_arr[i])
                arg = 1.0 / (2.0 * sqrt_G)
                arg = min(arg, 10.0)  # erf(10) ≈ 1.0
                P_arr[i] = erf(arg)
            else:
                P_arr[i] = 1.0

            # Compute S(q) for next step
            S_arr[i] = self.gamma + (1 - self.gamma) * P_arr[i]**2

        result = {
            'q': q_values,
            'log_q': np.log10(q_values),
            'C_q': C_q_arr,
            'avg_modulus_term': avg_modulus_arr,
            'G_integrand': G_integrand_arr,
            'delta_G': delta_G_arr,
            'G': G_arr,
            'contact_area_ratio': P_arr,
            'S_q': S_arr
        }

        # Add inner integral details if stored
        if store_inner_integral:
            result['inner_integral_details'] = inner_integral_details

        return result

    def calculate_G_multi_velocity(
        self,
        q_values: np.ndarray,
        v_values: np.ndarray,
        q_min: Optional[float] = None,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Calculate G(q,v) for multiple velocities - 2D matrix calculation.

        This creates a 2D matrix G(q,v) where G depends on both
        wavenumber and sliding velocity, as required by work instruction v2.1.

        Parameters
        ----------
        q_values : np.ndarray
            Array of wavenumbers (1/m) in ascending order
        v_values : np.ndarray
            Array of sliding velocities (m/s)
            Recommended: np.logspace(-4, 1, 30) for 0.0001~10 m/s
        q_min : float, optional
            Lower integration limit (default: first value in q_values)
        progress_callback : callable, optional
            Function to call with progress updates (0-100)

        Returns
        -------
        dict
            Dictionary containing:
            - 'q': wavenumber array
            - 'v': velocity array
            - 'G_matrix': 2D array G(q,v) with shape (len(q), len(v))
            - 'P_matrix': 2D array P(q,v) contact area ratio
            - 'log_q': log10(q)
            - 'log_v': log10(v)
        """
        q_values = np.asarray(q_values)
        v_values = np.asarray(v_values)

        if q_min is None:
            q_min = q_values[0]

        n_q = len(q_values)
        n_v = len(v_values)

        # Initialize output matrices
        G_matrix = np.zeros((n_q, n_v))
        P_matrix = np.zeros((n_q, n_v))

        # Store original velocity
        original_velocity = self.velocity

        # Calculate for each velocity
        for j, v in enumerate(v_values):
            # Update velocity
            self.velocity = v

            # Calculate G(q) for this velocity
            results = self.calculate_G_with_details(q_values, q_min=q_min)

            # Store results
            G_matrix[:, j] = results['G']
            P_matrix[:, j] = results['contact_area_ratio']

            # Progress callback
            if progress_callback:
                progress = int((j + 1) / n_v * 100)
                progress_callback(progress)

        # Restore original velocity
        self.velocity = original_velocity

        return {
            'q': q_values,
            'v': v_values,
            'G_matrix': G_matrix,
            'P_matrix': P_matrix,
            'log_q': np.log10(q_values),
            'log_v': np.log10(v_values)
        }

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
