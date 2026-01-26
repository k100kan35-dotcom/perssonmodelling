"""
DMA Master Curve Generation Module (with Vertical Shift)
=========================================================

Time-Temperature Superposition (TTS) implementation for generating
viscoelastic master curves from multi-temperature DMA data.

Key features:
- Horizontal shift (aT): frequency shift factor
- Vertical shift (bT): modulus shift factor (density/entropy elasticity correction)
- WLF equation fitting for aT
- Numerical optimization for shift factors

References:
- Ferry, J.D. "Viscoelastic Properties of Polymers"
- Persson friction theory applications
"""

import numpy as np
from scipy import optimize, interpolate
from scipy.signal import savgol_filter
from typing import Tuple, Optional, Dict, List, Callable
import warnings


class MasterCurveGenerator:
    """
    Generate viscoelastic master curves using Time-Temperature Superposition (TTS).

    Supports both horizontal (aT) and vertical (bT) shift factors.
    """

    def __init__(self, T_ref: float = 20.0):
        """
        Initialize master curve generator.

        Parameters
        ----------
        T_ref : float
            Reference temperature in Celsius (default: 20.0)
        """
        self.T_ref = T_ref
        self.T_ref_K = T_ref + 273.15  # Kelvin

        # Data storage
        self.temperatures = None  # Unique temperatures
        self.raw_data = None  # Dict: {T: {'f': array, 'E_storage': array, 'E_loss': array}}

        # Shift factors
        self.aT = None  # Dict: {T: aT_value}
        self.bT = None  # Dict: {T: bT_value}

        # Master curve
        self.master_f = None
        self.master_E_storage = None
        self.master_E_loss = None

        # WLF parameters
        self.C1 = None
        self.C2 = None

    def load_data(self, df, T_col='T', f_col='f', E_storage_col="E'", E_loss_col="E''"):
        """
        Load multi-temperature DMA data from DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with DMA data
        T_col : str
            Column name for temperature (Celsius)
        f_col : str
            Column name for frequency (Hz)
        E_storage_col : str
            Column name for storage modulus
        E_loss_col : str
            Column name for loss modulus
        """
        import pandas as pd

        # Get unique temperatures
        self.temperatures = np.sort(df[T_col].unique())

        # Store data by temperature
        self.raw_data = {}
        for T in self.temperatures:
            mask = df[T_col] == T
            T_data = df[mask].sort_values(by=f_col)

            self.raw_data[T] = {
                'f': T_data[f_col].values.astype(float),
                'E_storage': T_data[E_storage_col].values.astype(float),
                'E_loss': T_data[E_loss_col].values.astype(float)
            }

        # Initialize shift factors
        self.aT = {T: 1.0 for T in self.temperatures}
        self.bT = {T: 1.0 for T in self.temperatures}

        # Set reference temperature aT and bT to 1
        if self.T_ref in self.temperatures:
            self.aT[self.T_ref] = 1.0
            self.bT[self.T_ref] = 1.0

        return len(self.temperatures)

    def calculate_theoretical_bT(self, T: float) -> float:
        """
        Calculate theoretical bT based on entropy elasticity.

        bT = T * rho(T) / (T_ref * rho(T_ref)) ≈ T / T_ref (in Kelvin)

        For glassy region, bT approaches 1.

        Parameters
        ----------
        T : float
            Temperature in Celsius

        Returns
        -------
        float
            Theoretical bT value
        """
        T_K = T + 273.15
        return T_K / self.T_ref_K

    def _overlap_error(self, params, T1, T2, use_bT=True):
        """
        Calculate overlap error between two temperature curves.

        Parameters
        ----------
        params : tuple
            (log_aT, bT) or (log_aT,) depending on use_bT
        T1 : float
            First temperature (reference, already shifted)
        T2 : float
            Second temperature (to be shifted)
        use_bT : bool
            Whether to optimize bT as well

        Returns
        -------
        float
            Mean squared error in overlap region
        """
        if use_bT:
            log_aT, bT = params
        else:
            log_aT = params[0]
            bT = self.calculate_theoretical_bT(T2)

        aT = 10**log_aT

        # Get data
        f1 = self.raw_data[T1]['f'] * self.aT[T1]
        E1 = self.raw_data[T1]['E_storage'] / self.bT[T1]

        f2 = self.raw_data[T2]['f'] * aT
        E2 = self.raw_data[T2]['E_storage'] / bT

        # Find overlap region
        f_min_overlap = max(f1.min(), f2.min())
        f_max_overlap = min(f1.max(), f2.max())

        if f_min_overlap >= f_max_overlap:
            return 1e10  # No overlap

        # Interpolate in overlap region
        n_points = 20
        f_overlap = np.logspace(np.log10(f_min_overlap), np.log10(f_max_overlap), n_points)

        # Log-log interpolation
        interp1 = interpolate.interp1d(np.log10(f1), np.log10(E1),
                                        kind='linear', fill_value='extrapolate')
        interp2 = interpolate.interp1d(np.log10(f2), np.log10(E2),
                                        kind='linear', fill_value='extrapolate')

        log_E1_interp = interp1(np.log10(f_overlap))
        log_E2_interp = interp2(np.log10(f_overlap))

        # MSE in log space
        mse = np.mean((log_E1_interp - log_E2_interp)**2)

        return mse

    def optimize_shift_factors(self, use_bT=True, bT_mode='optimize', verbose=False):
        """
        Optimize shift factors aT and bT for all temperatures.

        Parameters
        ----------
        use_bT : bool
            Whether to apply vertical shift (default: True)
        bT_mode : str
            'optimize': numerically optimize bT
            'theoretical': use T/T_ref formula
        verbose : bool
            Print optimization progress

        Returns
        -------
        dict
            {'aT': {T: value}, 'bT': {T: value}}
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Sort temperatures
        temps_sorted = np.sort(self.temperatures)

        # Find index of reference temperature (or closest)
        ref_idx = np.argmin(np.abs(temps_sorted - self.T_ref))
        T_ref_actual = temps_sorted[ref_idx]

        # Set reference to 1.0
        self.aT[T_ref_actual] = 1.0
        self.bT[T_ref_actual] = 1.0

        if verbose:
            print(f"Reference temperature: {T_ref_actual}°C")

        # Shift temperatures below reference (working downward)
        for i in range(ref_idx - 1, -1, -1):
            T_current = temps_sorted[i]
            T_neighbor = temps_sorted[i + 1]

            # Initial guess
            log_aT_init = 0.0
            bT_init = self.calculate_theoretical_bT(T_current) if use_bT else 1.0

            if use_bT and bT_mode == 'optimize':
                # Optimize both aT and bT
                result = optimize.minimize(
                    self._overlap_error,
                    x0=[log_aT_init, bT_init],
                    args=(T_neighbor, T_current, True),
                    method='Nelder-Mead',
                    options={'maxiter': 500}
                )
                self.aT[T_current] = 10**result.x[0] * self.aT[T_neighbor]
                self.bT[T_current] = result.x[1]
            else:
                # Optimize only aT
                self.bT[T_current] = self.calculate_theoretical_bT(T_current) if use_bT else 1.0
                result = optimize.minimize(
                    self._overlap_error,
                    x0=[log_aT_init],
                    args=(T_neighbor, T_current, False),
                    method='Nelder-Mead',
                    options={'maxiter': 500}
                )
                self.aT[T_current] = 10**result.x[0] * self.aT[T_neighbor]

            if verbose:
                print(f"T={T_current}°C: aT={self.aT[T_current]:.4e}, bT={self.bT[T_current]:.4f}")

        # Shift temperatures above reference (working upward)
        for i in range(ref_idx + 1, len(temps_sorted)):
            T_current = temps_sorted[i]
            T_neighbor = temps_sorted[i - 1]

            # Initial guess
            log_aT_init = 0.0
            bT_init = self.calculate_theoretical_bT(T_current) if use_bT else 1.0

            if use_bT and bT_mode == 'optimize':
                result = optimize.minimize(
                    self._overlap_error,
                    x0=[log_aT_init, bT_init],
                    args=(T_neighbor, T_current, True),
                    method='Nelder-Mead',
                    options={'maxiter': 500}
                )
                self.aT[T_current] = 10**result.x[0] * self.aT[T_neighbor]
                self.bT[T_current] = result.x[1]
            else:
                self.bT[T_current] = self.calculate_theoretical_bT(T_current) if use_bT else 1.0
                result = optimize.minimize(
                    self._overlap_error,
                    x0=[log_aT_init],
                    args=(T_neighbor, T_current, False),
                    method='Nelder-Mead',
                    options={'maxiter': 500}
                )
                self.aT[T_current] = 10**result.x[0] * self.aT[T_neighbor]

            if verbose:
                print(f"T={T_current}°C: aT={self.aT[T_current]:.4e}, bT={self.bT[T_current]:.4f}")

        return {'aT': self.aT.copy(), 'bT': self.bT.copy()}

    def generate_master_curve(self, n_points=200, smooth=True, window_length=11):
        """
        Generate the master curve by combining shifted data.

        Parameters
        ----------
        n_points : int
            Number of points in output master curve
        smooth : bool
            Apply smoothing to the result
        window_length : int
            Savitzky-Golay filter window length

        Returns
        -------
        dict
            {'f': array, 'omega': array, 'E_storage': array, 'E_loss': array, 'tan_delta': array}
        """
        if self.aT is None:
            raise ValueError("Shift factors not calculated. Call optimize_shift_factors() first.")

        # Collect all shifted data
        all_f_reduced = []
        all_E_storage_shifted = []
        all_E_loss_shifted = []

        for T in self.temperatures:
            f = self.raw_data[T]['f']
            E_storage = self.raw_data[T]['E_storage']
            E_loss = self.raw_data[T]['E_loss']

            # Apply shifts
            f_reduced = f * self.aT[T]
            E_storage_shifted = E_storage / self.bT[T]
            E_loss_shifted = E_loss / self.bT[T]

            all_f_reduced.extend(f_reduced)
            all_E_storage_shifted.extend(E_storage_shifted)
            all_E_loss_shifted.extend(E_loss_shifted)

        # Convert to arrays and sort by frequency
        all_f_reduced = np.array(all_f_reduced)
        all_E_storage_shifted = np.array(all_E_storage_shifted)
        all_E_loss_shifted = np.array(all_E_loss_shifted)

        sort_idx = np.argsort(all_f_reduced)
        all_f_reduced = all_f_reduced[sort_idx]
        all_E_storage_shifted = all_E_storage_shifted[sort_idx]
        all_E_loss_shifted = all_E_loss_shifted[sort_idx]

        # Create uniform log-spaced frequency array
        f_min = all_f_reduced.min()
        f_max = all_f_reduced.max()
        self.master_f = np.logspace(np.log10(f_min), np.log10(f_max), n_points)

        # Interpolate in log-log space
        # Remove duplicates and invalid values
        valid_mask = (all_E_storage_shifted > 0) & (all_E_loss_shifted > 0) & (all_f_reduced > 0)
        f_valid = all_f_reduced[valid_mask]
        E_storage_valid = all_E_storage_shifted[valid_mask]
        E_loss_valid = all_E_loss_shifted[valid_mask]

        # Use spline interpolation in log space
        try:
            interp_storage = interpolate.UnivariateSpline(
                np.log10(f_valid), np.log10(E_storage_valid), s=len(f_valid)*0.01, k=3
            )
            interp_loss = interpolate.UnivariateSpline(
                np.log10(f_valid), np.log10(E_loss_valid), s=len(f_valid)*0.01, k=3
            )

            log_E_storage = interp_storage(np.log10(self.master_f))
            log_E_loss = interp_loss(np.log10(self.master_f))
        except:
            # Fallback to linear interpolation
            interp_storage = interpolate.interp1d(
                np.log10(f_valid), np.log10(E_storage_valid),
                kind='linear', fill_value='extrapolate'
            )
            interp_loss = interpolate.interp1d(
                np.log10(f_valid), np.log10(E_loss_valid),
                kind='linear', fill_value='extrapolate'
            )
            log_E_storage = interp_storage(np.log10(self.master_f))
            log_E_loss = interp_loss(np.log10(self.master_f))

        self.master_E_storage = 10**log_E_storage
        self.master_E_loss = 10**log_E_loss

        # Apply smoothing if requested
        if smooth and window_length > 3:
            try:
                self.master_E_storage = savgol_filter(self.master_E_storage, window_length, 2)
                self.master_E_loss = savgol_filter(self.master_E_loss, window_length, 2)
            except:
                pass  # Skip smoothing if it fails

        # Calculate tan delta
        master_tan_delta = self.master_E_loss / self.master_E_storage

        # Angular frequency
        master_omega = 2 * np.pi * self.master_f

        return {
            'f': self.master_f,
            'omega': master_omega,
            'E_storage': self.master_E_storage,
            'E_loss': self.master_E_loss,
            'tan_delta': master_tan_delta
        }

    def fit_wlf(self):
        """
        Fit WLF equation to the horizontal shift factors.

        log(aT) = -C1 * (T - T_ref) / (C2 + (T - T_ref))

        Returns
        -------
        dict
            {'C1': float, 'C2': float, 'r_squared': float}
        """
        if self.aT is None:
            raise ValueError("Shift factors not calculated.")

        temps = np.array(list(self.aT.keys()))
        log_aT = np.log10(np.array(list(self.aT.values())))

        # WLF equation: log(aT) = -C1*(T-Tref)/(C2+(T-Tref))
        def wlf_func(T, C1, C2):
            dT = T - self.T_ref
            return -C1 * dT / (C2 + dT)

        # Initial guess
        C1_init = 17.44
        C2_init = 51.6

        try:
            popt, pcov = optimize.curve_fit(
                wlf_func, temps, log_aT,
                p0=[C1_init, C2_init],
                bounds=([0, 0], [100, 500]),
                maxfev=5000
            )
            self.C1, self.C2 = popt

            # Calculate R-squared
            y_pred = wlf_func(temps, self.C1, self.C2)
            ss_res = np.sum((log_aT - y_pred)**2)
            ss_tot = np.sum((log_aT - np.mean(log_aT))**2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            return {
                'C1': self.C1,
                'C2': self.C2,
                'r_squared': r_squared,
                'T_ref': self.T_ref
            }
        except Exception as e:
            warnings.warn(f"WLF fitting failed: {e}")
            return {'C1': None, 'C2': None, 'r_squared': None, 'T_ref': self.T_ref}

    def get_shift_factor_table(self):
        """
        Get shift factor table as DataFrame.

        Returns
        -------
        pandas.DataFrame
            Table with T, aT, bT, log(aT) columns
        """
        import pandas as pd

        temps = sorted(self.temperatures)
        data = {
            'T (°C)': temps,
            'aT': [self.aT[T] for T in temps],
            'bT': [self.bT[T] for T in temps],
            'log10(aT)': [np.log10(self.aT[T]) for T in temps]
        }

        return pd.DataFrame(data)

    def get_shifted_data(self, T: float):
        """
        Get shifted data for a specific temperature.

        Parameters
        ----------
        T : float
            Temperature

        Returns
        -------
        dict
            {'f_reduced': array, 'E_storage_shifted': array, 'E_loss_shifted': array}
        """
        if T not in self.raw_data:
            raise ValueError(f"Temperature {T} not found in data")

        f = self.raw_data[T]['f']
        E_storage = self.raw_data[T]['E_storage']
        E_loss = self.raw_data[T]['E_loss']

        return {
            'f_reduced': f * self.aT[T],
            'E_storage_shifted': E_storage / self.bT[T],
            'E_loss_shifted': E_loss / self.bT[T]
        }


def load_multi_temp_dma(filename: str, skip_header: int = 1) -> 'pd.DataFrame':
    """
    Load multi-temperature DMA data from file.

    Expected columns: f(Hz), T(°C), f(Hz), Amplitude, E'(MPa), E''(MPa)

    Parameters
    ----------
    filename : str
        Path to data file (CSV, TXT, or Excel)
    skip_header : int
        Number of header rows to skip

    Returns
    -------
    pd.DataFrame
        Loaded data with standardized column names
    """
    import pandas as pd

    # Determine file type
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename, skiprows=skip_header)
    else:
        # Try different delimiters
        try:
            df = pd.read_csv(filename, skiprows=skip_header, sep='\t')
            if len(df.columns) < 4:
                df = pd.read_csv(filename, skiprows=skip_header, sep=',')
            if len(df.columns) < 4:
                df = pd.read_csv(filename, skiprows=skip_header, delim_whitespace=True)
        except:
            df = pd.read_csv(filename, skiprows=skip_header, delim_whitespace=True)

    # Standardize column names based on expected format
    # Expected: f(Hz), T(°C), f(Hz), Amplitude, E'(MPa), E''(MPa)
    if len(df.columns) >= 6:
        df.columns = ['f', 'T', 'f_reduced', 'Amplitude', "E'", "E''"][:len(df.columns)]
    elif len(df.columns) >= 4:
        # Minimal: f, T, E', E''
        df.columns = ['f', 'T', "E'", "E''"][:len(df.columns)]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN
    df = df.dropna()

    return df
