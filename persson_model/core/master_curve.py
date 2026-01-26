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

    def _calculate_overlap_error(self, log_aT_rel, T_neighbor, T_current,
                                  target='E_storage', bT_current=1.0):
        """
        Calculate MSE between two temperature curves in overlap region.

        The goal is to find log_aT_rel such that T_current's data, when shifted,
        overlaps smoothly with T_neighbor's data.

        Parameters
        ----------
        log_aT_rel : float
            Log10 of the RELATIVE shift factor (between adjacent temps)
        T_neighbor : float
            Adjacent temperature (already has accumulated aT assigned)
        T_current : float
            Current temperature being optimized
        target : str
            Which modulus to use: 'E_storage', 'E_loss', or 'tan_delta'
        bT_current : float
            Vertical shift for current temperature

        Returns
        -------
        float
            Mean squared error in log space
        """
        aT_rel = 10**log_aT_rel

        # Get neighbor's data (already shifted to master curve)
        f_neighbor = self.raw_data[T_neighbor]['f'] * self.aT[T_neighbor]

        if target == 'E_storage':
            E_neighbor = self.raw_data[T_neighbor]['E_storage'] / self.bT[T_neighbor]
        elif target == 'E_loss':
            E_neighbor = self.raw_data[T_neighbor]['E_loss'] / self.bT[T_neighbor]
        else:  # tan_delta
            E_neighbor = (self.raw_data[T_neighbor]['E_loss'] /
                         self.raw_data[T_neighbor]['E_storage'])

        # Get current temp's data with TOTAL shift
        aT_total = aT_rel * self.aT[T_neighbor]
        f_current = self.raw_data[T_current]['f'] * aT_total

        if target == 'E_storage':
            E_current = self.raw_data[T_current]['E_storage'] / bT_current
        elif target == 'E_loss':
            E_current = self.raw_data[T_current]['E_loss'] / bT_current
        else:  # tan_delta
            E_current = (self.raw_data[T_current]['E_loss'] /
                        self.raw_data[T_current]['E_storage'])

        # Find overlap region in reduced frequency
        f_min_overlap = max(f_neighbor.min(), f_current.min())
        f_max_overlap = min(f_neighbor.max(), f_current.max())

        # Check for valid overlap
        if f_min_overlap >= f_max_overlap * 0.99:
            # No overlap - penalize heavily
            return 1e6

        # Create common frequency grid in overlap region
        n_points = min(20, len(f_neighbor), len(f_current))
        f_common = np.logspace(np.log10(f_min_overlap * 1.01),
                               np.log10(f_max_overlap * 0.99), n_points)

        try:
            # Interpolate both curves to common grid (log-log space)
            # Make sure arrays are sorted and unique
            sort_idx_n = np.argsort(f_neighbor)
            f_neighbor_sorted = f_neighbor[sort_idx_n]
            E_neighbor_sorted = E_neighbor[sort_idx_n]

            sort_idx_c = np.argsort(f_current)
            f_current_sorted = f_current[sort_idx_c]
            E_current_sorted = E_current[sort_idx_c]

            # Remove any non-positive values
            valid_n = (f_neighbor_sorted > 0) & (E_neighbor_sorted > 0)
            valid_c = (f_current_sorted > 0) & (E_current_sorted > 0)

            if np.sum(valid_n) < 2 or np.sum(valid_c) < 2:
                return 1e6

            interp_neighbor = interpolate.interp1d(
                np.log10(f_neighbor_sorted[valid_n]),
                np.log10(E_neighbor_sorted[valid_n]),
                kind='linear', fill_value='extrapolate', bounds_error=False
            )
            interp_current = interpolate.interp1d(
                np.log10(f_current_sorted[valid_c]),
                np.log10(E_current_sorted[valid_c]),
                kind='linear', fill_value='extrapolate', bounds_error=False
            )

            log_E_neighbor = interp_neighbor(np.log10(f_common))
            log_E_current = interp_current(np.log10(f_common))

            # Check for invalid interpolation results
            if np.any(np.isnan(log_E_neighbor)) or np.any(np.isnan(log_E_current)):
                return 1e6

            # MSE in log space
            mse = np.mean((log_E_neighbor - log_E_current)**2)

            return mse

        except Exception as e:
            return 1e6

    def optimize_shift_factors(self, use_bT=True, bT_mode='theoretical',
                               target='E_storage', verbose=False):
        """
        Optimize shift factors aT and bT for all temperatures.

        Parameters
        ----------
        use_bT : bool
            Whether to apply vertical shift (default: True)
        bT_mode : str
            'optimize': numerically optimize bT
            'theoretical': use T/T_ref formula
        target : str
            Which to optimize: 'E_storage', 'E_loss', 'tan_delta'
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
        self.T_ref = T_ref_actual  # Update to actual reference

        # Set reference to 1.0
        self.aT[T_ref_actual] = 1.0
        self.bT[T_ref_actual] = 1.0

        if verbose:
            print(f"Reference temperature: {T_ref_actual}°C (index {ref_idx})")
            print(f"Target: {target}")

        # Process temperatures below reference (colder -> need aT > 1)
        for i in range(ref_idx - 1, -1, -1):
            T_current = temps_sorted[i]
            T_neighbor = temps_sorted[i + 1]

            # Calculate bT
            if use_bT:
                if bT_mode == 'theoretical':
                    self.bT[T_current] = self.calculate_theoretical_bT(T_current)
                else:
                    # For now, use theoretical as starting point
                    self.bT[T_current] = self.calculate_theoretical_bT(T_current)
            else:
                self.bT[T_current] = 1.0

            # Initial guess for relative shift (small positive value for colder temps)
            # Adjacent temps should have small relative shifts
            dT = T_current - T_neighbor  # negative for colder
            log_aT_init = -0.1 * dT / 10.0  # Rough estimate: 0.1 decades per 10°C

            # Optimize with bounded search
            # For colder temp than neighbor, aT_rel should be > 1 (positive log)
            result = optimize.minimize_scalar(
                self._calculate_overlap_error,
                args=(T_neighbor, T_current, target, self.bT[T_current]),
                bounds=(-0.5, 3.0),  # Reasonable range for relative shift
                method='bounded',
                options={'maxiter': 100}
            )

            log_aT_rel = result.x
            self.aT[T_current] = 10**log_aT_rel * self.aT[T_neighbor]

            if verbose:
                print(f"T={T_current:.1f}°C: aT={self.aT[T_current]:.4e}, "
                      f"bT={self.bT[T_current]:.4f}, log_aT_rel={log_aT_rel:.4f}, "
                      f"error={result.fun:.4e}")

        # Process temperatures above reference (hotter -> need aT < 1)
        for i in range(ref_idx + 1, len(temps_sorted)):
            T_current = temps_sorted[i]
            T_neighbor = temps_sorted[i - 1]

            # Calculate bT
            if use_bT:
                if bT_mode == 'theoretical':
                    self.bT[T_current] = self.calculate_theoretical_bT(T_current)
                else:
                    self.bT[T_current] = self.calculate_theoretical_bT(T_current)
            else:
                self.bT[T_current] = 1.0

            # Initial guess for relative shift (small negative value for hotter temps)
            dT = T_current - T_neighbor  # positive for hotter
            log_aT_init = -0.1 * dT / 10.0  # Rough estimate

            # Optimize with bounded search
            # For hotter temp than neighbor, aT_rel should be < 1 (negative log)
            result = optimize.minimize_scalar(
                self._calculate_overlap_error,
                args=(T_neighbor, T_current, target, self.bT[T_current]),
                bounds=(-3.0, 0.5),  # Reasonable range for relative shift
                method='bounded',
                options={'maxiter': 100}
            )

            log_aT_rel = result.x
            self.aT[T_current] = 10**log_aT_rel * self.aT[T_neighbor]

            if verbose:
                print(f"T={T_current:.1f}°C: aT={self.aT[T_current]:.4e}, "
                      f"bT={self.bT[T_current]:.4f}, log_aT_rel={log_aT_rel:.4f}, "
                      f"error={result.fun:.4e}")

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

        # Use linear interpolation in log space (more robust)
        try:
            interp_storage = interpolate.interp1d(
                np.log10(f_valid), np.log10(E_storage_valid),
                kind='linear', fill_value='extrapolate', bounds_error=False
            )
            interp_loss = interpolate.interp1d(
                np.log10(f_valid), np.log10(E_loss_valid),
                kind='linear', fill_value='extrapolate', bounds_error=False
            )

            log_E_storage = interp_storage(np.log10(self.master_f))
            log_E_loss = interp_loss(np.log10(self.master_f))
        except Exception as e:
            raise ValueError(f"Interpolation failed: {e}")

        self.master_E_storage = 10**log_E_storage
        self.master_E_loss = 10**log_E_loss

        # Apply smoothing if requested
        if smooth and window_length > 3 and len(self.master_f) > window_length:
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
            # Avoid division by zero
            denom = C2 + dT
            denom = np.where(np.abs(denom) < 0.1, np.sign(denom) * 0.1, denom)
            return -C1 * dT / denom

        # Initial guess
        C1_init = 10.0
        C2_init = 50.0

        try:
            popt, pcov = optimize.curve_fit(
                wlf_func, temps, log_aT,
                p0=[C1_init, C2_init],
                bounds=([0.1, 1], [50, 300]),
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
