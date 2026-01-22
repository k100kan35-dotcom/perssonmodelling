"""
Data Loading Utilities
=======================

Functions to load measured PSD and DMA data from files.
"""

import numpy as np
from typing import Tuple, Optional
import os


def load_psd_from_text(
    data_text: str,
    skip_header: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PSD data from text string.

    Parameters
    ----------
    data_text : str
        Text data with format: q(1/m)  C(q)(m^4)
        Can be tab or space separated
    skip_header : int, optional
        Number of header lines to skip

    Returns
    -------
    q : np.ndarray
        Wavenumber values (1/m)
    C_q : np.ndarray
        PSD values (m^4)
    """
    lines = data_text.strip().split('\n')[skip_header:]

    q_list = []
    C_list = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Split by tab or multiple spaces
        parts = line.split()
        if len(parts) >= 2:
            try:
                q_val = float(parts[0])
                C_val = float(parts[1])
                q_list.append(q_val)
                C_list.append(C_val)
            except ValueError:
                continue

    q = np.array(q_list)
    C_q = np.array(C_list)

    # Sort by q
    sort_idx = np.argsort(q)
    q = q[sort_idx]
    C_q = C_q[sort_idx]

    return q, C_q


def load_psd_from_file(
    filename: str,
    skip_header: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PSD data from file.

    File format (tab or space separated):
    q(1/m)  C(q)(m^4)
    100     1.23e-12
    200     4.56e-13
    ...

    Parameters
    ----------
    filename : str
        Path to PSD data file
    skip_header : int, optional
        Number of header lines to skip

    Returns
    -------
    q : np.ndarray
        Wavenumber values (1/m)
    C_q : np.ndarray
        PSD values (m^4)
    """
    with open(filename, 'r') as f:
        data_text = f.read()

    return load_psd_from_text(data_text, skip_header)


def load_dma_from_text(
    data_text: str,
    skip_header: int = 0,
    freq_unit: str = 'Hz',
    modulus_unit: str = 'MPa'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DMA (Dynamic Mechanical Analysis) data from text string.

    Parameters
    ----------
    data_text : str
        Text data with format: frequency  E'  E''
        Can be tab or space separated
    skip_header : int, optional
        Number of header lines to skip
    freq_unit : str, optional
        Frequency unit: 'Hz' or 'rad/s' (default: 'Hz')
    modulus_unit : str, optional
        Modulus unit: 'Pa', 'MPa', or 'GPa' (default: 'MPa')

    Returns
    -------
    omega : np.ndarray
        Angular frequency (rad/s)
    E_storage : np.ndarray
        Storage modulus E' (Pa)
    E_loss : np.ndarray
        Loss modulus E'' (Pa)
    """
    lines = data_text.strip().split('\n')[skip_header:]

    freq_list = []
    E_storage_list = []
    E_loss_list = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        parts = line.split()
        if len(parts) >= 3:
            try:
                freq = float(parts[0])
                E_prime = float(parts[1])
                E_double_prime = float(parts[2])

                freq_list.append(freq)
                E_storage_list.append(E_prime)
                E_loss_list.append(E_double_prime)
            except ValueError:
                continue

    freq = np.array(freq_list)
    E_storage = np.array(E_storage_list)
    E_loss = np.array(E_loss_list)

    # Convert frequency to rad/s
    if freq_unit.lower() == 'hz':
        omega = 2 * np.pi * freq
    else:
        omega = freq

    # Convert modulus to Pa
    if modulus_unit.lower() == 'mpa':
        E_storage *= 1e6
        E_loss *= 1e6
    elif modulus_unit.lower() == 'gpa':
        E_storage *= 1e9
        E_loss *= 1e9
    # else: already in Pa

    # Sort by frequency
    sort_idx = np.argsort(omega)
    omega = omega[sort_idx]
    E_storage = E_storage[sort_idx]
    E_loss = E_loss[sort_idx]

    return omega, E_storage, E_loss


def load_dma_from_file(
    filename: str,
    skip_header: int = 0,
    freq_unit: str = 'Hz',
    modulus_unit: str = 'MPa'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DMA data from file.

    File format (tab or space separated):
    frequency(Hz)  E'(MPa)  E''(MPa)
    0.01          10.5     1.2
    0.1           15.3     2.5
    ...

    Parameters
    ----------
    filename : str
        Path to DMA data file
    skip_header : int, optional
        Number of header lines to skip
    freq_unit : str, optional
        Frequency unit: 'Hz' or 'rad/s'
    modulus_unit : str, optional
        Modulus unit: 'Pa', 'MPa', or 'GPa'

    Returns
    -------
    omega : np.ndarray
        Angular frequency (rad/s)
    E_storage : np.ndarray
        Storage modulus E' (Pa)
    E_loss : np.ndarray
        Loss modulus E'' (Pa)
    """
    with open(filename, 'r') as f:
        data_text = f.read()

    return load_dma_from_text(data_text, skip_header, freq_unit, modulus_unit)


def create_material_from_dma(
    omega: np.ndarray,
    E_storage: np.ndarray,
    E_loss: np.ndarray,
    material_name: str = "Measured Material",
    reference_temp: float = 20.0
):
    """
    Create ViscoelasticMaterial from DMA data.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency (rad/s)
    E_storage : np.ndarray
        Storage modulus E' (Pa)
    E_loss : np.ndarray
        Loss modulus E'' (Pa)
    material_name : str, optional
        Name of the material
    reference_temp : float, optional
        Reference temperature (°C)

    Returns
    -------
    ViscoelasticMaterial
        Material object with loaded master curve
    """
    from ..core.viscoelastic import ViscoelasticMaterial

    material = ViscoelasticMaterial(
        frequencies=omega,
        storage_modulus=E_storage,
        loss_modulus=E_loss,
        reference_temp=reference_temp,
        name=material_name
    )

    return material


def create_psd_from_data(
    q: np.ndarray,
    C_q: np.ndarray,
    interpolation_kind: str = 'log-log'
):
    """
    Create PSD model from measured data.

    Parameters
    ----------
    q : np.ndarray
        Wavenumber values (1/m)
    C_q : np.ndarray
        PSD values (m^4)
    interpolation_kind : str, optional
        Interpolation method (default: 'log-log')

    Returns
    -------
    MeasuredPSD
        PSD model object
    """
    from ..core.psd_models import MeasuredPSD

    psd = MeasuredPSD(
        q_data=q,
        C_data=C_q,
        interpolation_kind=interpolation_kind,
        extrapolate=False,
        fill_value=0.0
    )

    return psd


def save_example_data(output_dir: str = 'examples/data'):
    """
    Save example measured data files.

    Creates example PSD and DMA data files in the specified directory.

    Parameters
    ----------
    output_dir : str, optional
        Output directory for data files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Example PSD data (rougher surface)
    psd_data = """# Surface PSD Data
# q(1/m)  C(q)(m^4)
2.0e+01  3.0e-09
2.0e+01  2.0e-09
3.0e+01  6.0e-10
5.0e+01  1.0e-10
1.0e+02  5.0e-12
5.0e+02  7.0e-14
1.0e+03  4.0e-15
5.0e+03  6.0e-17
1.0e+04  3.0e-18
5.0e+04  4.0e-20
1.0e+05  2.0e-21
5.0e+05  3.0e-23
1.0e+06  2.0e-24
5.0e+06  2.0e-26
1.0e+07  1.0e-27
5.0e+07  2.0e-29
1.0e+08  1.0e-30
5.0e+08  9.0e-33
1.0e+09  1.0e-33
"""

    psd_file = os.path.join(output_dir, 'example_psd.txt')
    with open(psd_file, 'w') as f:
        f.write(psd_data)

    # Example DMA data
    dma_data = """# DMA Master Curve Data
# Frequency(Hz)  E'(MPa)  E''(MPa)
0.01      6.7     0.7
0.1       7.8     1.0
1.0       8.9     1.2
10        10.0    1.6
100       12.3    2.1
1000      15.0    3.4
10000     20.7    7.0
100000    31.0    13.6
1000000   54.6    31.7
10000000  104     68.5
100000000 239     168
1000000000 613    358
10000000000 1280  606
"""

    dma_file = os.path.join(output_dir, 'example_dma.txt')
    with open(dma_file, 'w') as f:
        f.write(dma_data)

    print(f"Example data files created:")
    print(f"  {psd_file}")
    print(f"  {dma_file}")
