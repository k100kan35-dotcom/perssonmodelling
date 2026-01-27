"""
Enhanced Main GUI for Persson Friction Model (v3.0)
===================================================

Work Instruction v3.0 Implementation:
- Log-log cubic spline interpolation for viscoelastic data
- Inner integral visualization with savgol_filter smoothing
- G(q,v) heatmap using pcolormesh
- Korean labels for all graphs and UI elements
- Velocity range: 0.0001~10 m/s (log scale)
- G(q,v) 2D matrix calculation
- Input data verification tab
- Multi-velocity G(q) plotting
- Default measured data loading
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# Configure matplotlib to handle mathematical symbols properly
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign issue
matplotlib.rcParams['text.usetex'] = False  # Disable LaTeX to avoid font issues
# Standardize font sizes across all plots
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['legend.fontsize'] = 7
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from typing import Optional
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from persson_model.core.g_calculator import GCalculator
from persson_model.core.psd_models import FractalPSD, MeasuredPSD
from persson_model.core.viscoelastic import ViscoelasticMaterial
from persson_model.core.contact import ContactMechanics
from persson_model.utils.output import (
    save_calculation_details_csv,
    save_summary_txt,
    export_for_plotting,
    format_parameters_dict
)
from persson_model.utils.data_loader import (
    load_psd_from_file,
    load_dma_from_file,
    create_material_from_dma,
    create_psd_from_data,
    smooth_dma_data,
    load_strain_sweep_file,
    load_fg_curve_file,
    compute_fg_from_strain_sweep,
    create_fg_interpolator,
    average_fg_curves,
    create_strain_grid
)
from persson_model.core.friction import (
    FrictionCalculator,
    calculate_mu_visc_simple,
    apply_nonlinear_strain_correction,
    calculate_rms_slope_profile,
    calculate_strain_profile,
    calculate_hrms_profile,
    RMSSlopeCalculator
)
from persson_model.core.master_curve import MasterCurveGenerator, load_multi_temp_dma

# Configure matplotlib for better Korean font support
# IMPORTANT: Set unicode_minus FIRST to avoid minus sign warnings
matplotlib.rcParams['axes.unicode_minus'] = False

# Configure fonts
try:
    import matplotlib.font_manager as fm

    # Try to find Korean fonts on the system
    korean_fonts = []
    for font in fm.fontManager.ttflist:
        font_name = font.name
        # Check for common Korean font names
        if any(name in font_name for name in ['Malgun', 'NanumGothic', 'NanumBarun',
                                                'AppleGothic', 'Gulim', 'Dotum']):
            if font_name not in korean_fonts:
                korean_fonts.append(font_name)

    if korean_fonts:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = korean_fonts + ['DejaVu Sans', 'Arial', 'Helvetica']
    else:
        # Fallback to common fonts
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = ['Malgun Gothic', 'DejaVu Sans', 'Arial']
except:
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['figure.titleweight'] = 'bold'
matplotlib.rcParams['figure.titlesize'] = 14


class PerssonModelGUI_V2:
    """Enhanced GUI for Persson friction model (Work Instruction v2.1)."""

    def __init__(self, root):
        """Initialize enhanced GUI."""
        self.root = root
        self.root.title("Persson 마찰 계산기 v3.0")
        self.root.geometry("1600x1000")

        # Initialize variables
        self.material = None
        self.psd_model = None
        self.g_calculator = None
        self.results = {}
        self.raw_dma_data = None  # Store raw DMA data for plotting
        self.raw_psd_data = None  # Store raw PSD data for comparison plotting
        self.target_xi = None  # Target h'rms from Tab 2 PSD settings

        # Strain/mu_visc related variables
        self.strain_data = None  # Strain sweep raw data by temperature
        self.fg_by_T = None  # f,g curves by temperature
        self.fg_averaged = None  # Averaged f,g curves
        self.f_interpolator = None  # f(strain) function
        self.g_interpolator = None  # g(strain) function
        self.mu_visc_results = None  # mu_visc calculation results

        # h'rms / Local Strain related variables
        self.rms_slope_calculator = None  # RMSSlopeCalculator instance
        self.rms_slope_profiles = None  # Calculated profiles (q, xi, strain, hrms)
        self.local_strain_array = None  # Local strain for mu_visc calculation

        # Create UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

        # Load default measured data
        self._load_default_data()

    def _load_default_data(self):
        """Load default measured data on startup."""
        try:
            # Get data directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, 'examples', 'data')

            psd_file = os.path.join(data_dir, 'measured_psd.txt')
            dma_file = os.path.join(data_dir, 'measured_dma.txt')

            if os.path.exists(psd_file) and os.path.exists(dma_file):
                # Load DMA data
                omega_raw, E_storage_raw, E_loss_raw = load_dma_from_file(
                    dma_file,
                    skip_header=1,
                    freq_unit='Hz',
                    modulus_unit='MPa'
                )

                # Apply smoothing/fitting
                smoothed = smooth_dma_data(omega_raw, E_storage_raw, E_loss_raw)

                # Store raw data for visualization
                self.raw_dma_data = {
                    'omega': omega_raw,
                    'E_storage': E_storage_raw,
                    'E_loss': E_loss_raw
                }

                # Create material from smoothed data
                self.material = create_material_from_dma(
                    omega=smoothed['omega'],
                    E_storage=smoothed['E_storage_smooth'],
                    E_loss=smoothed['E_loss_smooth'],
                    material_name="Measured Rubber (smoothed)",
                    reference_temp=20.0
                )

                # Load PSD data
                q, C_q = load_psd_from_file(psd_file, skip_header=1)
                self.psd_model = create_psd_from_data(q, C_q, interpolation_kind='log-log')

                # Update UI
                self.q_min_var.set(f"{q[0]:.2e}")
                # Don't overwrite q_max - keep the default value (6.0e+4) for better initial calculations
                # User can manually adjust if they want to use the full PSD range
                # self.q_max_var.set(f"{q[-1]:.2e}")
                self.psd_type_var.set("measured")

                self._update_material_display()
                self._update_verification_plots()

                self.status_var.set(f"초기 데이터 로드 완료: PSD ({len(q)}개), DMA ({len(omega_raw)}개)")
            else:
                # Use example material
                self.material = ViscoelasticMaterial.create_example_sbr()
                self._update_material_display()
                self.status_var.set("예제 재료 (SBR) 로드됨")

        except Exception as e:
            print(f"Default data loading error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to example material
            self.material = ViscoelasticMaterial.create_example_sbr()
            self.status_var.set("예제 재료 (SBR) 로드됨")

    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load DMA Data", command=self._load_material)
        file_menu.add_command(label="Load PSD Data", command=self._load_psd_data)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results (CSV)", command=self._save_detailed_csv)
        file_menu.add_command(label="Export All", command=self._export_all_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self._show_help)
        help_menu.add_command(label="About", command=self._show_about)

    def _create_main_layout(self):
        """Create main application layout with tabs."""
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 0: Master Curve Generation (NEW - First Tab)
        self.tab_master_curve = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_master_curve, text="0. 마스터 커브 생성")
        self._create_master_curve_tab(self.tab_master_curve)

        # Tab 1: Input Data Verification
        self.tab_verification = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_verification, text="1. 입력 데이터 검증")
        self._create_verification_tab(self.tab_verification)

        # Tab 2: Calculation Parameters
        self.tab_parameters = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_parameters, text="2. 계산 설정")
        self._create_parameters_tab(self.tab_parameters)

        # Tab 3: G(q,v) Results
        self.tab_results = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_results, text="3. G(q,v) 결과")
        self._create_results_tab(self.tab_results)

        # Tab 4: h'rms / Local Strain
        self.tab_rms_slope = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_rms_slope, text="4. h'rms/Local Strain")
        self._create_rms_slope_tab(self.tab_rms_slope)

        # Tab 5: mu_visc Calculation
        self.tab_mu_visc = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_mu_visc, text="5. μ_visc 계산")
        self._create_mu_visc_tab(self.tab_mu_visc)

        # Tab 6: Local Strain Map (NEW)
        self.tab_strain_map = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_strain_map, text="6. Local Strain Map")
        self._create_strain_map_tab(self.tab_strain_map)

        # Tab 7: Integrand Visualization
        self.tab_integrand = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_integrand, text="7. 피적분함수 분석")
        self._create_integrand_tab(self.tab_integrand)

        # Tab 8: Equations Summary
        self.tab_equations = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_equations, text="8. 수식 정리")
        self._create_equations_tab(self.tab_equations)

        # Tab 9: Variable Relationship
        self.tab_variables = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_variables, text="9. 변수 관계")
        self._create_variables_tab(self.tab_variables)

        # Tab 10: Debug Log (NEW)
        self.tab_debug = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_debug, text="10. 디버그 로그")
        self._create_debug_tab(self.tab_debug)

        # Tab 11: Friction Factor Analysis (NEW)
        self.tab_friction_factors = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_friction_factors, text="11. 마찰계수 영향 인자")
        self._create_friction_factors_tab(self.tab_friction_factors)

        # Initialize debug log storage
        self.debug_log_messages = []

    def _create_verification_tab(self, parent):
        """Create input data verification tab."""
        # Instruction label
        instruction = ttk.LabelFrame(parent, text="탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "이 탭에서는 계산 전에 재료 물성과 표면 거칠기 데이터가 올바르게\n"
            "로드되었는지 확인합니다. E', E'', tan(δ) 및 C(q)를 검토하세요.",
            font=('Arial', 10)
        ).pack()

        # DMA data import controls
        import_frame = ttk.LabelFrame(parent, text="DMA 데이터 가져오기", padding=10)
        import_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create controls in a grid
        control_grid = ttk.Frame(import_frame)
        control_grid.pack(fill=tk.X)

        # Import from Master Curve button
        ttk.Button(
            control_grid,
            text="마스터 커브에서 가져오기 (Tab 0)",
            command=self._import_from_master_curve,
            width=30
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)

        # Status label
        self.dma_import_status_var = tk.StringVar(value="데이터 미로드")
        ttk.Label(
            control_grid,
            textvariable=self.dma_import_status_var,
            font=('Arial', 9),
            foreground='gray'
        ).grid(row=0, column=1, sticky=tk.W, padx=10, pady=3)

        # Two-column layout for DMA and PSD settings
        settings_container = ttk.Frame(parent)
        settings_container.pack(fill=tk.X, padx=10, pady=5)

        # Left column: DMA Smoothing/Extrapolation (compact)
        dma_frame = ttk.LabelFrame(settings_container, text="DMA Smoothing/Extrapolation", padding=5)
        dma_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Load DMA button row (inside labelframe, at top)
        dma_load_row = ttk.Frame(dma_frame)
        dma_load_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(dma_load_row, text="Load DMA", command=self._load_material, width=10).pack(side=tk.LEFT)

        # Smoothing row
        smooth_row = ttk.Frame(dma_frame)
        smooth_row.pack(fill=tk.X, pady=1)
        self.verify_smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(smooth_row, text="Smooth", variable=self.verify_smooth_var).pack(side=tk.LEFT)
        ttk.Label(smooth_row, text="Window:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(5, 2))
        self.verify_smooth_window_var = tk.IntVar(value=11)
        ttk.Entry(smooth_row, textvariable=self.verify_smooth_window_var, width=4).pack(side=tk.LEFT)

        # Extrapolation row with range
        extrap_row = ttk.Frame(dma_frame)
        extrap_row.pack(fill=tk.X, pady=1)
        self.verify_extrap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(extrap_row, text="Extrapolate", variable=self.verify_extrap_var).pack(side=tk.LEFT)
        ttk.Label(extrap_row, text="f_min:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(5, 2))
        self.dma_extrap_fmin_var = tk.StringVar(value="1e-2")
        ttk.Entry(extrap_row, textvariable=self.dma_extrap_fmin_var, width=6).pack(side=tk.LEFT)
        ttk.Label(extrap_row, text="f_max:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(5, 2))
        self.dma_extrap_fmax_var = tk.StringVar(value="1e12")
        ttk.Entry(extrap_row, textvariable=self.dma_extrap_fmax_var, width=6).pack(side=tk.LEFT)

        # Apply DMA button
        ttk.Button(dma_frame, text="Apply DMA", command=self._apply_dma_smoothing_extrapolation, width=12).pack(pady=2)

        # Right column: PSD Settings
        psd_frame = ttk.LabelFrame(settings_container, text="PSD Settings (Power Law)", padding=5)
        psd_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Load PSD button row (at top of PSD frame)
        psd_load_row = ttk.Frame(psd_frame)
        psd_load_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(psd_load_row, text="Load PSD", command=self._load_psd_data, width=10).pack(side=tk.LEFT)

        # q0, q1 row
        q_row = ttk.Frame(psd_frame)
        q_row.pack(fill=tk.X, pady=1)
        ttk.Label(q_row, text="q0:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.psd_q0_var = tk.StringVar(value="1e2")
        ttk.Entry(q_row, textvariable=self.psd_q0_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(q_row, text="q1:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(5, 0))
        self.psd_q1_var = tk.StringVar(value="1e8")
        ttk.Entry(q_row, textvariable=self.psd_q1_var, width=8).pack(side=tk.LEFT, padx=2)

        # H row
        h_row = ttk.Frame(psd_frame)
        h_row.pack(fill=tk.X, pady=1)
        ttk.Label(h_row, text="H (Hurst):", font=('Arial', 8)).pack(side=tk.LEFT)
        self.psd_H_var = tk.StringVar(value="0.8")
        ttk.Entry(h_row, textvariable=self.psd_H_var, width=6).pack(side=tk.LEFT, padx=2)

        # ξ target row - specify h'rms to auto-calculate C(q0)
        xi_row = ttk.Frame(psd_frame)
        xi_row.pack(fill=tk.X, pady=1)
        ttk.Label(xi_row, text="ξ (h'rms):", font=('Arial', 8)).pack(side=tk.LEFT)
        self.psd_xi_var = tk.StringVar(value="1.3")
        ttk.Entry(xi_row, textvariable=self.psd_xi_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(xi_row, text="→ C(q0) 계산", command=self._calc_Cq0_from_xi, width=10).pack(side=tk.LEFT, padx=(5, 0))

        # C(q0) row (can be manually overridden)
        cq_row = ttk.Frame(psd_frame)
        cq_row.pack(fill=tk.X, pady=1)
        ttk.Label(cq_row, text="C(q0):", font=('Arial', 8)).pack(side=tk.LEFT)
        self.psd_Cq0_var = tk.StringVar(value="1e-18")
        ttk.Entry(cq_row, textvariable=self.psd_Cq0_var, width=12).pack(side=tk.LEFT, padx=2)

        # Apply PSD button
        ttk.Button(psd_frame, text="Apply PSD", command=self._apply_psd_settings, width=12).pack(pady=2)

        # Plot area
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create figure with 2 subplots
        self.fig_verification = Figure(figsize=(14, 6), dpi=100)

        self.ax_master_curve = self.fig_verification.add_subplot(121)
        self.ax_psd = self.fig_verification.add_subplot(122)

        self.canvas_verification = FigureCanvasTkAgg(self.fig_verification, plot_frame)
        self.canvas_verification.draw()
        self.canvas_verification.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_verification, plot_frame)
        toolbar.update()

        # Refresh button
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btn_frame,
            text="그래프 새로고침",
            command=self._update_verification_plots
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="그래프 저장",
            command=lambda: self._save_plot(self.fig_verification, "verification_plot")
        ).pack(side=tk.LEFT, padx=5)

    def _create_master_curve_tab(self, parent):
        """Create Master Curve generation tab using Time-Temperature Superposition."""
        # Main container
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls (fixed width)
        left_frame = ttk.Frame(main_container, width=380)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # ============== Left Panel: Controls ==============

        # 1. Description
        desc_frame = ttk.LabelFrame(left_frame, text="탭 설명", padding=5)
        desc_frame.pack(fill=tk.X, pady=2, padx=3)

        desc_text = (
            "다중 온도 DMA 데이터로부터 마스터 커브를 생성합니다.\n"
            "시간-온도 중첩 원리(TTS) 적용:\n"
            "  - 수평 이동 aT: 주파수 시프트\n"
            "  - 수직 이동 bT: 모듈러스 시프트 (밀도/엔트로피 보정)"
        )
        ttk.Label(desc_frame, text=desc_text, font=('Arial', 9), justify=tk.LEFT).pack(anchor=tk.W)

        # 2. File Loading
        load_frame = ttk.LabelFrame(left_frame, text="데이터 로드", padding=5)
        load_frame.pack(fill=tk.X, pady=2, padx=3)

        ttk.Button(
            load_frame,
            text="다중 온도 DMA 데이터 로드 (CSV/Excel)",
            command=self._load_multi_temp_dma
        ).pack(fill=tk.X, pady=2)

        self.mc_data_info_var = tk.StringVar(value="데이터 미로드")
        ttk.Label(load_frame, textvariable=self.mc_data_info_var,
                  font=('Arial', 8), foreground='gray').pack(anchor=tk.W)

        # 3. Settings
        settings_frame = ttk.LabelFrame(left_frame, text="설정", padding=5)
        settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # Reference temperature
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="기준 온도 Tref (°C):", font=('Arial', 9)).pack(side=tk.LEFT)
        self.mc_tref_var = tk.StringVar(value="20.0")
        ttk.Entry(row1, textvariable=self.mc_tref_var, width=10).pack(side=tk.RIGHT)

        # bT mode
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="bT 계산 방법:", font=('Arial', 9)).pack(side=tk.LEFT)
        self.mc_bt_mode_var = tk.StringVar(value="optimize")
        bt_combo = ttk.Combobox(
            row2, textvariable=self.mc_bt_mode_var,
            values=["optimize", "theoretical"],
            width=12, state="readonly"
        )
        bt_combo.pack(side=tk.RIGHT)

        # Apply bT checkbox
        self.mc_use_bt_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="수직 이동 bT 적용 (Apply Vertical Shift)",
            variable=self.mc_use_bt_var
        ).pack(anchor=tk.W, pady=2)

        ttk.Label(settings_frame,
                  text="optimize: 수치 최적화 / theoretical: T/Tref 공식",
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # Optimization target selection
        row3 = ttk.Frame(settings_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="최적화 대상:", font=('Arial', 9)).pack(side=tk.LEFT)
        self.mc_target_var = tk.StringVar(value="E_storage")
        target_combo = ttk.Combobox(
            row3, textvariable=self.mc_target_var,
            values=["E_storage", "E_loss", "tan_delta"],
            width=12, state="readonly"
        )
        target_combo.pack(side=tk.RIGHT)
        ttk.Label(settings_frame,
                  text="E_storage: E' / E_loss: E'' / tan_delta: tanδ",
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # Smoothing control for master curve
        smooth_frame = ttk.LabelFrame(settings_frame, text="마스터 커브 스무딩", padding=3)
        smooth_frame.pack(fill=tk.X, pady=3)

        self.mc_smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            smooth_frame,
            text="스무딩 적용",
            variable=self.mc_smooth_var
        ).pack(anchor=tk.W)

        # Smoothing window slider
        slider_frame = ttk.Frame(smooth_frame)
        slider_frame.pack(fill=tk.X, pady=2)
        ttk.Label(slider_frame, text="스무딩 강도:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.mc_smooth_window_var = tk.IntVar(value=11)
        self.mc_smooth_slider = ttk.Scale(
            slider_frame,
            from_=5, to=51,
            orient=tk.HORIZONTAL,
            variable=self.mc_smooth_window_var,
            command=lambda v: self.mc_smooth_label.config(text=f"{int(float(v))}")
        )
        self.mc_smooth_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.mc_smooth_label = ttk.Label(slider_frame, text="11", width=3)
        self.mc_smooth_label.pack(side=tk.RIGHT)

        # bT comparison checkbox
        self.mc_compare_bt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            smooth_frame,
            text="bT 적용/미적용 비교 보기",
            variable=self.mc_compare_bt_var,
            command=self._toggle_bt_comparison
        ).pack(anchor=tk.W, pady=2)

        # 4. Calculate button
        calc_frame = ttk.Frame(settings_frame)
        calc_frame.pack(fill=tk.X, pady=5)

        self.mc_calc_btn = ttk.Button(
            calc_frame,
            text="마스터 커브 생성",
            command=self._generate_master_curve
        )
        self.mc_calc_btn.pack(fill=tk.X)

        # Progress bar
        self.mc_progress_var = tk.IntVar()
        self.mc_progress_bar = ttk.Progressbar(
            calc_frame, variable=self.mc_progress_var, maximum=100
        )
        self.mc_progress_bar.pack(fill=tk.X, pady=2)

        # 5. Results Summary
        results_frame = ttk.LabelFrame(left_frame, text="결과 요약", padding=5)
        results_frame.pack(fill=tk.X, pady=2, padx=3)

        self.mc_result_text = tk.Text(results_frame, height=10, font=("Courier", 8), wrap=tk.WORD)
        self.mc_result_text.pack(fill=tk.X)

        # 6. Shift Factor Table
        table_frame = ttk.LabelFrame(left_frame, text="Shift Factor 테이블", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=2, padx=3)

        # Create treeview for shift factors
        columns = ('T', 'aT', 'bT', 'log_aT')
        self.mc_shift_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        self.mc_shift_table.heading('T', text='T (°C)')
        self.mc_shift_table.heading('aT', text='aT')
        self.mc_shift_table.heading('bT', text='bT')
        self.mc_shift_table.heading('log_aT', text='log(aT)')

        self.mc_shift_table.column('T', width=60, anchor='center')
        self.mc_shift_table.column('aT', width=80, anchor='center')
        self.mc_shift_table.column('bT', width=60, anchor='center')
        self.mc_shift_table.column('log_aT', width=70, anchor='center')

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.mc_shift_table.yview)
        self.mc_shift_table.configure(yscrollcommand=scrollbar.set)

        self.mc_shift_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 7. Export buttons
        export_frame = ttk.LabelFrame(left_frame, text="내보내기", padding=5)
        export_frame.pack(fill=tk.X, pady=2, padx=3)

        ttk.Button(
            export_frame, text="마스터 커브 CSV 내보내기",
            command=self._export_master_curve
        ).pack(fill=tk.X, pady=1)

        ttk.Button(
            export_frame, text="마스터 커브 적용 (Tab 1로 전송)",
            command=self._apply_master_curve_to_verification
        ).pack(fill=tk.X, pady=1)

        # ============== Right Panel: Plots ==============

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_panel, text="시각화", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with 2x2 subplots
        self.fig_mc = Figure(figsize=(11, 8), dpi=100)

        # Top-left: Raw data (multi-temperature)
        self.ax_mc_raw = self.fig_mc.add_subplot(221)
        self.ax_mc_raw.set_title('원본 데이터 (온도별)', fontweight='bold')
        self.ax_mc_raw.set_xlabel('주파수 f (Hz)')
        self.ax_mc_raw.set_ylabel('E\', E\'\' (MPa)')
        self.ax_mc_raw.set_xscale('log')
        self.ax_mc_raw.set_yscale('log')
        self.ax_mc_raw.grid(True, alpha=0.3)

        # Top-right: Master curve
        self.ax_mc_master = self.fig_mc.add_subplot(222)
        self.ax_mc_master.set_title('마스터 커브 (Tref)', fontweight='bold')
        self.ax_mc_master.set_xlabel('Reduced Frequency (Hz)')
        self.ax_mc_master.set_ylabel('E\', E\'\' (MPa)')
        self.ax_mc_master.set_xscale('log')
        self.ax_mc_master.set_yscale('log')
        self.ax_mc_master.grid(True, alpha=0.3)

        # Bottom-left: aT vs Temperature
        self.ax_mc_aT = self.fig_mc.add_subplot(223)
        self.ax_mc_aT.set_title('수평 이동 계수 aT', fontweight='bold')
        self.ax_mc_aT.set_xlabel('온도 T (°C)')
        self.ax_mc_aT.set_ylabel('log10(aT)')
        self.ax_mc_aT.grid(True, alpha=0.3)

        # Bottom-right: bT vs Temperature
        self.ax_mc_bT = self.fig_mc.add_subplot(224)
        self.ax_mc_bT.set_title('수직 이동 계수 bT', fontweight='bold')
        self.ax_mc_bT.set_xlabel('온도 T (°C)')
        self.ax_mc_bT.set_ylabel('bT')
        self.ax_mc_bT.grid(True, alpha=0.3)

        self.fig_mc.tight_layout()

        self.canvas_mc = FigureCanvasTkAgg(self.fig_mc, plot_frame)
        self.canvas_mc.draw()
        self.canvas_mc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_mc, plot_frame)
        toolbar.update()

        # Initialize master curve generator
        self.master_curve_gen = None
        self.mc_raw_df = None

    def _load_multi_temp_dma(self):
        """Load multi-temperature DMA data for master curve generation."""
        filename = filedialog.askopenfilename(
            title="다중 온도 DMA 데이터 파일 선택",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            import pandas as pd

            # Load data
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                self.mc_raw_df = pd.read_excel(filename, skiprows=1)
            else:
                # Try different delimiters
                try:
                    self.mc_raw_df = pd.read_csv(filename, skiprows=1, sep='\t')
                    if len(self.mc_raw_df.columns) < 4:
                        self.mc_raw_df = pd.read_csv(filename, skiprows=1, sep=',')
                except:
                    self.mc_raw_df = pd.read_csv(filename, skiprows=1, delim_whitespace=True)

            # Standardize column names based on number of columns
            n_cols = len(self.mc_raw_df.columns)

            if n_cols >= 7:
                # Full format with |E*|: f, T, f_reduced, Amplitude, E', E'', |E*|
                col_names = ['f', 'T', 'f_reduced', 'Amplitude', "E'", "E''", "E_star"]
                # Extend with generic names if more columns exist
                while len(col_names) < n_cols:
                    col_names.append(f"col_{len(col_names)}")
                self.mc_raw_df.columns = col_names[:n_cols]
            elif n_cols == 6:
                # Full format: f, T, f_reduced, Amplitude, E', E''
                self.mc_raw_df.columns = ['f', 'T', 'f_reduced', 'Amplitude', "E'", "E''"]
            elif n_cols == 5:
                # Missing E'': f, T, f_reduced, Amplitude, E'
                self.mc_raw_df.columns = ['f', 'T', 'f_reduced', 'Amplitude', "E'"]
                # E'' is missing - show warning
                messagebox.showwarning(
                    "주의",
                    "E'' (손실 탄성률) 컬럼이 없습니다.\n"
                    "마스터 커브 생성은 E'만 사용합니다.\n\n"
                    "완전한 분석을 위해 6개 컬럼 데이터를 권장합니다:\n"
                    "f(Hz), T(°C), f_reduced, Amplitude, E'(MPa), E''(MPa)"
                )
                # Estimate E'' as 10% of E' (rough estimate for demonstration)
                self.mc_raw_df["E''"] = self.mc_raw_df["E'"] * 0.1
            elif n_cols == 4:
                # Minimal: f, T, E', E''
                self.mc_raw_df.columns = ['f', 'T', "E'", "E''"]
            elif n_cols == 3:
                # Very minimal: f, T, E' (no E'')
                self.mc_raw_df.columns = ['f', 'T', "E'"]
                self.mc_raw_df["E''"] = self.mc_raw_df["E'"] * 0.1
            else:
                raise ValueError(f"데이터 컬럼 수가 부족합니다 ({n_cols}개). 최소 3개 컬럼이 필요합니다.")

            # Convert to numeric and drop NaN
            for col in self.mc_raw_df.columns:
                self.mc_raw_df[col] = pd.to_numeric(self.mc_raw_df[col], errors='coerce')
            self.mc_raw_df = self.mc_raw_df.dropna()

            # Get unique temperatures
            temps = self.mc_raw_df['T'].unique()
            n_temps = len(temps)
            n_points = len(self.mc_raw_df)

            self.mc_data_info_var.set(f"로드됨: {n_points}개 데이터, {n_temps}개 온도")

            # Plot raw data
            self._plot_mc_raw_data()

            messagebox.showinfo(
                "성공",
                f"데이터 로드 완료:\n"
                f"  - 총 {n_points}개 데이터 포인트\n"
                f"  - {n_temps}개 온도: {temps.min():.1f}°C ~ {temps.max():.1f}°C\n"
                f"  - 주파수 범위: {self.mc_raw_df['f'].min():.2f} ~ {self.mc_raw_df['f'].max():.2f} Hz"
            )

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"데이터 로드 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _plot_mc_raw_data(self):
        """Plot raw multi-temperature DMA data."""
        if self.mc_raw_df is None:
            return

        self.ax_mc_raw.clear()
        self.ax_mc_raw.set_title('원본 데이터 (온도별)', fontweight='bold')
        self.ax_mc_raw.set_xlabel('주파수 f (Hz)')
        self.ax_mc_raw.set_ylabel('E\', E\'\' (MPa)')
        self.ax_mc_raw.set_xscale('log')
        self.ax_mc_raw.set_yscale('log')
        self.ax_mc_raw.grid(True, alpha=0.3)

        temps = np.sort(self.mc_raw_df['T'].unique())
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))

        for i, T in enumerate(temps):
            mask = self.mc_raw_df['T'] == T
            data = self.mc_raw_df[mask]

            self.ax_mc_raw.plot(data['f'], data["E'"], 'o-', color=colors[i],
                               markersize=3, linewidth=1, alpha=0.7,
                               label=f"E' ({T:.0f}°C)")
            self.ax_mc_raw.plot(data['f'], data["E''"], 's--', color=colors[i],
                               markersize=2, linewidth=0.8, alpha=0.5)

        # Add legend with limited entries
        if len(temps) <= 8:
            self.ax_mc_raw.legend(fontsize=6, loc='upper left', ncol=2)

        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

    def _generate_master_curve(self):
        """Generate master curve using TTS."""
        if self.mc_raw_df is None:
            messagebox.showwarning("경고", "먼저 다중 온도 DMA 데이터를 로드하세요.")
            return

        try:
            self.mc_calc_btn.config(state='disabled')
            self.mc_progress_var.set(10)
            self.root.update_idletasks()

            # Get settings
            T_ref = float(self.mc_tref_var.get())
            use_bT = self.mc_use_bt_var.get()
            bT_mode = self.mc_bt_mode_var.get()
            target = self.mc_target_var.get()

            # Create master curve generator
            self.master_curve_gen = MasterCurveGenerator(T_ref=T_ref)

            # Load data
            self.master_curve_gen.load_data(
                self.mc_raw_df,
                T_col='T', f_col='f',
                E_storage_col="E'", E_loss_col="E''"
            )

            self.mc_progress_var.set(30)
            self.root.update_idletasks()

            # Optimize shift factors
            self.master_curve_gen.optimize_shift_factors(
                use_bT=use_bT,
                bT_mode=bT_mode,
                target=target,
                verbose=False
            )

            self.mc_progress_var.set(60)
            self.root.update_idletasks()

            # Generate master curve with smoothing settings
            smooth_enabled = self.mc_smooth_var.get()
            smooth_window = self.mc_smooth_window_var.get()
            # Ensure window is odd
            if smooth_window % 2 == 0:
                smooth_window += 1

            master_curve = self.master_curve_gen.generate_master_curve(
                n_points=300, smooth=smooth_enabled, window_length=smooth_window
            )

            self.mc_progress_var.set(80)
            self.root.update_idletasks()

            # Fit WLF
            wlf_result = self.master_curve_gen.fit_wlf()

            self.mc_progress_var.set(90)
            self.root.update_idletasks()

            # Update plots
            self._update_mc_plots(master_curve, wlf_result, target)

            # Update results text
            self._update_mc_results(wlf_result, target)

            # Update shift factor table
            self._update_mc_shift_table()

            self.mc_progress_var.set(100)
            self.status_var.set("마스터 커브 생성 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"마스터 커브 생성 실패:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.mc_calc_btn.config(state='normal')

    def _update_mc_plots(self, master_curve, wlf_result, target='E_storage'):
        """Update master curve plots."""
        # Clear all axes except raw data
        self.ax_mc_master.clear()
        self.ax_mc_aT.clear()
        self.ax_mc_bT.clear()

        # Get data
        temps = np.sort(self.master_curve_gen.temperatures)
        aT = self.master_curve_gen.aT
        bT = self.master_curve_gen.bT
        T_ref = self.master_curve_gen.T_ref

        # Target display name
        target_names = {
            'E_storage': "E'",
            'E_loss': "E''",
            'tan_delta': "tanδ"
        }
        target_display = target_names.get(target, target)

        # Colors for temperatures
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))

        # Plot 1: Master curve with shifted data
        self.ax_mc_master.set_title(f'마스터 커브 (Tref={T_ref}°C, 최적화: {target_display})', fontweight='bold')
        self.ax_mc_master.set_xlabel('Reduced Frequency (Hz)')
        self.ax_mc_master.set_ylabel('E\', E\'\' (MPa)')
        self.ax_mc_master.set_xscale('log')
        self.ax_mc_master.set_yscale('log')
        self.ax_mc_master.grid(True, alpha=0.3)

        # Plot shifted data points
        for i, T in enumerate(temps):
            shifted = self.master_curve_gen.get_shifted_data(T)
            self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_storage_shifted'],
                                     c=[colors[i]], s=10, alpha=0.5, marker='o')
            self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_loss_shifted'],
                                     c=[colors[i]], s=8, alpha=0.3, marker='s')

        # Plot master curve
        self.ax_mc_master.plot(master_curve['f'], master_curve['E_storage'],
                              'k-', linewidth=2, label="E' (Master)")
        self.ax_mc_master.plot(master_curve['f'], master_curve['E_loss'],
                              'k--', linewidth=2, label="E'' (Master)")
        self.ax_mc_master.legend(fontsize=8)

        # Plot 2: aT vs Temperature
        self.ax_mc_aT.set_title('수평 이동 계수 aT', fontweight='bold')
        self.ax_mc_aT.set_xlabel('온도 T (°C)')
        self.ax_mc_aT.set_ylabel('log10(aT)')
        self.ax_mc_aT.grid(True, alpha=0.3)

        log_aT = [np.log10(aT[T]) for T in temps]
        self.ax_mc_aT.scatter(temps, log_aT, c='blue', s=50, zorder=3, label='Measured')

        # WLF fit line
        if wlf_result['C1'] is not None:
            T_fit = np.linspace(temps.min(), temps.max(), 100)
            log_aT_fit = -wlf_result['C1'] * (T_fit - T_ref) / (wlf_result['C2'] + (T_fit - T_ref))
            self.ax_mc_aT.plot(T_fit, log_aT_fit, 'r-', linewidth=2,
                              label=f"WLF (C1={wlf_result['C1']:.2f}, C2={wlf_result['C2']:.1f})")
        self.ax_mc_aT.legend(fontsize=8)
        self.ax_mc_aT.axhline(0, color='gray', linestyle=':', alpha=0.5)
        self.ax_mc_aT.axvline(T_ref, color='green', linestyle='--', alpha=0.5, label=f'Tref={T_ref}°C')

        # Plot 3: bT vs Temperature
        self.ax_mc_bT.set_title('수직 이동 계수 bT', fontweight='bold')
        self.ax_mc_bT.set_xlabel('온도 T (°C)')
        self.ax_mc_bT.set_ylabel('bT')
        self.ax_mc_bT.grid(True, alpha=0.3)

        bT_values = [bT[T] for T in temps]
        self.ax_mc_bT.scatter(temps, bT_values, c='blue', s=50, zorder=3, label='Measured')

        # Theoretical line T/Tref
        T_ref_K = T_ref + 273.15
        bT_theory = (temps + 273.15) / T_ref_K
        self.ax_mc_bT.plot(temps, bT_theory, 'r--', linewidth=1.5, label='T/Tref (이론)')

        self.ax_mc_bT.axhline(1, color='gray', linestyle=':', alpha=0.5)
        self.ax_mc_bT.axvline(T_ref, color='green', linestyle='--', alpha=0.5)
        self.ax_mc_bT.legend(fontsize=8)

        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

    def _update_mc_results(self, wlf_result, target='E_storage'):
        """Update master curve results text."""
        self.mc_result_text.delete('1.0', tk.END)

        T_ref = self.master_curve_gen.T_ref
        temps = self.master_curve_gen.temperatures

        # Target display name
        target_names = {
            'E_storage': "E' (Storage Modulus)",
            'E_loss': "E'' (Loss Modulus)",
            'tan_delta': "tanδ (Loss Factor)"
        }
        target_display = target_names.get(target, target)

        text = f"=== 마스터 커브 생성 결과 ===\n\n"
        text += f"기준 온도 Tref: {T_ref}°C\n"
        text += f"최적화 대상: {target_display}\n"
        text += f"온도 범위: {temps.min():.1f} ~ {temps.max():.1f}°C\n"
        text += f"온도 개수: {len(temps)}개\n\n"

        if wlf_result['C1'] is not None:
            text += f"=== WLF 파라미터 ===\n"
            text += f"C1 = {wlf_result['C1']:.4f}\n"
            text += f"C2 = {wlf_result['C2']:.4f}\n"
            text += f"R² = {wlf_result['r_squared']:.4f}\n\n"

        text += f"마스터 커브 주파수 범위:\n"
        text += f"  {self.master_curve_gen.master_f.min():.2e} ~ "
        text += f"{self.master_curve_gen.master_f.max():.2e} Hz\n"

        self.mc_result_text.insert(tk.END, text)

    def _toggle_bt_comparison(self):
        """Toggle bT comparison view in master curve plot."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            return

        show_comparison = self.mc_compare_bt_var.get()

        # Clear and redraw master curve plot
        self.ax_mc_master.clear()

        temps = np.sort(self.master_curve_gen.temperatures)
        T_ref = self.master_curve_gen.T_ref
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))

        self.ax_mc_master.set_xlabel('Reduced Frequency (Hz)')
        self.ax_mc_master.set_ylabel('E\', E\'\' (MPa)')
        self.ax_mc_master.set_xscale('log')
        self.ax_mc_master.set_yscale('log')
        self.ax_mc_master.grid(True, alpha=0.3)

        if show_comparison:
            # Show both with and without bT
            self.ax_mc_master.set_title(f'마스터 커브 비교: bT 적용 vs 미적용 (Tref={T_ref}°C)', fontweight='bold')

            # Plot WITH bT (solid lines)
            for i, T in enumerate(temps):
                shifted = self.master_curve_gen.get_shifted_data(T)
                self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_storage_shifted'],
                                         c=[colors[i]], s=12, alpha=0.7, marker='o')

            # Plot master curve with bT
            self.ax_mc_master.plot(self.master_curve_gen.master_f, self.master_curve_gen.master_E_storage,
                                  'k-', linewidth=2.5, label="E' (bT 적용)", zorder=10)
            self.ax_mc_master.plot(self.master_curve_gen.master_f, self.master_curve_gen.master_E_loss,
                                  'k--', linewidth=2.5, label="E'' (bT 적용)", zorder=10)

            # Plot WITHOUT bT (dashed, lighter)
            for i, T in enumerate(temps):
                f_reduced = self.master_curve_gen.raw_data[T]['f'] * self.master_curve_gen.aT[T]
                E_storage_no_bt = self.master_curve_gen.raw_data[T]['E_storage']  # No bT division
                E_loss_no_bt = self.master_curve_gen.raw_data[T]['E_loss']
                self.ax_mc_master.scatter(f_reduced, E_storage_no_bt,
                                         c=[colors[i]], s=8, alpha=0.3, marker='x')

            # Generate master curve without bT for comparison
            all_f = []
            all_E_storage_no_bt = []
            all_E_loss_no_bt = []
            for T in temps:
                f_reduced = self.master_curve_gen.raw_data[T]['f'] * self.master_curve_gen.aT[T]
                all_f.extend(f_reduced)
                all_E_storage_no_bt.extend(self.master_curve_gen.raw_data[T]['E_storage'])
                all_E_loss_no_bt.extend(self.master_curve_gen.raw_data[T]['E_loss'])

            all_f = np.array(all_f)
            all_E_storage_no_bt = np.array(all_E_storage_no_bt)
            all_E_loss_no_bt = np.array(all_E_loss_no_bt)
            sort_idx = np.argsort(all_f)

            self.ax_mc_master.plot(all_f[sort_idx], all_E_storage_no_bt[sort_idx],
                                  'b:', linewidth=1.5, alpha=0.7, label="E' (bT 미적용)")
            self.ax_mc_master.plot(all_f[sort_idx], all_E_loss_no_bt[sort_idx],
                                  'r:', linewidth=1.5, alpha=0.7, label="E'' (bT 미적용)")

        else:
            # Normal view with bT
            target = self.mc_target_var.get()
            target_names = {'E_storage': "E'", 'E_loss': "E''", 'tan_delta': "tanδ"}
            target_display = target_names.get(target, target)
            self.ax_mc_master.set_title(f'마스터 커브 (Tref={T_ref}°C, 최적화: {target_display})', fontweight='bold')

            for i, T in enumerate(temps):
                shifted = self.master_curve_gen.get_shifted_data(T)
                self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_storage_shifted'],
                                         c=[colors[i]], s=10, alpha=0.5, marker='o')
                self.ax_mc_master.scatter(shifted['f_reduced'], shifted['E_loss_shifted'],
                                         c=[colors[i]], s=8, alpha=0.3, marker='s')

            self.ax_mc_master.plot(self.master_curve_gen.master_f, self.master_curve_gen.master_E_storage,
                                  'k-', linewidth=2, label="E' (Master)")
            self.ax_mc_master.plot(self.master_curve_gen.master_f, self.master_curve_gen.master_E_loss,
                                  'k--', linewidth=2, label="E'' (Master)")

        self.ax_mc_master.legend(fontsize=8, loc='best')
        self.fig_mc.tight_layout()
        self.canvas_mc.draw()

    def _update_mc_shift_table(self):
        """Update shift factor table."""
        # Clear existing entries
        for item in self.mc_shift_table.get_children():
            self.mc_shift_table.delete(item)

        # Add new entries
        temps = np.sort(self.master_curve_gen.temperatures)
        for T in temps:
            aT = self.master_curve_gen.aT[T]
            bT = self.master_curve_gen.bT[T]
            log_aT = np.log10(aT)

            self.mc_shift_table.insert('', 'end', values=(
                f'{T:.1f}',
                f'{aT:.4e}',
                f'{bT:.4f}',
                f'{log_aT:.4f}'
            ))

    def _export_master_curve(self):
        """Export master curve data to CSV."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            messagebox.showwarning("경고", "먼저 마스터 커브를 생성하세요.")
            return

        filename = filedialog.asksaveasfilename(
            title="마스터 커브 저장",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            import pandas as pd

            # Master curve data
            mc_data = pd.DataFrame({
                'f (Hz)': self.master_curve_gen.master_f,
                'omega (rad/s)': 2 * np.pi * self.master_curve_gen.master_f,
                "E' (MPa)": self.master_curve_gen.master_E_storage,
                "E'' (MPa)": self.master_curve_gen.master_E_loss,
                'tan_delta': self.master_curve_gen.master_E_loss / self.master_curve_gen.master_E_storage
            })

            # Shift factors
            shift_data = self.master_curve_gen.get_shift_factor_table()

            # Save to CSV with both tables
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                f.write(f"# Master Curve (Tref = {self.master_curve_gen.T_ref}°C)\n")
                mc_data.to_csv(f, index=False)
                f.write("\n# Shift Factors\n")
                shift_data.to_csv(f, index=False)

            messagebox.showinfo("성공", f"마스터 커브 저장 완료:\n{filename}")

        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{str(e)}")

    def _apply_master_curve_to_verification(self):
        """Apply generated master curve to Tab 1 for friction calculation."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            messagebox.showwarning("경고", "먼저 마스터 커브를 생성하세요.")
            return

        try:
            # Get master curve data
            omega = 2 * np.pi * self.master_curve_gen.master_f
            E_storage = self.master_curve_gen.master_E_storage * 1e6  # MPa to Pa
            E_loss = self.master_curve_gen.master_E_loss * 1e6  # MPa to Pa
            T_ref = self.master_curve_gen.T_ref

            # Create material from master curve
            self.material = create_material_from_dma(
                omega=omega,
                E_storage=E_storage,
                E_loss=E_loss,
                material_name=f"Master Curve (Tref={T_ref}°C)",
                reference_temp=T_ref
            )

            # Store raw data for plotting (in omega units)
            self.raw_dma_data = {
                'omega': omega,
                'E_storage': E_storage,
                'E_loss': E_loss
            }

            # Store shift factor data for temperature conversion
            self.master_curve_shift_factors = {
                'aT': self.master_curve_gen.aT.copy(),
                'bT': self.master_curve_gen.bT.copy(),
                'T_ref': T_ref,
                'C1': self.master_curve_gen.C1,
                'C2': self.master_curve_gen.C2,
                'temperatures': list(self.master_curve_gen.temperatures)
            }

            # Update temperature entry with reference temperature
            self.temperature_var.set(str(T_ref))

            # Update status label in Tab 1
            f_min = self.master_curve_gen.master_f.min()
            f_max = self.master_curve_gen.master_f.max()
            self.dma_import_status_var.set(f"Master Curve (Tref={T_ref}°C, {f_min:.1e}~{f_max:.1e} Hz)")

            # Update verification plots
            self._update_verification_plots()
            self._update_material_display()

            # Switch to verification tab
            self.notebook.select(1)  # Tab 1

            # Build info message
            info_msg = f"마스터 커브가 Tab 1에 적용되었습니다.\n\n"
            info_msg += f"기준 온도 (Tref): {self.master_curve_gen.T_ref}°C\n"
            info_msg += f"주파수 범위: {self.master_curve_gen.master_f.min():.2e} ~ "
            info_msg += f"{self.master_curve_gen.master_f.max():.2e} Hz\n\n"

            if self.master_curve_gen.C1 is not None:
                info_msg += f"WLF 파라미터:\n"
                info_msg += f"  C1 = {self.master_curve_gen.C1:.2f}\n"
                info_msg += f"  C2 = {self.master_curve_gen.C2:.2f}°C\n"

            messagebox.showinfo("성공", info_msg)

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"적용 실패:\n{str(e)}\n\n{traceback.format_exc()}")

    def _create_parameters_tab(self, parent):
        """Create calculation parameters tab."""
        # Instruction
        instruction = ttk.LabelFrame(parent, text="탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "계산 매개변수를 설정합니다: 압력, 속도 범위 (로그 스케일), 온도.\n"
            "속도 범위: 0.0001~10 m/s (로그 간격)로 주파수 스윕을 수행합니다.",
            font=('Arial', 10)
        ).pack()

        # Create main container with 2 columns
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left panel for inputs
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        # Right panel for visualization
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Input panel in left column
        input_frame = ttk.LabelFrame(left_panel, text="계산 매개변수", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 5))

        # Create input fields
        row = 0

        # Nominal pressure
        ttk.Label(input_frame, text="공칭 압력 (MPa):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sigma_0_var = tk.StringVar(value="0.3")
        ttk.Entry(input_frame, textvariable=self.sigma_0_var, width=15).grid(row=row, column=1, pady=5)

        # Velocity range
        row += 1
        ttk.Label(input_frame, text="속도 범위:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(input_frame, text="로그 스케일: 0.0001~10 m/s").grid(row=row, column=1, sticky=tk.W, pady=5)

        row += 1
        ttk.Label(input_frame, text="  최소 속도 v_min (m/s):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.v_min_var = tk.StringVar(value="0.0001")
        ttk.Entry(input_frame, textvariable=self.v_min_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="  최대 속도 v_max (m/s):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.v_max_var = tk.StringVar(value="10.0")
        ttk.Entry(input_frame, textvariable=self.v_max_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="  속도 포인트 수:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_velocity_var = tk.StringVar(value="30")
        ttk.Entry(input_frame, textvariable=self.n_velocity_var, width=15).grid(row=row, column=1, pady=5)

        # Temperature
        row += 1
        ttk.Label(input_frame, text="온도 (°C):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.temperature_var = tk.StringVar(value="20")
        ttk.Entry(input_frame, textvariable=self.temperature_var, width=15).grid(row=row, column=1, pady=5)

        # Poisson ratio
        row += 1
        ttk.Label(input_frame, text="푸아송 비:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.poisson_var = tk.StringVar(value="0.5")
        ttk.Entry(input_frame, textvariable=self.poisson_var, width=15).grid(row=row, column=1, pady=5)

        # Wavenumber range
        row += 1
        ttk.Label(input_frame, text="최소 파수 q_min (1/m):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.q_min_var = tk.StringVar(value="2.00e-01")
        ttk.Entry(input_frame, textvariable=self.q_min_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="최대 파수 q_max (1/m):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.q_max_var = tk.StringVar(value="6.0e+4")
        ttk.Entry(input_frame, textvariable=self.q_max_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="파수 포인트 수:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_q_var = tk.StringVar(value="100")
        ttk.Entry(input_frame, textvariable=self.n_q_var, width=15).grid(row=row, column=1, pady=5)

        # ===== h'rms (ξ) / q1 모드 선택 섹션 =====
        # h'rms = ξ = RMS slope (경사), NOT h_rms (height)
        row += 1
        mode_frame = ttk.LabelFrame(input_frame, text="h'rms (ξ, slope) / q1 결정 모드", padding=5)
        mode_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)

        # 설명 라벨
        desc_label = ttk.Label(mode_frame,
            text="※ h'rms = ξ = RMS slope (경사), ξ² = 2π∫k³C(k)dk",
            font=('Arial', 7), foreground='gray')
        desc_label.pack(fill=tk.X, pady=(0, 5))

        # 모드 선택 라디오 버튼
        self.hrms_q1_mode_var = tk.StringVar(value="hrms_to_q1")  # 기본값: h'rms(ξ) → q1

        mode_row1 = ttk.Frame(mode_frame)
        mode_row1.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_row1, text="모드 1: h'rms (ξ) → q1 계산",
                       variable=self.hrms_q1_mode_var, value="hrms_to_q1",
                       command=self._on_hrms_q1_mode_changed).pack(side=tk.LEFT)

        mode_row2 = ttk.Frame(mode_frame)
        mode_row2.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_row2, text="모드 2: q1 → h'rms (ξ) 계산",
                       variable=self.hrms_q1_mode_var, value="q1_to_hrms",
                       command=self._on_hrms_q1_mode_changed).pack(side=tk.LEFT)

        # 구분선
        ttk.Separator(mode_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # h'rms(ξ) 입력 (모드 1용)
        self.hrms_input_frame = ttk.Frame(mode_frame)
        self.hrms_input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.hrms_input_frame, text="목표 h'rms (ξ):").pack(side=tk.LEFT)
        self.target_hrms_slope_var = tk.StringVar(value="1.3")
        self.hrms_entry = ttk.Entry(self.hrms_input_frame, textvariable=self.target_hrms_slope_var, width=12)
        self.hrms_entry.pack(side=tk.LEFT, padx=5)

        # q1 입력 (모드 2용)
        self.q1_input_frame = ttk.Frame(mode_frame)
        self.q1_input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(self.q1_input_frame, text="목표 q1 (1/m):").pack(side=tk.LEFT)
        self.input_q1_var = tk.StringVar(value="1.0e+08")
        self.q1_entry = ttk.Entry(self.q1_input_frame, textvariable=self.input_q1_var, width=12)
        self.q1_entry.pack(side=tk.LEFT, padx=5)

        # Add trace to sync target_hrms_slope_var with Tab 1's psd_xi_var and Tab 4's display
        self.target_hrms_slope_var.trace_add('write', self._on_target_hrms_changed)

        # 계산 버튼
        calc_btn_frame = ttk.Frame(mode_frame)
        calc_btn_frame.pack(fill=tk.X, pady=5)
        self.hrms_q1_calc_btn = ttk.Button(calc_btn_frame, text="h'rms/q1 계산",
                                           command=self._calculate_hrms_q1)
        self.hrms_q1_calc_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(calc_btn_frame, text="Tab 4로 전달",
                  command=self._send_hrms_q1_to_tab4).pack(side=tk.LEFT, padx=5)

        # 구분선
        ttk.Separator(mode_frame, orient='horizontal').pack(fill=tk.X, pady=5)

        # 결과 표시 영역
        result_frame = ttk.Frame(mode_frame)
        result_frame.pack(fill=tk.X, pady=2)

        # 계산된 q1 표시 (모드 1 결과)
        q1_result_row = ttk.Frame(result_frame)
        q1_result_row.pack(fill=tk.X, pady=2)
        ttk.Label(q1_result_row, text="→ 계산된 q1:").pack(side=tk.LEFT)
        self.calculated_q1_var = tk.StringVar(value="(계산 후 표시)")
        self.calculated_q1_label = ttk.Label(q1_result_row, textvariable=self.calculated_q1_var,
                                             font=('Arial', 9, 'bold'), foreground='blue')
        self.calculated_q1_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(q1_result_row, text="(1/m)").pack(side=tk.LEFT)

        # 계산된 h'rms(ξ) 표시 (모드 2 결과)
        hrms_result_row = ttk.Frame(result_frame)
        hrms_result_row.pack(fill=tk.X, pady=2)
        ttk.Label(hrms_result_row, text="→ 계산된 h'rms (ξ):").pack(side=tk.LEFT)
        self.calculated_hrms_var = tk.StringVar(value="(계산 후 표시)")
        self.calculated_hrms_label = ttk.Label(hrms_result_row, textvariable=self.calculated_hrms_var,
                                               font=('Arial', 9, 'bold'), foreground='green')
        self.calculated_hrms_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(hrms_result_row, text="(무차원)").pack(side=tk.LEFT)

        # 초기 모드에 따른 UI 상태 설정
        self._on_hrms_q1_mode_changed()

        # PSD type
        row += 1
        ttk.Label(input_frame, text="PSD 유형:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.psd_type_var = tk.StringVar(value="measured")
        ttk.Combobox(
            input_frame,
            textvariable=self.psd_type_var,
            values=["measured", "fractal"],
            state="readonly",
            width=12
        ).grid(row=row, column=1, pady=5)

        # ===== G(q,v) 계산 버튼 (input_frame 내부) =====
        row += 1
        calc_section = ttk.Frame(input_frame)
        calc_section.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=10)

        self.calc_button = ttk.Button(
            calc_section,
            text="▶ G(q,v) 계산 실행",
            command=self._run_calculation
        )
        self.calc_button.pack(fill=tk.X, pady=2)

        # Progress bar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(
            calc_section,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=2)

        # Calculation visualization area in right panel
        viz_frame = ttk.LabelFrame(right_panel, text="계산 과정 시각화", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True)

        # Status display for current calculation state
        status_display_frame = ttk.Frame(viz_frame)
        status_display_frame.pack(fill=tk.X, pady=(0, 5))

        self.calc_status_label = ttk.Label(
            status_display_frame,
            text="대기 중 | v = - m/s | q 범위 = - ~ - (1/m) | f 범위 = - ~ - (Hz)",
            font=('Arial', 10, 'bold'),
            foreground='blue'
        )
        self.calc_status_label.pack()

        # Create figure for calculation progress with three vertical subplots
        self.fig_calc_progress = Figure(figsize=(8, 15), dpi=100)

        # Top: PSD(q) - wavenumber based (static)
        self.ax_psd_q = self.fig_calc_progress.add_subplot(311)

        # Middle: DMA master curve (animated with frequency range)
        self.ax_dma_progress = self.fig_calc_progress.add_subplot(312)

        # Bottom: PSD(f) - frequency based (animated)
        self.ax_psd_f = self.fig_calc_progress.add_subplot(313)

        self.canvas_calc_progress = FigureCanvasTkAgg(self.fig_calc_progress, viz_frame)
        self.canvas_calc_progress.draw()
        self.canvas_calc_progress.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize PSD(q) plot - top
        self.ax_psd_q.set_xlabel('파수 q (1/m)', fontsize=9, fontweight='bold')
        self.ax_psd_q.set_ylabel('PSD C(q) (m⁴)', fontsize=9, fontweight='bold')
        self.ax_psd_q.set_xscale('log')
        self.ax_psd_q.set_yscale('log')
        self.ax_psd_q.grid(True, alpha=0.3)
        self.ax_psd_q.set_title('PSD (파수 기준)', fontsize=10, fontweight='bold')

        # Initialize DMA plot - middle
        self.ax_dma_progress.set_xlabel('주파수 f (Hz)', fontsize=9, fontweight='bold')
        self.ax_dma_progress.set_ylabel('탄성률 (Pa)', fontsize=9, fontweight='bold')
        self.ax_dma_progress.set_xscale('log')
        self.ax_dma_progress.set_yscale('log')
        self.ax_dma_progress.grid(True, alpha=0.3)
        self.ax_dma_progress.set_title('DMA 마스터 곡선', fontsize=10, fontweight='bold')

        # Initialize PSD(f) plot - bottom
        self.ax_psd_f.set_xlabel('주파수 f (Hz)', fontsize=9, fontweight='bold')
        self.ax_psd_f.set_ylabel('PSD C(f) (변환)', fontsize=9, fontweight='bold')
        self.ax_psd_f.set_xscale('log')
        self.ax_psd_f.set_yscale('log')
        self.ax_psd_f.grid(True, alpha=0.3)
        self.ax_psd_f.set_title('PSD (주파수 기준)', fontsize=10, fontweight='bold')

        self.fig_calc_progress.tight_layout()

        # Save button for calculation progress plot in left panel
        save_btn_frame = ttk.Frame(left_panel)
        save_btn_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(
            save_btn_frame,
            text="계산 과정 그래프 저장",
            command=lambda: self._save_plot(self.fig_calc_progress, "calculation_progress")
        ).pack(fill=tk.X)

    def _on_target_hrms_changed(self, *args):
        """Callback when target h'rms value is changed in Tab 2.
        Syncs the value to Tab 1's psd_xi_var and Tab 4's display."""
        try:
            new_value = self.target_hrms_slope_var.get()
            # Validate it's a number
            float(new_value)

            # Sync to Tab 1's psd_xi_var
            if hasattr(self, 'psd_xi_var'):
                self.psd_xi_var.set(new_value)

            # Update Tab 4's display if it exists
            if hasattr(self, 'rms_target_xi_display'):
                self.rms_target_xi_display.set(f"{float(new_value):.4f}")

            # Update target_xi for calculations
            self.target_xi = float(new_value)
        except (ValueError, AttributeError):
            # Invalid value or variables not yet initialized
            pass

    def _on_hrms_q1_mode_changed(self):
        """모드 변경 시 UI 상태 업데이트."""
        mode = self.hrms_q1_mode_var.get()
        if mode == "hrms_to_q1":
            # 모드 1: h'rms 입력 활성화, q1 입력 비활성화
            self.hrms_entry.config(state='normal')
            self.q1_entry.config(state='disabled')
            self.hrms_q1_calc_btn.config(text="h'rms → q1 계산")
        else:
            # 모드 2: q1 입력 활성화, h'rms 입력 비활성화
            self.hrms_entry.config(state='disabled')
            self.q1_entry.config(state='normal')
            self.hrms_q1_calc_btn.config(text="q1 → h'rms 계산")

    def _calculate_hrms_q1(self):
        """선택된 모드에 따라 h'rms(ξ) 또는 q1 계산.

        h'rms = ξ = RMS slope (경사), 무차원
        ξ²(q) = 2π ∫[q₀→q] k³ C(k) dk
        """
        if self.psd_model is None:
            messagebox.showwarning("경고", "PSD 데이터를 먼저 로드해주세요!")
            return

        try:
            mode = self.hrms_q1_mode_var.get()

            # PSD 데이터에서 q 범위 결정
            if hasattr(self.psd_model, 'q_data'):
                q_data = self.psd_model.q_data
                C_data = self.psd_model.C_data
            else:
                q_min = float(self.q_min_var.get())
                q_max = float(self.q_max_var.get())
                q_data = np.logspace(np.log10(q_min), np.log10(q_max), 500)
                C_data = self.psd_model(q_data)

            # 누적 h'rms(ξ) 계산: ξ²(q) = 2π∫[q0 to q] k³C(k)dk
            xi_squared_cumulative = np.zeros_like(q_data)
            for i in range(len(q_data)):
                q_int = q_data[:i+1]
                C_int = C_data[:i+1]
                xi_squared_cumulative[i] = 2 * np.pi * np.trapezoid(q_int**3 * C_int, q_int)
            xi_cumulative = np.sqrt(xi_squared_cumulative)  # ξ = h'rms

            if mode == "hrms_to_q1":
                # 모드 1: 주어진 h'rms(ξ)로 q1 계산
                target_xi = float(self.target_hrms_slope_var.get())

                # ξ 값이 도달 가능한지 확인
                if target_xi > xi_cumulative[-1]:
                    messagebox.showwarning("경고",
                        f"목표 ξ ({target_xi:.4f})가 최대 도달 가능한 값 ({xi_cumulative[-1]:.4f})보다 큽니다.\n"
                        f"q 범위를 늘리거나 목표 ξ를 줄이세요.")
                    return

                # 목표 ξ에 해당하는 q1 찾기 (보간 사용)
                from scipy.interpolate import interp1d
                f_interp = interp1d(xi_cumulative, q_data, kind='linear', fill_value='extrapolate')
                q1_calculated = float(f_interp(target_xi))

                # 결과 표시
                self.calculated_q1_var.set(f"{q1_calculated:.3e}")
                self.calculated_q1 = q1_calculated
                self.target_xi = target_xi

                self.status_var.set(f"계산 완료: ξ={target_xi:.4f} → q1={q1_calculated:.3e} (1/m)")
                messagebox.showinfo("계산 완료",
                    f"모드 1: h'rms (ξ) → q1 계산\n\n"
                    f"입력 ξ (h'rms): {target_xi:.4f}\n"
                    f"계산된 q1: {q1_calculated:.3e} (1/m)\n\n"
                    f"※ ξ² = 2π∫k³C(k)dk")

            else:
                # 모드 2: 주어진 q1로 h'rms(ξ) 계산
                target_q1 = float(self.input_q1_var.get())

                # q1이 범위 내에 있는지 확인
                if target_q1 < q_data[0] or target_q1 > q_data[-1]:
                    messagebox.showwarning("경고",
                        f"입력 q1 ({target_q1:.3e})이 PSD 데이터 범위 밖입니다.\n"
                        f"범위: {q_data[0]:.3e} ~ {q_data[-1]:.3e} (1/m)")
                    return

                # q1에 해당하는 ξ 찾기 (보간 사용)
                from scipy.interpolate import interp1d
                f_interp = interp1d(q_data, xi_cumulative, kind='linear', fill_value='extrapolate')
                xi_calculated = float(f_interp(target_q1))

                # 결과 표시
                self.calculated_hrms_var.set(f"{xi_calculated:.4f}")
                self.calculated_q1 = target_q1
                self.target_xi = xi_calculated

                # ξ 입력란에도 반영
                self.target_hrms_slope_var.set(f"{xi_calculated:.4f}")

                self.status_var.set(f"계산 완료: q1={target_q1:.3e} → ξ={xi_calculated:.4f}")
                messagebox.showinfo("계산 완료",
                    f"모드 2: q1 → h'rms (ξ) 계산\n\n"
                    f"입력 q1: {target_q1:.3e} (1/m)\n"
                    f"계산된 ξ (h'rms): {xi_calculated:.4f}\n\n"
                    f"※ ξ² = 2π∫k³C(k)dk")

        except ValueError as e:
            messagebox.showerror("오류", f"입력값이 유효하지 않습니다: {e}")
        except Exception as e:
            messagebox.showerror("오류", f"계산 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    def _send_hrms_q1_to_tab4(self):
        """계산된 h'rms(ξ)와 q1을 Tab 4로 전달."""
        try:
            # 계산된 q1이 있으면 Tab 4의 q_max에 전달
            if hasattr(self, 'calculated_q1') and self.calculated_q1 is not None:
                self.rms_q_max_var.set(f"{self.calculated_q1:.3e}")

            # target_xi를 Tab 4에 전달
            if self.target_xi is not None:
                if hasattr(self, 'rms_target_xi_display'):
                    self.rms_target_xi_display.set(f"{self.target_xi:.4f}")
                if hasattr(self, 'psd_xi_var'):
                    self.psd_xi_var.set(f"{self.target_xi:.4f}")

            # Tab 4로 전환
            self.notebook.select(4)

            self.status_var.set(f"Tab 4로 전달 완료: ξ={self.target_xi:.4f}, q1={self.calculated_q1:.3e}")
            messagebox.showinfo("전달 완료",
                f"Tab 4로 전달되었습니다.\n\n"
                f"ξ (h'rms): {self.target_xi:.4f}\n"
                f"q1: {self.calculated_q1:.3e} (1/m)\n\n"
                f"Tab 4에서 'h'rms slope 계산' 버튼을 클릭하세요.")

        except Exception as e:
            messagebox.showerror("오류", f"Tab 4로 전달 중 오류: {e}")

    def _create_results_tab(self, parent):
        """Create G(q,v) results tab."""
        # Instruction
        instruction = ttk.LabelFrame(parent, text="탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "G(q,v) 2D 행렬 계산 결과: 다중 속도 G(q) 곡선, 히트맵, 접촉 면적.\n"
            "모든 속도가 컬러 코딩되어 하나의 그래프에 표시됩니다.",
            font=('Arial', 10)
        ).pack()

        # Plot area
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig_results = Figure(figsize=(16, 10), dpi=100)
        self.canvas_results = FigureCanvasTkAgg(self.fig_results, plot_frame)
        self.canvas_results.draw()
        self.canvas_results.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_results, plot_frame)
        toolbar.update()

        # Save button
        save_btn_frame = ttk.Frame(parent)
        save_btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            save_btn_frame,
            text="결과 그래프 저장",
            command=lambda: self._save_plot(self.fig_results, "results_plot")
        ).pack(side=tk.LEFT, padx=5)

    def _create_status_bar(self):
        """Create status bar."""
        self.status_var = tk.StringVar(value="준비")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _update_verification_plots(self):
        """Update input data verification plots."""
        if self.material is None:
            return

        # Clear previous plots
        self.ax_master_curve.clear()
        self.ax_psd.clear()

        # Plot 1: Master Curve (E', E'') - X-axis in Hz
        # Use material's actual frequency range for plotting (respects extrapolation settings)
        omega_min = self.material._frequencies.min()
        omega_max = self.material._frequencies.max()
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), 500)
        f_Hz = omega / (2 * np.pi)  # Convert omega (rad/s) to f (Hz)
        E_storage = self.material.get_storage_modulus(omega)
        E_loss = self.material.get_loss_modulus(omega)

        ax1 = self.ax_master_curve

        # Plot smoothed data (from interpolator)
        ax1.loglog(f_Hz, E_storage/1e6, 'g-', linewidth=2.5, label="E' (보간/평활화)", alpha=0.9, zorder=2)
        ax1.loglog(f_Hz, E_loss/1e6, 'orange', linewidth=2.5, label="E'' (보간/평활화)", alpha=0.9, zorder=2)

        # Plot raw measured data if available
        if self.raw_dma_data is not None:
            f_raw = self.raw_dma_data['omega'] / (2 * np.pi)  # Convert to Hz
            ax1.scatter(f_raw, self.raw_dma_data['E_storage']/1e6,
                       c='darkgreen', s=20, alpha=0.5, label="E' (측정값)", zorder=1)
            ax1.scatter(f_raw, self.raw_dma_data['E_loss']/1e6,
                       c='darkorange', s=20, alpha=0.5, label="E'' (측정값)", zorder=1)

        ax1.set_xlabel('주파수 f (Hz)', fontweight='bold', fontsize=11, labelpad=5)
        ax1.set_ylabel('탄성률 (MPa)', fontweight='bold', fontsize=11, rotation=90, labelpad=10)
        ax1.set_title('점탄성 마스터 곡선', fontweight='bold', fontsize=12, pad=10)
        ax1.legend(loc='upper left', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Fix axis formatter to use superscript notation for all log axes
        from matplotlib.ticker import FuncFormatter
        def log_tick_formatter(val, pos=None):
            if val <= 0:
                return ''
            exponent = int(np.floor(np.log10(val)))
            # For cleaner display, show integer if it's exactly a power of 10
            if abs(val - 10**exponent) < 1e-10:
                return f'$10^{{{exponent}}}$'
            else:
                # For intermediate values, still show in exponential form
                mantissa = val / (10**exponent)
                if abs(mantissa - 1.0) < 0.01:
                    return f'$10^{{{exponent}}}$'
                else:
                    return f'${mantissa:.1f} \\times 10^{{{exponent}}}$'
        ax1.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        ax1.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        # Plot 2: PSD C(q) - compare raw data with applied power law
        has_raw_psd = self.raw_psd_data is not None
        has_psd_model = self.psd_model is not None

        if has_raw_psd or has_psd_model:
            # Plot raw PSD data first (if available)
            if has_raw_psd:
                q_raw = self.raw_psd_data['q']
                C_raw = self.raw_psd_data['C_q']
                self.ax_psd.loglog(q_raw, C_raw, 'ko', markersize=3, alpha=0.5,
                                  label='Raw PSD (측정값)', zorder=1)

            # Plot applied PSD model (power law or loaded)
            if has_psd_model:
                # Check if this is a power-law model (has q_data attribute from _apply_psd_settings)
                is_power_law = hasattr(self.psd_model, 'q_data')

                if is_power_law:
                    # Determine plot range: include plateau region if raw PSD data exists
                    if has_raw_psd:
                        q_plot_min = min(self.raw_psd_data['q'])
                    elif hasattr(self.psd_model, 'q0'):
                        q_plot_min = self.psd_model.q0 / 10
                    else:
                        q_plot_min = float(self.q_min_var.get())

                    q_plot_max = float(self.q_max_var.get())
                    q_plot = np.logspace(np.log10(q_plot_min), np.log10(q_plot_max), 300)
                    C_q = self.psd_model(q_plot)

                    # Plot power law model with different color
                    self.ax_psd.loglog(q_plot, C_q, 'r-', linewidth=2.5,
                                      label='적용된 PSD (Power Law)', alpha=0.9, zorder=2)
                else:
                    q_min = float(self.q_min_var.get())
                    q_max = float(self.q_max_var.get())
                    q_plot = np.logspace(np.log10(q_min), np.log10(q_max), 200)
                    C_q = self.psd_model(q_plot)

                    # Plot loaded/interpolated PSD
                    self.ax_psd.loglog(q_plot, C_q, 'b-', linewidth=2,
                                      label='적용된 PSD (보간)', alpha=0.9, zorder=2)

                    # Calculate Hurst exponent from power law fitting for display
                    fit_idx = (q_plot > q_min * 10) & (q_plot < q_max / 10)
                    if np.sum(fit_idx) > 10:
                        log_q_fit = np.log10(q_plot[fit_idx])
                        log_C_fit = np.log10(C_q[fit_idx])
                        coeffs = np.polyfit(log_q_fit, log_C_fit, 1)
                        slope = coeffs[0]
                        intercept = coeffs[1]
                        H = -slope / 2.0 - 1.0

                        # Plot fitted line
                        C_fit = 10**(intercept + slope * np.log10(q_plot))
                        self.ax_psd.loglog(q_plot, C_fit, 'g--', linewidth=1.5, alpha=0.7,
                                          label=f'Power law fit (H={H:.3f})')

            self.ax_psd.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11, labelpad=5)
            self.ax_psd.set_ylabel('PSD C(q) (m⁴)', fontweight='bold', fontsize=11,
                                   rotation=90, labelpad=10)
            self.ax_psd.set_title('표면 거칠기 PSD 비교', fontweight='bold', fontsize=12, pad=10)
            self.ax_psd.legend(fontsize=8, loc='upper right')
            self.ax_psd.grid(True, alpha=0.3)

            # Fix axis formatter
            self.ax_psd.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
            self.ax_psd.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        self.fig_verification.tight_layout(pad=2.0)
        self.canvas_verification.draw()

    def _update_material_display(self):
        """Update material information (if needed)."""
        pass  # Simplified for now

    def _load_material(self):
        """Load DMA data from file."""
        filename = filedialog.askopenfilename(
            title="Select DMA Data File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                omega_raw, E_storage_raw, E_loss_raw = load_dma_from_file(
                    filename, skip_header=1, freq_unit='Hz', modulus_unit='MPa'
                )

                # Apply smoothing/fitting
                smoothed = smooth_dma_data(omega_raw, E_storage_raw, E_loss_raw)

                # Store raw data for visualization
                self.raw_dma_data = {
                    'omega': omega_raw,
                    'E_storage': E_storage_raw,
                    'E_loss': E_loss_raw
                }

                # Create material from smoothed data
                self.material = create_material_from_dma(
                    omega=smoothed['omega'],
                    E_storage=smoothed['E_storage_smooth'],
                    E_loss=smoothed['E_loss_smooth'],
                    material_name=os.path.splitext(os.path.basename(filename))[0] + " (smoothed)",
                    reference_temp=float(self.temperature_var.get())
                )

                self._update_verification_plots()
                messagebox.showinfo("Success", f"DMA data loaded and smoothed: {len(omega_raw)} points")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load DMA data:\n{str(e)}")

    def _load_psd_data(self):
        """Load PSD data from file."""
        filename = filedialog.askopenfilename(
            title="Select PSD Data File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Auto-detect format (is_log_data=None)
                q, C_q = load_psd_from_file(filename, skip_header=1, is_log_data=None)

                # Validate the loaded data
                if len(q) == 0:
                    raise ValueError("No valid data points found in file")

                if np.any(q <= 0) or np.any(C_q <= 0):
                    raise ValueError("Invalid data: q and C must be positive after conversion")

                # Store raw PSD data for comparison plotting
                self.raw_psd_data = {
                    'q': q.copy(),
                    'C_q': C_q.copy()
                }

                self.psd_model = create_psd_from_data(q, C_q, interpolation_kind='log-log')

                self.q_min_var.set(f"{q[0]:.2e}")
                self.q_max_var.set(f"{q[-1]:.2e}")
                self.psd_type_var.set("measured")

                self._update_verification_plots()

                # Show info about loaded data
                messagebox.showinfo(
                    "Success",
                    f"PSD data loaded: {len(q)} points\n"
                    f"q 범위: {q[0]:.2e} ~ {q[-1]:.2e} 1/m\n"
                    f"C(q) 범위: {C_q.min():.2e} ~ {C_q.max():.2e} m⁴"
                )

            except Exception as e:
                import traceback
                messagebox.showerror("Error", f"Failed to load PSD data:\n{str(e)}\n\n{traceback.format_exc()}")

    def _import_from_master_curve(self):
        """Import DMA data from master curve tab (Tab 0)."""
        if self.master_curve_gen is None or self.master_curve_gen.master_f is None:
            messagebox.showwarning("경고", "먼저 Tab 0에서 마스터 커브를 생성하세요.")
            return

        try:
            # Get master curve data
            omega = 2 * np.pi * self.master_curve_gen.master_f
            E_storage = self.master_curve_gen.master_E_storage * 1e6  # MPa to Pa
            E_loss = self.master_curve_gen.master_E_loss * 1e6  # MPa to Pa
            T_ref = self.master_curve_gen.T_ref

            # Create material from master curve
            self.material = create_material_from_dma(
                omega=omega,
                E_storage=E_storage,
                E_loss=E_loss,
                material_name=f"Master Curve (Tref={T_ref}°C)",
                reference_temp=T_ref
            )

            # Store raw data for plotting
            self.raw_dma_data = {
                'omega': omega,
                'E_storage': E_storage,
                'E_loss': E_loss
            }

            # Store shift factor data
            self.master_curve_shift_factors = {
                'aT': self.master_curve_gen.aT.copy(),
                'bT': self.master_curve_gen.bT.copy(),
                'T_ref': T_ref,
                'C1': self.master_curve_gen.C1,
                'C2': self.master_curve_gen.C2,
                'temperatures': list(self.master_curve_gen.temperatures)
            }

            # Update temperature entry
            self.temperature_var.set(str(T_ref))

            # Update status
            f_min = self.master_curve_gen.master_f.min()
            f_max = self.master_curve_gen.master_f.max()
            self.dma_import_status_var.set(f"Master Curve (Tref={T_ref}°C, {f_min:.1e}~{f_max:.1e} Hz)")

            # Update plots
            self._update_material_display()
            self._update_verification_plots()

            self.status_var.set(f"마스터 커브 가져오기 완료")

        except Exception as e:
            messagebox.showerror("Error", f"스무딩 적용 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _apply_dma_smoothing_extrapolation(self):
        """Apply smoothing and/or extrapolation to DMA data in verification tab."""
        if self.raw_dma_data is None:
            messagebox.showwarning("경고", "먼저 DMA 데이터를 불러오세요.")
            return

        try:
            from scipy.signal import savgol_filter
            from scipy.interpolate import interp1d

            # Get raw data
            omega_raw = self.raw_dma_data['omega'].copy()
            E_storage_raw = self.raw_dma_data['E_storage'].copy()
            E_loss_raw = self.raw_dma_data['E_loss'].copy()

            # Sort by omega
            sort_idx = np.argsort(omega_raw)
            omega_sorted = omega_raw[sort_idx]
            E_storage_sorted = E_storage_raw[sort_idx]
            E_loss_sorted = E_loss_raw[sort_idx]

            # Apply smoothing if enabled
            if self.verify_smooth_var.get():
                window = self.verify_smooth_window_var.get()
                if window % 2 == 0:
                    window += 1  # Must be odd
                window = min(window, len(omega_sorted) - 1)
                if window < 5:
                    window = 5
                if len(omega_sorted) > window:
                    # Use log scale for smoothing
                    log_E_storage = np.log10(np.maximum(E_storage_sorted, 1e-10))
                    log_E_loss = np.log10(np.maximum(E_loss_sorted, 1e-10))
                    log_E_storage_smooth = savgol_filter(log_E_storage, window, 3)
                    log_E_loss_smooth = savgol_filter(log_E_loss, window, 3)
                    E_storage_smooth = 10**log_E_storage_smooth
                    E_loss_smooth = 10**log_E_loss_smooth
                else:
                    E_storage_smooth = E_storage_sorted
                    E_loss_smooth = E_loss_sorted
            else:
                E_storage_smooth = E_storage_sorted
                E_loss_smooth = E_loss_sorted

            # Apply extrapolation if enabled
            if self.verify_extrap_var.get():
                # Get user-specified frequency range
                f_min = float(self.dma_extrap_fmin_var.get())
                f_max = float(self.dma_extrap_fmax_var.get())
                omega_min_target = 2 * np.pi * f_min
                omega_max_target = 2 * np.pi * f_max

                # Create extended omega array
                omega_extended = np.logspace(
                    np.log10(omega_min_target),
                    np.log10(omega_max_target),
                    500
                )

                # Create interpolators (log-log space)
                log_omega = np.log10(omega_sorted)
                log_E_storage = np.log10(np.maximum(E_storage_smooth, 1e-10))
                log_E_loss = np.log10(np.maximum(E_loss_smooth, 1e-10))

                interp_storage = interp1d(log_omega, log_E_storage, kind='linear',
                                         fill_value='extrapolate')
                interp_loss = interp1d(log_omega, log_E_loss, kind='linear',
                                      fill_value='extrapolate')

                # Extrapolate
                log_omega_ext = np.log10(omega_extended)
                log_E_storage_ext = interp_storage(log_omega_ext)
                log_E_loss_ext = interp_loss(log_omega_ext)

                # Linear extrapolation in log-log space using edge slopes
                # For low frequencies: extrapolate using slope from first few points
                low_mask = log_omega_ext < log_omega.min()
                if np.any(low_mask):
                    # Use first 10 points (or less if not enough data) to estimate slope
                    n_fit = min(10, len(log_omega) // 4, len(log_omega) - 1)
                    if n_fit >= 2:
                        slope_storage_low = (log_E_storage[n_fit] - log_E_storage[0]) / (log_omega[n_fit] - log_omega[0])
                        slope_loss_low = (log_E_loss[n_fit] - log_E_loss[0]) / (log_omega[n_fit] - log_omega[0])
                        delta_omega = log_omega_ext[low_mask] - log_omega[0]
                        log_E_storage_ext[low_mask] = log_E_storage[0] + slope_storage_low * delta_omega
                        log_E_loss_ext[low_mask] = log_E_loss[0] + slope_loss_low * delta_omega

                # For high frequencies: extrapolate using slope from last few points
                high_mask = log_omega_ext > log_omega.max()
                if np.any(high_mask):
                    # Use last 10 points (or less if not enough data) to estimate slope
                    n_fit = min(10, len(log_omega) // 4, len(log_omega) - 1)
                    if n_fit >= 2:
                        slope_storage_high = (log_E_storage[-1] - log_E_storage[-n_fit-1]) / (log_omega[-1] - log_omega[-n_fit-1])
                        slope_loss_high = (log_E_loss[-1] - log_E_loss[-n_fit-1]) / (log_omega[-1] - log_omega[-n_fit-1])
                        delta_omega = log_omega_ext[high_mask] - log_omega[-1]
                        log_E_storage_ext[high_mask] = log_E_storage[-1] + slope_storage_high * delta_omega
                        log_E_loss_ext[high_mask] = log_E_loss[-1] + slope_loss_high * delta_omega

                omega_final = omega_extended
                E_storage_final = 10**log_E_storage_ext
                E_loss_final = 10**log_E_loss_ext
            else:
                omega_final = omega_sorted
                E_storage_final = E_storage_smooth
                E_loss_final = E_loss_smooth

            # Update material
            self.material = create_material_from_dma(
                omega=omega_final,
                E_storage=E_storage_final,
                E_loss=E_loss_final,
                material_name=self.material.name if self.material else "Processed DMA",
                reference_temp=float(self.temperature_var.get())
            )

            # Update status
            f_min = omega_final.min() / (2 * np.pi)
            f_max = omega_final.max() / (2 * np.pi)
            smooth_str = f"스무딩(w={self.verify_smooth_window_var.get()})" if self.verify_smooth_var.get() else ""
            extrap_str = "외삽" if self.verify_extrap_var.get() else ""
            process_str = "+".join(filter(None, [smooth_str, extrap_str])) or "원본"
            self.dma_import_status_var.set(f"처리됨 [{process_str}] ({f_min:.1e}~{f_max:.1e} Hz)")

            # Update plots
            self._update_verification_plots()
            self.status_var.set("DMA 스무딩/외삽 적용 완료")

            messagebox.showinfo("완료", f"DMA 데이터 처리 완료\n- 스무딩: {'적용' if self.verify_smooth_var.get() else '미적용'}\n- 외삽: {'적용' if self.verify_extrap_var.get() else '미적용'}\n- 주파수 범위: {f_min:.1e} ~ {f_max:.1e} Hz")

        except Exception as e:
            messagebox.showerror("Error", f"스무딩/외삽 적용 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _calc_Cq0_from_xi(self):
        """Calculate C(q0) from target h'rms (ξ).

        Formula derivation:
        ξ² = 2π ∫[q0→q1] q³ C(q) dq
        For power-law PSD: C(q) = C(q0) * (q/q0)^(-2(H+1))

        ξ² = 2π * C(q0) * q0^(2(H+1)) * [q^(2-2H) / (2-2H)]_{q0}^{q1}
           = 2π * C(q0) * q0^(2(H+1)) * (q1^(2-2H) - q0^(2-2H)) / (2-2H)

        Therefore:
        C(q0) = ξ² * (2-2H) / (2π * q0^(2(H+1)) * (q1^(2-2H) - q0^(2-2H)))
        """
        try:
            q0 = float(self.psd_q0_var.get())
            q1 = float(self.psd_q1_var.get())
            H = float(self.psd_H_var.get())
            xi_target = float(self.psd_xi_var.get())

            if q1 <= q0:
                messagebox.showerror("Error", "q1 must be greater than q0")
                return

            # Calculate C(q0) from target ξ
            exp_factor = 2 - 2 * H
            if abs(exp_factor) < 1e-10:
                # Special case H ≈ 1
                integral_factor = np.log(q1 / q0)
            else:
                integral_factor = (q1**exp_factor - q0**exp_factor) / exp_factor

            # ξ² = 2π * C(q0) * q0^(2(H+1)) * integral_factor
            # C(q0) = ξ² / (2π * q0^(2(H+1)) * integral_factor)
            C_q0 = xi_target**2 / (2 * np.pi * q0**(2*(H+1)) * integral_factor)

            # Update C(q0) entry
            self.psd_Cq0_var.set(f"{C_q0:.3e}")
            self.status_var.set(f"C(q0) = {C_q0:.3e} calculated for ξ = {xi_target}")

        except Exception as e:
            messagebox.showerror("Error", f"C(q0) 계산 실패:\n{str(e)}")

    def _apply_psd_settings(self):
        """Apply PSD power-law settings from user input."""
        try:
            from scipy.interpolate import interp1d

            # Get user parameters
            q0 = float(self.psd_q0_var.get())
            q1 = float(self.psd_q1_var.get())
            H = float(self.psd_H_var.get())
            C_q0 = float(self.psd_Cq0_var.get())

            if q1 <= q0:
                messagebox.showerror("Error", "q1 must be greater than q0")
                return

            if H < 0 or H > 1:
                messagebox.showwarning("Warning", "Hurst exponent H should be between 0 and 1")

            # Create power-law PSD: C(q) = C(q0) * (q/q0)^(-2(H+1))
            # Power law exponent: -2(H+1)
            exponent = -2 * (H + 1)

            # Create q array for power law region (q0 to q1)
            q_powerlaw = np.logspace(np.log10(q0), np.log10(q1), 500)
            C_powerlaw = C_q0 * (q_powerlaw / q0) ** exponent

            # Determine minimum q for plateau region
            # Use raw PSD data range if available, otherwise use q0/100
            if self.raw_psd_data is not None:
                q_min_plateau = min(self.raw_psd_data['q'])
            else:
                q_min_plateau = q0 / 100

            # Create plateau region (q < q0) with constant C(q0)
            if q_min_plateau < q0:
                q_plateau = np.logspace(np.log10(q_min_plateau), np.log10(q0), 100)[:-1]  # exclude q0 to avoid duplicate
                C_plateau = np.full_like(q_plateau, C_q0)

                # Combine plateau and power law regions
                q_array = np.concatenate([q_plateau, q_powerlaw])
                C_array = np.concatenate([C_plateau, C_powerlaw])
            else:
                q_array = q_powerlaw
                C_array = C_powerlaw

            # Create interpolator for PSD model
            log_q = np.log10(q_array)
            log_C = np.log10(C_array)

            # Store q0 and C_q0 for the model function
            _q0 = q0
            _C_q0 = C_q0
            _exponent = exponent

            def psd_model(q_input):
                """Power-law PSD model with plateau for q < q0."""
                q_input = np.atleast_1d(q_input)

                C_out = np.empty_like(q_input)

                # q < q0: plateau at C(q0)
                mask_plateau = q_input < _q0
                C_out[mask_plateau] = _C_q0

                # q >= q0: power law
                mask_powerlaw = ~mask_plateau
                C_out[mask_powerlaw] = _C_q0 * (q_input[mask_powerlaw] / _q0) ** _exponent

                return C_out

            # Store the PSD model with q_data and C_data attributes for RMS slope calculation
            self.psd_model = psd_model
            # Add attributes to function object so RMS slope calculation uses correct q range
            # IMPORTANT: q_data/C_data now include plateau region (q < q0) for complete integration
            self.psd_model.q_data = q_array.copy()  # Full range including plateau
            self.psd_model.C_data = C_array.copy()  # Full range including plateau
            # Also store full range for plotting (same as q_data/C_data now)
            self.psd_model.q_full = q_array.copy()
            self.psd_model.C_full = C_array.copy()
            # Store power law region separately for reference
            self.psd_model.q_powerlaw = q_powerlaw.copy()
            self.psd_model.C_powerlaw = C_powerlaw.copy()
            self.psd_model.q0 = q0
            self.psd_model.C_q0 = C_q0

            # Store q range for calculations
            self.q_min_var.set(str(q0))
            self.q_max_var.set(str(q1))

            # Calculate actual RMS slope from the PSD for verification
            # ξ² = 2π ∫ q³ C(q) dq
            q_calc = np.logspace(np.log10(q0), np.log10(q1), 1000)
            C_calc = psd_model(q_calc)
            integrand = q_calc**3 * C_calc
            xi_squared = 2 * np.pi * np.trapezoid(integrand, q_calc)
            xi_actual = np.sqrt(xi_squared)

            # Use user's input ξ value from Tab 2 directly as target_xi
            # (The user specified ξ, calculated C(q0) from it, so target ξ is the input value)
            try:
                xi_user_input = float(self.psd_xi_var.get())
            except:
                xi_user_input = xi_actual

            # Store user's target xi for consistency with Tab 4 (RMS Slope)
            self.target_xi = xi_user_input
            self.psd_model.target_xi = xi_user_input

            # Update plots
            self._update_verification_plots()
            self.status_var.set(f"PSD applied: ξ(target)={xi_user_input:.3f}, ξ(calc)={xi_actual:.3f}, H={H:.2f}")

            # Show both target and calculated ξ for transparency
            xi_diff_pct = abs(xi_user_input - xi_actual) / xi_user_input * 100 if xi_user_input > 0 else 0
            xi_info = f"- Target h'rms ξ = {xi_user_input:.4f}\n- Calculated h'rms ξ = {xi_actual:.4f}"
            if xi_diff_pct > 1:
                xi_info += f"\n  (차이: {xi_diff_pct:.1f}% - 수치 적분 오차)"

            messagebox.showinfo("Complete", f"PSD model applied:\n"
                              f"- q range: {q0:.1e} ~ {q1:.1e} 1/m\n"
                              f"- Hurst exponent H: {H:.3f}\n"
                              f"- C(q0): {C_q0:.1e} m^4\n"
                              f"- Power law: C(q) = C(q0)*(q/q0)^{exponent:.2f}\n"
                              f"{xi_info}")

        except Exception as e:
            messagebox.showerror("Error", f"PSD settings failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _save_plot(self, fig, default_name):
        """Save matplotlib figure to file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"{default_name}.png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"그래프가 저장되었습니다:\n{filename}")
                self.status_var.set(f"그래프 저장 완료: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"그래프 저장 실패:\n{str(e)}")

    def _run_calculation(self):
        """Run G(q,v) 2D calculation."""
        if self.material is None or self.psd_model is None:
            messagebox.showwarning("Warning", "Please load material and PSD data first!")
            return

        try:
            self.status_var.set("Calculating G(q,v)...")
            self.calc_button.config(state='disabled')
            self.root.update()

            # Get parameters
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            v_min = float(self.v_min_var.get())
            v_max = float(self.v_max_var.get())
            n_v = int(self.n_velocity_var.get())
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())
            n_q = int(self.n_q_var.get())
            poisson = float(self.poisson_var.get())
            temperature = float(self.temperature_var.get())

            # Create arrays
            v_array = np.logspace(np.log10(v_min), np.log10(v_max), n_v)
            q_array = np.logspace(np.log10(q_min), np.log10(q_max), n_q)

            # Create G calculator
            self.g_calculator = GCalculator(
                psd_func=self.psd_model,
                modulus_func=lambda w: self.material.get_modulus(w, temperature=temperature),
                sigma_0=sigma_0,
                velocity=v_array[0],  # Initial velocity
                poisson_ratio=poisson,
                n_angle_points=36,
                integration_method='trapz'
            )

            # Initialize calculation progress plots (3 subplots)
            try:
                # Clear all three subplots
                self.ax_psd_q.clear()
                self.ax_dma_progress.clear()
                self.ax_psd_f.clear()

                # TOP SUBPLOT: Plot PSD(q) - wavenumber based (static)
                # Include plateau region (q < q0) if available
                if self.psd_model is not None:
                    # Determine plot range including plateau region
                    if hasattr(self.psd_model, 'q_data') and len(self.psd_model.q_data) > 0:
                        # Use full q_data range which includes plateau
                        q_plot_min = min(self.psd_model.q_data)
                        q_plot_max = max(self.psd_model.q_data[self.psd_model.q_data <= q_max]) if np.any(self.psd_model.q_data <= q_max) else q_max
                        q_plot_max = max(q_plot_max, q_max)
                    elif hasattr(self.psd_model, 'q0'):
                        # If q0 is defined, extend plot range to include plateau
                        q_plot_min = self.psd_model.q0 / 10  # Show some plateau region
                        q_plot_max = q_max
                    else:
                        q_plot_min = q_min
                        q_plot_max = q_max

                    q_plot = np.logspace(np.log10(q_plot_min), np.log10(q_plot_max), 300)
                    C_q = self.psd_model(q_plot)

                    # Plot full PSD including plateau
                    self.ax_psd_q.loglog(q_plot, C_q, 'b-', linewidth=2, label='PSD C(q)')

                    # Highlight plateau region (q < q0) if q0 is defined
                    if hasattr(self.psd_model, 'q0'):
                        q0_psd = self.psd_model.q0
                        if q_plot_min < q0_psd:
                            self.ax_psd_q.axvspan(q_plot_min, q0_psd, alpha=0.2, facecolor='yellow',
                                                 edgecolor='orange', linewidth=1, label=f'플래토 (q < q0={q0_psd:.1e})')
                            # Mark q0 with vertical line
                            self.ax_psd_q.axvline(x=q0_psd, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

                    # Highlight the q range being used for calculation
                    self.ax_psd_q.axvspan(q_min, q_max, alpha=0.15, facecolor='cyan',
                                         edgecolor='blue', linewidth=1.5, label=f'계산 q 범위')

                    self.ax_psd_q.set_xlabel('파수 q (1/m)', fontsize=10, fontweight='bold')
                    self.ax_psd_q.set_ylabel('PSD C(q) (m⁴)', fontsize=10, fontweight='bold')
                    self.ax_psd_q.set_xscale('log')
                    self.ax_psd_q.set_yscale('log')
                    self.ax_psd_q.grid(True, alpha=0.3)
                    self.ax_psd_q.legend(loc='upper right', fontsize=7)
                    self.ax_psd_q.set_title('PSD (파수 기준) - 플래토 영역 포함', fontsize=11, fontweight='bold')

                # MIDDLE SUBPLOT: Plot DMA master curve
                if self.material is not None:
                    omega_plot = np.logspace(-2, 8, 200)
                    f_plot = omega_plot / (2 * np.pi)  # Convert to Hz
                    E_prime = self.material.get_storage_modulus(omega_plot)
                    E_double_prime = self.material.get_loss_modulus(omega_plot)

                    self.ax_dma_progress.plot(f_plot, E_prime, 'b-', linewidth=2, label="E' (저장 탄성률)")
                    self.ax_dma_progress.plot(f_plot, E_double_prime, 'r--', linewidth=2, label="E'' (손실 탄성률)")

                    self.ax_dma_progress.set_xlabel('주파수 f (Hz)', fontsize=10, fontweight='bold')
                    self.ax_dma_progress.set_ylabel('탄성률 (Pa)', fontsize=10, fontweight='bold')
                    self.ax_dma_progress.set_xscale('log')
                    self.ax_dma_progress.set_yscale('log')
                    self.ax_dma_progress.grid(True, alpha=0.3)
                    self.ax_dma_progress.legend(loc='best', fontsize=8)
                    self.ax_dma_progress.set_title('DMA 마스터 곡선', fontsize=11, fontweight='bold')

                # BOTTOM SUBPLOT: Initialize PSD(f) plot - will be updated during animation
                if self.psd_model is not None:
                    # Plot a placeholder that will be updated during calculation
                    # Use initial velocity to show what PSD(f) looks like
                    v_init = v_min
                    f_plot_range = np.logspace(-3, 6, 200)  # Wide frequency range
                    # Convert f to q: q = 2*pi*f/v
                    q_from_f = 2 * np.pi * f_plot_range / v_init
                    # Evaluate PSD at these q values (only for valid range)
                    C_f_init = np.zeros_like(f_plot_range)
                    valid_mask = (q_from_f >= q_min) & (q_from_f <= q_max)
                    C_f_init[valid_mask] = self.psd_model(q_from_f[valid_mask])
                    C_f_init[~valid_mask] = np.nan

                    self.ax_psd_f.loglog(f_plot_range, C_f_init, 'g-', linewidth=2,
                                        label=f'PSD(f) @ v={v_init:.4f} m/s', alpha=0.3)

                    self.ax_psd_f.set_xlabel('주파수 f (Hz)', fontsize=10, fontweight='bold')
                    self.ax_psd_f.set_ylabel('PSD C(f)', fontsize=10, fontweight='bold')
                    self.ax_psd_f.set_xscale('log')
                    self.ax_psd_f.set_yscale('log')
                    self.ax_psd_f.grid(True, alpha=0.3)
                    self.ax_psd_f.legend(loc='best', fontsize=8)
                    self.ax_psd_f.set_title('PSD (주파수 기준)', fontsize=11, fontweight='bold')

                self.fig_calc_progress.tight_layout()
                self.canvas_calc_progress.draw()
            except Exception as e:
                print(f"Error initializing plots: {e}")
                import traceback
                traceback.print_exc()

            # Calculate G(q,v) 2D matrix with real-time visualization
            def progress_callback(percent):
                self.progress_var.set(percent)
                # Update status to show which velocity is being calculated
                v_idx = int(percent / 100 * len(v_array))
                if v_idx < len(v_array):
                    current_v = v_array[v_idx]
                    # Calculate frequency and wavenumber ranges
                    omega_min = q_array[0] * current_v
                    omega_max = q_array[-1] * current_v
                    f_min = omega_min / (2 * np.pi)  # Convert to Hz
                    f_max = omega_max / (2 * np.pi)  # Convert to Hz
                    q_min_used = q_array[0]
                    q_max_used = q_array[-1]

                    # Update main status bar
                    self.status_var.set(f"계산 중... v={current_v:.4f} m/s (f: {f_min:.2e} ~ {f_max:.2e} Hz)")

                    # Update calculation status label
                    self.calc_status_label.config(
                        text=f"계산 중 ({percent:.0f}%) | v = {current_v:.4f} m/s | "
                             f"q 범위 = {q_min_used:.2e} ~ {q_max_used:.2e} (1/m) | "
                             f"f 범위 = {f_min:.2e} ~ {f_max:.2e} (Hz)",
                        foreground='red'
                    )

                    # Highlight frequency range on DMA plot (middle subplot)
                    try:
                        # Remove ALL previous highlight bands from DMA plot
                        for artist in self.ax_dma_progress.collections[:]:
                            if hasattr(artist, '_is_highlight'):
                                artist.remove()

                        # Add vertical band to show current frequency range on DMA
                        band_dma = self.ax_dma_progress.axvspan(f_min, f_max,
                                                                alpha=0.3, facecolor='yellow',
                                                                edgecolor='orange', linewidth=2,
                                                                zorder=0)  # Behind curves
                        band_dma._is_highlight = True

                        self.canvas_calc_progress.draw()
                    except Exception as e:
                        print(f"Error updating DMA highlight: {e}")

                    # Update PSD(f) plot with current velocity (bottom subplot)
                    try:
                        # Remove previous PSD(f) lines and highlights
                        for line in self.ax_psd_f.lines[:]:
                            if hasattr(line, '_is_psd_f_line'):
                                line.remove()
                        for artist in self.ax_psd_f.collections[:]:
                            if hasattr(artist, '_is_highlight'):
                                artist.remove()

                        # Plot PSD(f) for current velocity
                        f_plot_range = np.logspace(np.log10(f_min) - 2, np.log10(f_max) + 2, 300)
                        # Convert f to q: q = 2*pi*f/v
                        q_from_f = 2 * np.pi * f_plot_range / current_v
                        # Evaluate PSD at these q values
                        C_f = np.zeros_like(f_plot_range)
                        valid_mask = (q_from_f >= q_array[0]) & (q_from_f <= q_array[-1])
                        if np.any(valid_mask):
                            C_f[valid_mask] = self.psd_model(q_from_f[valid_mask])
                            C_f[~valid_mask] = np.nan

                            # Plot PSD(f) curve
                            line = self.ax_psd_f.loglog(f_plot_range, C_f, 'g-', linewidth=2,
                                                       label=f'v={current_v:.4f} m/s', alpha=0.7)[0]
                            line._is_psd_f_line = True

                            # Highlight the frequency range being used
                            band_psd = self.ax_psd_f.axvspan(f_min, f_max,
                                                            alpha=0.3, facecolor='yellow',
                                                            edgecolor='orange', linewidth=2,
                                                            zorder=0)
                            band_psd._is_highlight = True

                            self.ax_psd_f.legend(loc='best', fontsize=8)

                        self.canvas_calc_progress.draw()
                    except Exception as e:
                        print(f"Error updating PSD(f): {e}")
                        import traceback
                        traceback.print_exc()

                self.root.update()

            results_2d = self.g_calculator.calculate_G_multi_velocity(
                q_array, v_array, q_min=q_min, progress_callback=progress_callback
            )

            # Clear highlights after calculation and show completion message
            try:
                # Remove all highlights from DMA plot
                for artist in self.ax_dma_progress.collections[:]:
                    if hasattr(artist, '_is_highlight'):
                        artist.remove()

                # Remove all highlights and lines from PSD(f) plot
                for line in self.ax_psd_f.lines[:]:
                    if hasattr(line, '_is_psd_f_line'):
                        line.remove()
                for artist in self.ax_psd_f.collections[:]:
                    if hasattr(artist, '_is_highlight'):
                        artist.remove()

                # Update status label to show completion
                self.calc_status_label.config(
                    text=f"계산 완료! | 총 {len(v_array)}개 속도 × {len(q_array)}개 파수",
                    foreground='green'
                )

                self.canvas_calc_progress.draw()
            except Exception as e:
                print(f"Error clearing highlights: {e}")

            # Calculate detailed results for selected velocities (for visualization)
            # Select 5-8 velocities spanning the range
            n_detail_v = min(8, len(v_array))
            detail_v_indices = np.linspace(0, len(v_array)-1, n_detail_v, dtype=int)
            detailed_results_multi_v = []

            for v_idx in detail_v_indices:
                self.g_calculator.velocity = v_array[v_idx]
                detailed = self.g_calculator.calculate_G_with_details(
                    q_array, q_min=q_min, store_inner_integral=False
                )
                detailed['velocity'] = v_array[v_idx]
                detailed_results_multi_v.append(detailed)

            self.results = {
                '2d_results': results_2d,
                'detailed_results_multi_v': detailed_results_multi_v,
                'sigma_0': sigma_0,
                'temperature': temperature,
                'poisson': poisson
            }

            # Plot results
            self._plot_g_results()

            self.status_var.set("Calculation complete!")
            self.calc_button.config(state='normal')
            messagebox.showinfo("Success", f"G(q,v) calculated for {n_v} velocities and {n_q} wavenumbers")

        except Exception as e:
            self.calc_button.config(state='normal')
            messagebox.showerror("Error", f"Calculation failed:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _plot_g_results(self):
        """Plot G(q,v) 2D results with enhanced visualizations."""
        self.fig_results.clear()

        results_2d = self.results['2d_results']
        q = results_2d['q']
        v = results_2d['v']
        G_matrix = results_2d['G_matrix']
        P_matrix = results_2d['P_matrix']

        # Get detailed results if available
        has_detailed = 'detailed_results_multi_v' in self.results
        if has_detailed:
            detailed_multi_v = self.results['detailed_results_multi_v']

        # Create 2x3 subplot layout
        ax1 = self.fig_results.add_subplot(2, 3, 1)
        ax2 = self.fig_results.add_subplot(2, 3, 2)
        ax3 = self.fig_results.add_subplot(2, 3, 3)
        ax4 = self.fig_results.add_subplot(2, 3, 4)
        ax5 = self.fig_results.add_subplot(2, 3, 5)
        ax6 = self.fig_results.add_subplot(2, 3, 6)

        # Standard font settings for all plots
        TITLE_FONT = 9
        LABEL_FONT = 9
        LEGEND_FONT = 7
        TITLE_PAD = 5

        # Plot 1: Multi-velocity G(q) curves (다중 속도 G(q) 곡선)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / len(v)) for i in range(len(v))]

        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:  # Plot every 10th curve
                ax1.loglog(q, G_matrix[:, j], color=color, linewidth=1.5,
                          label=f'v={v_val:.4f} m/s')

        ax1.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
        ax1.set_ylabel('G(q)', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
        ax1.set_title('(a) 다중 속도에서의 G(q)', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
        ax1.legend(fontsize=LEGEND_FONT, ncol=2)
        ax1.grid(True, alpha=0.3)

        # Fix axis formatter to use superscript notation
        from matplotlib.ticker import FuncFormatter
        def log_tick_formatter(val, pos=None):
            if val <= 0:
                return ''
            exponent = int(np.floor(np.log10(val)))
            if abs(val - 10**exponent) < 1e-10:
                return f'$10^{{{exponent}}}$'
            else:
                mantissa = val / (10**exponent)
                if abs(mantissa - 1.0) < 0.01:
                    return f'$10^{{{exponent}}}$'
                else:
                    return f'${mantissa:.1f} \\times 10^{{{exponent}}}$'
        ax1.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        ax1.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        # Plot 2: Local stress probability distribution P(σ,q) for multiple wavenumbers
        # CHANGED: Plot vs wavenumber (q) instead of velocity, with v fixed at 1 m/s
        # Persson theory: P(σ,q) = 1/√(4πG_stress(q)) [exp(-(σ-σ0)²/4G_stress) - exp(-(σ+σ0)²/4G_stress)]
        # where G_stress(q) = (π/4) * (E*/σ₀)² * ∫[q0→q] k³C(k)dk  [dimensionless]
        sigma_0_Pa = self.results['sigma_0']  # Pa
        sigma_0_MPa = sigma_0_Pa / 1e6  # Convert Pa to MPa

        # Calculate G_stress(q) at FIXED velocity v = 1 m/s
        poisson = float(self.poisson_var.get())
        temperature = float(self.temperature_var.get())
        q_min = float(self.q_min_var.get())
        q_max = float(self.q_max_var.get())

        # Filter q array to user-specified range for stress distribution calculation
        # Use separate variable to avoid affecting other plots
        q_mask = (q >= q_min) & (q <= q_max)
        q_stress = q[q_mask]
        C_q_stress = self.psd_model(q_stress)

        print(f"Filtered q range for stress dist: {q_stress[0]:.2e} ~ {q_stress[-1]:.2e} (1/m), {len(q_stress)} points")

        # Fixed velocity for wavenumber analysis
        v_fixed = 0.01  # m/s (lower velocity to see clearer peak at σ₀)

        # Calculate G_stress(q) at fixed velocity
        # Calculate G_stress(q) at fixed velocity
        print("\n" + "="*80)
        print("DEBUG: G_stress Calculation at FIXED velocity v = 1 m/s")
        print("="*80)
        print(f"σ₀ = {sigma_0_MPa:.4f} MPa = {sigma_0_Pa:.2e} Pa")
        print(f"q range: {q_min:.2e} ~ {q_max:.2e} (1/m)")
        print(f"Fixed velocity: v = {v_fixed:.1f} m/s")
        print(f"Poisson ratio: {poisson:.3f}")

        # Check E values at this velocity
        omega_low = q_min * v_fixed
        omega_high = q_max * v_fixed
        E_low = self.material.get_storage_modulus(np.array([omega_low]))[0]
        E_high = self.material.get_storage_modulus(np.array([omega_high]))[0]
        E_star_low = E_low / (1 - poisson**2)
        E_star_high = E_high / (1 - poisson**2)

        print(f"\nω_low = {omega_low:.2e} rad/s  →  E = {E_low:.2e} Pa  →  E* = {E_star_low:.2e} Pa")
        print(f"ω_high = {omega_high:.2e} rad/s  →  E = {E_high:.2e} Pa  →  E* = {E_star_high:.2e} Pa")

        # Calculate integral of q³C(q)
        integrand = q_stress**3 * C_q_stress
        integral_full = np.trapezoid(integrand, q_stress)
        print(f"∫ q³C(q)dq = {integral_full:.4e} m⁴")

        # CRITICAL FIX: Normalize by σ₀ to make G_stress dimensionless
        E_normalized = E_star_low / sigma_0_Pa  # Normalize E by σ₀
        print(f"E_normalized = E*/σ₀ = {E_normalized:.2e}")

        # Calculate G_stress(q) array
        G_stress_array = np.zeros_like(q_stress)
        for i in range(1, len(q_stress)):
            integrand_partial = q_stress[:i+1]**3 * C_q_stress[:i+1]
            G_stress_array[i] = (np.pi / 4) * E_normalized**2 * np.trapezoid(integrand_partial, q_stress[:i+1])

        print(f"G_dimensionless(qmax) = {G_stress_array[-1]:.4e}")
        print(f"√G_dimensionless(qmax) = {np.sqrt(G_stress_array[-1]):.4f}")
        print(f"Peak location: ALWAYS at σ₀ = {sigma_0_MPa:.2f} MPa (independent of G)")
        print(f"Distribution width at qmax: √G × σ₀ = {np.sqrt(G_stress_array[-1]) * sigma_0_MPa:.4f} MPa")
        print("="*80 + "\n")

        # Set x-axis range based on MAXIMUM G to show full Gaussian shapes
        # Use the maximum G value to ensure all curves fit within the plot
        G_max = G_stress_array[-1]  # Use maximum G to capture widest distribution
        std_max = np.sqrt(G_max) * sigma_0_MPa
        sigma_max = sigma_0_MPa + 6 * std_max  # Use 6σ to show complete Gaussian tail
        sigma_min = -sigma_0_MPa - 6 * std_max  # Extended negative side for full mirror image

        # Ensure minimum x-axis range
        if sigma_max < 2 * sigma_0_MPa:
            sigma_max = 2 * sigma_0_MPa
            sigma_min = -sigma_max
        sigma_array = np.linspace(sigma_min, sigma_max, 800)  # Increased points for smoother curves

        # Debug: Print some values to verify calculations
        print(f"\n=== Debug: Stress Distribution at Fixed Velocity ===")
        print(f"σ0 = {sigma_0_MPa:.4f} MPa")
        print(f"Fixed velocity: v = {v_fixed:.1f} m/s")
        print(f"G_dimensionless(qmax) = {G_max:.4e}")
        print(f"√G_dimensionless(qmax) = {np.sqrt(G_max):.4f}")
        print(f"std_max = √G_max × σ₀ = {std_max:.4f} MPa")
        print(f"sigma_min (for x-axis) = {sigma_min:.2f} MPa")
        print(f"sigma_max (for x-axis) = {sigma_max:.2f} MPa")
        print(f"X-axis range: [{sigma_min:.2f}, {sigma_max:.2f}] MPa (±6σ from σ0)")

        # Select 3 wavenumbers to plot (first, middle, last)
        n_q_selected = 3
        q_indices = np.linspace(0, len(q_stress)-1, n_q_selected, dtype=int)

        # Create color map for wavenumbers
        cmap_q = plt.get_cmap('plasma')
        colors_q = [cmap_q(i / (n_q_selected-1)) for i in range(n_q_selected)]

        print(f"\nPlotting P(σ) for {n_q_selected} wavenumbers:")
        print("="*80)

        # Track maximum values for axis scaling
        max_P_sigma = 0
        max_term = 0

        # Plot stress distributions for selected wavenumbers
        for i, q_idx in enumerate(q_indices):
            color = colors_q[i]
            q_val = q_stress[q_idx]
            G_norm_q = G_stress_array[q_idx]

            # Calculate stress distribution at this wavenumber
            if G_norm_q > 1e-10:
                # Normalize σ by σ₀ for calculation
                sigma_norm = sigma_array / sigma_0_MPa

                # Calculate individual terms to show mirror image
                normalization = 1 / (sigma_0_MPa * np.sqrt(4 * np.pi * G_norm_q))
                term1 = normalization * np.exp(-(sigma_norm - 1)**2 / (4 * G_norm_q))  # Main peak at σ₀
                term2 = normalization * np.exp(-(sigma_norm + 1)**2 / (4 * G_norm_q))  # Mirror at -σ₀

                # Final P(σ) is difference - clip to ensure non-negative (probability cannot be negative)
                P_sigma = np.maximum(0, term1 - term2)

                # Track maximum values for axis scaling
                max_P_sigma = max(max_P_sigma, np.max(P_sigma))
                max_term = max(max_term, np.max(term1), np.max(term2))

                # Calculate P(σ > 0): probability of positive stress
                positive_indices = sigma_array > 0
                P_positive = np.trapezoid(P_sigma[positive_indices], sigma_array[positive_indices])
                P_positive_percent = P_positive * 100  # Convert to percentage

                # Total integral for verification
                integral_total = np.trapezoid(P_sigma, sigma_array)

                # Plot the final distribution (solid line) with P(σ>0) in label
                label_name = ['최소 q', '중간 q', '최대 q'][i]
                ax2.plot(sigma_array, P_sigma, color=color, linewidth=2.5,
                        label=f'{label_name}: P(σ>0)={P_positive_percent:.1f}%', alpha=0.9)

                # Plot individual terms (term1, term2) for all 3 curves
                # Use shared labels for clarity
                if i == 0:
                    ax2.plot(sigma_array, term1, color=color, linewidth=1.3,
                            linestyle='--', alpha=0.6, label=f'term1: exp[-(σ-σ₀)²/4G]')
                    ax2.plot(sigma_array, term2, color=color, linewidth=1.3,
                            linestyle=':', alpha=0.6, label=f'term2: exp[-(σ+σ₀)²/4G] 거울상')
                else:
                    ax2.plot(sigma_array, term1, color=color, linewidth=1.3,
                            linestyle='--', alpha=0.6)
                    ax2.plot(sigma_array, term2, color=color, linewidth=1.3,
                            linestyle=':', alpha=0.6)

                # Print debug information
                print(f"\n[q = {q_val:.2e} 1/m]")
                print(f"  G = {G_norm_q:.4e} (무차원)")
                print(f"  √G = {np.sqrt(G_norm_q):.4f}")
                print(f"  분포 폭 (√G × σ₀) = {np.sqrt(G_norm_q) * sigma_0_MPa:.4f} MPa")
                print(f"  피크 위치: σ = {sigma_array[np.argmax(P_sigma)]:.4f} MPa")
                print(f"  ∫P(σ)dσ (전체) = {integral_total:.4f}")
                print(f"  P(σ > 0) = {P_positive_percent:.2f}% ← 양의 응력 확률")
                print(f"  P(σ ≤ 0) = {(1 - P_positive)*100:.2f}%")

        # Add vertical line for nominal pressure
        ax2.axvline(sigma_0_MPa, color='black', linestyle='--', linewidth=2,
                   label=f'σ0 = {sigma_0_MPa:.2f} MPa', alpha=0.7)

        ax2.set_xlabel('응력 σ (MPa)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
        ax2.set_ylabel('응력 분포 P(σ)', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
        ax2.set_title(f'(b) 파수별 국소 응력 확률 분포 (v={v_fixed:.2f} m/s 고정)', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
        ax2.legend(fontsize=LEGEND_FONT, ncol=2, loc='upper right')
        ax2.grid(True, alpha=0.3)
        # Set axis limits to show FULL Gaussian distribution shapes
        # X-axis: show full calculated range to capture complete Gaussian tails
        ax2.set_xlim(sigma_min, sigma_max)
        # Y-axis: use maximum value from all curves with some headroom
        ax2.set_ylim(0, max(max_P_sigma, max_term) * 1.15)

        print("="*80)

        # Plot 3: Contact Area P(q,v) (접촉 면적)
        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:
                # Filter out values very close to 1.0 for better visualization
                P_curve = P_matrix[:, j].copy()
                # Clip very small differences from 1.0
                P_curve = np.clip(P_curve, 0, 0.999)

                ax3.semilogx(q, P_curve, color=color, linewidth=1.5,
                            label=f'v={v_val:.4f} m/s')

        ax3.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
        ax3.set_ylabel('접촉 면적 비율 P(q)', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
        ax3.set_title('(c) 다중 속도에서의 접촉 면적', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
        ax3.legend(fontsize=LEGEND_FONT, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.0)  # Set y-axis limit for better visualization

        # Fix axis formatter
        ax3.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        # Plot 4: Final contact area vs velocity (속도에 따른 최종 접촉 면적)
        P_final = P_matrix[-1, :]

        # Create gradient color for all velocity points
        scatter = ax4.scatter(v, P_final, c=np.arange(len(v)), cmap='viridis',
                             s=50, zorder=3, edgecolors='black', linewidth=0.5)

        # Add connecting line with gradient using line segments
        from matplotlib.collections import LineCollection
        points = np.array([v, P_final]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, len(v)-1)
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2, alpha=0.6)
        lc.set_array(np.arange(len(v)))
        ax4.add_collection(lc)

        ax4.set_xscale('log')
        ax4.set_xlabel('속도 v (m/s)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
        ax4.set_ylabel('최종 접촉 면적 P(q_max)', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
        ax4.set_title('(d) 속도에 따른 접촉 면적', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
        ax4.grid(True, alpha=0.3)

        # Fix axis formatter
        ax4.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        # Plot 5: Inner integral vs q for multiple velocities (다중 속도에서의 내부 적분)
        # Physical meaning: ∫dφ |E(qvcosφ)|² - 슬립 방향 성분의 점탄성 응답
        if has_detailed:
            for i, detail_result in enumerate(detailed_multi_v):
                v_val = detail_result['velocity']
                # Find matching velocity index in v array to use consistent color
                v_idx = np.argmin(np.abs(v - v_val))
                color = colors[v_idx]  # Use same color scheme as other plots
                ax5.loglog(detail_result['q'], detail_result['avg_modulus_term'],
                          color=color, linewidth=1.5, label=f'v={v_val:.4f} m/s', alpha=0.8)

            ax5.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
            ax5.set_ylabel('각도 적분 ∫dφ|E/(1-ν²)σ0|²', fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
            ax5.set_title('(e) 내부 적분: 상대적 강성비', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)
            ax5.legend(fontsize=LEGEND_FONT, ncol=2)
            ax5.grid(True, alpha=0.3)

            # Add physical interpretation text box
            textstr = ('물리적 의미: ∫dφ|E*/(1-ν²)σ0|²\n'
                      '= 외부 압력 대비 고무의 단단함을 나타내는 척도\n'
                      'E*: 복소 탄성률 (주파수 ω=qv cosφ)\n'
                      '(1-ν²)σ0: 평면 변형률 보정 + 명목 압력\n'
                      '높을수록 → 고무가 단단 → 응력 불균일 증가\n'
                      '낮을수록 → 고무가 말랑 → 완전 접촉에 가까움')
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            ax5.text(0.98, 0.02, textstr, transform=ax5.transAxes, fontsize=7,
                    verticalalignment='bottom', horizontalalignment='right', bbox=props)

            # Fix axis formatter
            ax5.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
            ax5.yaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        else:
            ax5.text(0.5, 0.5, '내부 적분 데이터 없음',
                    ha='center', va='center', transform=ax5.transAxes, fontsize=LABEL_FONT)
            ax5.set_title('(e) 내부 적분', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)

        # Plot 6: Parseval theorem - Cumulative RMS slope with q1 determination
        # Slope²(q) = 2π∫[0 to q] k³C(k)dk
        if self.psd_model is not None:
            # Calculate cumulative RMS slope
            q_parse = np.logspace(np.log10(q[0]), np.log10(q[-1]), 1000)
            C_q_parse = self.psd_model(q_parse)

            # Cumulative integration using correct formula
            # Slope²(q) = 2π∫[qmin to q] k³C(k)dk
            slope_squared_cumulative = np.zeros_like(q_parse)

            for i in range(len(q_parse)):
                q_int = q_parse[:i+1]
                C_int = C_q_parse[:i+1]
                # Integrate q³C(q) with 2π factor
                slope_squared_cumulative[i] = 2 * np.pi * np.trapezoid(q_int**3 * C_int, q_int)

            slope_rms_cumulative = np.sqrt(slope_squared_cumulative)

            # Find q1 where slope_rms = target (from parameter settings)
            target_slope_rms = float(self.target_hrms_slope_var.get())
            q1_idx = np.argmax(slope_rms_cumulative >= target_slope_rms)

            if q1_idx > 0:
                # Use interpolation for more accurate q1
                from scipy.interpolate import interp1d
                # Create interpolator
                f_interp = interp1d(slope_rms_cumulative[q1_idx-10:q1_idx+10],
                                   q_parse[q1_idx-10:q1_idx+10],
                                   kind='linear', fill_value='extrapolate')
                q1_determined = float(f_interp(target_slope_rms))
            else:
                # If target not reached, use extrapolation with Hurst exponent
                # Fit power law to last portion of data
                log_q_fit = np.log10(q_parse[-50:])
                log_C_fit = np.log10(C_q_parse[-50:])
                slope_fit = np.polyfit(log_q_fit, log_C_fit, 1)[0]
                H = -slope_fit / 2 - 1  # Hurst exponent

                # Extrapolate C(q) = A * q^(-2(H+1))
                A = C_q_parse[-1] / (q_parse[-1]**(-2*(H+1)))

                # Find q1 by solving integral equation
                # This is approximate; could use root finding for precision
                q1_determined = q_parse[-1] * 1.5  # Placeholder
                messagebox.showinfo("Info", f"Target slope {target_slope_rms} not reached. Extrapolating with H={H:.3f}")

            # Plot cumulative h'rms (RMS slope)
            ax6.semilogx(q_parse, slope_rms_cumulative, 'b-', linewidth=2.5, label="누적 h'rms")

            # Add horizontal line at target h'rms
            ax6.axhline(target_slope_rms, color='red', linestyle='--', linewidth=2,
                       label=f"목표 h'rms = {target_slope_rms}", alpha=0.7, zorder=5)

            # Add vertical line at q1
            if q1_idx > 0:
                ax6.axvline(q1_determined, color='green', linestyle='--', linewidth=2,
                           label=f'결정된 q1 = {q1_determined:.2e} (1/m)', alpha=0.7, zorder=5)

                # Mark intersection point
                ax6.plot(q1_determined, target_slope_rms, 'ro', markersize=12,
                        markeredgecolor='black', markeredgewidth=2, zorder=10,
                        label='교차점')

                # Update calculated q1 display in Tab 3
                self.calculated_q1_var.set(f"{q1_determined:.3e}")

                # Pass q1 and hrms_slope to Tab 4 (RMS Slope tab)
                self.rms_q_max_var.set(f"{q1_determined:.3e}")

                # Store calculated q1 for other uses
                self.calculated_q1 = q1_determined

            ax6.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=LABEL_FONT, labelpad=3)
            ax6.set_ylabel("누적 h'rms √(Slope²)", fontweight='bold', fontsize=LABEL_FONT, rotation=90, labelpad=5)
            ax6.set_title(f"(f) Parseval 정리: q1 자동 결정 (목표 h'rms={target_slope_rms})", fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)

            # Legend with better positioning
            ax6.legend(fontsize=LEGEND_FONT, loc='lower right', framealpha=0.9)

            ax6.grid(True, alpha=0.3)

            # Add annotation box
            if q1_idx > 0:
                textstr = (f"파서벌 정리:\nh'rms²(q) = 2π∫k³C(k)dk\n\n"
                          f"결정된 q1 = {q1_determined:.2e} 1/m\n"
                          f"해당 h'rms = {target_slope_rms:.2f}")
            else:
                textstr = (f"파서벌 정리:\nh'rms²(q) = 2π∫k³C(k)dk\n\n"
                          f"최종 h'rms = {slope_rms_cumulative[-1]:.3f}\n"
                          f"(목표 {target_slope_rms} 미달)")

            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black')
            ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)

            # Fix axis formatter
            ax6.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        else:
            ax6.text(0.5, 0.5, 'PSD 데이터 없음',
                    ha='center', va='center', transform=ax6.transAxes, fontsize=LABEL_FONT)
            ax6.set_title('(f) Parseval 정리', fontweight='bold', fontsize=TITLE_FONT, pad=TITLE_PAD)

        self.fig_results.suptitle('G(q,v) 2D 행렬 계산 결과', fontweight='bold', fontsize=11, y=0.98)
        self.fig_results.tight_layout(rect=[0, 0.01, 1, 0.97], pad=1.5, h_pad=2.0, w_pad=1.5)
        self.canvas_results.draw()

    def _save_detailed_csv(self):
        """Save detailed CSV results."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to save. Run calculation first!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            # Implementation here
            messagebox.showinfo("Info", "CSV save functionality to be implemented")

    def _export_all_results(self):
        """Export all results."""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export. Run calculation first!")
            return

        output_dir = filedialog.askdirectory(title="Select output directory")
        if output_dir:
            messagebox.showinfo("Info", "Export functionality to be implemented")

    def _show_help(self):
        """Show help dialog."""
        help_text = """
Persson Friction Calculator v2.1 - User Guide

1. Input Data Verification Tab:
   - Check that E', E'', tan(δ) and C(q) are correctly loaded
   - Verify master curve and PSD before calculation

2. Calculation Setup Tab:
   - Set pressure, velocity range (log scale: 0.0001~10 m/s)
   - Configure q range and number of points
   - Click "Run G(q,v) Calculation"

3. G(q,v) Results Tab:
   - View multi-velocity G(q) curves
   - Analyze G(q,v) heatmap
   - Check contact area P(q,v)

4. Friction Analysis Tab:
   - Velocity-dependent friction coefficient
   - Contact area ratio analysis
        """
        messagebox.showinfo("User Guide", help_text)

    def _show_about(self):
        """Show about dialog."""
        about_text = """
Persson Friction Calculator v2.1

Work Instruction v2.1 Implementation:
- Velocity range: 0.0001~10 m/s (log scale)
- G(q,v) 2D matrix calculation
- Multi-velocity plotting
- Input data verification

Based on:
Persson, B.N.J. (2001, 2006)
Rubber friction theory
        """
        messagebox.showinfo("About", about_text)

    def _create_equations_tab(self, parent):
        """Create equations reference tab with all formulas used in calculations."""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg='white')
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind('<Enter>', _bind_mousewheel)
        canvas.bind('<Leave>', _unbind_mousewheel)

        # Create matplotlib figure for equations
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import platform

        # Set Korean font based on OS
        system = platform.system()
        if system == 'Windows':
            korean_font = 'Malgun Gothic'
        elif system == 'Darwin':  # macOS
            korean_font = 'AppleGothic'
        else:  # Linux
            korean_font = 'NanumGothic'

        fig = Figure(figsize=(12, 20), facecolor='white')
        fig.suptitle('Persson 마찰 이론 - 계산 수식 정리', fontsize=18, fontweight='bold', y=0.985,
                    fontproperties={'family': korean_font})

        # Single axis for all equations
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Equation text with LaTeX and detailed Korean explanations
        # Based on user's comprehensive formula documentation
        equations_text = r"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【0. 기본 물리량 정의】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

주파수 (고무가 느끼는 진동수):
  $\omega = q \cdot v \cdot \cos\phi$
  (응력 분포 계산 등 각도 적분 없는 경우: $\omega = q \cdot v$)

유효 탄성률 (평면 변형 상태):
  $E^*(\omega) = \frac{E(\omega)}{1-\nu^2}$
  (여기서 $E(\omega)$는 DMA 데이터에서 보간한 복소 탄성률 크기)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【1. 마찰 및 접촉 면적 계산용】(무차원, $\sigma_0$ 포함)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A. 면적 계산용 파워 스펙트럼 적분함수 $G_{area}(q)$:

$G_{area}(q) = \frac{1}{8} \int_{q_0}^{q} dq' (q')^3 C(q') \int_{0}^{2\pi} d\phi | \frac{E(q'v\cos\phi)}{(1-\nu^2)\sigma_0} |^2$

  변수: $E$는 $\cos\phi$에 따라 주파수가 변하므로 각도 적분 내부에서 업데이트
  단위: 무차원 (dimensionless)
  물리적 의미: 거칠기에 의한 고무 변형 정도 (접촉 불균일성 지표)
  용도: 그래프 (a), (c)의 G(q) 곡선

B. 실접촉 면적 비율 $P(q)$:

$\frac{A(q)}{A_0} = P(q) \approx \mathrm{erf}( \frac{1}{2\sqrt{G_{area}(q)}} )$

  물리적 의미: 배율 q에서 고무가 바닥과 닿아있는 면적 비율
  범위: 0 <= P <= 1
  용도: 그래프 (c), (d)의 접촉 면적 곡선

C. 점탄성 마찰 계수 $\mu_{visc}$:

$\mu_{visc} \approx \frac{1}{2} \int_{q_0}^{q_1} dq \, q^3 C(q) S(q) P(q) \int_{0}^{2\pi} d\phi \, \cos\phi \, \mathrm{Im}( \frac{E(qv\cos\phi)}{(1-\nu^2)\sigma_0} )$

  물리적 의미: 에너지 손실에 의한 마찰 계수 (접촉 면적 $P(q)$가 가중치)
  영향 요인: 접촉 면적 $P(q)$, 에너지 소산, 재료 물성 ($E'$, $E''$)

보정 계수 $S(q)$ (대변형 시 접촉 면적 감소 보정):
  $S(q) = \gamma + (1-\gamma)P^2(q)$  (보통 $\gamma \approx 0.5$)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【2. 응력 분포 계산용】(Pa² 단위, $\sigma_0$ 없음, 각도 적분 없음)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A. 응력 분산 함수 $G_{stress}(q)$:

$G_{stress}(q) = \frac{\pi}{4} \int_{q_0}^{q} dq' (q')^3 C(q') | \frac{E(q'v)}{1-\nu^2} |^2$

  주의: 각도 적분 없음! $\sigma_0$ 분모에 없음!
  단위: Pa² (압력의 제곱)
  물리적 의미: 접촉 압력 분포의 분산 (variance)
    - $G_{stress}$ 크다 = 응력이 넓게 퍼짐 (일부 매우 높은 압력)
    - $G_{stress}$ 작다 = 응력이 명목 압력 근처에 집중
  용도: 그래프 (b)의 응력 확률 분포 계산

  속도 의존성 정확도 향상 팁:
    일부 문헌에서는 E를 상수로 가정하여 적분 밖으로 빼지만,
    속도에 따른 변화를 정확히 보려면 E(q'v)를 적분 안에 넣고
    누적 적분(cumtrapz) 필요

B. 국소 응력 확률 분포 $P(\sigma, q)$:

$P(\sigma, q) = \frac{1}{\sqrt{4\pi G_{stress}(q)}} [ \exp( -\frac{(\sigma - \sigma_0)^2}{4 G_{stress}(q)} ) - \exp( -\frac{(\sigma + \sigma_0)^2}{4 G_{stress}(q)} ) ]$

  변수:
    - $\sigma$: 국소 접촉 응력 (특정 지점에서 실제 받는 압력)
    - $\sigma_0$: 명목 압력 (평균 압력, 분포의 중심)

  물리적 의미:
    특정 압력값을 받는 접촉점의 확률 분포
    평균은 $\sigma_0$이지만 일부 돌기는 매우 높은 압력 (10 MPa 이상)

  핵심 특징:
    1. 피크 위치 = $\sigma_0$ (속도 무관, 항상 명목 압력에서 최대)
    2. $G_{stress}$ 증가 시 분포 넓어짐 (고압 구간 증가)
    3. $\sigma \to 0$일 때 P $\to$ 0 (음의 응력 불가능)

  용도: 그래프 (b)의 속도별 응력 분포 곡선


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【핵심 비교: $G_{area}$ vs $G_{stress}$】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

$\begin{array}{lcc}
\hline
\text{항목} & G_{area}(q,v) & G_{stress}(q,v) \\
\hline
\text{단위} & \text{무차원} & \text{Pa}^2 \\
\text{각도 적분} & \text{있음}\ (\int_0^{2\pi} d\phi) & \text{없음} \\
\text{탄성률 항} & |E/((1-\nu^2)\sigma_0)|^2 & |E/(1-\nu^2)|^2 \\
\text{주파수} & \omega = qv\cos\phi & \omega = qv \\
\text{물리적 의미} & \text{접촉 불균일성} & \text{응력 분산} \\
\text{용도} & P(q), \mu\ \text{계산} & P(\sigma)\ \text{계산} \\
\text{그래프} & (a), (c), (d) & (b) \\
\hline
\end{array}$

주의: 같은 이름 "G"를 사용하지만 완전히 다른 물리량!
  $G_{area}$: "얼마나 띄엄띄엄 닿는가" (접촉의 불균일성)
  $G_{stress}$: "압력이 얼마나 들쭉날쭉한가" (응력의 분산)
"""

        # Render text with proper LaTeX support
        # matplotlib's text() requires each $...$ to be recognized as math mode
        lines = equations_text.strip().split('\n')
        y_position = 0.98
        line_spacing = 0.018  # Spacing between lines

        import re

        for line in lines:
            line_stripped = line.strip()
            if line_stripped:
                # Check if line contains LaTeX (starts with $ or contains inline $...$)
                if line_stripped.startswith('$') and line_stripped.endswith('$'):
                    # Full LaTeX line - render as math
                    ax.text(0.02, y_position, line_stripped, transform=ax.transAxes,
                           fontsize=12, verticalalignment='top', horizontalalignment='left',
                           usetex=False)
                elif '$' in line_stripped:
                    # Mixed content - split and render parts separately
                    # For simplicity, render whole line and let matplotlib handle it
                    ax.text(0.02, y_position, line_stripped, transform=ax.transAxes,
                           fontsize=12, verticalalignment='top', horizontalalignment='left',
                           family=korean_font, usetex=False)
                else:
                    # Plain text - use Korean font
                    ax.text(0.02, y_position, line_stripped, transform=ax.transAxes,
                           fontsize=12, verticalalignment='top', horizontalalignment='left',
                           family=korean_font, usetex=False)
                y_position -= line_spacing
            else:
                y_position -= line_spacing * 0.5  # Half spacing for empty lines

        # Embed in tkinter
        canvas_eq = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas_eq.draw()
        canvas_eq.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pack scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_rms_slope_tab(self, parent):
        """Create h'rms / Local Strain calculation tab."""
        # Main container
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls (fixed width)
        left_frame = ttk.Frame(main_container, width=320)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # ============== Left Panel: Controls ==============

        # 1. Description
        desc_frame = ttk.LabelFrame(left_frame, text="설명", padding=5)
        desc_frame.pack(fill=tk.X, pady=2, padx=3)

        desc_text = (
            "PSD 데이터로부터 h'rms(ξ)와\n"
            "Local Strain(ε)을 계산합니다.\n\n"
            "수식:\n"
            "  ξ²(q) = 2π ∫[q₀→q] k³C(k)dk\n"
            "  ε(q) = factor × ξ(q)"
        )
        ttk.Label(desc_frame, text=desc_text, font=('Arial', 9), justify=tk.LEFT).pack(anchor=tk.W)

        # 2. Calculation Settings
        settings_frame = ttk.LabelFrame(left_frame, text="계산 설정", padding=5)
        settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # Strain factor
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Strain Factor:", font=('Arial', 9)).pack(side=tk.LEFT)
        self.strain_factor_var = tk.StringVar(value="0.5")
        ttk.Entry(row1, textvariable=self.strain_factor_var, width=8).pack(side=tk.RIGHT)

        ttk.Label(settings_frame, text="(ε = factor × ξ, Persson 권장: 0.5)",
                  font=('Arial', 8), foreground='gray').pack(anchor=tk.W)

        # q range
        q_frame = ttk.LabelFrame(settings_frame, text="q 범위", padding=3)
        q_frame.pack(fill=tk.X, pady=3)

        row_q1 = ttk.Frame(q_frame)
        row_q1.pack(fill=tk.X, pady=1)
        ttk.Label(row_q1, text="q_min (1/m):", font=('Arial', 8)).pack(side=tk.LEFT)
        self.rms_q_min_var = tk.StringVar(value="auto")
        ttk.Entry(row_q1, textvariable=self.rms_q_min_var, width=10).pack(side=tk.RIGHT)

        row_q2 = ttk.Frame(q_frame)
        row_q2.pack(fill=tk.X, pady=1)
        ttk.Label(row_q2, text="q_max (1/m):", font=('Arial', 8)).pack(side=tk.LEFT)
        self.rms_q_max_var = tk.StringVar(value="auto")
        ttk.Entry(row_q2, textvariable=self.rms_q_max_var, width=10).pack(side=tk.RIGHT)

        ttk.Label(q_frame, text="'auto' = PSD 데이터 범위 사용",
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # Target h'rms display (synced with Tab 2)
        target_frame = ttk.LabelFrame(settings_frame, text="목표 h'rms (Tab 2 연동)", padding=3)
        target_frame.pack(fill=tk.X, pady=3)

        row_target = ttk.Frame(target_frame)
        row_target.pack(fill=tk.X, pady=1)
        ttk.Label(row_target, text="목표 ξ:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.rms_target_xi_display = tk.StringVar(value="(Tab 2에서 설정)")
        self.rms_target_xi_label = ttk.Label(row_target, textvariable=self.rms_target_xi_display,
                                             font=('Arial', 9, 'bold'), foreground='blue')
        self.rms_target_xi_label.pack(side=tk.RIGHT)

        # Refresh button for target value
        ttk.Button(target_frame, text="Tab 2 값 불러오기", command=self._sync_target_xi_from_tab2,
                   width=15).pack(pady=2)

        ttk.Label(target_frame, text="※ Tab 2에서 '목표 h'rms' 변경 시\n   이 버튼을 눌러 동기화하세요",
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # Calculate button
        calc_frame = ttk.Frame(settings_frame)
        calc_frame.pack(fill=tk.X, pady=5)

        self.rms_calc_btn = ttk.Button(
            calc_frame,
            text="h'rms slope / Local Strain 계산",
            command=self._calculate_rms_slope
        )
        self.rms_calc_btn.pack(fill=tk.X)

        # Progress bar
        self.rms_progress_var = tk.IntVar()
        self.rms_progress_bar = ttk.Progressbar(
            calc_frame,
            variable=self.rms_progress_var,
            maximum=100
        )
        self.rms_progress_bar.pack(fill=tk.X, pady=2)

        # 3. Results Summary
        results_frame = ttk.LabelFrame(left_frame, text="결과 요약", padding=5)
        results_frame.pack(fill=tk.X, pady=2, padx=3)

        self.rms_result_text = tk.Text(results_frame, height=12, font=("Courier", 8), wrap=tk.WORD)
        self.rms_result_text.pack(fill=tk.X)

        # 4. Export / Apply buttons
        action_frame = ttk.LabelFrame(left_frame, text="작업", padding=5)
        action_frame.pack(fill=tk.X, pady=2, padx=3)

        self.apply_strain_btn = ttk.Button(
            action_frame,
            text="μ_visc 탭에 Local Strain 적용",
            command=self._apply_local_strain_to_mu_visc
        )
        self.apply_strain_btn.pack(fill=tk.X, pady=2)

        ttk.Button(
            action_frame,
            text="CSV 내보내기",
            command=self._export_rms_slope_data
        ).pack(fill=tk.X, pady=2)

        # ============== Right Panel: Plots ==============

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_panel, text="그래프", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with 2x2 subplots
        self.fig_rms = Figure(figsize=(9, 7), dpi=100)

        # Top-left: h'rms vs q
        self.ax_rms_slope = self.fig_rms.add_subplot(221)
        self.ax_rms_slope.set_title("h'rms ξ(q)", fontweight='bold')
        self.ax_rms_slope.set_xlabel('파수 q (1/m)')
        self.ax_rms_slope.set_ylabel("ξ (h'rms)")
        self.ax_rms_slope.set_xscale('log')
        self.ax_rms_slope.set_yscale('log')
        self.ax_rms_slope.grid(True, alpha=0.3)

        # Top-right: Local Strain vs q
        self.ax_local_strain = self.fig_rms.add_subplot(222)
        self.ax_local_strain.set_title('Local Strain ε(q)', fontweight='bold')
        self.ax_local_strain.set_xlabel('파수 q (1/m)')
        self.ax_local_strain.set_ylabel('ε (fraction)')
        self.ax_local_strain.set_xscale('log')
        self.ax_local_strain.set_yscale('log')
        self.ax_local_strain.grid(True, alpha=0.3)

        # Bottom-left: RMS Height vs q
        self.ax_rms_height = self.fig_rms.add_subplot(223)
        self.ax_rms_height.set_title('RMS Height h_rms(q)', fontweight='bold')
        self.ax_rms_height.set_xlabel('파수 q (1/m)')
        self.ax_rms_height.set_ylabel('h_rms (m)')
        self.ax_rms_height.set_xscale('log')
        self.ax_rms_height.set_yscale('log')
        self.ax_rms_height.grid(True, alpha=0.3)

        # Bottom-right: PSD (for reference)
        self.ax_psd_ref = self.fig_rms.add_subplot(224)
        self.ax_psd_ref.set_title('PSD C(q) (참조)', fontweight='bold')
        self.ax_psd_ref.set_xlabel('파수 q (1/m)')
        self.ax_psd_ref.set_ylabel('C(q) (m⁴)')
        self.ax_psd_ref.set_xscale('log')
        self.ax_psd_ref.set_yscale('log')
        self.ax_psd_ref.grid(True, alpha=0.3)

        self.fig_rms.tight_layout()

        self.canvas_rms = FigureCanvasTkAgg(self.fig_rms, plot_frame)
        self.canvas_rms.draw()
        self.canvas_rms.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_rms, plot_frame)
        toolbar.update()

    def _sync_target_xi_from_tab2(self):
        """Sync target h'rms from Tab 2 to Tab 4 display and update target_xi."""
        try:
            # Get target h'rms from Tab 2
            target_xi_str = self.target_hrms_slope_var.get()
            target_xi = float(target_xi_str)

            # Update Tab 4 display
            self.rms_target_xi_display.set(f"{target_xi:.4f}")

            # Update target_xi for calculations
            self.target_xi = target_xi

            # Also sync Tab 1's psd_xi_var for consistency
            self.psd_xi_var.set(target_xi_str)

            self.status_var.set(f"목표 h'rms 동기화 완료: ξ = {target_xi:.4f}")
        except ValueError:
            self.rms_target_xi_display.set("(유효하지 않은 값)")
            self.status_var.set("오류: Tab 2의 목표 h'rms 값이 유효하지 않습니다.")

    def _calculate_rms_slope(self):
        """Calculate h'rms and Local Strain from PSD data."""
        # Check if PSD data is available
        if self.psd_model is None:
            messagebox.showwarning("경고", "먼저 PSD 데이터를 로드하세요.")
            return

        try:
            # Sync target_xi from Tab 2 before calculation
            self._sync_target_xi_from_tab2()

            self.rms_calc_btn.config(state='disabled')
            self.rms_progress_var.set(10)
            self.root.update_idletasks()

            # Get PSD data - MeasuredPSD uses q_data and C_data attributes
            if hasattr(self.psd_model, 'q_data'):
                q_array = self.psd_model.q_data
                C_q_array = self.psd_model.C_data
            elif hasattr(self.psd_model, 'q'):
                q_array = self.psd_model.q
                C_q_array = self.psd_model.C_iso if hasattr(self.psd_model, 'C_iso') else self.psd_model(self.psd_model.q)
            else:
                # Generate q array and call PSD model
                q_array = np.logspace(2, 8, 200)  # Default range
                C_q_array = self.psd_model(q_array)

            # Get strain factor
            strain_factor = float(self.strain_factor_var.get())

            # Get q range - use Tab 1/2 settings when 'auto' for consistency
            q_min_str = self.rms_q_min_var.get().strip().lower()
            q_max_str = self.rms_q_max_var.get().strip().lower()

            if q_min_str == 'auto':
                # Use Tab 1/2 q_min for consistency with G(q) calculation
                try:
                    q_min = float(self.q_min_var.get())
                except:
                    q_min = q_array[0]
            else:
                q_min = float(q_min_str)

            if q_max_str == 'auto':
                # Use Tab 1/2 q_max for consistency with G(q) calculation
                try:
                    q_max = float(self.q_max_var.get())
                except:
                    q_max = q_array[-1]
            else:
                q_max = float(q_max_str)

            # Filter q range
            mask = (q_array >= q_min) & (q_array <= q_max)
            q_filtered = q_array[mask]
            C_filtered = C_q_array[mask]

            if len(q_filtered) < 3:
                messagebox.showerror("오류", "q 범위에 데이터 포인트가 부족합니다.")
                self.rms_calc_btn.config(state='normal')
                return

            self.rms_progress_var.set(30)
            self.root.update_idletasks()

            # Create RMS slope calculator
            self.rms_slope_calculator = RMSSlopeCalculator(
                q_filtered, C_filtered, strain_factor=strain_factor
            )

            self.rms_progress_var.set(60)
            self.root.update_idletasks()

            # Store profiles
            self.rms_slope_profiles = self.rms_slope_calculator.get_profiles()
            self.local_strain_array = self.rms_slope_profiles['strain'].copy()

            # Update plots
            self._update_rms_slope_plots()

            self.rms_progress_var.set(80)
            self.root.update_idletasks()

            # Update result text
            self._update_rms_result_text()

            self.rms_progress_var.set(100)
            self.status_var.set("h'rms slope / Local Strain 계산 완료")

            # Use target_xi from Tab 2 if available for consistency
            xi_max_display = self.target_xi if self.target_xi is not None else self.rms_slope_profiles['xi'][-1]
            messagebox.showinfo("완료",
                f"h'rms slope / Local Strain 계산 완료!\n\n"
                f"ξ_max (h'rms) = {xi_max_display:.4f}\n"
                f"ε_max = {self.rms_slope_profiles['strain'][-1]*100:.2f}%\n"
                f"h_rms = {self.rms_slope_profiles['hrms'][-1]*1e6:.2f} μm"
            )

        except Exception as e:
            messagebox.showerror("오류", f"계산 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.rms_calc_btn.config(state='normal')

    def _update_rms_slope_plots(self):
        """Update RMS slope plots."""
        if self.rms_slope_profiles is None:
            return

        profiles = self.rms_slope_profiles
        q = profiles['q']
        xi = profiles['xi']
        strain = profiles['strain']
        hrms = profiles['hrms']
        C_q = profiles['C_q']

        # Clear all subplots
        self.ax_rms_slope.clear()
        self.ax_local_strain.clear()
        self.ax_rms_height.clear()
        self.ax_psd_ref.clear()

        # Plot 1: h'rms
        valid_xi = xi > 0
        if np.any(valid_xi):
            self.ax_rms_slope.loglog(q[valid_xi], xi[valid_xi], 'b-', linewidth=2)
        self.ax_rms_slope.set_title("h'rms ξ(q)", fontweight='bold')
        self.ax_rms_slope.set_xlabel('파수 q (1/m)')
        self.ax_rms_slope.set_ylabel("ξ (h'rms)")
        self.ax_rms_slope.grid(True, alpha=0.3)

        # Add final value annotation - use target_xi from Tab 2 if available
        if len(xi) > 0 and xi[-1] > 0:
            # Use target_xi from PSD settings (Tab 2) for consistency
            xi_max_display = self.target_xi if self.target_xi is not None else xi[-1]
            self.ax_rms_slope.axhline(y=xi_max_display, color='r', linestyle='--', alpha=0.5)
            self.ax_rms_slope.annotate(f'ξ_max={xi_max_display:.4f}',
                xy=(q[-1], xi_max_display), xytext=(0.7, 0.9),
                textcoords='axes fraction', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

        # Plot 2: Local Strain
        valid_strain = strain > 0
        if np.any(valid_strain):
            self.ax_local_strain.loglog(q[valid_strain], strain[valid_strain]*100, 'r-', linewidth=2)
        self.ax_local_strain.set_title('Local Strain ε(q)', fontweight='bold')
        self.ax_local_strain.set_xlabel('파수 q (1/m)')
        self.ax_local_strain.set_ylabel('ε (%)')
        self.ax_local_strain.grid(True, alpha=0.3)

        # Add strain thresholds
        self.ax_local_strain.axhline(y=1, color='g', linestyle=':', alpha=0.5, label='1%')
        self.ax_local_strain.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='10%')
        self.ax_local_strain.axhline(y=100, color='red', linestyle=':', alpha=0.5, label='100%')
        self.ax_local_strain.legend(loc='lower right', fontsize=7)

        if len(strain) > 0 and strain[-1] > 0:
            self.ax_local_strain.annotate(f'ε_max={strain[-1]*100:.2f}%',
                xy=(q[-1], strain[-1]*100), xytext=(0.7, 0.9),
                textcoords='axes fraction', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))

        # Plot 3: RMS Height
        valid_hrms = hrms > 0
        if np.any(valid_hrms):
            self.ax_rms_height.loglog(q[valid_hrms], hrms[valid_hrms]*1e6, 'g-', linewidth=2)
        self.ax_rms_height.set_title('RMS Height h_rms(q)', fontweight='bold')
        self.ax_rms_height.set_xlabel('파수 q (1/m)')
        self.ax_rms_height.set_ylabel('h_rms (μm)')
        self.ax_rms_height.grid(True, alpha=0.3)

        if len(hrms) > 0 and hrms[-1] > 0:
            self.ax_rms_height.annotate(f'h_rms={hrms[-1]*1e6:.2f}μm',
                xy=(q[-1], hrms[-1]*1e6), xytext=(0.7, 0.9),
                textcoords='axes fraction', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.5))

        # Plot 4: PSD Reference
        valid_C = C_q > 0
        if np.any(valid_C):
            self.ax_psd_ref.loglog(q[valid_C], C_q[valid_C], 'k-', linewidth=1.5)
        self.ax_psd_ref.set_title('PSD C(q) (참조)', fontweight='bold')
        self.ax_psd_ref.set_xlabel('파수 q (1/m)')
        self.ax_psd_ref.set_ylabel('C(q) (m⁴)')
        self.ax_psd_ref.grid(True, alpha=0.3)

        self.fig_rms.tight_layout()
        self.canvas_rms.draw()

    def _update_rms_result_text(self):
        """Update RMS slope result text."""
        self.rms_result_text.delete(1.0, tk.END)

        if self.rms_slope_calculator is None:
            return

        summary = self.rms_slope_calculator.get_summary()
        profiles = self.rms_slope_profiles

        self.rms_result_text.insert(tk.END, "=" * 35 + "\n")
        self.rms_result_text.insert(tk.END, "h'rms slope / Local Strain 결과\n")
        self.rms_result_text.insert(tk.END, "=" * 35 + "\n\n")

        self.rms_result_text.insert(tk.END, "[입력 데이터]\n")
        self.rms_result_text.insert(tk.END, f"  q_min: {summary['q_min']:.2e} 1/m\n")
        self.rms_result_text.insert(tk.END, f"  q_max: {summary['q_max']:.2e} 1/m\n")
        self.rms_result_text.insert(tk.END, f"  데이터 점: {summary['n_points']}\n")
        self.rms_result_text.insert(tk.END, f"  Strain Factor: {summary['strain_factor']}\n\n")

        self.rms_result_text.insert(tk.END, "[h'rms]\n")
        # Show both target ξ (user input from Tab 2) and calculated ξ
        if self.target_xi is not None:
            self.rms_result_text.insert(tk.END, f"  ξ_target (Tab 2 입력값): {self.target_xi:.4f}\n")
        self.rms_result_text.insert(tk.END, f"  ξ_calc (적분 계산값): {summary['xi_max']:.4f}\n")
        self.rms_result_text.insert(tk.END, f"  ξ(q_max): {summary['xi_at_qmax']:.4f}\n\n")

        self.rms_result_text.insert(tk.END, "[Local Strain]\n")
        self.rms_result_text.insert(tk.END, f"  ε_max: {summary['strain_max']*100:.2f}%\n")
        self.rms_result_text.insert(tk.END, f"  ε(q_max): {summary['strain_at_qmax']*100:.2f}%\n\n")

        self.rms_result_text.insert(tk.END, "[RMS Height]\n")
        self.rms_result_text.insert(tk.END, f"  h_rms: {summary['hrms_total']*1e6:.2f} μm\n\n")

        # Show strain at representative q values
        self.rms_result_text.insert(tk.END, "[파수별 Local Strain]\n")
        q = profiles['q']
        strain = profiles['strain']
        indices = np.linspace(0, len(q)-1, min(8, len(q)), dtype=int)
        for idx in indices:
            self.rms_result_text.insert(tk.END,
                f"  q={q[idx]:.2e}: ε={strain[idx]*100:.3f}%\n")

    def _apply_local_strain_to_mu_visc(self):
        """Apply calculated local strain to mu_visc calculation."""
        if self.local_strain_array is None or self.rms_slope_profiles is None:
            messagebox.showwarning("경고", "먼저 h'rms를 계산하세요.")
            return

        # Store for use in mu_visc tab
        self.status_var.set("Local Strain이 μ_visc 탭에 적용 준비됨")

        messagebox.showinfo("완료",
            f"Local Strain 데이터가 μ_visc 계산에 사용될 준비가 되었습니다.\n\n"
            f"데이터 점: {len(self.local_strain_array)}\n"
            f"ε 범위: {self.local_strain_array[0]*100:.4f}% ~ {self.local_strain_array[-1]*100:.2f}%\n\n"
            f"μ_visc 탭에서 '비선형 f,g 보정'을 활성화하고\n"
            f"Strain 추정 방법을 'rms_slope'로 설정하세요."
        )

    def _export_rms_slope_data(self):
        """Export h'rms data to CSV file."""
        if self.rms_slope_profiles is None:
            messagebox.showwarning("경고", "먼저 h'rms를 계산하세요.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="hrms_slope_data.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            import csv
            profiles = self.rms_slope_profiles

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["# h'rms slope / Local Strain Data"])
                writer.writerow(["# q (1/m)", "C(q) (m^4)", "xi^2", "xi (h'rms)",
                                "strain (fraction)", "strain (%)", "h_rms^2 (m^2)", "h_rms (m)"])

                for i in range(len(profiles['q'])):
                    writer.writerow([
                        f"{profiles['q'][i]:.6e}",
                        f"{profiles['C_q'][i]:.6e}",
                        f"{profiles['xi_squared'][i]:.6e}",
                        f"{profiles['xi'][i]:.6e}",
                        f"{profiles['strain'][i]:.6e}",
                        f"{profiles['strain'][i]*100:.4f}",
                        f"{profiles['hrms_squared'][i]:.6e}",
                        f"{profiles['hrms'][i]:.6e}"
                    ])

            messagebox.showinfo("성공", f"데이터 저장 완료:\n{filename}")
            self.status_var.set(f"h'rms 데이터 저장: {filename}")

        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{str(e)}")

    def _create_mu_visc_tab(self, parent):
        """Create enhanced Strain/mu_visc calculation tab with piecewise averaging."""
        # Create main container with 2 columns
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        # Left panel for inputs (scrollable) - fixed width
        left_frame = ttk.Frame(main_container, width=350)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)  # Keep fixed width

        # Create canvas and scrollbar for left panel
        left_canvas = tk.Canvas(left_frame, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        left_panel = ttk.Frame(left_canvas)

        left_panel.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )

        left_canvas.create_window((0, 0), window=left_panel, anchor="nw", width=330)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        # Pack scrollbar and canvas
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mouse wheel scroll binding for the canvas
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                left_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                left_canvas.yview_scroll(1, "units")

        # Bind mouse wheel events
        left_canvas.bind("<MouseWheel>", _on_mousewheel)  # Windows/Mac
        left_canvas.bind("<Button-4>", _on_mousewheel_linux)  # Linux scroll up
        left_canvas.bind("<Button-5>", _on_mousewheel_linux)  # Linux scroll down

        # Also bind to the left_panel for when mouse is over widgets
        def _bind_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", _on_mousewheel_linux)
            widget.bind("<Button-5>", _on_mousewheel_linux)
            for child in widget.winfo_children():
                _bind_mousewheel(child)

        left_panel.bind("<Map>", lambda e: _bind_mousewheel(left_panel))

        # ============== Left Panel: Controls ==============

        # 1. Strain Data Loading
        strain_frame = ttk.LabelFrame(left_panel, text="1) Strain 데이터", padding=5)
        strain_frame.pack(fill=tk.X, pady=2, padx=3)

        ttk.Button(
            strain_frame,
            text="Strain Sweep 로드",
            command=self._load_strain_sweep_data,
            width=20
        ).pack(anchor=tk.W, pady=1)

        self.strain_file_label = ttk.Label(strain_frame, text="(파일 없음)", font=('Arial', 8))
        self.strain_file_label.pack(anchor=tk.W)

        ttk.Button(
            strain_frame,
            text="f,g 곡선 로드",
            command=self._load_fg_curve_data,
            width=20
        ).pack(anchor=tk.W, pady=1)

        self.fg_file_label = ttk.Label(strain_frame, text="(파일 없음)", font=('Arial', 8))
        self.fg_file_label.pack(anchor=tk.W)

        # 2. f,g Calculation Settings
        fg_settings_frame = ttk.LabelFrame(left_panel, text="2) f,g 계산", padding=5)
        fg_settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # Target frequency and E0 points in one row
        row1 = ttk.Frame(fg_settings_frame)
        row1.pack(fill=tk.X, pady=1)
        ttk.Label(row1, text="주파수:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.fg_target_freq_var = tk.StringVar(value="1.0")
        ttk.Entry(row1, textvariable=self.fg_target_freq_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="Hz  E0점:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.e0_points_var = tk.StringVar(value="5")
        ttk.Entry(row1, textvariable=self.e0_points_var, width=4).pack(side=tk.LEFT, padx=2)

        # Strain is percent and clip checkboxes
        self.strain_is_percent_var = tk.BooleanVar(value=True)
        self.clip_fg_var = tk.BooleanVar(value=True)
        check_row = ttk.Frame(fg_settings_frame)
        check_row.pack(fill=tk.X, pady=1)
        ttk.Checkbutton(check_row, text="% 단위", variable=self.strain_is_percent_var).pack(side=tk.LEFT)
        ttk.Checkbutton(check_row, text="Clip ≤1", variable=self.clip_fg_var).pack(side=tk.LEFT)

        # Grid max strain and Persson grid
        row2 = ttk.Frame(fg_settings_frame)
        row2.pack(fill=tk.X, pady=1)
        ttk.Label(row2, text="Grid Max(%):", font=('Arial', 8)).pack(side=tk.LEFT)
        self.extend_strain_var = tk.StringVar(value="40")
        ttk.Entry(row2, textvariable=self.extend_strain_var, width=5).pack(side=tk.LEFT, padx=2)
        self.use_persson_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Persson Grid", variable=self.use_persson_grid_var).pack(side=tk.LEFT)

        # Compute f,g button
        ttk.Button(
            fg_settings_frame,
            text="f,g 계산",
            command=self._compute_fg_curves,
            width=15
        ).pack(anchor=tk.W, pady=2)

        # 3. Piecewise Temperature Selection (Group A / Group B)
        piecewise_frame = ttk.LabelFrame(left_panel, text="3) Piecewise 온도", padding=5)
        piecewise_frame.pack(fill=tk.X, pady=2, padx=3)

        # Split strain setting
        split_row = ttk.Frame(piecewise_frame)
        split_row.pack(fill=tk.X, pady=1)
        ttk.Label(split_row, text="Split(%):", font=('Arial', 8)).pack(side=tk.LEFT)
        self.split_strain_var = tk.StringVar(value="15.0")
        ttk.Entry(split_row, textvariable=self.split_strain_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(split_row, text="(A≤Split, B>Split)", font=('Arial', 7), foreground='gray').pack(side=tk.LEFT)

        # Group A temperatures (low strain)
        group_a_frame = ttk.LabelFrame(piecewise_frame, text="Group A (저변형)", padding=2)
        group_a_frame.pack(fill=tk.X, pady=2)

        self.temp_listbox_A = tk.Listbox(
            group_a_frame,
            height=3,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            font=('Arial', 8)
        )
        self.temp_listbox_A.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar_A = ttk.Scrollbar(group_a_frame, command=self.temp_listbox_A.yview)
        scrollbar_A.pack(side=tk.RIGHT, fill=tk.Y)
        self.temp_listbox_A.config(yscrollcommand=scrollbar_A.set)

        # Group B temperatures (high strain)
        group_b_frame = ttk.LabelFrame(piecewise_frame, text="Group B (고변형)", padding=2)
        group_b_frame.pack(fill=tk.X, pady=2)

        self.temp_listbox_B = tk.Listbox(
            group_b_frame,
            height=3,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            font=('Arial', 8)
        )
        self.temp_listbox_B.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar_B = ttk.Scrollbar(group_b_frame, command=self.temp_listbox_B.yview)
        scrollbar_B.pack(side=tk.RIGHT, fill=tk.Y)
        self.temp_listbox_B.config(yscrollcommand=scrollbar_B.set)

        # Buttons for temperature selection
        temp_btn_frame = ttk.Frame(piecewise_frame)
        temp_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(temp_btn_frame, text="전체 선택", command=self._select_all_temps, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(temp_btn_frame, text="Piecewise 평균", command=self._piecewise_average_fg_curves, width=15).pack(side=tk.LEFT, padx=1)

        # Legacy simple averaging (keep for compatibility)
        self.temp_listbox_frame = ttk.Frame(left_panel)
        self.temp_listbox = tk.Listbox(
            self.temp_listbox_frame,
            height=0,
            selectmode=tk.MULTIPLE,
            exportselection=False
        )

        # 4. mu_visc Calculation Settings
        mu_settings_frame = ttk.LabelFrame(left_panel, text="4) μ_visc 계산", padding=5)
        mu_settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # Nonlinear correction - single row
        nonlinear_row = ttk.Frame(mu_settings_frame)
        nonlinear_row.pack(fill=tk.X, pady=1)
        self.use_fg_correction_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(nonlinear_row, text="비선형 f,g 보정", variable=self.use_fg_correction_var).pack(side=tk.LEFT)

        # Strain estimation in same frame
        strain_row = ttk.Frame(mu_settings_frame)
        strain_row.pack(fill=tk.X, pady=1)
        ttk.Label(strain_row, text="Strain:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.strain_est_method_var = tk.StringVar(value="rms_slope")
        strain_combo = ttk.Combobox(strain_row, textvariable=self.strain_est_method_var,
                     values=["rms_slope", "fixed", "persson", "simple"], width=10, state="readonly")
        strain_combo.pack(side=tk.LEFT, padx=2)

        self.fixed_strain_var = tk.StringVar(value="1.0")
        self.fixed_strain_entry = ttk.Entry(strain_row, textvariable=self.fixed_strain_var, width=5)
        self.fixed_strain_entry.pack(side=tk.LEFT)
        self.fixed_strain_label = ttk.Label(strain_row, text="%", font=('Arial', 8))
        self.fixed_strain_label.pack(side=tk.LEFT)

        # Callback to show/hide fixed strain entry based on method
        def on_strain_method_change(*args):
            method = self.strain_est_method_var.get()
            if method == 'rms_slope':
                self.fixed_strain_entry.config(state='disabled')
                self.fixed_strain_label.config(foreground='gray')
            else:
                self.fixed_strain_entry.config(state='normal')
                self.fixed_strain_label.config(foreground='black')

        self.strain_est_method_var.trace_add('write', on_strain_method_change)
        on_strain_method_change()  # Initialize state

        # Integration parameters in single row
        integ_row = ttk.Frame(mu_settings_frame)
        integ_row.pack(fill=tk.X, pady=1)
        ttk.Label(integ_row, text="γ:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.gamma_var = tk.StringVar(value="0.5")
        ttk.Entry(integ_row, textvariable=self.gamma_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(integ_row, text="φ점:", font=('Arial', 8)).pack(side=tk.LEFT)
        self.n_phi_var = tk.StringVar(value="144")
        ttk.Entry(integ_row, textvariable=self.n_phi_var, width=5).pack(side=tk.LEFT, padx=2)

        # Smoothing in single row
        smooth_row = ttk.Frame(mu_settings_frame)
        smooth_row.pack(fill=tk.X, pady=1)
        self.smooth_mu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(smooth_row, text="스무딩", variable=self.smooth_mu_var).pack(side=tk.LEFT)
        self.smooth_window_var = tk.StringVar(value="5")
        ttk.Combobox(smooth_row, textvariable=self.smooth_window_var,
                     values=["3", "5", "7", "9", "11"], width=4, state="readonly").pack(side=tk.LEFT, padx=2)

        # Calculate button and progress bar
        calc_row = ttk.Frame(mu_settings_frame)
        calc_row.pack(fill=tk.X, pady=2)
        self.mu_calc_button = ttk.Button(calc_row, text="μ_visc 계산", command=self._calculate_mu_visc, width=15)
        self.mu_calc_button.pack(side=tk.LEFT, padx=2)

        self.mu_progress_var = tk.IntVar()
        self.mu_progress_bar = ttk.Progressbar(calc_row, variable=self.mu_progress_var, maximum=100, length=150)
        self.mu_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 5. Results Display
        results_frame = ttk.LabelFrame(left_panel, text="5) 결과", padding=5)
        results_frame.pack(fill=tk.X, pady=2, padx=3)

        self.mu_result_text = tk.Text(results_frame, height=8, font=("Courier", 8), wrap=tk.WORD)
        self.mu_result_text.pack(fill=tk.X)

        # Export buttons
        export_btn_frame = ttk.Frame(results_frame)
        export_btn_frame.pack(fill=tk.X, pady=2)

        ttk.Button(export_btn_frame, text="μ CSV", command=self._export_mu_visc_results, width=10).pack(side=tk.LEFT, padx=1)
        ttk.Button(export_btn_frame, text="f,g CSV", command=self._export_fg_curves, width=10).pack(side=tk.LEFT, padx=1)

        # ============== Right Panel: Plots ==============

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_panel, text="그래프", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with 2x2 subplots
        self.fig_mu_visc = Figure(figsize=(9, 7), dpi=100)

        # Top-left: f,g curves
        self.ax_fg_curves = self.fig_mu_visc.add_subplot(221)
        self.ax_fg_curves.set_title('f(ε), g(ε) 곡선', fontweight='bold')
        self.ax_fg_curves.set_xlabel('변형률 ε (fraction)')
        self.ax_fg_curves.set_ylabel('보정 계수')
        self.ax_fg_curves.grid(True, alpha=0.3)

        # Top-right: mu_visc vs velocity
        self.ax_mu_v = self.fig_mu_visc.add_subplot(222)
        self.ax_mu_v.set_title('μ_visc(v) 곡선', fontweight='bold')
        self.ax_mu_v.set_xlabel('속도 v (m/s)')
        self.ax_mu_v.set_ylabel('마찰 계수 μ_visc')
        self.ax_mu_v.set_xscale('log')
        self.ax_mu_v.grid(True, alpha=0.3)

        # Bottom-left: Contact Area Ratio vs Velocity
        self.ax_mu_cumulative = self.fig_mu_visc.add_subplot(223)
        self.ax_mu_cumulative.set_title('실접촉 면적비율 P(v)', fontweight='bold')
        self.ax_mu_cumulative.set_xlabel('속도 v (m/s)')
        self.ax_mu_cumulative.set_ylabel('평균 P(q)')
        self.ax_mu_cumulative.set_xscale('log')
        self.ax_mu_cumulative.grid(True, alpha=0.3)

        # Bottom-right: P(q) and S(q)
        self.ax_ps = self.fig_mu_visc.add_subplot(224)
        self.ax_ps.set_title('P(q), S(q) 분포', fontweight='bold')
        self.ax_ps.set_xlabel('파수 q (1/m)')
        self.ax_ps.set_ylabel('P(q), S(q)')
        self.ax_ps.set_xscale('log')
        self.ax_ps.grid(True, alpha=0.3)

        self.fig_mu_visc.tight_layout()

        self.canvas_mu_visc = FigureCanvasTkAgg(self.fig_mu_visc, plot_frame)
        self.canvas_mu_visc.draw()
        self.canvas_mu_visc.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_mu_visc, plot_frame)
        toolbar.update()

        # Initialize piecewise result storage
        self.piecewise_result = None

    def _select_all_temps(self):
        """Select all temperatures in both Group A and B listboxes."""
        # Select all in Group A
        for i in range(self.temp_listbox_A.size()):
            self.temp_listbox_A.selection_set(i)
        # Select all in Group B
        for i in range(self.temp_listbox_B.size()):
            self.temp_listbox_B.selection_set(i)

    def _piecewise_average_fg_curves(self):
        """Perform piecewise averaging with Group A/B temperatures."""
        if self.fg_by_T is None:
            messagebox.showwarning("경고", "먼저 f,g 곡선을 계산하세요.")
            return

        try:
            # Get selected temperatures from Group A and B
            temps = sorted(self.fg_by_T.keys())

            sel_A = self.temp_listbox_A.curselection()
            sel_B = self.temp_listbox_B.curselection()

            if not sel_A or not sel_B:
                messagebox.showwarning("경고", "Group A와 B 모두에서 최소 1개 온도를 선택하세요.")
                return

            temps_A = [temps[i] for i in sel_A]
            temps_B = [temps[i] for i in sel_B]

            # Get split strain
            split_percent = float(self.split_strain_var.get())
            split_strain = split_percent / 100.0

            # Get max strain for grid
            extend_percent = float(self.extend_strain_var.get())
            max_strain = extend_percent / 100.0

            # Create strain grid
            use_persson = self.use_persson_grid_var.get()
            grid_strain = create_strain_grid(30, max_strain, use_persson_grid=use_persson)

            # Average curves for Group A
            result_A = average_fg_curves(
                self.fg_by_T,
                temps_A,
                grid_strain,
                interp_kind='loglog_linear',
                avg_mode='mean',
                clip_leq_1=self.clip_fg_var.get()
            )

            # Average curves for Group B
            result_B = average_fg_curves(
                self.fg_by_T,
                temps_B,
                grid_strain,
                interp_kind='loglog_linear',
                avg_mode='mean',
                clip_leq_1=self.clip_fg_var.get()
            )

            if result_A is None or result_B is None:
                messagebox.showerror("오류", "평균화 실패: 데이터가 부족합니다.")
                return

            # Stitch results: use A for strain <= split, B for strain > split
            mask_A = grid_strain <= split_strain
            mask_B = ~mask_A

            f_stitched = np.empty_like(result_A['f_avg'])
            g_stitched = np.empty_like(result_A['g_avg'])
            n_eff_stitched = np.empty_like(result_A['n_eff'])

            f_stitched[mask_A] = result_A['f_avg'][mask_A]
            f_stitched[mask_B] = result_B['f_avg'][mask_B]
            g_stitched[mask_A] = result_A['g_avg'][mask_A]
            g_stitched[mask_B] = result_B['g_avg'][mask_B]
            n_eff_stitched[mask_A] = result_A['n_eff'][mask_A]
            n_eff_stitched[mask_B] = result_B['n_eff'][mask_B]

            # Handle any remaining NaN values with interpolation/fill
            if np.any(~np.isfinite(f_stitched)) or np.any(~np.isfinite(g_stitched)):
                # Forward-backward fill for NaN values
                valid_f = np.isfinite(f_stitched)
                valid_g = np.isfinite(g_stitched)

                if np.any(valid_f):
                    # Forward fill
                    last_val = f_stitched[valid_f][0]
                    for i in range(len(f_stitched)):
                        if valid_f[i]:
                            last_val = f_stitched[i]
                        else:
                            f_stitched[i] = last_val
                    # Backward fill for beginning
                    first_valid_idx = np.argmax(valid_f)
                    f_stitched[:first_valid_idx] = f_stitched[first_valid_idx]
                else:
                    f_stitched[:] = 1.0  # Default

                if np.any(valid_g):
                    last_val = g_stitched[valid_g][0]
                    for i in range(len(g_stitched)):
                        if valid_g[i]:
                            last_val = g_stitched[i]
                        else:
                            g_stitched[i] = last_val
                    first_valid_idx = np.argmax(valid_g)
                    g_stitched[:first_valid_idx] = g_stitched[first_valid_idx]
                else:
                    g_stitched[:] = 1.0  # Default

            # Force monotonic decrease: after minimum, hold the minimum value
            # (Payne effect: f,g should decrease with strain, not increase)
            f_min_idx = np.argmin(f_stitched)
            g_min_idx = np.argmin(g_stitched)
            f_min_val = f_stitched[f_min_idx]
            g_min_val = g_stitched[g_min_idx]

            # After minimum, ALL values become the minimum (flat plateau)
            f_stitched[f_min_idx:] = f_min_val
            g_stitched[g_min_idx:] = g_min_val

            # Extend to 100% strain with hold extrapolation
            max_data_strain = grid_strain[-1]
            original_len = len(grid_strain)  # Store original length before extension
            if max_data_strain < 1.0:
                # Add points up to 100% strain holding the last value
                extend_strains = np.array([0.5, 0.7, 1.0])
                extend_strains = extend_strains[extend_strains > max_data_strain]
                if len(extend_strains) > 0:
                    grid_strain = np.concatenate([grid_strain, extend_strains])
                    f_stitched = np.concatenate([f_stitched, np.full(len(extend_strains), f_stitched[-1])])
                    g_stitched = np.concatenate([g_stitched, np.full(len(extend_strains), g_stitched[-1])])
                    n_eff_stitched = np.concatenate([n_eff_stitched, np.full(len(extend_strains), n_eff_stitched[-1])])

            # Store piecewise result
            self.piecewise_result = {
                'strain': grid_strain.copy(),
                'strain_original_len': original_len,  # For plotting Group A/B
                'f_avg': f_stitched,
                'g_avg': g_stitched,
                'n_eff': n_eff_stitched,
                'split': split_strain,
                'temps_A': temps_A,
                'temps_B': temps_B,
                'result_A': result_A,
                'result_B': result_B
            }

            # Also set as main averaged result for mu_visc calculation
            self.fg_averaged = {
                'strain': grid_strain.copy(),
                'f_avg': f_stitched,
                'g_avg': g_stitched,
                'Ts_used': list(set(temps_A + temps_B)),
                'n_eff': n_eff_stitched
            }

            # Create interpolators with 'hold' extrapolation to avoid NaN at edges
            self.f_interpolator, self.g_interpolator = create_fg_interpolator(
                grid_strain, f_stitched, g_stitched,
                interp_kind='loglog_linear', extrap_mode='hold'
            )

            # Update plot
            self._update_fg_plot_piecewise()

            self.status_var.set(
                f"Piecewise 평균화 완료: Split={split_percent:.1f}%, "
                f"A={len(temps_A)}개, B={len(temps_B)}개 온도"
            )

        except Exception as e:
            messagebox.showerror("오류", f"Piecewise 평균화 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_fg_plot_piecewise(self):
        """Update f,g curves plot with piecewise averaging visualization."""
        self.ax_fg_curves.clear()
        self.ax_fg_curves.set_title('f(ε), g(ε) 곡선 (Piecewise 평균화)', fontweight='bold')
        self.ax_fg_curves.set_xlabel('변형률 ε (fraction)')
        self.ax_fg_curves.set_ylabel('보정 계수')
        self.ax_fg_curves.grid(True, alpha=0.3)

        # Plot individual temperature curves (thin, low alpha)
        if self.fg_by_T is not None:
            for T, data in self.fg_by_T.items():
                s = data['strain']
                f = data['f']
                g = data['g']
                self.ax_fg_curves.plot(s, f, 'b-', alpha=0.15, linewidth=0.8)
                self.ax_fg_curves.plot(s, g, 'r-', alpha=0.15, linewidth=0.8)

        # Plot piecewise results
        if self.piecewise_result is not None:
            s = self.piecewise_result['strain']
            split = self.piecewise_result['split']

            # Stitched (final) result only - Group A/B removed
            f_final = self.piecewise_result['f_avg']
            g_final = self.piecewise_result['g_avg']
            self.ax_fg_curves.plot(s, f_final, 'b-', linewidth=3.5, label='STITCHED f(ε)')
            self.ax_fg_curves.plot(s, g_final, 'r-', linewidth=3.5, label='STITCHED g(ε)')

            # Split line
            self.ax_fg_curves.axvline(split, color='green', linewidth=2, linestyle=':', alpha=0.8,
                                      label=f'Split @ {split*100:.1f}%')

        elif self.fg_averaged is not None:
            # Fallback to simple averaged plot
            s = self.fg_averaged['strain']
            f_avg = self.fg_averaged['f_avg']
            g_avg = self.fg_averaged['g_avg']
            self.ax_fg_curves.plot(s, f_avg, 'b-', linewidth=3, label='f(ε) 평균')
            self.ax_fg_curves.plot(s, g_avg, 'r-', linewidth=3, label='g(ε) 평균')

        self.ax_fg_curves.set_xlim(0, 1.0)  # Always show up to 100% strain
        self.ax_fg_curves.set_ylim(0, 1.1)
        self.ax_fg_curves.legend(loc='upper right', fontsize=7, ncol=2)

        self.canvas_mu_visc.draw()

    def _export_fg_curves(self):
        """Export f,g curves to CSV file."""
        if self.fg_averaged is None and self.piecewise_result is None:
            messagebox.showwarning("경고", "먼저 f,g 곡선을 계산하세요.")
            return

        result = self.piecewise_result if self.piecewise_result is not None else self.fg_averaged

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="fg_curves.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        try:
            import csv

            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['# f,g 곡선 데이터'])
                if self.piecewise_result is not None:
                    split = self.piecewise_result['split']
                    writer.writerow([f'# Split Strain: {split*100:.2f}%'])
                    writer.writerow([f'# Group A temps: {self.piecewise_result["temps_A"]}'])
                    writer.writerow([f'# Group B temps: {self.piecewise_result["temps_B"]}'])
                writer.writerow(['strain_fraction', 'f_value', 'g_value', 'n_eff'])

                for i in range(len(result['strain'])):
                    writer.writerow([
                        f'{result["strain"][i]:.6e}',
                        f'{result["f_avg"][i]:.6f}',
                        f'{result["g_avg"][i]:.6f}',
                        f'{result["n_eff"][i]:.0f}'
                    ])

            messagebox.showinfo("성공", f"f,g 곡선 저장 완료:\n{filename}")
            self.status_var.set(f"f,g 곡선 저장: {filename}")

        except Exception as e:
            messagebox.showerror("오류", f"저장 실패:\n{str(e)}")

    def _load_strain_sweep_data(self):
        """Load strain sweep data from file."""
        filename = filedialog.askopenfilename(
            title="Strain Sweep 파일 선택",
            filetypes=[
                ("All supported", "*.txt *.csv *.xlsx *.xls"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            self.strain_data = load_strain_sweep_file(filename)

            if not self.strain_data:
                messagebox.showerror("오류", "유효한 데이터를 찾을 수 없습니다.")
                return

            # Update label
            self.strain_file_label.config(
                text=f"{os.path.basename(filename)} ({len(self.strain_data)}개 온도)"
            )

            # Populate temperature listboxes (Group A and B)
            temps = sorted(self.strain_data.keys())

            # Clear and populate Group A
            self.temp_listbox_A.delete(0, tk.END)
            for T in temps:
                self.temp_listbox_A.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox_A.selection_set(tk.END)

            # Clear and populate Group B
            self.temp_listbox_B.delete(0, tk.END)
            for T in temps:
                self.temp_listbox_B.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox_B.selection_set(tk.END)

            # Also update legacy listbox for compatibility
            self.temp_listbox.delete(0, tk.END)
            for T in temps:
                self.temp_listbox.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox.selection_set(tk.END)

            self.status_var.set(f"Strain 데이터 로드 완료: {len(self.strain_data)}개 온도")
            messagebox.showinfo("성공", f"Strain 데이터 로드 완료\n온도 블록: {len(self.strain_data)}개")

        except Exception as e:
            messagebox.showerror("오류", f"Strain 데이터 로드 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _load_fg_curve_data(self):
        """Load pre-computed f,g curve data from file."""
        filename = filedialog.askopenfilename(
            title="f,g 곡선 파일 선택",
            filetypes=[
                ("Text/CSV files", "*.txt *.csv *.dat"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            fg_data = load_fg_curve_file(
                filename,
                strain_is_percent=self.strain_is_percent_var.get()
            )

            if fg_data is None:
                messagebox.showerror("오류", "유효한 f,g 데이터를 찾을 수 없습니다.")
                return

            # Create interpolators
            self.f_interpolator, self.g_interpolator = create_fg_interpolator(
                fg_data['strain'],
                fg_data['f'],
                fg_data['g'] if fg_data['g'] is not None else fg_data['f']
            )

            # Store for plotting
            self.fg_averaged = {
                'strain': fg_data['strain'],
                'f_avg': fg_data['f'],
                'g_avg': fg_data['g'] if fg_data['g'] is not None else fg_data['f']
            }

            # Update label
            self.fg_file_label.config(text=os.path.basename(filename))

            # Update plot
            self._update_fg_plot()

            self.status_var.set(f"f,g 곡선 로드 완료: {len(fg_data['strain'])}개 점")
            messagebox.showinfo("성공", f"f,g 곡선 로드 완료\n데이터 포인트: {len(fg_data['strain'])}개")

        except Exception as e:
            messagebox.showerror("오류", f"f,g 곡선 로드 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _compute_fg_curves(self):
        """Compute f,g curves from strain sweep data."""
        if self.strain_data is None:
            messagebox.showwarning("경고", "먼저 Strain 데이터를 로드하세요.")
            return

        try:
            target_freq = float(self.fg_target_freq_var.get())
            e0_points = int(self.e0_points_var.get())
            strain_is_percent = self.strain_is_percent_var.get()
            clip_fg = self.clip_fg_var.get()

            # Compute f,g curves
            self.fg_by_T = compute_fg_from_strain_sweep(
                self.strain_data,
                target_freq=target_freq,
                freq_mode='nearest',
                strain_is_percent=strain_is_percent,
                e0_n_points=e0_points,
                clip_leq_1=clip_fg
            )

            if not self.fg_by_T:
                messagebox.showerror("오류", "f,g 계산 실패: 유효한 데이터가 없습니다.")
                return

            temps = sorted(self.fg_by_T.keys())

            # Update Group A listbox
            self.temp_listbox_A.delete(0, tk.END)
            for T in temps:
                self.temp_listbox_A.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox_A.selection_set(tk.END)

            # Update Group B listbox
            self.temp_listbox_B.delete(0, tk.END)
            for T in temps:
                self.temp_listbox_B.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox_B.selection_set(tk.END)

            # Update legacy temperature listbox (for compatibility)
            self.temp_listbox.delete(0, tk.END)
            for T in temps:
                self.temp_listbox.insert(tk.END, f"{T:.2f} °C")
                self.temp_listbox.selection_set(tk.END)

            # Plot individual curves
            self._update_fg_plot()

            self.status_var.set(f"f,g 곡선 계산 완료: {len(self.fg_by_T)}개 온도")

        except Exception as e:
            messagebox.showerror("오류", f"f,g 계산 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _average_fg_curves(self):
        """Average f,g curves from selected temperatures."""
        if self.fg_by_T is None:
            messagebox.showwarning("경고", "먼저 f,g 곡선을 계산하세요.")
            return

        try:
            # Get selected temperatures
            selections = self.temp_listbox.curselection()
            if not selections:
                messagebox.showwarning("경고", "최소 1개의 온도를 선택하세요.")
                return

            temps = sorted(self.fg_by_T.keys())
            selected_temps = [temps[i] for i in selections]

            # Create strain grid
            max_strain = max(
                np.max(self.fg_by_T[T]['strain']) for T in selected_temps
            )
            grid_strain = create_strain_grid(30, max_strain, use_persson_grid=True)

            # Average curves
            self.fg_averaged = average_fg_curves(
                self.fg_by_T,
                selected_temps,
                grid_strain,
                interp_kind='loglog_linear',
                avg_mode='mean',
                clip_leq_1=self.clip_fg_var.get()
            )

            if self.fg_averaged is None:
                messagebox.showerror("오류", "평균화 실패")
                return

            # Create interpolators
            self.f_interpolator, self.g_interpolator = create_fg_interpolator(
                self.fg_averaged['strain'],
                self.fg_averaged['f_avg'],
                self.fg_averaged['g_avg']
            )

            # Update plot
            self._update_fg_plot()

            self.status_var.set(f"f,g 평균화 완료: {len(selected_temps)}개 온도 사용")

        except Exception as e:
            messagebox.showerror("오류", f"평균화 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_fg_plot(self):
        """Update f,g curves plot."""
        self.ax_fg_curves.clear()
        self.ax_fg_curves.set_title('f(ε), g(ε) 곡선', fontweight='bold')
        self.ax_fg_curves.set_xlabel('변형률 ε (fraction)')
        self.ax_fg_curves.set_ylabel('보정 계수')
        self.ax_fg_curves.grid(True, alpha=0.3)

        # Plot individual temperature curves
        if self.fg_by_T is not None:
            for T, data in self.fg_by_T.items():
                s = data['strain']
                f = data['f']
                g = data['g']
                self.ax_fg_curves.plot(s, f, 'b-', alpha=0.3, linewidth=1)
                self.ax_fg_curves.plot(s, g, 'r-', alpha=0.3, linewidth=1)

        # Plot averaged curves
        if self.fg_averaged is not None:
            s = self.fg_averaged['strain']
            f_avg = self.fg_averaged['f_avg']
            g_avg = self.fg_averaged['g_avg']
            self.ax_fg_curves.plot(s, f_avg, 'b-', linewidth=3, label='f(ε) 평균')
            self.ax_fg_curves.plot(s, g_avg, 'r-', linewidth=3, label='g(ε) 평균')
            self.ax_fg_curves.legend(loc='upper right')

        self.ax_fg_curves.set_xlim(left=0)
        self.ax_fg_curves.set_ylim(0, 1.1)

        self.canvas_mu_visc.draw()

    def _calculate_mu_visc(self):
        """Calculate viscoelastic friction coefficient mu_visc.

        Implements the full Persson formula:
        μ_visc = (1/2) ∫[q0→q1] dq · q³ · C(q) · P(q) · S(q)
                 · ∫[0→2π] dφ · cosφ · Im[E(qv·cosφ, T)] / ((1-ν²)σ₀)

        where:
        - P(q) = erf(1/(2√G(q))) : contact area ratio
        - S(q) = γ + (1-γ)P(q)² : contact correction factor
        - Im[E(ω,T)] : loss modulus (optionally corrected by g(strain))
        """
        if self.material is None or self.psd_model is None:
            messagebox.showwarning("경고", "먼저 재료(DMA)와 PSD 데이터를 로드하세요.")
            return

        if not self.results or '2d_results' not in self.results:
            messagebox.showwarning("경고", "먼저 G(q,v) 계산을 실행하세요 (탭 2).")
            return

        try:
            self.status_var.set("μ_visc 계산 중...")
            self.mu_calc_button.config(state='disabled')
            self.mu_progress_var.set(0)  # Initialize progress bar
            self.root.update()

            # Get parameters
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            temperature = float(self.temperature_var.get())
            poisson = float(self.poisson_var.get())
            gamma = float(self.gamma_var.get())
            n_phi = int(self.n_phi_var.get())
            use_fg = self.use_fg_correction_var.get()
            strain_est_method = self.strain_est_method_var.get()
            fixed_strain = float(self.fixed_strain_var.get()) / 100.0  # Convert % to fraction

            # Check h'rms data if using rms_slope method
            if strain_est_method == 'rms_slope':
                if self.rms_slope_calculator is None or self.rms_slope_profiles is None:
                    messagebox.showwarning("경고",
                        "h'rms 데이터가 없습니다.\n\n"
                        "Tab 4 (h'rms/Local Strain)에서\n"
                        "'h'rms slope / Local Strain 계산' 버튼을 먼저 실행하세요.")
                    self.mu_calc_button.config(state='normal')
                    return
                else:
                    # Show info about strain range being used
                    strain_min = self.rms_slope_profiles['strain'][0]
                    strain_max = self.rms_slope_profiles['strain'][-1]
                    self.status_var.set(f"h'rms 기반 strain 적용: {strain_min*100:.3f}% ~ {strain_max*100:.1f}%")
                    self.root.update()

            # Get G(q,v) results
            results_2d = self.results['2d_results']
            q = results_2d['q']
            v = results_2d['v']
            G_matrix = results_2d['G_matrix']

            # Get PSD values
            C_q = self.psd_model(q)

            # Precompute E' for strain estimation (using mid-frequency)
            omega_mid = 2 * np.pi * 1.0  # 1 Hz
            E_prime_ref = self.material.get_storage_modulus(omega_mid, temperature=temperature)

            # Prepare RMS slope strain interpolator if using that method
            rms_strain_interp = None
            if strain_est_method == 'rms_slope' and self.rms_slope_calculator is not None:
                from scipy.interpolate import interp1d
                rms_q = self.rms_slope_profiles['q']
                rms_strain = self.rms_slope_profiles['strain']
                # Use log-log interpolation for better accuracy
                log_q = np.log10(rms_q)
                log_strain = np.log10(np.maximum(rms_strain, 1e-10))
                rms_strain_interp = interp1d(log_q, log_strain, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')

            # Create enhanced loss modulus function with strain-dependent correction
            def loss_modulus_func_enhanced(omega, T, q_val=None, G_val=None, C_val=None):
                """Loss modulus with optional nonlinear strain correction."""
                E_loss = self.material.get_loss_modulus(omega, temperature=T)

                if use_fg and self.g_interpolator is not None:
                    # Estimate local strain based on method
                    if strain_est_method == 'rms_slope' and rms_strain_interp is not None and q_val is not None:
                        # Use pre-calculated RMS slope based local strain
                        try:
                            strain_estimate = 10 ** rms_strain_interp(np.log10(q_val))
                            strain_estimate = np.clip(strain_estimate, 0.0, 1.0)
                        except:
                            strain_estimate = fixed_strain
                    elif strain_est_method == 'fixed':
                        strain_estimate = fixed_strain
                    elif strain_est_method == 'persson' and q_val is not None and C_val is not None:
                        # Persson's approach: strain ~ sqrt(C(q)*q^4) * sigma0/E'
                        from persson_model.core.friction import estimate_local_strain
                        E_prime = self.material.get_storage_modulus(omega, temperature=T)
                        strain_estimate = estimate_local_strain(
                            G_val if G_val is not None else 0.1,
                            C_val, q_val, sigma_0, E_prime, method='persson'
                        )
                    elif strain_est_method == 'simple' and G_val is not None:
                        # Simple estimate: strain ~ sqrt(G) * sigma0/E
                        E_prime = self.material.get_storage_modulus(omega, temperature=T)
                        strain_estimate = np.sqrt(max(G_val, 1e-10)) * sigma_0 / max(E_prime, 1e3)
                        strain_estimate = np.clip(strain_estimate, 0.0, 1.0)
                    else:
                        strain_estimate = fixed_strain

                    # Get g correction factor
                    g_val = self.g_interpolator(strain_estimate)
                    g_val = np.clip(g_val, 0.0, 1.0)
                    E_loss = E_loss * g_val

                return E_loss

            # Simple wrapper for FrictionCalculator compatibility
            def loss_modulus_func(omega, T):
                return self.material.get_loss_modulus(omega, temperature=T)

            # Set up g_interpolator for nonlinear correction
            g_interp = self.g_interpolator if use_fg else None

            # Create strain estimator function based on method
            def strain_estimator_func(q_arr, G_arr, velocity):
                """Return strain array for given q values."""
                n = len(q_arr)
                if strain_est_method == 'rms_slope' and rms_strain_interp is not None:
                    # Use pre-calculated RMS slope based local strain
                    strain_arr = np.zeros(n)
                    for i, qi in enumerate(q_arr):
                        try:
                            strain_arr[i] = 10 ** rms_strain_interp(np.log10(qi))
                        except:
                            strain_arr[i] = fixed_strain
                    return np.clip(strain_arr, 0.0, 1.0)
                elif strain_est_method == 'fixed':
                    return np.full(n, fixed_strain)
                elif strain_est_method == 'persson':
                    from persson_model.core.friction import estimate_local_strain
                    strain_arr = np.zeros(n)
                    C_q_local = self.psd_model(q_arr)
                    for i, (qi, Gi, Ci) in enumerate(zip(q_arr, G_arr, C_q_local)):
                        omega_i = qi * velocity
                        E_prime = self.material.get_storage_modulus(omega_i, temperature=temperature)
                        strain_arr[i] = estimate_local_strain(Gi, Ci, qi, sigma_0, E_prime, method='persson')
                    return np.clip(strain_arr, 0.0, 1.0)
                elif strain_est_method == 'simple':
                    strain_arr = np.sqrt(np.maximum(G_arr, 1e-10)) * sigma_0 / max(E_prime_ref, 1e3)
                    return np.clip(strain_arr, 0.0, 1.0)
                else:
                    return np.full(n, fixed_strain)

            # Apply nonlinear correction to G(q) if enabled
            # Recalculate G with f(ε), g(ε) applied INSIDE the integral:
            # G(q) = (1/8) ∫∫ q'³ C(q') |E_eff(q'v cosφ)|² / ((1-ν²)σ₀)² dφ dq'
            # where |E_eff|² = (E'×f(ε))² + (E''×g(ε))²
            G_matrix_corrected = G_matrix.copy()

            if use_fg and self.f_interpolator is not None and self.g_interpolator is not None:
                self.status_var.set("비선형 G(q) 재계산 중 (적분 내 보정)...")
                self.root.update()

                # Get strain array for nonlinear correction
                if strain_est_method == 'rms_slope' and rms_strain_interp is not None:
                    strain_for_G = np.zeros(len(q))
                    for i, qi in enumerate(q):
                        try:
                            strain_for_G[i] = 10 ** rms_strain_interp(np.log10(qi))
                        except:
                            strain_for_G[i] = fixed_strain
                    strain_for_G = np.clip(strain_for_G, 0.0, 1.0)
                else:
                    strain_for_G = np.full(len(q), fixed_strain)

                # Set nonlinear correction on g_calculator
                # This applies f(ε), g(ε) INSIDE the angle integral
                self.g_calculator.storage_modulus_func = lambda w: self.material.get_storage_modulus(w, temperature=temperature)
                self.g_calculator.loss_modulus_func = lambda w: self.material.get_loss_modulus(w, temperature=temperature)
                self.g_calculator.set_nonlinear_correction(
                    f_interpolator=self.f_interpolator,
                    g_interpolator=self.g_interpolator,
                    strain_array=strain_for_G,
                    strain_q_array=q
                )

                # Recalculate G(q,v) with nonlinear correction inside integral
                q_min = float(self.q_min_var.get())
                for j, v_j in enumerate(v):
                    self.g_calculator.velocity = v_j
                    results_nl = self.g_calculator.calculate_G_with_details(q, q_min=q_min)
                    G_matrix_corrected[:, j] = results_nl['G']

                    # Progress update
                    if j % max(1, len(v) // 10) == 0:
                        progress = int((j + 1) / len(v) * 50)
                        self.mu_progress_var.set(progress)
                        self.root.update()

                # Clear nonlinear correction after calculation
                self.g_calculator.clear_nonlinear_correction()

                self.status_var.set("비선형 G(q) 재계산 완료 - μ_visc 계산 중...")
                self.root.update()

            # Create friction calculator with g_interpolator
            friction_calc = FrictionCalculator(
                psd_func=self.psd_model,
                loss_modulus_func=loss_modulus_func,
                sigma_0=sigma_0,
                velocity=v[0],
                temperature=temperature,
                poisson_ratio=poisson,
                gamma=gamma,
                n_angle_points=n_phi,
                g_interpolator=g_interp,
                strain_estimate=fixed_strain
            )

            # Calculate mu_visc for all velocities
            # Scale progress to 50-100% if nonlinear correction was applied (Stage 1 used 0-50%)
            def progress_callback(percent):
                if use_fg and self.f_interpolator is not None and self.g_interpolator is not None:
                    # Stage 2: scale 0-100% to 50-100%
                    scaled_percent = 50 + int(percent * 0.5)
                else:
                    # No Stage 1, so use full 0-100%
                    scaled_percent = percent
                self.mu_progress_var.set(scaled_percent)
                self.root.update()

            # Use strain_estimator if nonlinear correction is enabled
            strain_est = strain_estimator_func if use_fg else None

            # Use corrected G_matrix (will be same as original if nonlinear not applied)
            mu_array_raw, details = friction_calc.calculate_mu_visc_multi_velocity(
                q, G_matrix_corrected, v, C_q, progress_callback, strain_estimator=strain_est
            )

            # Apply smoothing if enabled
            smooth_mu = self.smooth_mu_var.get()
            if smooth_mu and len(mu_array_raw) >= 5:
                window = int(self.smooth_window_var.get())
                # Ensure window is odd and not larger than array
                window = min(window, len(mu_array_raw))
                if window % 2 == 0:
                    window -= 1
                window = max(3, window)

                # Apply Savitzky-Golay filter for smoothing
                mu_array = savgol_filter(mu_array_raw, window, 2)
            else:
                mu_array = mu_array_raw

            # Store results (both raw and smoothed)
            self.mu_visc_results = {
                'v': v,
                'mu': mu_array,
                'mu_raw': mu_array_raw,
                'details': details,
                'smoothed': smooth_mu
            }

            # Update plots
            self._update_mu_visc_plots(v, mu_array, details, use_nonlinear=use_fg)

            # Update result text with detailed information
            self.mu_result_text.delete(1.0, tk.END)
            self.mu_result_text.insert(tk.END, "=" * 40 + "\n")
            self.mu_result_text.insert(tk.END, "μ_visc 계산 결과 (Persson 이론)\n")
            self.mu_result_text.insert(tk.END, "=" * 40 + "\n\n")

            # Parameters used
            self.mu_result_text.insert(tk.END, "[계산 파라미터]\n")
            self.mu_result_text.insert(tk.END, f"  σ₀ (공칭 압력): {sigma_0/1e6:.3f} MPa\n")
            self.mu_result_text.insert(tk.END, f"  T (온도): {temperature:.1f} °C\n")
            self.mu_result_text.insert(tk.END, f"  ν (푸아송비): {poisson:.2f}\n")
            self.mu_result_text.insert(tk.END, f"  γ (접촉 보정): {gamma:.2f}\n")
            self.mu_result_text.insert(tk.END, f"  각도 적분점: {n_phi}\n")

            # Smoothing info
            if smooth_mu:
                self.mu_result_text.insert(tk.END, f"  결과 스무딩: 적용 (윈도우={self.smooth_window_var.get()})\n")
            else:
                self.mu_result_text.insert(tk.END, "  결과 스무딩: 미적용\n")

            # f,g correction info - more prominent
            self.mu_result_text.insert(tk.END, "\n[비선형 보정]\n")
            if use_fg and self.f_interpolator is not None and self.g_interpolator is not None:
                self.mu_result_text.insert(tk.END, f"  상태: *** 적용됨 ***\n")
                self.mu_result_text.insert(tk.END, f"  Strain 추정: {strain_est_method}\n")
                if strain_est_method == 'fixed':
                    self.mu_result_text.insert(tk.END, f"  고정 Strain: {fixed_strain*100:.2f}%\n")
                if self.piecewise_result is not None:
                    split = self.piecewise_result['split']
                    self.mu_result_text.insert(tk.END, f"  Piecewise Split: {split*100:.1f}%\n")
                self.mu_result_text.insert(tk.END, "\n  [보정 적용 항목]\n")
                self.mu_result_text.insert(tk.END, "  • E'(ω) → E'(ω) × f(ε)  (저장탄성률)\n")
                self.mu_result_text.insert(tk.END, "  • E''(ω) → E''(ω) × g(ε) (손실탄성률)\n")
                self.mu_result_text.insert(tk.END, "  • |E*|² → (E'×f)² + (E''×g)²\n")
                self.mu_result_text.insert(tk.END, "  • G(q) → G(q) × |E*_eff|²/|E*_lin|²\n")
                self.mu_result_text.insert(tk.END, "  • P(q) = erf(1/(2√G_eff)) : 비선형 G(q) 기반\n")
                self.mu_result_text.insert(tk.END, "  • S(q) = γ + (1-γ)P(q)² : 비선형 P(q) 기반\n")
            elif use_fg and self.g_interpolator is not None:
                self.mu_result_text.insert(tk.END, f"  상태: *** 부분 적용 (g만) ***\n")
                self.mu_result_text.insert(tk.END, "  ※ f 곡선이 없어 손실탄성률만 보정됨\n")
            else:
                self.mu_result_text.insert(tk.END, "  상태: 미적용 (선형 계산)\n")
                if self.g_interpolator is None:
                    self.mu_result_text.insert(tk.END, "  ※ f,g 곡선이 없음\n")

            # Helper for smart formatting
            def smart_fmt(val, threshold=0.001):
                if abs(val) < threshold and val != 0:
                    return f'{val:.2e}'
                return f'{val:.4f}'

            self.mu_result_text.insert(tk.END, "\n[결과]\n")
            self.mu_result_text.insert(tk.END, f"  속도: {v[0]:.2e} ~ {v[-1]:.2e} m/s\n")
            mu_min, mu_max = np.min(mu_array), np.max(mu_array)
            self.mu_result_text.insert(tk.END, f"  μ_visc: {smart_fmt(mu_min)} ~ {smart_fmt(mu_max)}\n")

            # Find peak
            peak_idx = np.argmax(mu_array)
            peak_mu = mu_array[peak_idx]
            self.mu_result_text.insert(tk.END, f"  최대: μ={smart_fmt(peak_mu)} @ v={v[peak_idx]:.4f} m/s\n")

            # Show comprehensive diagnostic info
            if details and 'details' in details and len(details['details']) > 0:
                self.mu_result_text.insert(tk.END, f"\n[진단 정보 - 중간 속도]\n")
                mid_detail = details['details'][len(details['details']) // 2]
                mid_v = mid_detail.get('velocity', v[len(v)//2])
                self.mu_result_text.insert(tk.END, f"  속도: {mid_v:.2e} m/s\n")

                # G(q) values
                if 'G' in mid_detail:
                    G = mid_detail['G']
                    self.mu_result_text.insert(tk.END, f"  G(q) 범위: {np.min(G):.2e} ~ {np.max(G):.2e}\n")

                # P(q) - contact area ratio (critical for mu)
                if 'P' in mid_detail:
                    P = mid_detail['P']
                    P_mean = np.mean(P)
                    P_min, P_max = np.min(P), np.max(P)
                    self.mu_result_text.insert(tk.END, f"  P(q) 범위: {P_min:.4f} ~ {P_max:.4f} (평균: {P_mean:.4f})\n")
                    if P_max < 0.1:
                        self.mu_result_text.insert(tk.END, f"  ※ 경고: P(q)가 매우 작음 - G(q)가 너무 클 수 있음\n")
                        self.mu_result_text.insert(tk.END, f"     → σ₀를 높이거나 표면 거칠기를 확인하세요\n")

                # S(q) - contact correction factor
                if 'S' in mid_detail:
                    S = mid_detail['S']
                    self.mu_result_text.insert(tk.END, f"  S(q) 범위: {np.min(S):.4f} ~ {np.max(S):.4f}\n")

                # Angle integral values
                if 'angle_integral' in mid_detail:
                    angle_int = mid_detail['angle_integral']
                    self.mu_result_text.insert(tk.END, f"  각도적분 범위: {np.min(angle_int):.2e} ~ {np.max(angle_int):.2e}\n")

                # Full integrand
                if 'integrand' in mid_detail:
                    integ = mid_detail['integrand']
                    self.mu_result_text.insert(tk.END, f"  피적분함수 max: {np.max(integ):.2e}\n")

                # q³C(q)P(q)S(q) term
                if 'q3CPS' in mid_detail:
                    q3CPS = mid_detail['q3CPS']
                    self.mu_result_text.insert(tk.END, f"  q³C(q)P(q)S(q) max: {np.max(q3CPS):.2e}\n")

            self.mu_result_text.insert(tk.END, "\n[속도별 μ_visc]\n")
            step = max(1, len(v) // 8)
            for i in range(0, len(v), step):
                self.mu_result_text.insert(tk.END, f"  v={v[i]:.2e}: μ={smart_fmt(mu_array[i])}\n")

            self.status_var.set("μ_visc 계산 완료")
            self.mu_calc_button.config(state='normal')

            messagebox.showinfo("성공", f"μ_visc 계산 완료\n"
                               f"범위: {smart_fmt(mu_min)} ~ {smart_fmt(mu_max)}\n"
                               f"최대: μ={smart_fmt(peak_mu)} @ v={v[peak_idx]:.4f} m/s")

        except Exception as e:
            self.mu_calc_button.config(state='normal')
            messagebox.showerror("오류", f"μ_visc 계산 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _update_mu_visc_plots(self, v, mu_array, details, use_nonlinear=False):
        """Update mu_visc plots."""
        try:
            # Sanitize input arrays - replace NaN/Inf with safe values
            mu_array = np.nan_to_num(mu_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Clear all subplots
            self.ax_mu_v.clear()
            self.ax_mu_cumulative.clear()

            # Remove any existing twin axes from ax_ps
            for ax in self.fig_mu_visc.axes:
                if ax is not self.ax_fg_curves and ax is not self.ax_mu_v and \
                   ax is not self.ax_mu_cumulative and ax is not self.ax_ps:
                    ax.remove()
            self.ax_ps.clear()

            # Helper function for smart formatting
            def smart_format(val, threshold=0.001):
                if abs(val) < threshold and val != 0:
                    return f'{val:.2e}'
                return f'{val:.4f}'

            # Plot 1: mu_visc vs velocity (handle NaN values)
            valid_mask = np.isfinite(mu_array)
            if np.any(valid_mask):
                self.ax_mu_v.semilogx(v[valid_mask], mu_array[valid_mask], 'b-', linewidth=2.5, marker='o', markersize=4)
            else:
                self.ax_mu_v.semilogx(v, np.zeros_like(v), 'b-', linewidth=2.5, marker='o', markersize=4)
            self.ax_mu_v.set_title('μ_visc(v) 곡선', fontweight='bold')
            self.ax_mu_v.set_xlabel('속도 v (m/s)')
            self.ax_mu_v.set_ylabel('마찰 계수 μ_visc')
            self.ax_mu_v.grid(True, alpha=0.3)

            # Find peak (handle NaN values)
            mu_for_peak = np.where(np.isfinite(mu_array), mu_array, -np.inf)
            peak_idx = np.argmax(mu_for_peak)
            peak_mu = mu_array[peak_idx] if np.isfinite(mu_array[peak_idx]) else 0.0
            peak_v = v[peak_idx]
            self.ax_mu_v.plot(peak_v, peak_mu, 'r*', markersize=15,
                             label=f'최대값: μ={smart_format(peak_mu)} @ v={peak_v:.4f} m/s')

            # Find and mark μ at v=1 m/s (important reference point)
            if np.min(v) <= 1.0 <= np.max(v):
                from scipy.interpolate import interp1d
                # Interpolate to find μ at exactly 1 m/s
                valid_for_interp = np.isfinite(mu_array)
                if np.sum(valid_for_interp) >= 2:
                    f_interp = interp1d(np.log10(v[valid_for_interp]), mu_array[valid_for_interp],
                                       kind='linear', fill_value='extrapolate')
                    mu_at_1ms = float(f_interp(0))  # log10(1) = 0
                    self.ax_mu_v.plot(1.0, mu_at_1ms, 'go', markersize=12, markeredgecolor='black',
                                     markeredgewidth=1.5, zorder=10,
                                     label=f'v=1m/s: μ={smart_format(mu_at_1ms)}')
                    # Add vertical line at v=1 m/s
                    self.ax_mu_v.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1)

            self.ax_mu_v.legend(loc='upper left', fontsize=7)

            # Plot 2: Real Contact Area Ratio A/A₀ = P(q_max) vs velocity
            P_qmax_array = np.zeros(len(v))

            for i, det in enumerate(details['details']):
                P = det.get('P', np.zeros(1))
                P_qmax_array[i] = P[-1] if len(P) > 0 else 0

            # Sanitize P_qmax_array
            P_qmax_array = np.nan_to_num(P_qmax_array, nan=0.0, posinf=1.0, neginf=0.0)

            # Color based on nonlinear correction
            if use_nonlinear:
                label_str = 'A/A₀ - 비선형 G(q)'
                color = 'r'
                title_suffix = ' (f,g 보정 적용)'
            else:
                label_str = 'A/A₀ - 선형 G(q)'
                color = 'b'
                title_suffix = ''

            # Plot A/A₀ = P(q_max)
            self.ax_mu_cumulative.semilogx(v, P_qmax_array, f'{color}-', linewidth=2,
                                            marker='s', markersize=4, label=label_str)

            self.ax_mu_cumulative.set_title(f'실접촉 면적비율 A/A₀{title_suffix}', fontweight='bold', fontsize=8)
            self.ax_mu_cumulative.set_xlabel('속도 v (m/s)')
            self.ax_mu_cumulative.set_ylabel('A/A₀ = P(q_max)')
            self.ax_mu_cumulative.legend(loc='best', fontsize=7)
            self.ax_mu_cumulative.grid(True, alpha=0.3)

            # Set y-axis to show data with padding
            y_max = max(np.max(P_qmax_array) * 1.2, 0.05)
            if not np.isfinite(y_max):
                y_max = 1.0
            self.ax_mu_cumulative.set_ylim(0, y_max)

            # Plot 3: P(q), S(q) for middle velocity
            mid_idx = len(details['details']) // 2
            detail = details['details'][mid_idx]
            q = detail['q']
            P = detail['P']
            S = detail['S']
            cumulative = detail.get('cumulative_mu', np.zeros_like(q))

            # Handle NaN values in P, S, cumulative
            P = np.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
            S = np.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0)
            cumulative = np.nan_to_num(cumulative, nan=0.0, posinf=0.0, neginf=0.0)

            # Use twin axis for cumulative
            ax_twin = self.ax_ps.twinx()

            self.ax_ps.semilogx(q, P, 'b-', linewidth=1.5, label='P(q)')
            self.ax_ps.semilogx(q, S, 'r--', linewidth=1.5, label='S(q)')
            ax_twin.semilogx(q, cumulative, 'g-', linewidth=1.5, alpha=0.7, label='누적μ')

            self.ax_ps.set_title('P(q), S(q) / 누적 μ', fontweight='bold', fontsize=9)
            self.ax_ps.set_xlabel('파수 q (1/m)')
            self.ax_ps.set_ylabel('P(q), S(q)', color='blue')
            ax_twin.set_ylabel('누적 μ', color='green')
            self.ax_ps.legend(loc='upper left', fontsize=7)
            ax_twin.legend(loc='upper right', fontsize=7)
            self.ax_ps.grid(True, alpha=0.3)
            self.ax_ps.set_ylim(0, 1.1)

            # Set twin axis limits safely
            cumulative_max = np.max(cumulative) if len(cumulative) > 0 else 0.1
            if not np.isfinite(cumulative_max) or cumulative_max <= 0:
                cumulative_max = 0.1
            ax_twin.set_ylim(0, cumulative_max * 1.2)

            self.fig_mu_visc.tight_layout()
            self.canvas_mu_visc.draw()

        except Exception as e:
            # Fallback: clear plots and show error message
            print(f"Plot update error: {e}")
            import traceback
            traceback.print_exc()
            self.ax_mu_v.clear()
            self.ax_mu_cumulative.clear()
            self.ax_ps.clear()
            self.ax_mu_v.text(0.5, 0.5, f'플롯 오류: {str(e)[:50]}',
                             ha='center', va='center', transform=self.ax_mu_v.transAxes)
            self.canvas_mu_visc.draw()

    def _export_mu_visc_results(self):
        """Export mu_visc results to CSV files with selection dialog."""
        if self.mu_visc_results is None:
            messagebox.showwarning("경고", "먼저 μ_visc를 계산하세요.")
            return

        # Create dialog for selecting data to export
        dialog = tk.Toplevel(self.root)
        dialog.title("CSV 내보내기 - μ_visc 데이터 선택")
        dialog.geometry("450x480")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 450) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 480) // 2
        dialog.geometry(f"+{x}+{y}")

        # Description
        desc_frame = ttk.Frame(dialog, padding=10)
        desc_frame.pack(fill=tk.X)
        ttk.Label(desc_frame, text="내보낼 데이터를 선택하세요.\n각 데이터는 별도의 CSV 파일로 저장됩니다.",
                  font=('Arial', 10)).pack(anchor=tk.W)

        # Checkbox frame
        check_frame = ttk.LabelFrame(dialog, text="데이터 선택", padding=10)
        check_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Data options - main results
        main_label = ttk.Label(check_frame, text="[기본 결과 (v vs 값)]", font=('Arial', 9, 'bold'))
        main_label.pack(anchor=tk.W, pady=(0, 5))

        main_options = [
            ("μ_visc(v) - 마찰계수", "mu_v", True),
            ("μ_visc_raw(v) - 스무딩 전", "mu_raw_v", False),
        ]

        # q-dependent data options
        q_label = ttk.Label(check_frame, text="\n[q 의존성 데이터 (특정 속도)]", font=('Arial', 9, 'bold'))
        q_label.pack(anchor=tk.W, pady=(5, 5))

        q_options = [
            ("P(q) - 접촉 면적비", "P_q", False),
            ("S(q) - 접촉 보정 인자", "S_q", False),
            ("G(q) - 누적 G 값", "G_q", False),
            ("C(q) - PSD 값", "C_q", False),
            ("Integrand(q) - μ 피적분함수", "integrand_q", False),
            ("Cumulative μ(q) - 누적 기여", "cumulative_q", False),
            ("Angle Integral(q) - 각도 적분", "angle_int_q", False),
        ]

        # Create checkbox variables
        check_vars = {}

        for display_name, key, default in main_options:
            var = tk.BooleanVar(value=default)
            check_vars[key] = var
            cb = ttk.Checkbutton(check_frame, text=display_name, variable=var)
            cb.pack(anchor=tk.W, pady=1)

        for display_name, key, default in q_options:
            var = tk.BooleanVar(value=default)
            check_vars[key] = var
            cb = ttk.Checkbutton(check_frame, text=display_name, variable=var)
            cb.pack(anchor=tk.W, pady=1)

        # Velocity selection for q-dependent data
        v_frame = ttk.Frame(check_frame)
        v_frame.pack(fill=tk.X, pady=5)
        ttk.Label(v_frame, text="q 데이터 속도 인덱스:", font=('Arial', 8)).pack(side=tk.LEFT)

        v_array = self.mu_visc_results['v']
        n_v = len(v_array)
        # Default to index closest to 1 m/s
        default_idx = np.argmin(np.abs(np.log10(v_array) - 0))  # log10(1) = 0
        self.export_v_idx_var = tk.StringVar(value=str(default_idx))
        v_spin = ttk.Spinbox(v_frame, from_=0, to=n_v-1, textvariable=self.export_v_idx_var, width=5)
        v_spin.pack(side=tk.LEFT, padx=5)

        # Show current velocity
        def update_v_label(*args):
            try:
                idx = int(self.export_v_idx_var.get())
                if 0 <= idx < n_v:
                    v_val = v_array[idx]
                    v_info_label.config(text=f"(v = {v_val:.2e} m/s)")
            except:
                pass

        v_info_label = ttk.Label(v_frame, text=f"(v = {v_array[default_idx]:.2e} m/s)", font=('Arial', 8))
        v_info_label.pack(side=tk.LEFT)
        self.export_v_idx_var.trace('w', update_v_label)

        # Select all / Deselect all buttons
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(fill=tk.X)

        def select_all():
            for var in check_vars.values():
                var.set(True)

        def deselect_all():
            for var in check_vars.values():
                var.set(False)

        ttk.Button(btn_frame, text="전체 선택", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="전체 해제", command=deselect_all).pack(side=tk.LEFT, padx=5)

        # Export button frame
        export_frame = ttk.Frame(dialog, padding=10)
        export_frame.pack(fill=tk.X)

        def do_export():
            # Check if any data is selected
            selected = [key for key, var in check_vars.items() if var.get()]
            if not selected:
                messagebox.showwarning("경고", "내보낼 데이터를 선택하세요.", parent=dialog)
                return

            # Ask for save directory
            save_dir = filedialog.askdirectory(
                title="CSV 파일 저장 폴더 선택",
                parent=dialog
            )
            if not save_dir:
                return

            try:
                v = self.mu_visc_results['v']
                mu = self.mu_visc_results['mu']
                mu_raw = self.mu_visc_results.get('mu_raw', mu)
                details = self.mu_visc_results.get('details', {})

                # Get velocity index for q-dependent data
                v_idx = int(self.export_v_idx_var.get())
                v_idx = max(0, min(v_idx, len(v) - 1))

                exported_files = []

                # Export main results (v vs value)
                if check_vars['mu_v'].get():
                    filename = "mu_visc_vs_velocity.csv"
                    filepath = os.path.join(save_dir, filename)
                    lines = ["velocity [m/s],mu_visc"]
                    for vi, mui in zip(v, mu):
                        lines.append(f"{vi:.6e},{mui:.6f}")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write("\n".join(lines))
                    exported_files.append(filename)

                if check_vars['mu_raw_v'].get():
                    filename = "mu_visc_raw_vs_velocity.csv"
                    filepath = os.path.join(save_dir, filename)
                    lines = ["velocity [m/s],mu_visc_raw"]
                    for vi, mui in zip(v, mu_raw):
                        lines.append(f"{vi:.6e},{mui:.6f}")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write("\n".join(lines))
                    exported_files.append(filename)

                # Export q-dependent data
                all_details = details.get('details', [])
                if v_idx < len(all_details):
                    det = all_details[v_idx]
                    q_arr = det.get('q', np.array([]))
                    v_selected = v[v_idx]

                    q_data_map = {
                        'P_q': ('P', 'P_q'),
                        'S_q': ('S', 'S_q'),
                        'G_q': ('G', 'G_q'),
                        'C_q': ('C_q', 'C_q'),
                        'integrand_q': ('integrand', 'integrand_q'),
                        'cumulative_q': ('cumulative_mu', 'cumulative_mu_q'),
                        'angle_int_q': ('angle_integral', 'angle_integral_q'),
                    }

                    for key, (data_key, file_suffix) in q_data_map.items():
                        if check_vars[key].get():
                            data = det.get(data_key)
                            if data is not None and len(data) == len(q_arr):
                                filename = f"{file_suffix}_v{v_idx}_{v_selected:.2e}.csv"
                                filepath = os.path.join(save_dir, filename)
                                lines = [f"# velocity = {v_selected:.6e} m/s", f"q [1/m],{data_key}"]
                                for qi, di in zip(q_arr, data):
                                    lines.append(f"{qi:.6e},{di:.6e}")
                                with open(filepath, 'w', encoding='utf-8') as f:
                                    f.write("\n".join(lines))
                                exported_files.append(filename)

                dialog.destroy()
                messagebox.showinfo("완료", f"CSV 파일 내보내기 완료:\n\n" + "\n".join(exported_files) + f"\n\n저장 위치: {save_dir}")
                self.status_var.set(f"CSV 내보내기 완료: {len(exported_files)}개 파일")

            except Exception as e:
                import traceback
                messagebox.showerror("오류", f"내보내기 실패:\n{str(e)}\n\n{traceback.format_exc()}", parent=dialog)

        ttk.Button(export_frame, text="내보내기", command=do_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(export_frame, text="취소", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _create_strain_map_tab(self, parent):
        """Create Local Strain Map visualization tab."""
        # Main container
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Local Strain Map 설정", padding=5)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Description
        desc_text = (
            "각 파수 q와 슬립 속도 v에서의 국소 변형률 ε(q,v)와 감소된 모듈러스를 시각화합니다.\n"
            "ω = q·v·cos(φ) 로 주파수가 결정되며, 해당 주파수에서의 모듈러스가 변형률에 따라 감소합니다."
        )
        ttk.Label(control_frame, text=desc_text, font=('Arial', 9)).pack(anchor=tk.W)

        # Control row
        ctrl_row = ttk.Frame(control_frame)
        ctrl_row.pack(fill=tk.X, pady=5)

        # Number of q points
        ttk.Label(ctrl_row, text="q 분할 수:").pack(side=tk.LEFT, padx=5)
        self.strain_map_nq_var = tk.StringVar(value="32")
        ttk.Entry(ctrl_row, textvariable=self.strain_map_nq_var, width=6).pack(side=tk.LEFT)

        # Number of v points
        ttk.Label(ctrl_row, text="  v 분할 수:").pack(side=tk.LEFT, padx=5)
        self.strain_map_nv_var = tk.StringVar(value="32")
        ttk.Entry(ctrl_row, textvariable=self.strain_map_nv_var, width=6).pack(side=tk.LEFT)

        # Strain estimation method - default to rms_slope
        ttk.Label(ctrl_row, text="  변형률 추정:").pack(side=tk.LEFT, padx=5)
        self.strain_map_method_var = tk.StringVar(value="rms_slope")
        method_combo = ttk.Combobox(
            ctrl_row, textvariable=self.strain_map_method_var,
            values=["rms_slope", "persson", "simple", "fixed"],
            width=10, state="readonly"
        )
        method_combo.pack(side=tk.LEFT)

        # Fixed strain value
        ttk.Label(ctrl_row, text="  고정 ε (%):").pack(side=tk.LEFT, padx=5)
        self.strain_map_fixed_var = tk.StringVar(value="1.0")
        ttk.Entry(ctrl_row, textvariable=self.strain_map_fixed_var, width=6).pack(side=tk.LEFT)

        # Calculate button
        ttk.Button(
            ctrl_row, text="계산 및 시각화",
            command=self._calculate_strain_map
        ).pack(side=tk.LEFT, padx=20)

        # CSV Export button
        ttk.Button(
            ctrl_row, text="CSV 내보내기",
            command=self._export_strain_map_csv
        ).pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.strain_map_progress = ttk.Progressbar(control_frame, mode='determinate')
        self.strain_map_progress.pack(fill=tk.X, pady=3)

        # Plot area - 2x4 grid for 8 heatmaps
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_strain_map = Figure(figsize=(18, 9), dpi=100)

        # 2x4 subplots layout:
        # Row 1: Local Strain | E' Storage | E''*g Loss | E'*f Storage
        # Row 2: G Integrand (linear) | G Integrand (nonlinear) | A/A0 (linear) | A/A0 (nonlinear)
        self.ax_strain_contour = self.fig_strain_map.add_subplot(241)
        self.ax_E_storage = self.fig_strain_map.add_subplot(242)
        self.ax_E_loss_nonlinear = self.fig_strain_map.add_subplot(243)
        self.ax_E_storage_nonlinear = self.fig_strain_map.add_subplot(244)
        self.ax_G_integrand_linear = self.fig_strain_map.add_subplot(245)
        self.ax_G_integrand_nonlinear = self.fig_strain_map.add_subplot(246)
        self.ax_contact_linear = self.fig_strain_map.add_subplot(247)
        self.ax_contact_nonlinear = self.fig_strain_map.add_subplot(248)

        self.canvas_strain_map = FigureCanvasTkAgg(self.fig_strain_map, plot_frame)
        self.canvas_strain_map.draw()
        self.canvas_strain_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_strain_map, plot_frame)
        toolbar.update()

        # Initialize plots
        self._init_strain_map_plots()

    def _init_strain_map_plots(self):
        """Initialize strain map plots with placeholder data."""
        for ax, title in [
            (self.ax_strain_contour, 'Local Strain [%]'),
            (self.ax_E_storage, "E' Storage [log Pa]"),
            (self.ax_E_loss_nonlinear, "E''*g Loss [log Pa]"),
            (self.ax_E_storage_nonlinear, "E'*f Storage [log Pa]"),
            (self.ax_G_integrand_linear, "G Integrand (linear)"),
            (self.ax_G_integrand_nonlinear, "G Integrand (f applied)"),
            (self.ax_contact_linear, "A/A0 Contact (linear)"),
            (self.ax_contact_nonlinear, "A/A0 Contact (f applied)")
        ]:
            ax.set_title(title, fontweight='bold', fontsize=9)
            ax.set_xlabel('log10(v) [m/s]', fontsize=8)
            ax.set_ylabel('log10(q) [1/m]', fontsize=8)
            ax.text(0.5, 0.5, 'No data',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='gray')

        self.fig_strain_map.tight_layout()
        self.canvas_strain_map.draw()

    def _calculate_strain_map(self):
        """Calculate and visualize local strain map."""
        if self.material is None or self.psd_model is None:
            messagebox.showwarning("경고", "먼저 재료(DMA)와 PSD 데이터를 로드하세요.")
            return

        if not self.results or '2d_results' not in self.results:
            messagebox.showwarning("경고", "먼저 G(q,v) 계산을 실행하세요 (탭 2).")
            return

        try:
            self.status_var.set("Local Strain Map 계산 중...")
            self.root.update()

            # Get parameters
            n_q = int(self.strain_map_nq_var.get())
            n_v = int(self.strain_map_nv_var.get())
            method = self.strain_map_method_var.get()
            fixed_strain = float(self.strain_map_fixed_var.get()) / 100.0

            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            temperature = float(self.temperature_var.get())
            poisson = float(self.poisson_var.get())

            # Get q and v ranges from G(q,v) results
            results_2d = self.results['2d_results']
            q_min = results_2d['q'].min()
            q_max = results_2d['q'].max()
            v_min = results_2d['v'].min()
            v_max = results_2d['v'].max()

            # Create q and v arrays
            q_array = np.logspace(np.log10(q_min), np.log10(q_max), n_q)
            v_array = np.logspace(np.log10(v_min), np.log10(v_max), n_v)

            # Get PSD values
            C_q = self.psd_model(q_array)

            # Prepare RMS slope interpolator if available
            rms_strain_interp = None
            if method == 'rms_slope' and hasattr(self, 'rms_slope_profiles') and self.rms_slope_profiles is not None:
                from scipy.interpolate import interp1d
                rms_q = self.rms_slope_profiles['q']
                rms_strain = self.rms_slope_profiles['strain']
                log_q = np.log10(rms_q)
                log_strain = np.log10(np.maximum(rms_strain, 1e-10))
                rms_strain_interp = interp1d(log_q, log_strain, kind='linear',
                                             bounds_error=False, fill_value='extrapolate')

            # Initialize matrices
            strain_matrix = np.zeros((n_q, n_v))
            E_storage_matrix = np.zeros((n_q, n_v))  # E' storage modulus
            E_loss_linear = np.zeros((n_q, n_v))
            E_loss_nonlinear = np.zeros((n_q, n_v))
            E_storage_nonlinear = np.zeros((n_q, n_v))  # E'·f(ε)

            # NEW: G integrand and contact area matrices
            G_integrand_linear = np.zeros((n_q, n_v))
            G_integrand_nonlinear = np.zeros((n_q, n_v))
            contact_linear = np.zeros((n_q, n_v))
            contact_nonlinear = np.zeros((n_q, n_v))

            # Calculate for each (q, v) pair
            total = n_q * n_v
            count = 0

            for i, q in enumerate(q_array):
                for j, v in enumerate(v_array):
                    # Characteristic frequency: ω = q * v (simplified, ignoring cos(φ))
                    omega = q * v

                    # Get linear E' and E'' at this frequency
                    E_loss = self.material.get_loss_modulus(np.array([omega]), temperature=temperature)[0]
                    E_storage = self.material.get_storage_modulus(np.array([omega]), temperature=temperature)[0]
                    E_loss_linear[i, j] = E_loss
                    E_storage_matrix[i, j] = E_storage

                    # Estimate local strain
                    if method == 'fixed':
                        strain = fixed_strain
                    elif method == 'rms_slope' and rms_strain_interp is not None:
                        try:
                            strain = 10 ** rms_strain_interp(np.log10(q))
                            # Fix NaN issue
                            if not np.isfinite(strain):
                                strain = fixed_strain
                        except:
                            strain = fixed_strain
                    elif method == 'persson':
                        # Persson approach: ε ~ sqrt(C(q) * q^4) * σ₀ / E'
                        C_val = self.psd_model(q)
                        strain = np.sqrt(max(C_val * q**4, 1e-20)) * sigma_0 / max(E_storage, 1e3)
                    elif method == 'simple':
                        # Simple: ε ~ σ₀ / E'
                        strain = sigma_0 / max(E_storage, 1e3)
                    else:
                        strain = fixed_strain

                    # Ensure finite value
                    if not np.isfinite(strain):
                        strain = fixed_strain
                    strain = np.clip(strain, 0.0, 1.0)
                    strain_matrix[i, j] = strain

                    # Apply f(ε), g(ε) correction for nonlinear E', E''
                    if self.g_interpolator is not None:
                        g_val = self.g_interpolator(strain)
                        g_val = np.clip(g_val, 0.0, 1.0)
                        E_loss_nonlinear[i, j] = E_loss * g_val
                    else:
                        E_loss_nonlinear[i, j] = E_loss

                    if self.f_interpolator is not None:
                        f_val = self.f_interpolator(strain)
                        f_val = np.clip(f_val, 0.0, 1.0)
                        E_storage_nonlinear[i, j] = E_storage * f_val
                    else:
                        f_val = 1.0
                        E_storage_nonlinear[i, j] = E_storage

                    # Calculate G integrand: q^3 * C(q) * |E*|^2 / ((1-nu^2)*sigma0)^2
                    # Linear: E* = E' + iE''
                    # Nonlinear: E*_eff = E'*f + iE''*g
                    C_val = C_q[i]
                    prefactor = 1.0 / ((1 - poisson**2) * sigma_0)**2

                    # Linear |E*|^2 = E'^2 + E''^2
                    E_star_sq_linear = E_storage**2 + E_loss**2
                    G_integrand_linear[i, j] = q**3 * C_val * E_star_sq_linear * prefactor

                    # Nonlinear |E*_eff|^2
                    E_prime_eff = E_storage_nonlinear[i, j]
                    E_loss_eff = E_loss_nonlinear[i, j]
                    E_star_sq_nonlinear = E_prime_eff**2 + E_loss_eff**2
                    G_integrand_nonlinear[i, j] = q**3 * C_val * E_star_sq_nonlinear * prefactor

                    # Calculate contact area ratio A/A0 = P(q) = erf(1/(2*sqrt(G)))
                    # G ~ cumulative integral of q^3*C(q)*|E*|^2
                    # Approximate G at this point for visualization
                    from scipy.special import erf
                    G_linear_approx = max(G_integrand_linear[i, j] * (q / n_q), 1e-20)
                    G_nonlinear_approx = max(G_integrand_nonlinear[i, j] * (q / n_q), 1e-20)

                    arg_linear = 1.0 / (2.0 * np.sqrt(G_linear_approx))
                    arg_nonlinear = 1.0 / (2.0 * np.sqrt(G_nonlinear_approx))
                    contact_linear[i, j] = erf(min(arg_linear, 10.0))
                    contact_nonlinear[i, j] = erf(min(arg_nonlinear, 10.0))

                    count += 1
                    if count % (total // 20 + 1) == 0:
                        self.strain_map_progress['value'] = (count / total) * 100
                        self.root.update()

            # Store results
            self.strain_map_results = {
                'q': q_array,
                'v': v_array,
                'strain': strain_matrix,
                'C_q': C_q,
                'E_storage': E_storage_matrix,
                'E_storage_nonlinear': E_storage_nonlinear,
                'E_loss_linear': E_loss_linear,
                'E_loss_nonlinear': E_loss_nonlinear,
                'G_integrand_linear': G_integrand_linear,
                'G_integrand_nonlinear': G_integrand_nonlinear,
                'contact_linear': contact_linear,
                'contact_nonlinear': contact_nonlinear
            }

            # Update plots
            self._update_strain_map_plots()

            self.strain_map_progress['value'] = 100
            self.status_var.set("Local Strain Map 계산 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"계산 실패:\n{str(e)}\n\n{traceback.format_exc()}")
            self.status_var.set("오류 발생")

    def _update_strain_map_plots(self):
        """Update strain map heatmap plots (8 plots total)."""
        if not hasattr(self, 'strain_map_results') or self.strain_map_results is None:
            return

        q = self.strain_map_results['q']
        v = self.strain_map_results['v']
        strain = self.strain_map_results['strain']
        E_storage = self.strain_map_results['E_storage']
        E_storage_nl = self.strain_map_results['E_storage_nonlinear']
        E_loss_nl = self.strain_map_results['E_loss_nonlinear']
        G_int_lin = self.strain_map_results.get('G_integrand_linear')
        G_int_nl = self.strain_map_results.get('G_integrand_nonlinear')
        contact_lin = self.strain_map_results.get('contact_linear')
        contact_nl = self.strain_map_results.get('contact_nonlinear')

        # Create meshgrid for pcolormesh
        log_v = np.log10(v)
        log_q = np.log10(q)
        V, Q = np.meshgrid(log_v, log_q)

        # Remove existing colorbars
        if hasattr(self, '_strain_map_colorbars'):
            for cbar in self._strain_map_colorbars:
                try:
                    cbar.remove()
                except:
                    pass
        self._strain_map_colorbars = []

        # Clear all 8 axes
        all_axes = [self.ax_strain_contour, self.ax_E_storage,
                    self.ax_E_loss_nonlinear, self.ax_E_storage_nonlinear,
                    self.ax_G_integrand_linear, self.ax_G_integrand_nonlinear,
                    self.ax_contact_linear, self.ax_contact_nonlinear]
        for ax in all_axes:
            ax.clear()

        # Color maps
        strain_cmap = 'YlOrRd'
        modulus_cmap = 'viridis'
        contact_cmap = 'plasma'

        # Fix NaN in strain for statistics
        strain_valid = np.nan_to_num(strain, nan=0.0)

        # === Row 1 ===
        # Plot 1: Local Strain with contours
        im1 = self.ax_strain_contour.pcolormesh(V, Q, strain_valid * 100, cmap=strain_cmap, shading='auto')
        self.ax_strain_contour.set_title('Local Strain [%]', fontweight='bold', fontsize=9)
        self.ax_strain_contour.set_xlabel('log10(v)', fontsize=8)
        self.ax_strain_contour.set_ylabel('log10(q)', fontsize=8)
        cbar1 = self.fig_strain_map.colorbar(im1, ax=self.ax_strain_contour)
        self._strain_map_colorbars.append(cbar1)
        try:
            cs = self.ax_strain_contour.contour(V, Q, strain_valid * 100, levels=[1, 5, 10], colors='k', linewidths=0.5)
            self.ax_strain_contour.clabel(cs, inline=True, fontsize=7, fmt='%.0f%%')
        except:
            pass
        strain_mean = np.nanmean(strain) * 100
        strain_max = np.nanmax(strain) * 100
        self.ax_strain_contour.text(0.02, 0.98, f'Mean:{strain_mean:.1f}%\nMax:{strain_max:.1f}%',
            transform=self.ax_strain_contour.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Plot 2: E' Storage Modulus
        E_s_safe = np.maximum(E_storage, 1e-10)
        im2 = self.ax_E_storage.pcolormesh(V, Q, np.log10(E_s_safe), cmap=modulus_cmap, shading='auto')
        self.ax_E_storage.set_title("E' Storage [log Pa]", fontweight='bold', fontsize=9)
        self.ax_E_storage.set_xlabel('log10(v)', fontsize=8)
        self.ax_E_storage.set_ylabel('log10(q)', fontsize=8)
        cbar2 = self.fig_strain_map.colorbar(im2, ax=self.ax_E_storage)
        self._strain_map_colorbars.append(cbar2)

        # Plot 3: E''*g Loss Modulus
        E_l_safe = np.maximum(E_loss_nl, 1e-10)
        im3 = self.ax_E_loss_nonlinear.pcolormesh(V, Q, np.log10(E_l_safe), cmap=modulus_cmap, shading='auto')
        self.ax_E_loss_nonlinear.set_title("E''*g Loss [log Pa]", fontweight='bold', fontsize=9)
        self.ax_E_loss_nonlinear.set_xlabel('log10(v)', fontsize=8)
        self.ax_E_loss_nonlinear.set_ylabel('log10(q)', fontsize=8)
        cbar3 = self.fig_strain_map.colorbar(im3, ax=self.ax_E_loss_nonlinear)
        self._strain_map_colorbars.append(cbar3)
        if self.g_interpolator is not None:
            E_loss_lin = self.strain_map_results['E_loss_linear']
            avg_g = np.mean(E_loss_nl / np.maximum(E_loss_lin, 1e-10))
            self.ax_E_loss_nonlinear.text(0.02, 0.98, f'Avg g:{avg_g:.1%}',
                transform=self.ax_E_loss_nonlinear.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Plot 4: E'*f Storage Modulus
        E_snl_safe = np.maximum(E_storage_nl, 1e-10)
        im4 = self.ax_E_storage_nonlinear.pcolormesh(V, Q, np.log10(E_snl_safe), cmap=modulus_cmap, shading='auto')
        self.ax_E_storage_nonlinear.set_title("E'*f Storage [log Pa]", fontweight='bold', fontsize=9)
        self.ax_E_storage_nonlinear.set_xlabel('log10(v)', fontsize=8)
        self.ax_E_storage_nonlinear.set_ylabel('log10(q)', fontsize=8)
        cbar4 = self.fig_strain_map.colorbar(im4, ax=self.ax_E_storage_nonlinear)
        self._strain_map_colorbars.append(cbar4)
        if self.f_interpolator is not None:
            avg_f = np.mean(E_storage_nl / np.maximum(E_storage, 1e-10))
            self.ax_E_storage_nonlinear.text(0.02, 0.98, f'Avg f:{avg_f:.1%}',
                transform=self.ax_E_storage_nonlinear.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # === Row 2 ===
        # Plot 5: G Integrand (linear)
        if G_int_lin is not None:
            G_lin_safe = np.maximum(G_int_lin, 1e-30)
            im5 = self.ax_G_integrand_linear.pcolormesh(V, Q, np.log10(G_lin_safe), cmap='inferno', shading='auto')
            self.ax_G_integrand_linear.set_title('G Integrand (linear)', fontweight='bold', fontsize=9)
            self.ax_G_integrand_linear.set_xlabel('log10(v)', fontsize=8)
            self.ax_G_integrand_linear.set_ylabel('log10(q)', fontsize=8)
            cbar5 = self.fig_strain_map.colorbar(im5, ax=self.ax_G_integrand_linear)
            self._strain_map_colorbars.append(cbar5)

        # Plot 6: G Integrand (nonlinear with f applied)
        if G_int_nl is not None:
            G_nl_safe = np.maximum(G_int_nl, 1e-30)
            im6 = self.ax_G_integrand_nonlinear.pcolormesh(V, Q, np.log10(G_nl_safe), cmap='inferno', shading='auto')
            self.ax_G_integrand_nonlinear.set_title('G Integrand (f applied)', fontweight='bold', fontsize=9)
            self.ax_G_integrand_nonlinear.set_xlabel('log10(v)', fontsize=8)
            self.ax_G_integrand_nonlinear.set_ylabel('log10(q)', fontsize=8)
            cbar6 = self.fig_strain_map.colorbar(im6, ax=self.ax_G_integrand_nonlinear)
            self._strain_map_colorbars.append(cbar6)
            if G_int_lin is not None:
                ratio = np.mean(G_int_nl / np.maximum(G_int_lin, 1e-30))
                self.ax_G_integrand_nonlinear.text(0.02, 0.98, f'Ratio:{ratio:.1%}',
                    transform=self.ax_G_integrand_nonlinear.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Plot 7: Contact area ratio A/A0 (linear)
        if contact_lin is not None:
            im7 = self.ax_contact_linear.pcolormesh(V, Q, contact_lin, cmap=contact_cmap, shading='auto', vmin=0, vmax=1)
            self.ax_contact_linear.set_title('A/A0 Contact (linear)', fontweight='bold', fontsize=9)
            self.ax_contact_linear.set_xlabel('log10(v)', fontsize=8)
            self.ax_contact_linear.set_ylabel('log10(q)', fontsize=8)
            cbar7 = self.fig_strain_map.colorbar(im7, ax=self.ax_contact_linear)
            self._strain_map_colorbars.append(cbar7)
            avg_P = np.mean(contact_lin)
            self.ax_contact_linear.text(0.02, 0.98, f'Avg P:{avg_P:.2f}',
                transform=self.ax_contact_linear.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        # Plot 8: Contact area ratio A/A0 (nonlinear)
        if contact_nl is not None:
            im8 = self.ax_contact_nonlinear.pcolormesh(V, Q, contact_nl, cmap=contact_cmap, shading='auto', vmin=0, vmax=1)
            self.ax_contact_nonlinear.set_title('A/A0 Contact (f applied)', fontweight='bold', fontsize=9)
            self.ax_contact_nonlinear.set_xlabel('log10(v)', fontsize=8)
            self.ax_contact_nonlinear.set_ylabel('log10(q)', fontsize=8)
            cbar8 = self.fig_strain_map.colorbar(im8, ax=self.ax_contact_nonlinear)
            self._strain_map_colorbars.append(cbar8)
            avg_P_nl = np.mean(contact_nl)
            self.ax_contact_nonlinear.text(0.02, 0.98, f'Avg P:{avg_P_nl:.2f}',
                transform=self.ax_contact_nonlinear.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        self.fig_strain_map.tight_layout()
        self.canvas_strain_map.draw()

    def _export_strain_map_csv(self):
        """Export Local Strain Map data to CSV files with selection dialog."""
        if not hasattr(self, 'strain_map_results') or self.strain_map_results is None:
            messagebox.showwarning("경고", "먼저 계산을 실행하세요.")
            return

        # Create dialog for selecting data to export
        dialog = tk.Toplevel(self.root)
        dialog.title("CSV 내보내기 - 데이터 선택")
        dialog.geometry("400x400")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 400) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 400) // 2
        dialog.geometry(f"+{x}+{y}")

        # Description
        desc_frame = ttk.Frame(dialog, padding=10)
        desc_frame.pack(fill=tk.X)
        ttk.Label(desc_frame, text="내보낼 데이터를 선택하세요.\n각 데이터는 별도의 CSV 파일로 저장됩니다.",
                  font=('Arial', 10)).pack(anchor=tk.W)

        # Checkbox frame
        check_frame = ttk.LabelFrame(dialog, text="데이터 선택", padding=10)
        check_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Data options with display names and keys
        data_options = [
            ("Local Strain [%]", "strain", True),
            ("E' Storage [Pa]", "E_storage", True),
            ("E''*g Loss [Pa]", "E_loss_nonlinear", True),
            ("E'*f Storage [Pa]", "E_storage_nonlinear", True),
            ("G Integrand (linear)", "G_integrand_linear", True),
            ("G Integrand (nonlinear)", "G_integrand_nonlinear", True),
            ("A/A0 Contact (linear)", "contact_linear", True),
            ("A/A0 Contact (nonlinear)", "contact_nonlinear", True),
        ]

        # Create checkbox variables
        check_vars = {}
        for display_name, key, default in data_options:
            var = tk.BooleanVar(value=default)
            check_vars[key] = var
            cb = ttk.Checkbutton(check_frame, text=display_name, variable=var)
            cb.pack(anchor=tk.W, pady=2)

        # Select all / Deselect all buttons
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(fill=tk.X)

        def select_all():
            for var in check_vars.values():
                var.set(True)

        def deselect_all():
            for var in check_vars.values():
                var.set(False)

        ttk.Button(btn_frame, text="전체 선택", command=select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="전체 해제", command=deselect_all).pack(side=tk.LEFT, padx=5)

        # Export button frame
        export_frame = ttk.Frame(dialog, padding=10)
        export_frame.pack(fill=tk.X)

        def do_export():
            # Check if any data is selected
            selected = [key for key, var in check_vars.items() if var.get()]
            if not selected:
                messagebox.showwarning("경고", "내보낼 데이터를 선택하세요.", parent=dialog)
                return

            # Ask for save directory
            from tkinter import filedialog
            save_dir = filedialog.askdirectory(
                title="CSV 파일 저장 폴더 선택",
                parent=dialog
            )
            if not save_dir:
                return

            try:
                q = self.strain_map_results['q']
                v = self.strain_map_results['v']

                exported_files = []

                for key in selected:
                    data = self.strain_map_results.get(key)
                    if data is None:
                        continue

                    # Special handling for strain (convert to %)
                    if key == "strain":
                        data = data * 100

                    # Create filename
                    filename = f"strain_map_{key}.csv"
                    filepath = os.path.join(save_dir, filename)

                    # Build CSV content
                    # First row: header with v values
                    # First column: q values
                    # Format: rows = q, columns = v
                    lines = []

                    # Header row: empty cell + v values
                    header = ["q \\ v"] + [f"{vi:.6e}" for vi in v]
                    lines.append(",".join(header))

                    # Data rows
                    for i, qi in enumerate(q):
                        row_data = [f"{qi:.6e}"] + [f"{data[i, j]:.6e}" for j in range(len(v))]
                        lines.append(",".join(row_data))

                    # Write file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write("\n".join(lines))

                    exported_files.append(filename)

                dialog.destroy()
                messagebox.showinfo("완료", f"CSV 파일 내보내기 완료:\n\n" + "\n".join(exported_files) + f"\n\n저장 위치: {save_dir}")
                self.status_var.set(f"CSV 내보내기 완료: {len(exported_files)}개 파일")

            except Exception as e:
                import traceback
                messagebox.showerror("오류", f"내보내기 실패:\n{str(e)}\n\n{traceback.format_exc()}", parent=dialog)

        ttk.Button(export_frame, text="내보내기", command=do_export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(export_frame, text="취소", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _create_integrand_tab(self, parent):
        """Create Integrand visualization tab for G(q) and μ_visc analysis."""
        # Main container
        main_container = ttk.Frame(parent)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel for controls
        left_frame = ttk.Frame(main_container, width=320)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_frame.pack_propagate(False)

        # ============== Left Panel: Controls ==============

        # 1. Description
        desc_frame = ttk.LabelFrame(left_frame, text="설명", padding=5)
        desc_frame.pack(fill=tk.X, pady=2, padx=3)

        desc_text = (
            "G(q) 및 μ_visc 계산의 피적분함수를\n"
            "시각화합니다.\n\n"
            "1. G 각도 적분: |E(qv cosφ)|² vs φ\n"
            "2. G(q) 피적분: q³C(q)×(각도적분) vs q\n"
            "3. μ_visc 각도 적분 vs 속도"
        )
        ttk.Label(desc_frame, text=desc_text, font=('Arial', 9), justify=tk.LEFT).pack(anchor=tk.W)

        # 2. Calculation Settings
        settings_frame = ttk.LabelFrame(left_frame, text="계산 설정", padding=5)
        settings_frame.pack(fill=tk.X, pady=2, padx=3)

        # q value selection for angle integrand
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="q 값 (1/m):", font=('Arial', 9)).pack(side=tk.LEFT)
        self.integrand_q_var = tk.StringVar(value="1e4, 1e5, 1e6")
        ttk.Entry(row1, textvariable=self.integrand_q_var, width=15).pack(side=tk.RIGHT)

        ttk.Label(settings_frame, text="(쉼표로 구분, 예: 1e4, 1e5, 1e6)",
                  font=('Arial', 7), foreground='gray').pack(anchor=tk.W)

        # Velocity selection
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="속도 v (m/s):", font=('Arial', 9)).pack(side=tk.LEFT)
        self.integrand_v_var = tk.StringVar(value="0.01")
        ttk.Entry(row2, textvariable=self.integrand_v_var, width=15).pack(side=tk.RIGHT)

        # Number of angle points
        row3 = ttk.Frame(settings_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="각도 분할 수:", font=('Arial', 9)).pack(side=tk.LEFT)
        self.integrand_nangle_var = tk.StringVar(value="72")
        ttk.Entry(row3, textvariable=self.integrand_nangle_var, width=8).pack(side=tk.RIGHT)

        # Apply nonlinear correction checkbox
        self.integrand_use_fg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            settings_frame,
            text="비선형 보정 f(ε), g(ε) 적용",
            variable=self.integrand_use_fg_var
        ).pack(anchor=tk.W, pady=2)

        # Calculate button
        calc_frame = ttk.Frame(settings_frame)
        calc_frame.pack(fill=tk.X, pady=5)

        self.integrand_calc_btn = ttk.Button(
            calc_frame,
            text="피적분함수 계산",
            command=self._calculate_integrand_visualization
        )
        self.integrand_calc_btn.pack(fill=tk.X)

        # Progress bar
        self.integrand_progress_var = tk.IntVar()
        self.integrand_progress_bar = ttk.Progressbar(
            calc_frame,
            variable=self.integrand_progress_var,
            maximum=100
        )
        self.integrand_progress_bar.pack(fill=tk.X, pady=2)

        # 3. Results Summary
        results_frame = ttk.LabelFrame(left_frame, text="결과 요약", padding=5)
        results_frame.pack(fill=tk.X, pady=2, padx=3)

        self.integrand_result_text = tk.Text(results_frame, height=16, font=("Courier", 8), wrap=tk.WORD)
        self.integrand_result_text.pack(fill=tk.X)

        # 4. Frequency range info
        freq_frame = ttk.LabelFrame(left_frame, text="주파수 범위 (ω = qv cosφ)", padding=5)
        freq_frame.pack(fill=tk.X, pady=2, padx=3)

        self.freq_range_text = tk.Text(freq_frame, height=6, font=("Courier", 8), wrap=tk.WORD)
        self.freq_range_text.pack(fill=tk.X)

        # ============== Right Panel: Plots ==============

        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right_panel, text="피적분함수 그래프", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create figure with 2x2 subplots
        self.fig_integrand = Figure(figsize=(10, 8), dpi=100)

        # Top-left: Angle integrand |E(qv cosφ)|² vs φ
        self.ax_angle_integrand = self.fig_integrand.add_subplot(221)
        self.ax_angle_integrand.set_title('G 각도 피적분함수: |E(qv cosφ)|² vs φ', fontweight='bold')
        self.ax_angle_integrand.set_xlabel('φ (rad)')
        self.ax_angle_integrand.set_ylabel('|E(ω)/((1-ν²)σ₀)|²')
        self.ax_angle_integrand.grid(True, alpha=0.3)

        # Top-right: G(q) integrand vs q
        self.ax_q_integrand = self.fig_integrand.add_subplot(222)
        self.ax_q_integrand.set_title('G(q) 피적분함수: q³C(q)×(각도적분) vs q', fontweight='bold')
        self.ax_q_integrand.set_xlabel('q (1/m)')
        self.ax_q_integrand.set_ylabel('q³C(q)×∫|E|²dφ')
        self.ax_q_integrand.set_xscale('log')
        self.ax_q_integrand.set_yscale('log')
        self.ax_q_integrand.grid(True, alpha=0.3)

        # Bottom-left: μ_visc integrand vs φ (Im[E] × cosφ)
        self.ax_mu_integrand = self.fig_integrand.add_subplot(223)
        self.ax_mu_integrand.set_title('μ_visc 각도 피적분함수: cosφ × Im[E] vs φ', fontweight='bold')
        self.ax_mu_integrand.set_xlabel('φ (rad)')
        self.ax_mu_integrand.set_ylabel('cosφ × E\'\'/((1-ν²)σ₀)')
        self.ax_mu_integrand.grid(True, alpha=0.3)

        # Bottom-right: Frequency range vs velocity
        self.ax_freq_range = self.fig_integrand.add_subplot(224)
        self.ax_freq_range.set_title('속도별 주파수 스캔 범위', fontweight='bold')
        self.ax_freq_range.set_xlabel('속도 v (m/s)')
        self.ax_freq_range.set_ylabel('주파수 ω (rad/s)')
        self.ax_freq_range.set_xscale('log')
        self.ax_freq_range.set_yscale('log')
        self.ax_freq_range.grid(True, alpha=0.3)

        self.fig_integrand.tight_layout()

        self.canvas_integrand = FigureCanvasTkAgg(self.fig_integrand, plot_frame)
        self.canvas_integrand.draw()
        self.canvas_integrand.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_integrand, plot_frame)
        toolbar.update()

    def _calculate_integrand_visualization(self):
        """Calculate and visualize integrands for G(q) and μ_visc."""
        # Check if data is available
        if self.psd_model is None or self.material is None:
            messagebox.showwarning("경고", "먼저 PSD와 DMA 데이터를 로드하세요.")
            return

        try:
            self.integrand_calc_btn.config(state='disabled')
            self.integrand_progress_var.set(10)
            self.root.update_idletasks()

            # Parse q values
            q_str = self.integrand_q_var.get().strip()
            q_values = [float(x.strip()) for x in q_str.split(',')]

            # Get velocity
            v = float(self.integrand_v_var.get())

            # Get number of angle points
            n_angle = int(self.integrand_nangle_var.get())

            # Get parameters
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            poisson_ratio = float(self.poisson_var.get())
            prefactor = 1.0 / ((1 - poisson_ratio**2) * sigma_0)

            # Check for nonlinear correction
            use_fg = self.integrand_use_fg_var.get()

            # Clear previous results
            self.integrand_result_text.delete('1.0', tk.END)
            self.freq_range_text.delete('1.0', tk.END)

            # Clear axes
            self.ax_angle_integrand.clear()
            self.ax_q_integrand.clear()
            self.ax_mu_integrand.clear()
            self.ax_freq_range.clear()

            # Set up axes labels and titles again
            self.ax_angle_integrand.set_title('G 각도 피적분함수: |E(qv cosφ)|² vs φ', fontweight='bold')
            self.ax_angle_integrand.set_xlabel('φ (rad)')
            self.ax_angle_integrand.set_ylabel('|E(ω)/((1-ν²)σ₀)|²')
            self.ax_angle_integrand.grid(True, alpha=0.3)

            self.ax_q_integrand.set_title('G(q) 피적분함수: q³C(q)×(각도적분) vs q', fontweight='bold')
            self.ax_q_integrand.set_xlabel('q (1/m)')
            self.ax_q_integrand.set_ylabel('q³C(q)×∫|E|²dφ')
            self.ax_q_integrand.set_xscale('log')
            self.ax_q_integrand.set_yscale('log')
            self.ax_q_integrand.grid(True, alpha=0.3)

            self.ax_mu_integrand.set_title('μ_visc 각도 피적분함수: cosφ × Im[E] vs φ', fontweight='bold')
            self.ax_mu_integrand.set_xlabel('φ (rad)')
            self.ax_mu_integrand.set_ylabel('cosφ × E\'\'/((1-ν²)σ₀)')
            self.ax_mu_integrand.grid(True, alpha=0.3)

            self.ax_freq_range.set_title('속도별 주파수 스캔 범위', fontweight='bold')
            self.ax_freq_range.set_xlabel('속도 v (m/s)')
            self.ax_freq_range.set_ylabel('주파수 ω (rad/s)')
            self.ax_freq_range.set_xscale('log')
            self.ax_freq_range.set_yscale('log')
            self.ax_freq_range.grid(True, alpha=0.3)

            self.integrand_progress_var.set(20)
            self.root.update_idletasks()

            # Get f, g interpolators if using nonlinear correction
            f_interp = None
            g_interp = None
            if use_fg and hasattr(self, 'f_interpolator') and self.f_interpolator is not None:
                f_interp = self.f_interpolator
                g_interp = self.g_interpolator

            # Create angle array
            phi = np.linspace(0, 2 * np.pi, n_angle)

            colors = plt.cm.viridis(np.linspace(0, 0.9, len(q_values)))

            self.integrand_result_text.insert(tk.END, "=== G 각도 피적분함수 분석 ===\n\n")

            # Calculate angle integrand for each q value
            for idx, q in enumerate(q_values):
                # Calculate frequencies: ω = q * v * cos(φ)
                omega = q * v * np.cos(phi)
                omega_eval = np.abs(omega)
                omega_eval[omega_eval < 1e-10] = 1e-10

                # Get strain for nonlinear correction (if available)
                strain_at_q = 0.01  # default
                if use_fg and hasattr(self, 'local_strain_array') and self.local_strain_array is not None:
                    try:
                        from scipy.interpolate import interp1d
                        if hasattr(self, 'rms_q_array') and self.rms_q_array is not None:
                            log_q = np.log10(self.rms_q_array)
                            log_strain = np.log10(np.maximum(self.local_strain_array, 1e-10))
                            interp_func = interp1d(log_q, log_strain, kind='linear',
                                                   bounds_error=False, fill_value='extrapolate')
                            strain_at_q = 10 ** interp_func(np.log10(q))
                            strain_at_q = np.clip(strain_at_q, 0.0, 1.0)
                    except:
                        pass

                # Calculate |E|² integrand for G
                integrand_G = np.zeros_like(phi)
                integrand_mu = np.zeros_like(phi)

                for i, w in enumerate(omega_eval):
                    E_prime = self.material.get_storage_modulus(np.array([w]))[0]
                    E_loss = self.material.get_loss_modulus(np.array([w]))[0]

                    if use_fg and f_interp is not None and g_interp is not None:
                        f_val = np.clip(f_interp(strain_at_q), 0.0, 1.0)
                        g_val = np.clip(g_interp(strain_at_q), 0.0, 1.0)
                        E_prime_eff = E_prime * f_val
                        E_loss_eff = E_loss * g_val
                    else:
                        E_prime_eff = E_prime
                        E_loss_eff = E_loss

                    # |E_eff|² for G integrand
                    E_star_sq = E_prime_eff**2 + E_loss_eff**2
                    integrand_G[i] = E_star_sq * prefactor**2

                    # cosφ × E'' for μ_visc integrand
                    integrand_mu[i] = np.cos(phi[i]) * E_loss_eff * prefactor

                # Plot G angle integrand
                self.ax_angle_integrand.plot(phi, integrand_G, '-', color=colors[idx],
                                             label=f'q = {q:.1e} 1/m', linewidth=1.5)

                # Plot μ_visc angle integrand
                self.ax_mu_integrand.plot(phi, integrand_mu, '-', color=colors[idx],
                                          label=f'q = {q:.1e} 1/m', linewidth=1.5)

                # Calculate angle integral result
                angle_integral_G = np.trapz(integrand_G, phi)
                angle_integral_mu = np.trapz(integrand_mu, phi)

                # Summary text
                self.integrand_result_text.insert(tk.END, f"q = {q:.2e} 1/m:\n")
                self.integrand_result_text.insert(tk.END, f"  ω_max = qv = {q*v:.2e} rad/s\n")
                self.integrand_result_text.insert(tk.END, f"  ∫|E|²dφ = {angle_integral_G:.4e}\n")
                self.integrand_result_text.insert(tk.END, f"  ∫cosφ×E\'\'dφ = {angle_integral_mu:.4e}\n")
                if use_fg:
                    self.integrand_result_text.insert(tk.END, f"  ε(q) = {strain_at_q:.4f}\n")
                self.integrand_result_text.insert(tk.END, "\n")

            self.ax_angle_integrand.legend(fontsize=8)
            self.ax_mu_integrand.legend(fontsize=8)

            self.integrand_progress_var.set(50)
            self.root.update_idletasks()

            # === G(q) integrand plot ===
            # Get PSD q range
            if hasattr(self.psd_model, 'q_data'):
                q_array = self.psd_model.q_data
            elif hasattr(self.psd_model, 'q'):
                q_array = self.psd_model.q
            else:
                q_array = np.logspace(2, 8, 200)

            # Calculate G(q) integrand for each q
            q_plot = np.logspace(np.log10(q_array.min()), np.log10(q_array.max()), 100)
            G_integrand_values = np.zeros_like(q_plot)

            for i, q in enumerate(q_plot):
                # Get PSD value
                C_q = self.psd_model(np.array([q]))[0]

                # Calculate angle integral
                omega = q * v * np.cos(phi)
                omega_eval = np.abs(omega)
                omega_eval[omega_eval < 1e-10] = 1e-10

                integrand = np.zeros_like(phi)
                for j, w in enumerate(omega_eval):
                    E_prime = self.material.get_storage_modulus(np.array([w]))[0]
                    E_loss = self.material.get_loss_modulus(np.array([w]))[0]

                    if use_fg and f_interp is not None and g_interp is not None:
                        strain_q = 0.01  # simplified
                        f_val = np.clip(f_interp(strain_q), 0.0, 1.0)
                        g_val = np.clip(g_interp(strain_q), 0.0, 1.0)
                        E_prime_eff = E_prime * f_val
                        E_loss_eff = E_loss * g_val
                    else:
                        E_prime_eff = E_prime
                        E_loss_eff = E_loss

                    E_star_sq = E_prime_eff**2 + E_loss_eff**2
                    integrand[j] = E_star_sq * prefactor**2

                angle_int = np.trapz(integrand, phi)
                G_integrand_values[i] = q**3 * C_q * angle_int

            # Plot G(q) integrand
            self.ax_q_integrand.plot(q_plot, G_integrand_values, 'b-', linewidth=1.5,
                                     label=f'v = {v:.2e} m/s')

            # Mark the selected q values
            for q in q_values:
                if q >= q_plot.min() and q <= q_plot.max():
                    idx = np.abs(q_plot - q).argmin()
                    self.ax_q_integrand.axvline(q, color='r', linestyle='--', alpha=0.5)
                    self.ax_q_integrand.plot(q, G_integrand_values[idx], 'ro', markersize=8)

            self.ax_q_integrand.legend(fontsize=8)

            self.integrand_progress_var.set(70)
            self.root.update_idletasks()

            # === Frequency range vs velocity plot ===
            v_range = np.logspace(-4, 1, 50)  # 0.0001 to 10 m/s
            q_ref = float(q_values[0])  # Use first q value as reference

            # ω_min = 0 (at cosφ = 0), ω_max = q*v (at cosφ = 1)
            omega_max = q_ref * v_range
            omega_min = 1e-10 * np.ones_like(v_range)  # small but not zero

            # Plot frequency range
            self.ax_freq_range.fill_between(v_range, omega_min, omega_max, alpha=0.3, label=f'q = {q_ref:.1e} 1/m')
            self.ax_freq_range.plot(v_range, omega_max, 'b-', linewidth=1.5, label='ω_max = qv')

            # Show DMA frequency range
            if hasattr(self, 'raw_dma_data') and self.raw_dma_data is not None:
                omega_dma_min = self.raw_dma_data['omega'].min()
                omega_dma_max = self.raw_dma_data['omega'].max()
                self.ax_freq_range.axhline(omega_dma_min, color='r', linestyle='--', alpha=0.7, label=f'DMA ω_min = {omega_dma_min:.1e}')
                self.ax_freq_range.axhline(omega_dma_max, color='r', linestyle='--', alpha=0.7, label=f'DMA ω_max = {omega_dma_max:.1e}')

            # Mark current velocity
            self.ax_freq_range.axvline(v, color='g', linestyle='-', linewidth=2, alpha=0.7, label=f'현재 v = {v:.2e}')

            self.ax_freq_range.legend(fontsize=7, loc='lower right')

            # Frequency range info text
            self.freq_range_text.insert(tk.END, f"선택 q = {q_ref:.2e} 1/m, v = {v:.2e} m/s\n")
            self.freq_range_text.insert(tk.END, f"ω 범위: 0 ~ {q_ref * v:.2e} rad/s\n\n")
            if hasattr(self, 'raw_dma_data') and self.raw_dma_data is not None:
                self.freq_range_text.insert(tk.END, f"DMA 데이터 범위:\n")
                self.freq_range_text.insert(tk.END, f"  {omega_dma_min:.2e} ~ {omega_dma_max:.2e} rad/s\n")

            self.integrand_progress_var.set(100)
            self.fig_integrand.tight_layout()
            self.canvas_integrand.draw()

            self.status_var.set("피적분함수 계산 완료")

        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"계산 실패:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            self.integrand_calc_btn.config(state='normal')

    def _create_variables_tab(self, parent):
        """Create variable relationship explanation tab."""
        # Main scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Title
        title_frame = ttk.LabelFrame(scrollable_frame, text="Persson 마찰 이론 - 변수 관계 및 데이터 흐름", padding=15)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        # Content text with variable relationships
        content = """
════════════════════════════════════════════════════════════════════════════════
                     PERSSON 마찰 이론 변수 관계도
════════════════════════════════════════════════════════════════════════════════

【1. 입력 데이터】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────────────────────────────────────────────────┐
│  DMA 데이터 (재료 물성)                                                      │
│  ├─ ω (각진동수, rad/s)                                                      │
│  ├─ E'(ω) : 저장 탄성률 [Pa]                                                 │
│  ├─ E''(ω) : 손실 탄성률 [Pa]                                                │
│  └─ tan(δ) = E''/E' : 손실 탄젠트                                            │
│                                                                              │
│  PSD 데이터 (표면 거칠기)                                                    │
│  ├─ q (파수, 1/m)                                                            │
│  └─ C(q) : 파워 스펙트럼 밀도 [m⁴]                                           │
│                                                                              │
│  Strain Sweep 데이터 (비선형 보정용, 선택)                                   │
│  ├─ γ (strain, %)                                                            │
│  ├─ f(γ) = E'(γ)/E'(0) : 저장 탄성률 감소율                                  │
│  └─ g(γ) = E''(γ)/E''(0) : 손실 탄성률 감소율 (Payne 효과)                   │
└─────────────────────────────────────────────────────────────────────────────┘

【2. 계산 파라미터】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  σ₀  : 공칭 접촉 압력 [Pa]
  v   : 슬라이딩 속도 [m/s]
  T   : 온도 [°C]
  ν   : 푸아송 비 (일반적으로 0.5)
  γ   : 접촉 보정 인자 (일반적으로 0.5)

  q₀ ~ q₁ : PSD 적분 범위 (파수)

【3. 중간 계산 변수】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ G(q) 계산 ──────────────────────────────────────────────────────────────────┐
│                                                                              │
│  복소 탄성률:  E*(ω) = E'(ω) + i·E''(ω)                                      │
│                                                                              │
│  진동수:       ω = q · v · cos(φ)   (슬라이딩에 의한 진동)                   │
│                                                                              │
│  G(q) = (π/4) × (E*/(σ₀(1-ν²)))² × ∫[q₀→q] k³ C(k) ∫[0→2π] cos²φ dφ dk     │
│                                                                              │
│       = (π/2) × (|E(q·v)|/(σ₀(1-ν²)))² × ∫[q₀→q] k³ C(k) dk                 │
│                                                                              │
│                                                                              │
│  [비선형 보정 적용 시]                                                       │
│  E'_eff = E' × f(ε),  E''_eff = E'' × g(ε)                                  │
│  |E*_eff|² = (E'×f)² + (E''×g)²                                             │
│  G_eff(q) = G_linear(q) × |E*_eff|²/|E*_linear|²                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ P(q) 실접촉 면적비율 ───────────────────────────────────────────────────────┐
│                                                                              │
│  P(q) = erf(1 / (2√G(q)))                                                    │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────┐              │
│  │ G → 0  : P → 1.0 (완전 접촉)                              │              │
│  │ G → ∞  : P → 0.0 (접촉 없음)                              │              │
│  └───────────────────────────────────────────────────────────┘              │
│                                                                              │
│  ※ 비선형 모드: G_eff 사용 → P도 비선형 보정됨                               │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ S(q) 접촉 보정 인자 ────────────────────────────────────────────────────────┐
│                                                                              │
│  S(q) = γ + (1-γ) × P(q)²                                                    │
│                                                                              │
│  ※ S(q)는 P(q)에서 유도됨 → P가 비선형이면 S도 비선형                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ h'rms & Local Strain (Tab 4) ────────────────────────────────────────────────┐
│                                                                              │
│  RMS 경사:    ξ²(q) = 2π ∫[q₀→q] k³ C(k) dk                                  │
│  로컬 변형률: ε(q) = factor × ξ(q)    (factor ≈ 0.5, Persson 권장)           │
│                                                                              │
│  ※ 이 ε(q)는 비선형 보정 g(ε)에 사용됨                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

【4. 최종 출력: μ_visc 계산】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ μ_visc 공식 ────────────────────────────────────────────────────────────────┐
│                                                                              │
│  μ_visc = (1/2) × ∫[q₀→q₁] q³ C(q) P(q) S(q)                                │
│                   × ∫[0→2π] cosφ × Im[E(qv·cosφ)] / ((1-ν²)σ₀) dφ dq        │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ 비선형 보정 적용 시:                                          │          │
│  │   E'_eff = E' × f(ε)                                          │          │
│  │   E''_eff = E'' × g(ε)                                        │          │
│  │   G_eff = G_linear × |E*_eff|²/|E*_linear|²                   │          │
│  │   P(q), S(q) ← G_eff 기반으로 재계산                          │          │
│  │   여기서 ε는 Tab 4에서 계산된 local strain                    │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

【5. 데이터 흐름도】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌────────────┐     ┌────────────┐     ┌──────────────────┐
  │ DMA 데이터 │     │ PSD 데이터 │     │ Strain Sweep     │
  │ E'(ω),E''(ω)│     │ C(q)       │     │ f(γ), g(γ)       │
  └─────┬──────┘     └─────┬──────┘     └────────┬─────────┘
        │                  │                      │
        └────────┬─────────┘                      │
                 ▼                                │
  ┌──────────────────────────────┐                │
  │  Tab 1: 입력 데이터 검증     │                │
  │  - Master Curve 확인         │                │
  │  - PSD 확인                  │                │
  └──────────────┬───────────────┘                │
                 ▼                                │
  ┌──────────────────────────────┐                │
  │  Tab 2: 계산 설정            │                │
  │  - σ₀, v, T, ν 설정          │                │
  │  - q 범위 설정               │                │
  └──────────────┬───────────────┘                │
                 ▼                                │
  ┌──────────────────────────────┐                │
  │  Tab 3: G(q,v) 계산          │                │
  │  - G(q) = f(|E*|, C(q), σ₀)  │◄───────────────┤
  │  - P(q) = erf(1/(2√G))       │                │
  │  - 다중 속도 G_matrix 생성   │                │
  │                              │                │
  │  ※ 선형 |E*| 기반 계산       │                │
  └──────────────┬───────────────┘                │
                 ▼                                │
  ┌──────────────────────────────┐                │
  │  Tab 4: h'rms 계산           │                │
  │  - ξ(q) from PSD             │                │
  │  - ε(q) = factor × ξ(q)      │────────────────┤
  └──────────────┬───────────────┘                │
                 ▼                                ▼
  ┌──────────────────────────────────────────────────┐
  │  Tab 5: μ_visc 계산                              │
  │  ┌─────────────────────────────────────────────┐ │
  │  │ 선형 모드:                                  │ │
  │  │   G(q), P(q), S(q) ← Tab 3 기반 (선형)      │ │
  │  │   Im[E] = Im[E_linear]                      │ │
  │  └─────────────────────────────────────────────┘ │
  │  ┌─────────────────────────────────────────────┐ │
  │  │ 비선형 모드:                                │ │
  │  │   ε(q) ← Tab 4의 h'rms 기반                  │ │
  │  │   E'_eff = E' × f(ε), E''_eff = E'' × g(ε)  │ │
  │  │   G_eff = G × |E*_eff|²/|E*_lin|²           │ │
  │  │   P(q), S(q) ← G_eff 기반 (비선형)          │ │
  │  │   Im[E_eff] = Im[E_linear] × g(ε)           │ │
  │  └─────────────────────────────────────────────┘ │
  │                                                  │
  │  ※ 비선형 모드: G(q), P(q), S(q) 모두 보정됨   │
  └──────────────────────────────────────────────────┘

【6. 비선형 보정 적용 (현재 구현)】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [선형 모드]
  ───────────
  • Tab 3에서 계산된 G(q), P(q), S(q) 그대로 사용
  • Im[E] = Im[E_linear]

  [비선형 모드]
  ───────────
  • Tab 4에서 계산된 ε(q) = factor × ξ(q) 사용
  • E'_eff = E' × f(ε)     (저장 탄성률 보정)
  • E''_eff = E'' × g(ε)    (손실 탄성률 보정)
  • |E*_eff|² = (E'×f)² + (E''×g)²
  • G_eff = G_linear × |E*_eff|²/|E*_linear|²
  • P(q) = erf(1/(2√G_eff))  ← 비선형
  • S(q) = γ + (1-γ)P²       ← 비선형
  • Im[E_eff] = Im[E_linear] × g(ε)  ← 적분에 사용

  ※ 비선형 보정의 물리적 의미:
     f(ε), g(ε) < 1 이면 |E*|가 감소 → G가 감소 → P가 증가 → 접촉 면적 증가
     → Payne 효과: 높은 변형에서 재료가 부드러워짐

【7. 단위 정리】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  변수              단위              설명
  ────────────────────────────────────────────────────────────
  q                 1/m              파수 (wavenumber)
  C(q)              m⁴               2D 등방성 PSD
  E', E''           Pa               탄성률
  σ₀                Pa               공칭 압력
  v                 m/s              슬라이딩 속도
  ω = q·v           rad/s            각진동수
  G(q)              무차원           면적 함수
  P(q)              무차원 (0~1)     접촉 면적 비율
  S(q)              무차원           접촉 보정 인자
  ξ(q)              무차원           RMS 경사
  ε(q)              무차원 (0~1)     로컬 변형률
  μ_visc            무차원           점탄성 마찰 계수

════════════════════════════════════════════════════════════════════════════════
"""

        text_widget = tk.Text(title_frame, wrap=tk.WORD, font=('Courier New', 9), height=50, width=90)
        text_widget.insert(tk.END, content)
        text_widget.config(state='disabled')  # Read-only
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_debug_tab(self, parent):
        """Create debug log tab for monitoring calculation values."""
        # Instruction label
        instruction = ttk.LabelFrame(parent, text="디버그 로그 탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "이 탭에서는 μ_visc 계산 과정의 모든 중간 변수 값을 확인할 수 있습니다.\n"
            "문제 진단 및 계산 검증에 사용하세요. '진단 실행' 버튼으로 상세 분석을 수행합니다.",
            font=('Arial', 10)
        ).pack()

        # Control buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btn_frame,
            text="진단 실행",
            command=self._run_debug_diagnostic,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="로그 지우기",
            command=self._clear_debug_log,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            btn_frame,
            text="로그 저장",
            command=self._save_debug_log,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        # Debug log text area with scrollbar
        log_frame = ttk.LabelFrame(parent, text="계산 로그", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create text widget with scrollbar
        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.debug_log_text = tk.Text(
            log_frame,
            wrap=tk.WORD,
            font=('Courier New', 9),
            yscrollcommand=log_scroll.set
        )
        self.debug_log_text.pack(fill=tk.BOTH, expand=True)
        log_scroll.config(command=self.debug_log_text.yview)

        # Initialize with welcome message
        self.debug_log_text.insert(tk.END, "=" * 70 + "\n")
        self.debug_log_text.insert(tk.END, "  디버그 로그 탭 - μ_visc 계산 진단 도구\n")
        self.debug_log_text.insert(tk.END, "=" * 70 + "\n\n")
        self.debug_log_text.insert(tk.END, "  '진단 실행' 버튼을 클릭하여 계산 상세 분석을 시작하세요.\n")
        self.debug_log_text.insert(tk.END, "  이 탭에서는 다음 값들을 확인할 수 있습니다:\n\n")
        self.debug_log_text.insert(tk.END, "  • 마스터 커브 데이터 (E', E'', 주파수 범위)\n")
        self.debug_log_text.insert(tk.END, "  • PSD 데이터 (C(q), q 범위)\n")
        self.debug_log_text.insert(tk.END, "  • G(q) 계산 중간값\n")
        self.debug_log_text.insert(tk.END, "  • P(q), S(q) 값\n")
        self.debug_log_text.insert(tk.END, "  • 각도 적분값 (angle integral)\n")
        self.debug_log_text.insert(tk.END, "  • 피적분함수 (integrand)\n")
        self.debug_log_text.insert(tk.END, "  • μ_visc 최종값 및 진단 결과\n")

    def _log_debug(self, message: str):
        """Add a message to the debug log."""
        if hasattr(self, 'debug_log_text'):
            self.debug_log_text.insert(tk.END, message + "\n")
            self.debug_log_text.see(tk.END)
            self.root.update()

    def _clear_debug_log(self):
        """Clear the debug log."""
        if hasattr(self, 'debug_log_text'):
            self.debug_log_text.delete(1.0, tk.END)
            self._log_debug("로그가 지워졌습니다.\n")

    def _save_debug_log(self):
        """Save debug log to file."""
        if hasattr(self, 'debug_log_text'):
            from tkinter import filedialog
            filepath = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="디버그 로그 저장"
            )
            if filepath:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.debug_log_text.get(1.0, tk.END))
                messagebox.showinfo("저장 완료", f"로그가 저장되었습니다:\n{filepath}")

    def _run_debug_diagnostic(self):
        """Run comprehensive debug diagnostic for mu_visc calculation."""
        self._clear_debug_log()
        self._log_debug("=" * 70)
        self._log_debug("  μ_visc 계산 진단 시작")
        self._log_debug("=" * 70 + "\n")

        # 1. Check Material (DMA) data
        self._log_debug("[1] 재료 데이터 (DMA) 검사")
        self._log_debug("-" * 50)

        if self.material is None:
            self._log_debug("  ❌ 오류: 재료 데이터가 없습니다!")
            self._log_debug("     → Tab 0에서 마스터 커브를 생성하고 Tab 1에서 가져오세요.\n")
            return
        else:
            self._log_debug("  ✓ 재료 데이터 로드됨")

            # Check material attributes
            if hasattr(self.material, '_omega') and self.material._omega is not None:
                omega = self.material._omega
                self._log_debug(f"  • 주파수 범위: {omega[0]:.2e} ~ {omega[-1]:.2e} rad/s")
                self._log_debug(f"  • 데이터 점 수: {len(omega)}")

            # Sample E' and E'' at various frequencies
            test_freqs = [1e-4, 1e-2, 1, 1e2, 1e4, 1e6]
            self._log_debug("\n  주파수별 E', E'' 샘플:")
            self._log_debug("  " + "-" * 45)
            self._log_debug("  {:<12} {:<15} {:<15}".format("w (rad/s)", "E' (Pa)", "E'' (Pa)"))
            self._log_debug("  " + "-" * 45)

            temperature = float(self.temperature_var.get()) if hasattr(self, 'temperature_var') else 20.0
            any_zero_E = False
            any_nan_E = False

            for freq in test_freqs:
                try:
                    E_prime = self.material.get_storage_modulus(freq, temperature=temperature)
                    E_loss = self.material.get_loss_modulus(freq, temperature=temperature)

                    if not np.isfinite(E_prime) or not np.isfinite(E_loss):
                        any_nan_E = True
                        self._log_debug(f"  {freq:<12.2e} {'NaN!':<15} {'NaN!':<15} ⚠️")
                    elif E_prime < 1e3 or E_loss < 1e2:
                        any_zero_E = True
                        self._log_debug(f"  {freq:<12.2e} {E_prime:<15.2e} {E_loss:<15.2e} ⚠️ 너무 작음")
                    else:
                        self._log_debug(f"  {freq:<12.2e} {E_prime:<15.2e} {E_loss:<15.2e}")
                except Exception as e:
                    self._log_debug(f"  {freq:<12.2e} {'오류!':<15} {str(e)[:15]:<15}")

            if any_nan_E:
                self._log_debug("\n  ❌ 경고: 일부 주파수에서 E' 또는 E''가 NaN입니다!")
            if any_zero_E:
                self._log_debug("\n  ⚠️ 경고: 일부 주파수에서 E' 또는 E''가 너무 작습니다!")

            # Check master curve frequency range vs expected omega range
            self._log_debug("\n  마스터 커브 주파수 범위 분석:")
            if hasattr(self.material, '_frequencies') and self.material._frequencies is not None:
                omega_data_min = np.min(self.material._frequencies)
                omega_data_max = np.max(self.material._frequencies)
                self._log_debug(f"  • 마스터 커브 ω 범위: {omega_data_min:.2e} ~ {omega_data_max:.2e} rad/s")

                # Estimate typical omega range for mu_visc calculation
                q_typical = [1e2, 1e4, 1e6]  # typical q values
                v_typical = [0.001, 0.1, 10]  # typical velocities
                self._log_debug("\n  μ_visc 계산 시 예상 ω 범위 (ω = q × v):")
                for q in q_typical:
                    for v in v_typical:
                        omega_calc = q * v
                        in_range = omega_data_min <= omega_calc <= omega_data_max
                        status = "✓" if in_range else "⚠️ 범위 밖"
                        self._log_debug(f"    q={q:.0e}, v={v:.0e} → ω={omega_calc:.2e} {status}")

        # 2. Check PSD data
        self._log_debug("\n\n[2] PSD 데이터 검사")
        self._log_debug("-" * 50)

        if self.psd_model is None:
            self._log_debug("  ❌ 오류: PSD 데이터가 없습니다!")
            self._log_debug("     → Tab 1에서 PSD 데이터를 로드하세요.\n")
            return
        else:
            self._log_debug("  ✓ PSD 데이터 로드됨")

            # Sample C(q) at various wavenumbers
            test_qs = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
            self._log_debug("\n  파수별 C(q) 샘플:")
            self._log_debug("  " + "-" * 35)
            self._log_debug(f"  {'q (1/m)':<12} {'C(q) (m^4)':<15}")
            self._log_debug("  " + "-" * 35)

            for q in test_qs:
                try:
                    C_q = self.psd_model(np.array([q]))[0]
                    if np.isfinite(C_q) and C_q > 0:
                        self._log_debug(f"  {q:<12.2e} {C_q:<15.2e}")
                    else:
                        self._log_debug(f"  {q:<12.2e} {'무효!':<15} ⚠️")
                except Exception as e:
                    self._log_debug(f"  {q:<12.2e} {'오류!':<15}")

        # 3. Check G(q,v) results
        self._log_debug("\n\n[3] G(q,v) 계산 결과 검사")
        self._log_debug("-" * 50)

        if not hasattr(self, 'results') or self.results is None or '2d_results' not in self.results:
            self._log_debug("  ❌ 오류: G(q,v) 계산 결과가 없습니다!")
            self._log_debug("     → Tab 2에서 G(q,v) 계산을 먼저 실행하세요.\n")
            return
        else:
            results_2d = self.results['2d_results']
            q = results_2d['q']
            v = results_2d['v']
            G_matrix = results_2d['G_matrix']

            self._log_debug("  ✓ G(q,v) 계산 결과 있음")
            self._log_debug(f"  • q 범위: {q[0]:.2e} ~ {q[-1]:.2e} (1/m), {len(q)}점")
            self._log_debug(f"  • v 범위: {v[0]:.2e} ~ {v[-1]:.2e} (m/s), {len(v)}점")
            self._log_debug(f"  • G 행렬 크기: {G_matrix.shape}")

            # G statistics
            G_min = np.nanmin(G_matrix)
            G_max = np.nanmax(G_matrix)
            G_mean = np.nanmean(G_matrix)

            self._log_debug(f"\n  G(q,v) 통계:")
            self._log_debug(f"  • 최소값: {G_min:.4e}")
            self._log_debug(f"  • 최대값: {G_max:.4e}")
            self._log_debug(f"  • 평균값: {G_mean:.4e}")

            # Check for NaN values
            nan_count = np.sum(~np.isfinite(G_matrix))
            if nan_count > 0:
                self._log_debug(f"  ⚠️ 경고: G 행렬에 {nan_count}개의 NaN/Inf 값이 있습니다!")

            # Calculate P(q) from G
            self._log_debug("\n  G(q) → P(q) 변환 (중간 속도):")
            mid_v_idx = len(v) // 2
            G_mid = G_matrix[:, mid_v_idx]
            from scipy.special import erf

            # Safe P calculation
            P_mid = np.zeros_like(G_mid)
            valid_G = np.isfinite(G_mid) & (G_mid > 0)
            P_mid[~valid_G] = 1.0  # Full contact for invalid G
            if np.any(valid_G):
                sqrt_G = np.sqrt(G_mid[valid_G])
                arg = 1.0 / (2.0 * np.minimum(sqrt_G, 1e5))
                P_mid[valid_G] = erf(np.minimum(arg, 10.0))

            P_min = np.min(P_mid)
            P_max = np.max(P_mid)
            P_mean = np.mean(P_mid)

            self._log_debug(f"  • P(q) 최소값: {P_min:.6f}")
            self._log_debug(f"  • P(q) 최대값: {P_max:.6f}")
            self._log_debug(f"  • P(q) 평균값: {P_mean:.6f}")

            if P_max < 0.001:
                self._log_debug("  ❌ 심각: P(q)가 거의 0! G(q)가 너무 큽니다.")
                self._log_debug("     → σ₀ (공칭 압력)를 높이거나 PSD를 확인하세요.")

        # 4. Check calculation parameters
        self._log_debug("\n\n[4] 계산 파라미터 검사")
        self._log_debug("-" * 50)

        sigma_0 = float(self.sigma_0_var.get()) * 1e6 if hasattr(self, 'sigma_0_var') else 0.3e6
        temperature = float(self.temperature_var.get()) if hasattr(self, 'temperature_var') else 20.0
        poisson = float(self.poisson_var.get()) if hasattr(self, 'poisson_var') else 0.5
        gamma = float(self.gamma_var.get()) if hasattr(self, 'gamma_var') else 0.5
        n_phi = int(self.n_phi_var.get()) if hasattr(self, 'n_phi_var') else 72

        self._log_debug(f"  • σ₀ (공칭 압력): {sigma_0/1e6:.3f} MPa = {sigma_0:.2e} Pa")
        self._log_debug(f"  • T (온도): {temperature:.1f} °C")
        self._log_debug(f"  • ν (푸아송비): {poisson:.2f}")
        self._log_debug(f"  • γ (접촉 보정): {gamma:.2f}")
        self._log_debug(f"  • n_φ (각도 적분점): {n_phi}")

        prefactor = 1.0 / ((1 - poisson**2) * sigma_0)
        self._log_debug(f"  • prefactor = 1/((1-ν²)σ₀) = {prefactor:.4e}")

        # Check if prefactor is reasonable
        # For E'' ~ 1e7 Pa and prefactor ~ 4e-6, angle_integral ~ 4 * (pi/2) * 1e7 * 4e-6 ~ 0.25
        # This should give non-zero mu_visc
        if prefactor > 1e-4:
            self._log_debug("  ⚠️ 경고: prefactor가 큼 (σ₀가 너무 작음)")
        elif prefactor < 1e-8:
            self._log_debug("  ⚠️ 경고: prefactor가 작음 (σ₀가 너무 큼)")

        # 5. Test single point calculation
        self._log_debug("\n\n[5] 단일점 계산 테스트")
        self._log_debug("-" * 50)

        # Pick a middle q and v
        mid_q_idx = len(q) // 2
        mid_v_idx = len(v) // 2
        test_q = q[mid_q_idx]
        test_v = v[mid_v_idx]

        self._log_debug(f"  테스트 지점:")
        self._log_debug(f"  • q = {test_q:.2e} (1/m)")
        self._log_debug(f"  • v = {test_v:.4f} (m/s)")

        # Calculate omega range for this q and v
        omega_max = test_q * test_v  # φ = 0
        omega_min = test_q * test_v * np.cos(np.pi/2 - 0.001)  # near φ = π/2

        self._log_debug(f"  • ω 범위: {omega_min:.2e} ~ {omega_max:.2e} rad/s")

        # Get E' and E'' at omega_max
        try:
            E_prime_test = self.material.get_storage_modulus(omega_max, temperature=temperature)
            E_loss_test = self.material.get_loss_modulus(omega_max, temperature=temperature)
            self._log_debug(f"\n  ω={omega_max:.2e} 에서:")
            self._log_debug(f"  • E' = {E_prime_test:.4e} Pa")
            self._log_debug(f"  • E'' = {E_loss_test:.4e} Pa")

            if E_loss_test < 1:
                self._log_debug("  ❌ 심각: E''가 거의 0! 마스터 커브를 확인하세요.")
            elif E_loss_test < 1e4:
                self._log_debug(f"  ⚠️ 경고: E''가 작음 ({E_loss_test:.2e})")
        except Exception as e:
            self._log_debug(f"  ❌ 오류: {e}")

        # Calculate angle integral contribution
        self._log_debug("\n  각도 적분 테스트:")
        phi_test = np.linspace(0, np.pi/2, n_phi)
        omega_test = test_q * test_v * np.cos(phi_test)
        cos_phi_test = np.cos(phi_test)

        integrand_test = np.zeros(n_phi)
        ImE_test = np.zeros(n_phi)

        for i, (w, c) in enumerate(zip(omega_test, cos_phi_test)):
            w = max(w, 1e-10)
            try:
                E_loss = self.material.get_loss_modulus(w, temperature=temperature)
                ImE_test[i] = E_loss
                integrand_test[i] = c * E_loss * prefactor
            except:
                ImE_test[i] = 0
                integrand_test[i] = 0

        # Show some sample values
        self._log_debug(f"\n  φ별 샘플 (5개):")
        self._log_debug("  {:<10} {:<10} {:<12} {:<12} {:<12}".format("phi (rad)", "cos(phi)", "w (rad/s)", "E'' (Pa)", "integrand"))
        for i in [0, n_phi//4, n_phi//2, 3*n_phi//4, n_phi-1]:
            self._log_debug(f"  {phi_test[i]:<10.4f} {cos_phi_test[i]:<10.4f} {omega_test[i]:<12.2e} {ImE_test[i]:<12.2e} {integrand_test[i]:<12.4e}")

        # Calculate angle integral
        from scipy.integrate import simpson
        angle_integral = 4.0 * simpson(integrand_test, x=phi_test)
        self._log_debug(f"\n  각도 적분 결과: {angle_integral:.6e}")

        if abs(angle_integral) < 1e-10:
            self._log_debug("  ❌ 심각: 각도 적분이 0! E''가 문제입니다.")

        # 6. Calculate full integrand for this q
        self._log_debug("\n\n[6] 전체 피적분함수 테스트")
        self._log_debug("-" * 50)

        C_q_test = self.psd_model(np.array([test_q]))[0]
        G_test = G_matrix[mid_q_idx, mid_v_idx]

        # Calculate P and S
        if G_test > 0 and np.isfinite(G_test):
            P_test = erf(1.0 / (2.0 * np.sqrt(G_test)))
        else:
            P_test = 1.0
        S_test = gamma + (1 - gamma) * P_test**2

        q3_test = test_q**3
        qCPS_test = q3_test * C_q_test * P_test * S_test
        full_integrand = qCPS_test * angle_integral

        self._log_debug(f"  q = {test_q:.2e}")
        self._log_debug(f"  C(q) = {C_q_test:.4e}")
        self._log_debug(f"  G(q) = {G_test:.4e}")
        self._log_debug(f"  P(q) = erf(1/(2√G)) = {P_test:.6f}")
        self._log_debug(f"  S(q) = γ + (1-γ)P² = {S_test:.6f}")
        self._log_debug(f"  q³ = {q3_test:.4e}")
        self._log_debug(f"  q³·C·P·S = {qCPS_test:.4e}")
        self._log_debug(f"  angle_integral = {angle_integral:.4e}")
        self._log_debug(f"  피적분함수 = q³·C·P·S × angle_int = {full_integrand:.6e}")

        if abs(full_integrand) < 1e-15:
            self._log_debug("\n  ❌ 피적분함수가 거의 0!")
            if P_test < 0.01:
                self._log_debug("     → P(q)가 너무 작음 (G가 너무 큼)")
            if abs(angle_integral) < 1e-10:
                self._log_debug("     → 각도 적분이 0 (E''가 문제)")

        # 7. Existing mu_visc results
        self._log_debug("\n\n[7] μ_visc 결과 확인")
        self._log_debug("-" * 50)

        if hasattr(self, 'mu_visc_results') and self.mu_visc_results is not None:
            mu_array = self.mu_visc_results.get('mu', [])
            v_array = self.mu_visc_results.get('v', [])

            if len(mu_array) > 0:
                mu_min = np.min(mu_array)
                mu_max = np.max(mu_array)
                mu_mean = np.mean(mu_array)

                self._log_debug(f"  ✓ μ_visc 결과 있음")
                self._log_debug(f"  • μ_visc 범위: {mu_min:.6f} ~ {mu_max:.6f}")
                self._log_debug(f"  • μ_visc 평균: {mu_mean:.6f}")

                if mu_max < 1e-6:
                    self._log_debug("\n  ❌ 심각: μ_visc가 거의 0!")
                    self._log_debug("     가능한 원인:")
                    self._log_debug("     1. E''(손실탄성률)가 너무 작음")
                    self._log_debug("     2. P(q)(접촉면적비)가 너무 작음 → G(q)가 너무 큼")
                    self._log_debug("     3. 각도 적분이 0")
                    self._log_debug("     4. 마스터 커브 주파수 범위가 맞지 않음")
        else:
            self._log_debug("  ℹ️ μ_visc 계산 결과가 아직 없습니다.")
            self._log_debug("     → Tab 5에서 μ_visc 계산을 실행하세요.")

        # 8. Summary and recommendations
        self._log_debug("\n\n" + "=" * 70)
        self._log_debug("  진단 요약 및 권장 사항")
        self._log_debug("=" * 70)

        issues_found = False

        # Check for common issues
        if P_max < 0.01:
            issues_found = True
            self._log_debug("\n  ⚠️ P(q)가 매우 작음 (< 0.01)")
            self._log_debug("     권장: σ₀ (공칭 압력)를 높이거나 표면 거칠기 확인")

        if abs(angle_integral) < 1e-10:
            issues_found = True
            self._log_debug("\n  ⚠️ 각도 적분이 0에 가까움")
            self._log_debug("     권장: 마스터 커브 데이터 (특히 E'') 확인")
            self._log_debug("     → ω 범위가 qv 범위를 포함하는지 확인")

        if any_nan_E:
            issues_found = True
            self._log_debug("\n  ⚠️ E' 또는 E''에 NaN 값 존재")
            self._log_debug("     권장: 마스터 커브 생성 과정 확인")

        if not issues_found:
            self._log_debug("\n  ✓ 주요 문제가 발견되지 않았습니다.")
            self._log_debug("    계산이 정상적으로 진행되어야 합니다.")

        self._log_debug("\n\n진단 완료.\n")

    def _create_friction_factors_tab(self, parent):
        """Create friction factors analysis tab - explains how to increase/decrease μ_visc."""
        # Main scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Mouse wheel scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Title
        title_frame = ttk.LabelFrame(scrollable_frame, text="μ_visc 영향 인자 분석 - 0.1~10 m/s 범위에서 마찰계수 조절 가이드", padding=15)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        content = """
════════════════════════════════════════════════════════════════════════════════
                   μ_visc 마찰계수 영향 인자 분석
════════════════════════════════════════════════════════════════════════════════

【μ_visc 공식】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  μ_visc = (1/2) × ∫[q₀→q₁] q³ C(q) P(q) S(q) × [∫₀²π cosφ × Im[E(qv·cosφ)] dφ] / ((1-ν²)σ₀) dq

  분해하면:
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ μ_visc ∝ q³ × C(q) × P(q) × S(q) × Im[E(ω)] / σ₀                           │
  └─────────────────────────────────────────────────────────────────────────────┘


【영향 인자별 상세 분석】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─ 1. C(q) - PSD 표면 거칠기 ──────────────────────────────────────────────────┐
│                                                                              │
│  【μ 증가】 C(q) ↑                                                           │
│  ├─ 표면이 더 거칠면 → C(q) 증가 → μ 증가                                    │
│  ├─ C(q₀) 값 증가 (PSD 설정에서)                                             │
│  └─ Hurst 지수 H 감소 → 고주파 성분 증가 → μ 증가                            │
│                                                                              │
│  【코드 위치】                                                                │
│  ├─ Tab 1: PSD 파일 로드                                                     │
│  └─ Tab 2: PSD 설정 (q₀, C(q₀), H)                                           │
│                                                                              │
│  ※ 영향도: ★★★★★ (가장 큰 영향)                                              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 2. P(q) - 접촉 면적 비율 ───────────────────────────────────────────────────┐
│                                                                              │
│  P(q) = erf(1 / (2√G(q)))                                                    │
│                                                                              │
│  【μ 증가】 P(q) ↑ ← G(q) ↓                                                   │
│  ├─ G(q)가 작으면 → P(q) ≈ 1 (완전 접촉)                                     │
│  ├─ σ₀ (압력) ↑ → G(q) ↓ → P(q) ↑ → μ ↑                                      │
│  │   (단, 분모의 σ₀도 증가하므로 순효과 확인 필요)                            │
│  └─ |E*| ↓ (더 부드러운 재료) → G(q) ↓ → P(q) ↑                              │
│                                                                              │
│  【P(q) = 1로 단순화하면】                                                    │
│  일부 구현에서는 P(q)=1, S(q)=1로 가정 → μ가 더 높게 계산됨!                 │
│  → 이것이 0.43 vs 0.6 차이의 주요 원인일 수 있음                             │
│                                                                              │
│  ※ 영향도: ★★★★☆                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 3. S(q) - 접촉 보정 인자 ───────────────────────────────────────────────────┐
│                                                                              │
│  S(q) = γ + (1-γ) × P(q)²                                                    │
│                                                                              │
│  【μ 증가】 S(q) ↑                                                            │
│  ├─ γ = 1.0 설정 → S(q) = 1 (항상 최대)                                      │
│  ├─ γ = 0.5 (기본값) → S(q) = 0.5 + 0.5×P²                                   │
│  └─ γ = 0 설정 → S(q) = P² (최소)                                            │
│                                                                              │
│  【S(q) = 1로 단순화하면】                                                    │
│  γ=1 또는 S(q)=1 가정 → μ 증가                                               │
│                                                                              │
│  【코드 위치】                                                                │
│  └─ persson_model/core/friction.py: gamma 파라미터                           │
│                                                                              │
│  ※ 영향도: ★★★☆☆                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 4. Im[E(ω)] - 손실 탄성률 ──────────────────────────────────────────────────┐
│                                                                              │
│  ω = q × v × cosφ  (주파수 = 파수 × 속도)                                    │
│                                                                              │
│  【μ 증가】 Im[E(ω)] = E''(ω) ↑                                               │
│  ├─ E'' 피크가 높은 재료 → tan(δ)_max 큰 재료                                │
│  ├─ E'' 피크 위치가 qv 범위 내에 있어야 함                                   │
│  │   - v = 1 m/s, q = 10⁴ → ω ≈ 10⁴ rad/s → f ≈ 1600 Hz                     │
│  │   - v = 1 m/s, q = 10⁶ → ω ≈ 10⁶ rad/s → f ≈ 160 kHz                     │
│  └─ 마스터 커브가 이 주파수 범위를 포함해야 함                               │
│                                                                              │
│  【0.1~10 m/s 범위에서 μ 증가】                                               │
│  ├─ 이 속도 범위의 ω 범위: ~10² ~ 10⁸ rad/s                                  │
│  ├─ E'' 피크가 이 범위에 있으면 μ 증가                                       │
│  └─ 온도 ↓ → WLF shift → E'' 피크 이동 (고주파 쪽)                           │
│                                                                              │
│  【코드 위치】                                                                │
│  ├─ Tab 0: 마스터 커브 생성 (DMA 데이터)                                     │
│  └─ Tab 2: 온도 설정 (WLF shift)                                             │
│                                                                              │
│  ※ 영향도: ★★★★★ (C(q)와 함께 가장 큰 영향)                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 5. σ₀ - 공칭 압력 ──────────────────────────────────────────────────────────┐
│                                                                              │
│  μ_visc ∝ 1/σ₀  (분모에 있음)                                                │
│                                                                              │
│  【μ 증가】 σ₀ ↓                                                              │
│  ├─ 압력 감소 → μ 증가 (직접적)                                              │
│  ├─ 단, G(q) ∝ 1/σ₀² → σ₀↓ 시 G↑ → P↓ (간접적으로 μ 감소)                   │
│  └─ 순효과는 상황에 따라 다름                                                │
│                                                                              │
│  【코드 위치】                                                                │
│  └─ Tab 2: 공칭 압력 (MPa) 설정                                              │
│                                                                              │
│  ※ 영향도: ★★★☆☆                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─ 6. q 적분 범위 (q₀ ~ q₁) ───────────────────────────────────────────────────┐
│                                                                              │
│  【μ 증가】 적분 범위 확대                                                    │
│  ├─ q₁ ↑ → 더 미세한 거칠기 포함 → μ 증가                                    │
│  ├─ q₀ ↓ → 더 큰 스케일 거칠기 포함                                          │
│  └─ q₁은 h'rms(ξ) 목표값으로 결정                                            │
│                                                                              │
│  【주의】                                                                     │
│  └─ q₁이 너무 크면 PSD 데이터 범위 초과 → 외삽 필요                          │
│                                                                              │
│  【코드 위치】                                                                │
│  └─ Tab 2: q_min, q_max 설정 또는 h'rms(ξ)/q₁ 모드                           │
│                                                                              │
│  ※ 영향도: ★★★☆☆                                                             │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘


【구현 차이로 인한 μ 값 차이 원인】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  이 프로그램 μ ≈ 0.43  vs  다른 구현 μ ≈ 0.6 차이 원인:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. P(q), S(q) 포함 여부                                                     │
  │    ├─ 이 프로그램: P(q) = erf(...), S(q) = γ + (1-γ)P² 적용                │
  │    └─ 단순 구현: P(q) = 1, S(q) = 1로 가정 → μ 더 높음 (약 1.3~1.5배)      │
  │                                                                             │
  │ 2. γ 값 차이                                                                │
  │    ├─ 이 프로그램: γ = 0.5 (기본값)                                         │
  │    └─ 다른 구현: γ = 1.0 → S(q) = 1                                         │
  │                                                                             │
  │ 3. G(q) 계산 방식                                                           │
  │    ├─ 누적 적분 vs 단순 공식                                                │
  │    └─ |E*|² 계산 시 평균 방식 차이                                          │
  │                                                                             │
  │ 4. 각도 적분 처리                                                           │
  │    ├─ 이 프로그램: ∫cosφ × Im[E(qv·cosφ)] dφ 정확히 계산                   │
  │    └─ 단순화: 평균값 근사 사용                                              │
  └─────────────────────────────────────────────────────────────────────────────┘


【0.1~10 m/s에서 μ 높이는 체크리스트】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  □ 1. PSD C(q) 증가
     ├─ C(q₀) 값 증가
     ├─ Hurst 지수 H 감소 (0.8 → 0.6)
     └─ 더 거친 표면 데이터 사용

  □ 2. 재료 E'' 최적화
     ├─ tan(δ)_max가 큰 재료 선택
     ├─ E'' 피크가 계산 주파수 범위에 있는지 확인
     └─ 온도 조정으로 E'' 피크 이동

  □ 3. 압력 σ₀ 조정
     └─ σ₀ 감소 시도 (단, P(q) 영향 고려)

  □ 4. q 범위 확대
     └─ q₁ (목표 h'rms 또는 직접 입력) 증가

  □ 5. γ 값 조정 (고급)
     └─ friction.py에서 gamma=1.0 으로 설정 시 S(q)=1

  □ 6. P(q)=1, S(q)=1 단순화 (비교용)
     └─ 다른 구현과 비교 시 이 가정 사용


【결론】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  μ_visc = f(C(q), P(q), S(q), E''(ω), σ₀, q범위)

  • C(q)와 E''(ω)가 가장 큰 영향 (재료 및 표면 특성)
  • P(q), S(q) 적용 여부가 구현 간 차이의 주요 원인
  • 단순 구현 (P=1, S=1)은 μ를 과대평가할 수 있음

════════════════════════════════════════════════════════════════════════════════
"""

        text_widget = tk.Text(title_frame, wrap=tk.WORD, font=('Courier New', 9), height=50, width=90)
        text_widget.insert(tk.END, content)
        text_widget.config(state='disabled')  # Read-only
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


def main():
    """Run the enhanced application."""
    root = tk.Tk()
    app = PerssonModelGUI_V2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
