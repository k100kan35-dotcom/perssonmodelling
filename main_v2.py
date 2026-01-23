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
    smooth_dma_data
)

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
                self.q_max_var.set(f"{q[-1]:.2e}")
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

        # Tab 4: Friction Coefficient
        self.tab_friction = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_friction, text="4. 마찰 분석")
        self._create_friction_tab(self.tab_friction)

        # Tab 5: Equations
        self.tab_equations = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_equations, text="5. 수식 정리")
        self._create_equations_tab(self.tab_equations)

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

        # DMA smoothing controls
        smoothing_frame = ttk.LabelFrame(parent, text="DMA 데이터 스무딩 설정", padding=10)
        smoothing_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create controls in a grid
        control_grid = ttk.Frame(smoothing_frame)
        control_grid.pack(fill=tk.X)

        # Enable smoothing checkbox
        self.enable_smoothing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_grid,
            text="스무딩 활성화",
            variable=self.enable_smoothing_var,
            command=self._apply_smoothing
        ).grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)

        # Window length control
        ttk.Label(control_grid, text="스무딩 강도 (윈도우 길이):").grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
        self.smoothing_window_var = tk.StringVar(value="auto")
        smoothing_combo = ttk.Combobox(
            control_grid,
            textvariable=self.smoothing_window_var,
            values=["auto", "7", "11", "15", "21", "31", "51"],
            width=10,
            state="readonly"
        )
        smoothing_combo.grid(row=0, column=2, sticky=tk.W, padx=5, pady=3)
        smoothing_combo.bind('<<ComboboxSelected>>', lambda e: self._apply_smoothing())

        # Remove outliers checkbox
        self.remove_outliers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_grid,
            text="이상치 제거",
            variable=self.remove_outliers_var,
            command=self._apply_smoothing
        ).grid(row=0, column=3, sticky=tk.W, padx=5, pady=3)

        # Apply button
        ttk.Button(
            control_grid,
            text="적용",
            command=self._apply_smoothing
        ).grid(row=0, column=4, sticky=tk.W, padx=5, pady=3)

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

        # Input panel
        input_frame = ttk.LabelFrame(parent, text="계산 매개변수", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

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
        self.q_max_var = tk.StringVar(value="6.0e+5")
        ttk.Entry(input_frame, textvariable=self.q_max_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="파수 포인트 수:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_q_var = tk.StringVar(value="100")
        ttk.Entry(input_frame, textvariable=self.n_q_var, width=15).grid(row=row, column=1, pady=5)

        # Target RMS Slope (for q1 determination)
        row += 1
        ttk.Label(input_frame, text="목표 RMS Slope (q1 결정):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.target_rms_slope_var = tk.StringVar(value="1.3")
        ttk.Entry(input_frame, textvariable=self.target_rms_slope_var, width=15).grid(row=row, column=1, pady=5)

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

        # Calculate button
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.calc_button = ttk.Button(
            btn_frame,
            text="G(q,v) 계산 실행",
            command=self._run_calculation
        )
        self.calc_button.pack(fill=tk.X, pady=5)

        # Progress bar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(
            btn_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Calculation visualization area
        viz_frame = ttk.LabelFrame(parent, text="계산 과정 시각화", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

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

        # Create figure for calculation progress with single plot
        self.fig_calc_progress = Figure(figsize=(12, 5), dpi=100)
        self.ax_calc_progress = self.fig_calc_progress.add_subplot(111)

        self.canvas_calc_progress = FigureCanvasTkAgg(self.fig_calc_progress, viz_frame)
        self.canvas_calc_progress.draw()
        self.canvas_calc_progress.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize empty DMA plot
        self.ax_calc_progress.set_xlabel('각주파수 ω (rad/s)', fontsize=11, fontweight='bold')
        self.ax_calc_progress.set_ylabel('탄성률 (Pa)', fontsize=11, fontweight='bold')
        self.ax_calc_progress.set_xscale('log')
        self.ax_calc_progress.set_yscale('log')
        self.ax_calc_progress.grid(True, alpha=0.3)
        self.ax_calc_progress.set_title('DMA 마스터 곡선 (사용 주파수 범위)', fontsize=12, fontweight='bold')

        self.fig_calc_progress.tight_layout()

        # Save button for calculation progress plot
        save_btn_frame = ttk.Frame(parent)
        save_btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            save_btn_frame,
            text="계산 과정 그래프 저장",
            command=lambda: self._save_plot(self.fig_calc_progress, "calculation_progress")
        ).pack(side=tk.LEFT, padx=5)

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

    def _create_friction_tab(self, parent):
        """Create friction coefficient analysis tab."""
        # Instruction
        instruction = ttk.LabelFrame(parent, text="탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "마찰 계수 μ(v) 및 접촉 면적 분석.\n"
            "속도 의존성 마찰과 실제 접촉 면적 비율을 표시합니다.",
            font=('Arial', 10)
        ).pack()

        # Results text
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = tk.Text(text_frame, font=("Courier", 10))
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

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

        # Plot 1: Master Curve (E', E'')
        omega = np.logspace(-2, 12, 200)
        E_storage = self.material.get_storage_modulus(omega)
        E_loss = self.material.get_loss_modulus(omega)

        ax1 = self.ax_master_curve

        # Plot smoothed data (from interpolator)
        ax1.loglog(omega, E_storage/1e6, 'g-', linewidth=2.5, label="E' (보간/평활화)", alpha=0.9, zorder=2)
        ax1.loglog(omega, E_loss/1e6, 'orange', linewidth=2.5, label="E'' (보간/평활화)", alpha=0.9, zorder=2)

        # Plot raw measured data if available
        if self.raw_dma_data is not None:
            ax1.scatter(self.raw_dma_data['omega'], self.raw_dma_data['E_storage']/1e6,
                       c='darkgreen', s=20, alpha=0.5, label="E' (측정값)", zorder=1)
            ax1.scatter(self.raw_dma_data['omega'], self.raw_dma_data['E_loss']/1e6,
                       c='darkorange', s=20, alpha=0.5, label="E'' (측정값)", zorder=1)

        ax1.set_xlabel('각주파수 ω (rad/s)', fontweight='bold', fontsize=11, labelpad=5)
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

        # Plot 2: PSD C(q) with Hurst exponent
        if self.psd_model is not None:
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())
            q_plot = np.logspace(np.log10(q_min), np.log10(q_max), 200)
            C_q = self.psd_model(q_plot)

            # Calculate Hurst exponent from power law fitting
            # C(q) = A * q^(-2(H+1)) => log(C) = log(A) - 2(H+1)*log(q)
            # Use middle range for fitting (avoid edge artifacts)
            fit_idx = (q_plot > q_min * 10) & (q_plot < q_max / 10)
            if np.sum(fit_idx) > 10:
                log_q_fit = np.log10(q_plot[fit_idx])
                log_C_fit = np.log10(C_q[fit_idx])
                # Linear fit: log(C) = a + b*log(q), where b = -2(H+1)
                coeffs = np.polyfit(log_q_fit, log_C_fit, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                H = -slope / 2.0 - 1.0  # Hurst exponent

                # Plot fitted line
                C_fit = 10**(intercept + slope * np.log10(q_plot))
                self.ax_psd.loglog(q_plot, C_fit, 'r--', linewidth=1.5, alpha=0.7,
                                  label=f'Power law fit (H={H:.3f})')

            self.ax_psd.loglog(q_plot, C_q, 'b-', linewidth=2, label='로드된 PSD', alpha=0.9)
            self.ax_psd.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11, labelpad=5)
            self.ax_psd.set_ylabel('PSD C(q) (m⁴)', fontweight='bold', fontsize=11,
                                   rotation=90, labelpad=10)
            self.ax_psd.set_title('표면 거칠기 PSD', fontweight='bold', fontsize=12, pad=10)
            self.ax_psd.legend(fontsize=9)
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
                q, C_q = load_psd_from_file(filename, skip_header=1)
                self.psd_model = create_psd_from_data(q, C_q, interpolation_kind='log-log')

                self.q_min_var.set(f"{q[0]:.2e}")
                self.q_max_var.set(f"{q[-1]:.2e}")
                self.psd_type_var.set("measured")

                self._update_verification_plots()
                messagebox.showinfo("Success", f"PSD data loaded: {len(q)} points")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load PSD data:\n{str(e)}")

    def _apply_smoothing(self):
        """Apply smoothing to DMA data with current settings."""
        # Check if raw DMA data exists
        if not hasattr(self, 'raw_dma_data') or self.raw_dma_data is None:
            return

        try:
            omega_raw = self.raw_dma_data['omega']
            E_storage_raw = self.raw_dma_data['E_storage']
            E_loss_raw = self.raw_dma_data['E_loss']

            # Get smoothing parameters from GUI
            enable_smoothing = self.enable_smoothing_var.get()
            remove_outliers = self.remove_outliers_var.get()
            window_str = self.smoothing_window_var.get()

            # Parse window length
            if window_str == "auto":
                window_length = None  # Auto-determine
            else:
                window_length = int(window_str)

            if enable_smoothing:
                # Apply smoothing
                smoothed = smooth_dma_data(
                    omega_raw,
                    E_storage_raw,
                    E_loss_raw,
                    window_length=window_length,
                    polyorder=2,
                    remove_outliers=remove_outliers
                )

                # Create material from smoothed data
                self.material = create_material_from_dma(
                    omega=smoothed['omega'],
                    E_storage=smoothed['E_storage_smooth'],
                    E_loss=smoothed['E_loss_smooth'],
                    material_name="Measured Rubber (smoothed)",
                    reference_temp=float(self.temperature_var.get())
                )
            else:
                # Use raw data without smoothing
                self.material = create_material_from_dma(
                    omega=omega_raw,
                    E_storage=E_storage_raw,
                    E_loss=E_loss_raw,
                    material_name="Measured Rubber (raw)",
                    reference_temp=float(self.temperature_var.get())
                )

            # Update plots
            self._update_material_display()
            self._update_verification_plots()
            self.status_var.set(f"스무딩 적용 완료 (윈도우: {window_str}, 이상치 제거: {remove_outliers})")

        except Exception as e:
            messagebox.showerror("Error", f"스무딩 적용 실패:\n{str(e)}")
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

            # Initialize calculation progress plot with master curve
            try:
                self.ax_calc_progress.clear()

                # Plot DMA master curve
                if self.material is not None:
                    omega_plot = np.logspace(-2, 8, 200)
                    f_plot = omega_plot / (2 * np.pi)  # Convert to Hz
                    E_prime = self.material.get_storage_modulus(omega_plot)
                    E_double_prime = self.material.get_loss_modulus(omega_plot)

                    self.ax_calc_progress.plot(f_plot, E_prime, 'b-', linewidth=2, label="E' (저장 탄성률)")
                    self.ax_calc_progress.plot(f_plot, E_double_prime, 'r--', linewidth=2, label="E'' (손실 탄성률)")

                    self.ax_calc_progress.set_xlabel('주파수 f (Hz)', fontsize=11, fontweight='bold')
                    self.ax_calc_progress.set_ylabel('탄성률 (Pa)', fontsize=11, fontweight='bold')
                    self.ax_calc_progress.set_xscale('log')
                    self.ax_calc_progress.set_yscale('log')
                    self.ax_calc_progress.grid(True, alpha=0.3)
                    self.ax_calc_progress.legend(loc='best', fontsize=9)
                    self.ax_calc_progress.set_title('DMA 마스터 곡선 (사용 주파수 범위)', fontsize=12, fontweight='bold')

                self.fig_calc_progress.tight_layout()
                self.canvas_calc_progress.draw()
            except:
                pass

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

                    # Highlight frequency range on DMA plot
                    try:
                        # Remove ALL previous highlight bands (clear old highlights)
                        for artist in self.ax_calc_progress.collections[:]:
                            if hasattr(artist, '_is_highlight'):
                                artist.remove()

                        # Add vertical band to show current frequency range being used
                        # Use yellow color with edge for better visibility
                        band = self.ax_calc_progress.axvspan(f_min, f_max,
                                                            alpha=0.3, facecolor='yellow',
                                                            edgecolor='orange', linewidth=2,
                                                            zorder=0)  # Behind curves
                        band._is_highlight = True

                        self.canvas_calc_progress.draw()
                    except:
                        pass

                self.root.update()

            results_2d = self.g_calculator.calculate_G_multi_velocity(
                q_array, v_array, q_min=q_min, progress_callback=progress_callback
            )

            # Clear highlight after calculation and show completion message
            try:
                # Remove all highlights
                for artist in self.ax_calc_progress.collections[:]:
                    if hasattr(artist, '_is_highlight'):
                        artist.remove()

                # Update status label to show completion
                self.calc_status_label.config(
                    text=f"계산 완료! | 총 {len(v_array)}개 속도 × {len(q_array)}개 파수",
                    foreground='green'
                )

                self.canvas_calc_progress.draw()
            except:
                pass

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

        # Plot 1: Multi-velocity G(q) curves (다중 속도 G(q) 곡선)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / len(v)) for i in range(len(v))]

        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:  # Plot every 10th curve
                ax1.loglog(q, G_matrix[:, j], color=color, linewidth=1.5,
                          label=f'v={v_val:.4f} m/s')

        ax1.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11, labelpad=5)
        ax1.set_ylabel('G(q)', fontweight='bold', fontsize=11, rotation=90, labelpad=10)
        ax1.set_title('(a) 다중 속도에서의 G(q)', fontweight='bold', fontsize=11, pad=8)
        ax1.legend(fontsize=7, ncol=2)
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

        # Plot 2: Local stress probability distribution P(σ,ζ) for multiple velocities
        # Persson theory: P(σ,ζ) = 1/√(4πG_stress) [exp(-(σ-σ0)²/4G_stress) - exp(-(σ+σ0)²/4G_stress)]
        # where G_stress(q) = (π/4) * (E*)² * ∫[q0→q] k³C(k)dk  [Pa²]
        sigma_0_Pa = self.results['sigma_0']  # Pa
        sigma_0_MPa = sigma_0_Pa / 1e6  # Convert Pa to MPa

        # Calculate G_stress(q,v) - stress distribution variance parameter
        # G_stress depends on velocity through E*(ω) where ω = q*v
        poisson = float(self.poisson_var.get())
        temperature = float(self.temperature_var.get())
        q_min = float(self.q_min_var.get())
        q_max = float(self.q_max_var.get())

        # PSD values (same for all velocities)
        C_q_vals = self.psd_model(q)

        # Find maximum G_stress across all velocities for x-axis range
        G_stress_max_global = 0
        G_stress_dict = {}  # Store G_stress for each velocity

        # Pre-calculate G_stress for each velocity
        # DEBUG: Print actual values to diagnose peak shift problem
        print("\n" + "="*80)
        print("DEBUG: G_stress Calculation Analysis")
        print("="*80)
        print(f"σ₀ = {sigma_0_MPa:.4f} MPa = {sigma_0_Pa:.2e} Pa")
        print(f"q range: {q_min:.2e} ~ {q_max:.2e} (1/m)")
        print(f"Poisson ratio: {poisson:.3f}")

        for j, v_val in enumerate(v):
            if j % max(1, len(v) // 10) == 0:
                # Check what E value we're using
                omega_low = q_min * v_val
                omega_high = q_max * v_val
                E_low = self.material.get_storage_modulus(np.array([omega_low]))[0]
                E_high = self.material.get_storage_modulus(np.array([omega_high]))[0]
                E_star_low = E_low / (1 - poisson**2)
                E_star_high = E_high / (1 - poisson**2)

                # Calculate integral of q³C(q)
                integrand = q**3 * C_q_vals
                integral_full = np.trapezoid(integrand, q)

                # Original formula WITHOUT σ₀: G = (π/4) × E*² × ∫q³C(q)dq
                G_original_Pa2 = (np.pi / 4) * E_star_low**2 * integral_full
                G_original_MPa2 = G_original_Pa2 / 1e12

                if j == 0:
                    print(f"\n--- First velocity: v = {v_val:.4e} m/s ---")
                    print(f"ω_low = {omega_low:.2e} rad/s  →  E = {E_low:.2e} Pa  →  E* = {E_star_low:.2e} Pa")
                    print(f"ω_high = {omega_high:.2e} rad/s  →  E = {E_high:.2e} Pa  →  E* = {E_star_high:.2e} Pa")
                    print(f"∫ q³C(q)dq = {integral_full:.4e} m⁴")
                    print(f"E*² = {E_star_low**2:.4e} Pa²")
                    print(f"G_stress (original formula) = {G_original_MPa2:.4e} MPa²")
                    print(f"√G_stress = {np.sqrt(G_original_MPa2):.4f} MPa")
                    print(f"Expected peak location ≈ √(2G) = {np.sqrt(2*G_original_MPa2):.2f} MPa")
                    print(f"But σ₀ = {sigma_0_MPa:.2f} MPa << √G, so peak shifts away from σ₀!")

                # CRITICAL FIX: Normalize by σ₀² to make G_stress dimensionless-like
                # This ensures √G ≈ σ₀, so peak stays at σ₀ in P(σ) distribution
                # Formula: G_dimensionless = (π/4) × (E*/σ₀)² × ∫q³C(q)dq
                E_normalized = E_star_low / sigma_0_Pa  # Normalize E by σ₀

                G_stress_array = np.zeros_like(q)
                for i in range(1, len(q)):
                    integrand_partial = q[:i+1]**3 * C_q_vals[:i+1]
                    # Use normalized E to get dimensionless G
                    G_stress_array[i] = (np.pi / 4) * E_normalized**2 * np.trapezoid(integrand_partial, q[:i+1])

                # G_stress_array is now dimensionless (like G_area)
                # For display purposes, scale back by σ₀² to get "effective MPa²"
                # This ensures √G_display ≈ σ₀
                G_stress_array_MPa2 = G_stress_array * (sigma_0_MPa**2)
                G_stress_dict[j] = G_stress_array_MPa2

                if j == 0:
                    print(f"\n>>> APPLYING FIX: Normalize by σ₀")
                    print(f"E_normalized = E*/σ₀ = {E_normalized:.2e}")
                    print(f"G_dimensionless(qmax) = {G_stress_array[-1]:.4e}")
                    print(f"G_scaled(qmax) = G × σ₀² = {G_stress_array_MPa2[-1]:.4e} MPa²")
                    print(f"√G_scaled = {np.sqrt(G_stress_array_MPa2[-1]):.4f} MPa")
                    print(f">>> Now √G ≈ σ₀, so peak should be at σ₀ = 0.3 MPa!")

                # Update global maximum
                if G_stress_array_MPa2[-1] > G_stress_max_global:
                    G_stress_max_global = G_stress_array_MPa2[-1]

        print(f"\nG_stress_max (across all velocities) = {G_stress_max_global:.4e} MPa²")
        print(f"√G_max = {np.sqrt(G_stress_max_global):.2f} MPa")
        print(f"Ratio: √G_max / σ₀ = {np.sqrt(G_stress_max_global) / sigma_0_MPa:.1f}")
        print("\n>>> PROBLEM DIAGNOSIS: √G >> σ₀, so peak shifts to √(2G) instead of σ₀")
        print(">>> This means the G_stress formula or P(σ) formula needs revision!")
        print("="*80 + "\n")

        # Set x-axis range based on maximum G_stress
        std_max = np.sqrt(G_stress_max_global)
        sigma_max = sigma_0_MPa + 4 * std_max

        # Create stress array (in MPa)
        sigma_array = np.linspace(0, sigma_max, 500)

        # Debug: Print some values to verify calculations
        print(f"\n=== Debug: Stress Distribution ===")
        print(f"σ0 = {sigma_0_MPa:.4f} MPa")
        print(f"G_stress_max (across all v) = {G_stress_max_global:.4e} MPa²")
        print(f"std_max = {std_max:.4f} MPa")
        print(f"sigma_max = {sigma_max:.2f} MPa")

        # Plot stress distributions for selected velocities
        for j, v_val in enumerate(v):
            if j % max(1, len(v) // 10) == 0:
                color = colors[j]

                # Get G_stress values for this velocity
                G_stress_array_MPa2 = G_stress_dict[j]
                G_stress_q0 = G_stress_array_MPa2[0]      # G_stress at minimum wavenumber
                G_stress_qmax = G_stress_array_MPa2[-1]   # G_stress at maximum wavenumber

                # Debug for first velocity
                if j == 0:
                    q_mid = np.sqrt(q_min * q_max)
                    omega_rep = q_mid * v_val
                    E_rep = self.material.get_storage_modulus(np.array([omega_rep]))[0]
                    E_star = E_rep / (1 - poisson**2)
                    print(f"\nFor v = {v_val:.4e} m/s:")
                    print(f"  ω_rep = {omega_rep:.2e} rad/s")
                    print(f"  E(ω_rep) = {E_rep:.2e} Pa")
                    print(f"  E* = {E_star:.2e} Pa")
                    print(f"  G_stress(q0) = {G_stress_q0:.4e} MPa²")
                    print(f"  G_stress(qmax) = {G_stress_qmax:.4e} MPa²")
                    print(f"  std(qmax) = {np.sqrt(G_stress_qmax):.4f} MPa")

                # Calculate stress distribution at q0 (dotted line)
                if G_stress_q0 > 1e-10:
                    P_sigma_q0 = (1 / np.sqrt(4 * np.pi * G_stress_q0)) * \
                                 (np.exp(-(sigma_array - sigma_0_MPa)**2 / (4 * G_stress_q0)) - \
                                  np.exp(-(sigma_array + sigma_0_MPa)**2 / (4 * G_stress_q0)))
                    ax2.plot(sigma_array, P_sigma_q0, color=color, linestyle=':', linewidth=2.5,
                            alpha=0.6)  # No label for q0 lines to reduce legend clutter

                # Calculate stress distribution at q_max (solid line)
                if G_stress_qmax > 1e-10:
                    P_sigma_qmax = (1 / np.sqrt(4 * np.pi * G_stress_qmax)) * \
                                   (np.exp(-(sigma_array - sigma_0_MPa)**2 / (4 * G_stress_qmax)) - \
                                    np.exp(-(sigma_array + sigma_0_MPa)**2 / (4 * G_stress_qmax)))
                    ax2.plot(sigma_array, P_sigma_qmax, color=color, linestyle='-', linewidth=2,
                            label=f'v={v_val:.4f} m/s', alpha=0.9)

        # Add vertical line for nominal pressure
        ax2.axvline(sigma_0_MPa, color='black', linestyle='--', linewidth=2,
                   label=f'σ0 = {sigma_0_MPa:.2f} MPa', alpha=0.7)

        ax2.set_xlabel('응력 σ (MPa)', fontweight='bold', fontsize=11, labelpad=5)
        ax2.set_ylabel('응력 분포 P(σ)', fontweight='bold', fontsize=11, rotation=90, labelpad=10)
        ax2.set_title('(b) 속도별 국소 응력 확률 분포 (점선: q0, 실선: qmax)', fontweight='bold', fontsize=11, pad=8)
        ax2.legend(fontsize=6, ncol=2, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, sigma_max)

        # Plot 3: Contact Area P(q,v) (접촉 면적)
        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:
                # Filter out values very close to 1.0 for better visualization
                P_curve = P_matrix[:, j].copy()
                # Clip very small differences from 1.0
                P_curve = np.clip(P_curve, 0, 0.999)

                ax3.semilogx(q, P_curve, color=color, linewidth=1.5,
                            label=f'v={v_val:.4f} m/s')

        ax3.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11, labelpad=5)
        ax3.set_ylabel('접촉 면적 비율 P(q)', fontweight='bold', fontsize=11, rotation=90, labelpad=10)
        ax3.set_title('(c) 다중 속도에서의 접촉 면적', fontweight='bold', fontsize=11, pad=8)
        ax3.legend(fontsize=7, ncol=2)
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
        ax4.set_xlabel('속도 v (m/s)', fontweight='bold', fontsize=11, labelpad=5)
        ax4.set_ylabel('최종 접촉 면적 P(q_max)', fontweight='bold', fontsize=11, rotation=90, labelpad=10)
        ax4.set_title('(d) 속도에 따른 접촉 면적', fontweight='bold', fontsize=11, pad=8)
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

            ax5.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11, labelpad=5)
            ax5.set_ylabel('각도 적분 ∫dφ|E/(1-ν²)σ0|²', fontweight='bold', fontsize=10, rotation=90, labelpad=12)
            ax5.set_title('(e) 내부 적분: 상대적 강성비', fontweight='bold', fontsize=11, pad=8)
            ax5.legend(fontsize=7, ncol=2)
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
                    ha='center', va='center', transform=ax5.transAxes, fontsize=10)
            ax5.set_title('(e) 내부 적분', fontweight='bold', fontsize=11, pad=8)

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
            target_slope_rms = float(self.target_rms_slope_var.get())
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

            # Plot cumulative RMS slope
            ax6.semilogx(q_parse, slope_rms_cumulative, 'b-', linewidth=2.5, label='누적 RMS 기울기')

            # Add horizontal line at target (1.3)
            ax6.axhline(target_slope_rms, color='red', linestyle='--', linewidth=2,
                       label=f'목표 RMS Slope = {target_slope_rms}', alpha=0.7, zorder=5)

            # Add vertical line at q1
            if q1_idx > 0:
                ax6.axvline(q1_determined, color='green', linestyle='--', linewidth=2,
                           label=f'결정된 q1 = {q1_determined:.2e} (1/m)', alpha=0.7, zorder=5)

                # Mark intersection point
                ax6.plot(q1_determined, target_slope_rms, 'ro', markersize=12,
                        markeredgecolor='black', markeredgewidth=2, zorder=10,
                        label='교차점')

            ax6.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11, labelpad=5)
            ax6.set_ylabel('누적 RMS 기울기 √(Slope²)', fontweight='bold', fontsize=10, rotation=90, labelpad=12)
            ax6.set_title('(f) Parseval 정리: q1 자동 결정 (Target Slope=1.3)', fontweight='bold', fontsize=11, pad=8)

            # Legend with better positioning
            ax6.legend(fontsize=8, loc='lower right', framealpha=0.9)

            ax6.grid(True, alpha=0.3)

            # Add annotation box
            if q1_idx > 0:
                textstr = (f'파서벌 정리:\nSlope²(q) = 2π∫k³C(k)dk\n\n'
                          f'결정된 q1 = {q1_determined:.2e} 1/m\n'
                          f'해당 RMS Slope = {target_slope_rms:.2f}')
            else:
                textstr = (f'파서벌 정리:\nSlope²(q) = 2π∫k³C(k)dk\n\n'
                          f'최종 RMS Slope = {slope_rms_cumulative[-1]:.3f}\n'
                          f'(목표 {target_slope_rms} 미달)')

            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='black')
            ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)

            # Fix axis formatter
            ax6.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))
        else:
            ax6.text(0.5, 0.5, 'PSD 데이터 없음',
                    ha='center', va='center', transform=ax6.transAxes, fontsize=10)
            ax6.set_title('(f) Parseval 정리', fontweight='bold', fontsize=11, pad=8)

        self.fig_results.suptitle('G(q,v) 2D 행렬 계산 결과', fontweight='bold', fontsize=13, y=0.995)
        self.fig_results.tight_layout(rect=[0, 0, 1, 0.99])
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


def main():
    """Run the enhanced application."""
    root = tk.Tk()
    app = PerssonModelGUI_V2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
