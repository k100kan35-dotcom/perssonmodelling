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
import matplotlib.pyplot as plt
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

        # Tab 5: RMS Analysis (Parseval theorem)
        self.tab_rms = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_rms, text="5. RMS 분석")
        self._create_rms_tab(self.tab_rms)

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
        self.sigma_0_var = tk.StringVar(value="1.0")
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
        self.q_min_var = tk.StringVar(value="2e1")
        ttk.Entry(input_frame, textvariable=self.q_min_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="최대 파수 q_max (1/m):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.q_max_var = tk.StringVar(value="1e9")
        ttk.Entry(input_frame, textvariable=self.q_max_var, width=15).grid(row=row, column=1, pady=5)

        row += 1
        ttk.Label(input_frame, text="파수 포인트 수:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.n_q_var = tk.StringVar(value="100")
        ttk.Entry(input_frame, textvariable=self.n_q_var, width=15).grid(row=row, column=1, pady=5)

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
            text="대기 중 | v = - m/s | q 범위 = - ~ - (1/m) | ω 범위 = - ~ - (rad/s)",
            font=('Arial', 10, 'bold'),
            foreground='blue'
        )
        self.calc_status_label.pack()

        # Create figure for calculation progress with 2 subplots
        self.fig_calc_progress = Figure(figsize=(14, 4), dpi=100)
        self.ax_calc_dma = self.fig_calc_progress.add_subplot(121)
        self.ax_calc_psd = self.fig_calc_progress.add_subplot(122)

        self.canvas_calc_progress = FigureCanvasTkAgg(self.fig_calc_progress, viz_frame)
        self.canvas_calc_progress.draw()
        self.canvas_calc_progress.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialize empty DMA plot
        self.ax_calc_dma.set_xlabel('각주파수 ω (rad/s)')
        self.ax_calc_dma.set_ylabel('저장 탄성률 E\' (Pa)')
        self.ax_calc_dma.set_xscale('log')
        self.ax_calc_dma.set_yscale('log')
        self.ax_calc_dma.grid(True, alpha=0.3)
        self.ax_calc_dma.set_title('DMA 마스터 곡선 (사용 주파수 범위)', fontsize=10, fontweight='bold')

        # Initialize empty PSD plot
        self.ax_calc_psd.set_xlabel('파수 q (1/m)')
        self.ax_calc_psd.set_ylabel('PSD C(q) (m⁴)')
        self.ax_calc_psd.set_xscale('log')
        self.ax_calc_psd.set_yscale('log')
        self.ax_calc_psd.grid(True, alpha=0.3)
        self.ax_calc_psd.set_title('PSD (사용 파수 범위)', fontsize=10, fontweight='bold')

        self.fig_calc_progress.tight_layout()

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

    def _create_rms_tab(self, parent):
        """Create RMS analysis tab based on Parseval theorem."""
        # Instruction
        instruction = ttk.LabelFrame(parent, text="탭 설명", padding=10)
        instruction.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(instruction, text=
            "Parseval 정리를 이용한 RMS 높이 및 기울기 분석.\n"
            "PSD로부터 표면 거칠기 특성을 정량화합니다.",
            font=('Arial', 10)
        ).pack()

        # Plot area
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create figure with 3 subplots
        self.fig_rms = Figure(figsize=(14, 10), dpi=100)

        self.ax_rms_cumulative = self.fig_rms.add_subplot(311)
        self.ax_rms_contribution = self.fig_rms.add_subplot(312)
        self.ax_rms_spectrum = self.fig_rms.add_subplot(313)

        self.canvas_rms = FigureCanvasTkAgg(self.fig_rms, plot_frame)
        self.canvas_rms.draw()
        self.canvas_rms.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas_rms, plot_frame)
        toolbar.update()

        # Calculate button
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            btn_frame,
            text="RMS 분석 실행",
            command=self._calculate_rms_analysis
        ).pack(side=tk.LEFT, padx=5)

        # Results summary
        self.rms_summary_var = tk.StringVar(value="분석 전")
        summary_label = ttk.Label(
            btn_frame,
            textvariable=self.rms_summary_var,
            font=('Arial', 10, 'bold'),
            foreground='blue'
        )
        summary_label.pack(side=tk.LEFT, padx=20)

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

            # Initialize calculation progress plots with master curves
            try:
                self.ax_calc_dma.clear()
                self.ax_calc_psd.clear()

                # Plot DMA master curve
                if self.material is not None:
                    omega_plot = np.logspace(-2, 8, 200)
                    E_prime = self.material.get_storage_modulus(omega_plot)
                    E_double_prime = self.material.get_loss_modulus(omega_plot)

                    self.ax_calc_dma.plot(omega_plot, E_prime, 'b-', linewidth=1.5, label="E' (저장 탄성률)")
                    self.ax_calc_dma.plot(omega_plot, E_double_prime, 'r--', linewidth=1.5, label="E'' (손실 탄성률)")

                    self.ax_calc_dma.set_xlabel('각주파수 ω (rad/s)', fontsize=10)
                    self.ax_calc_dma.set_ylabel('탄성률 (Pa)', fontsize=10)
                    self.ax_calc_dma.set_xscale('log')
                    self.ax_calc_dma.set_yscale('log')
                    self.ax_calc_dma.grid(True, alpha=0.3)
                    self.ax_calc_dma.legend(loc='best', fontsize=8)
                    self.ax_calc_dma.set_title('DMA 마스터 곡선 (사용 주파수 범위)', fontsize=10, fontweight='bold')

                # Plot PSD
                if self.psd_model is not None:
                    q_plot = np.logspace(np.log10(q_min), np.log10(q_max), 200)
                    C_q = self.psd_model(q_plot)

                    self.ax_calc_psd.loglog(q_plot, C_q, 'g-', linewidth=2, label='C(q)')
                    self.ax_calc_psd.set_xlabel('파수 q (1/m)', fontsize=10)
                    self.ax_calc_psd.set_ylabel('PSD C(q) (m⁴)', fontsize=10)
                    self.ax_calc_psd.grid(True, alpha=0.3)
                    self.ax_calc_psd.legend(loc='best', fontsize=8)
                    self.ax_calc_psd.set_title('PSD (사용 파수 범위)', fontsize=10, fontweight='bold')

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
                    q_min_used = q_array[0]
                    q_max_used = q_array[-1]

                    # Update main status bar
                    self.status_var.set(f"계산 중... v={current_v:.4f} m/s (ω: {omega_min:.2e} ~ {omega_max:.2e} rad/s)")

                    # Update calculation status label
                    self.calc_status_label.config(
                        text=f"계산 중 ({percent:.0f}%) | v = {current_v:.4f} m/s | "
                             f"q 범위 = {q_min_used:.2e} ~ {q_max_used:.2e} (1/m) | "
                             f"ω 범위 = {omega_min:.2e} ~ {omega_max:.2e} (rad/s)",
                        foreground='red'
                    )

                    # Highlight frequency and wavenumber ranges
                    try:
                        # Remove ALL previous highlight bands (clear old highlights)
                        for artist in self.ax_calc_dma.collections[:]:
                            if hasattr(artist, '_is_highlight'):
                                artist.remove()
                        for artist in self.ax_calc_psd.collections[:]:
                            if hasattr(artist, '_is_highlight'):
                                artist.remove()

                        # Add vertical band to DMA plot (current frequency range)
                        band_dma = self.ax_calc_dma.axvspan(omega_min, omega_max,
                                                            alpha=0.2, color='red', zorder=0)
                        band_dma._is_highlight = True

                        # Add vertical band to PSD plot (current wavenumber range)
                        band_psd = self.ax_calc_psd.axvspan(q_min_used, q_max_used,
                                                            alpha=0.2, color='red', zorder=0)
                        band_psd._is_highlight = True

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
                for artist in self.ax_calc_dma.collections[:]:
                    if hasattr(artist, '_is_highlight'):
                        artist.remove()
                for artist in self.ax_calc_psd.collections[:]:
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

        # Plot 2: Gaussian stress distribution p(σ) for multiple velocities
        # Based on Persson theory: p(σ) = (1/√(2πσ₀²G)) * exp(-(σ-σ₀)²/(2σ₀²G))
        # Get nominal pressure from calculation settings
        sigma_0_MPa = results['sigma_0'] / 1e6  # Convert Pa to MPa

        # Create stress array (in MPa)
        sigma_array = np.linspace(0, 3 * sigma_0_MPa, 500)

        for j, (v_val, color) in enumerate(zip(v, colors)):
            if j % max(1, len(v) // 10) == 0:
                # Get G value at the last wavenumber (most relevant for overall contact)
                G_val = G_matrix[-1, j]

                # Gaussian stress distribution
                # p(σ) = (1/√(2πσ₀²G)) * exp(-(σ-σ₀)²/(2σ₀²G))
                if G_val > 1e-10:
                    variance = (sigma_0_MPa**2) * G_val
                    p_sigma = (1 / np.sqrt(2 * np.pi * variance)) * \
                              np.exp(-(sigma_array - sigma_0_MPa)**2 / (2 * variance))

                    # Plot line
                    ax2.plot(sigma_array, p_sigma, color=color, linewidth=1.5,
                            label=f'v={v_val:.4f} m/s (G={G_val:.3f})', alpha=0.9)

                    # Fill area where stress > 0 (compressive region)
                    positive_mask = sigma_array >= 0
                    ax2.fill_between(sigma_array[positive_mask], 0, p_sigma[positive_mask],
                                    color=color, alpha=0.2)

        # Add vertical line for nominal pressure
        ax2.axvline(sigma_0_MPa, color='black', linestyle='--', linewidth=2,
                   label=f'σ₀ = {sigma_0_MPa:.2f} MPa', alpha=0.7)

        ax2.set_xlabel('응력 σ (MPa)', fontweight='bold', fontsize=11, labelpad=5)
        ax2.set_ylabel('응력 분포 p(σ) (1/MPa)', fontweight='bold', fontsize=11, rotation=90, labelpad=10)
        ax2.set_title('(b) 속도별 가우시안 응력 분포', fontweight='bold', fontsize=11, pad=8)
        ax2.legend(fontsize=7, ncol=2, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 3 * sigma_0_MPa)

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
        ax4.semilogx(v, P_final, 'ro-', linewidth=2, markersize=4)
        ax4.set_xlabel('속도 v (m/s)', fontweight='bold', fontsize=11, labelpad=5)
        ax4.set_ylabel('최종 접촉 면적 P(q_max)', fontweight='bold', fontsize=11, rotation=90, labelpad=10)
        ax4.set_title('(d) 속도에 따른 접촉 면적', fontweight='bold', fontsize=11, pad=8)
        ax4.grid(True, alpha=0.3)

        # Fix axis formatter
        ax4.xaxis.set_major_formatter(FuncFormatter(log_tick_formatter))

        # Plot 5: Inner integral vs q for multiple velocities (다중 속도에서의 내부 적분)
        # Physical meaning: ∫dφ |E(qvcosφ)|² - 슬립 방향 성분의 점탄성 응답
        if has_detailed:
            cmap_detail = plt.get_cmap('plasma')
            for i, detail_result in enumerate(detailed_multi_v):
                v_val = detail_result['velocity']
                color = cmap_detail(i / len(detailed_multi_v))
                ax5.loglog(detail_result['q'], detail_result['avg_modulus_term'],
                          color=color, linewidth=1.5, label=f'v={v_val:.4f} m/s', alpha=0.8)

            ax5.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11, labelpad=5)
            ax5.set_ylabel('각도 적분 ∫dφ|E(qvcosφ)|²', fontweight='bold', fontsize=10, rotation=90, labelpad=12)
            ax5.set_title('(e) 내부 적분: 슬립 방향 점탄성 응답', fontweight='bold', fontsize=11, pad=8)
            ax5.legend(fontsize=7, ncol=2)
            ax5.grid(True, alpha=0.3)

            # Add physical interpretation text box
            textstr = ('물리적 의미: 모든 슬립 방향(φ=0~2π)에 대해\n'
                      '탄성률의 제곱을 적분 → 에너지 손실량 측정\n'
                      '높은 값 = 더 많은 변형 에너지 소산 = 높은 마찰\n'
                      '각 파수(거칠기 스케일)의 마찰 기여도')
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

        # Plot 6: Parseval theorem - Cumulative h_rms and slope_rms from PSD
        # h_rms² = ∫ C(q) dq
        # slope_rms² = ∫ q² C(q) dq
        if self.psd_model is not None:
            # Calculate cumulative RMS height and slope
            q_parse = np.logspace(np.log10(q[0]), np.log10(q[-1]), 500)
            C_q_parse = self.psd_model(q_parse)

            # Cumulative integration (from q_min to each q)
            h_rms_cumulative = np.zeros_like(q_parse)
            slope_rms_cumulative = np.zeros_like(q_parse)

            for i in range(len(q_parse)):
                # Integrate from start to current q
                q_int = q_parse[:i+1]
                C_int = C_q_parse[:i+1]

                # Trapezoidal integration
                h_rms_cumulative[i] = np.sqrt(np.trapz(C_int, q_int))
                slope_rms_cumulative[i] = np.sqrt(np.trapz(q_int**2 * C_int, q_int))

            # Plot on twin axes
            ax6_twin = ax6.twinx()

            line1 = ax6.semilogx(q_parse, h_rms_cumulative * 1e6, 'b-', linewidth=2,
                                label='누적 h_rms', alpha=0.8)
            line2 = ax6_twin.semilogx(q_parse, slope_rms_cumulative, 'r-', linewidth=2,
                                     label='누적 slope_rms', alpha=0.8)

            ax6.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11, labelpad=5)
            ax6.set_ylabel('누적 h_rms (μm)', fontweight='bold', fontsize=10, color='b', rotation=90, labelpad=12)
            ax6_twin.set_ylabel('누적 RMS 기울기', fontweight='bold', fontsize=10, color='r', rotation=90, labelpad=12)
            ax6.set_title('(f) Parseval 정리: 누적 RMS 높이 및 기울기', fontweight='bold', fontsize=11, pad=8)

            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax6.legend(lines, labels, fontsize=8, loc='upper left')

            ax6.grid(True, alpha=0.3)
            ax6.tick_params(axis='y', labelcolor='b')
            ax6_twin.tick_params(axis='y', labelcolor='r')

            # Add final values as text
            textstr = (f'최종 h_rms = {h_rms_cumulative[-1]*1e6:.2f} μm\n'
                      f'최종 slope_rms = {slope_rms_cumulative[-1]:.3f}')
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            ax6.text(0.02, 0.98, textstr, transform=ax6.transAxes, fontsize=9,
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

    def _calculate_rms_analysis(self):
        """Calculate and plot RMS analysis based on Parseval theorem."""
        if self.psd_model is None:
            messagebox.showwarning("Warning", "No PSD data loaded!")
            return

        try:
            # Get q range from settings
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())

            # Create fine q array for integration
            q_array = np.logspace(np.log10(q_min), np.log10(q_max), 1000)
            C_q = self.psd_model(q_array)

            # Calculate cumulative RMS height and slope
            h_rms_cumulative = np.zeros_like(q_array)
            slope_rms_cumulative = np.zeros_like(q_array)

            for i in range(len(q_array)):
                q_int = q_array[:i+1]
                C_int = C_q[:i+1]

                # Parseval theorem: h_rms² = ∫ C(q) dq
                h_rms_cumulative[i] = np.sqrt(np.trapz(C_int, q_int))
                # RMS slope: slope² = ∫ q² C(q) dq
                slope_rms_cumulative[i] = np.sqrt(np.trapz(q_int**2 * C_int, q_int))

            # Calculate incremental contribution per decade
            log_q = np.log10(q_array)
            dh_rms_dlogq = np.gradient(h_rms_cumulative**2, log_q)
            dslope_rms_dlogq = np.gradient(slope_rms_cumulative**2, log_q)

            # Clear previous plots
            self.ax_rms_cumulative.clear()
            self.ax_rms_contribution.clear()
            self.ax_rms_spectrum.clear()

            # Plot 1: Cumulative RMS values
            ax1 = self.ax_rms_cumulative
            ax1_twin = ax1.twinx()

            line1 = ax1.semilogx(q_array, h_rms_cumulative * 1e6, 'b-', linewidth=2.5, label='누적 h_rms')
            line2 = ax1_twin.semilogx(q_array, slope_rms_cumulative, 'r-', linewidth=2.5, label='누적 slope_rms')

            ax1.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11)
            ax1.set_ylabel('누적 h_rms (μm)', fontweight='bold', fontsize=11, color='b')
            ax1_twin.set_ylabel('누적 RMS 기울기', fontweight='bold', fontsize=11, color='r')
            ax1.set_title('(a) Parseval 정리: 누적 RMS 값', fontweight='bold', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1_twin.tick_params(axis='y', labelcolor='r')

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left', fontsize=10)

            # Plot 2: Incremental contribution per decade
            ax2 = self.ax_rms_contribution
            ax2_twin = ax2.twinx()

            line3 = ax2.semilogx(q_array, dh_rms_dlogq * 1e12, 'b-', linewidth=2, label='dh²/d(log q)')
            line4 = ax2_twin.semilogx(q_array, dslope_rms_dlogq, 'r-', linewidth=2, label='d(slope²)/d(log q)')

            ax2.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11)
            ax2.set_ylabel('h² 기여도 (μm²/decade)', fontweight='bold', fontsize=11, color='b')
            ax2_twin.set_ylabel('slope² 기여도 (1/decade)', fontweight='bold', fontsize=11, color='r')
            ax2.set_title('(b) 파수별 기여도 (각 decade의 기여량)', fontweight='bold', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='y', labelcolor='b')
            ax2_twin.tick_params(axis='y', labelcolor='r')

            lines2 = line3 + line4
            labels2 = [l.get_label() for l in lines2]
            ax2.legend(lines2, labels2, loc='upper left', fontsize=10)

            # Plot 3: PSD spectrum with annotations
            ax3 = self.ax_rms_spectrum
            ax3.loglog(q_array, C_q, 'g-', linewidth=2.5, label='C(q)')

            # Calculate and show Hurst exponent
            mid_idx_start = len(q_array) // 3
            mid_idx_end = 2 * len(q_array) // 3
            log_q_fit = np.log10(q_array[mid_idx_start:mid_idx_end])
            log_C_fit = np.log10(C_q[mid_idx_start:mid_idx_end])
            slope_fit = np.polyfit(log_q_fit, log_C_fit, 1)[0]
            H = -slope_fit / 2 - 1

            ax3.set_xlabel('파수 q (1/m)', fontweight='bold', fontsize=11)
            ax3.set_ylabel('PSD C(q) (m⁴)', fontweight='bold', fontsize=11)
            ax3.set_title(f'(c) PSD 스펙트럼 (Hurst 지수 H ≈ {H:.3f})', fontweight='bold', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper right', fontsize=10)

            # Add wavelength scale annotations
            wavelengths = [1e-3, 1e-4, 1e-5, 1e-6]  # 1mm, 100μm, 10μm, 1μm
            for lam in wavelengths:
                q_lam = 2 * np.pi / lam
                if q_min < q_lam < q_max:
                    ax3.axvline(q_lam, color='gray', linestyle='--', alpha=0.5)
                    if lam >= 1e-3:
                        ax3.text(q_lam, ax3.get_ylim()[0] * 1.5, f'{lam*1e3:.0f}mm',
                                rotation=90, fontsize=8, alpha=0.7)
                    else:
                        ax3.text(q_lam, ax3.get_ylim()[0] * 1.5, f'{lam*1e6:.0f}μm',
                                rotation=90, fontsize=8, alpha=0.7)

            self.fig_rms.tight_layout()
            self.canvas_rms.draw()

            # Update summary
            final_h_rms = h_rms_cumulative[-1] * 1e6  # μm
            final_slope_rms = slope_rms_cumulative[-1]
            self.rms_summary_var.set(
                f"h_rms = {final_h_rms:.3f} μm | slope_rms = {final_slope_rms:.4f} | Hurst H = {H:.3f}"
            )
            self.status_var.set(f"RMS 분석 완료: h_rms={final_h_rms:.3f}μm, slope={final_slope_rms:.4f}")

        except Exception as e:
            messagebox.showerror("Error", f"RMS 분석 실패:\n{str(e)}")
            import traceback
            traceback.print_exc()

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


def main():
    """Run the enhanced application."""
    root = tk.Tk()
    app = PerssonModelGUI_V2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
