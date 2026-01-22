"""
Main GUI Window for Persson Friction Model
===========================================

Provides interactive interface for:
- Material property input
- Surface roughness specification
- G(q) calculation
- Contact mechanics analysis
- Result visualization
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import json
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.g_calculator import GCalculator
from core.psd_models import FractalPSD, MeasuredPSD
from core.viscoelastic import ViscoelasticMaterial
from core.contact import ContactMechanics
from utils.output import (
    save_calculation_details_csv,
    save_summary_txt,
    export_for_plotting,
    format_parameters_dict
)


class PerssonModelGUI:
    """Main GUI application for Persson friction model."""

    def __init__(self, root):
        """Initialize GUI."""
        self.root = root
        self.root.title("Persson 마찰 모델 계산기")
        self.root.geometry("1400x900")

        # Initialize variables
        self.material = None
        self.psd_model = None
        self.g_calculator = None
        self.contact_mechanics = None
        self.results = {}

        # Create UI
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

        # Load default material
        self._load_default_material()

    def _create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="재료 물성 불러오기", command=self._load_material)
        file_menu.add_command(label="PSD 데이터 불러오기", command=self._load_psd_data)
        file_menu.add_separator()
        file_menu.add_command(label="결과 요약 저장 (TXT)", command=self._save_results)
        file_menu.add_command(label="상세 결과 저장 (CSV)", command=self._save_detailed_csv)
        file_menu.add_command(label="모든 결과 내보내기", command=self._export_all_results)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)

        # Material menu
        material_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="재료", menu=material_menu)
        material_menu.add_command(label="SBR (예제)", command=lambda: self._load_example_material("SBR"))
        material_menu.add_command(label="PDMS (예제)", command=lambda: self._load_example_material("PDMS"))

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="사용법", command=self._show_help)
        help_menu.add_command(label="정보", command=self._show_about)

    def _create_main_layout(self):
        """Create main application layout."""
        # Create main container with paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Input parameters
        left_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(left_frame, weight=1)
        self._create_input_panel(left_frame)

        # Right panel: Results and plots
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        self._create_results_panel(right_frame)

    def _create_input_panel(self, parent):
        """Create input parameter panel."""
        # Notebook for different input sections
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Operating conditions
        conditions_frame = ttk.Frame(notebook)
        notebook.add(conditions_frame, text="작동 조건")
        self._create_conditions_inputs(conditions_frame)

        # Tab 2: Surface roughness
        roughness_frame = ttk.Frame(notebook)
        notebook.add(roughness_frame, text="표면 거칠기")
        self._create_roughness_inputs(roughness_frame)

        # Tab 3: Material properties
        material_frame = ttk.Frame(notebook)
        notebook.add(material_frame, text="재료 물성")
        self._create_material_display(material_frame)

        # Tab 4: Calculation settings
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="계산 설정")
        self._create_settings_inputs(settings_frame)

        # Calculate button
        calc_frame = ttk.Frame(parent)
        calc_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(
            calc_frame,
            text="계산 실행",
            command=self._run_calculation,
            style="Accent.TButton"
        ).pack(fill=tk.X, pady=5)

    def _create_conditions_inputs(self, parent):
        """Create operating conditions input fields."""
        frame = ttk.LabelFrame(parent, text="작동 조건 입력", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Nominal pressure
        row = 0
        ttk.Label(frame, text="명목 접촉 압력 (MPa):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sigma_0_var = tk.StringVar(value="1.0")
        ttk.Entry(frame, textvariable=self.sigma_0_var, width=15).grid(row=row, column=1, pady=5)

        # Sliding velocity
        row += 1
        ttk.Label(frame, text="미끄럼 속도 (m/s):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.velocity_var = tk.StringVar(value="0.01")
        ttk.Entry(frame, textvariable=self.velocity_var, width=15).grid(row=row, column=1, pady=5)

        # Temperature
        row += 1
        ttk.Label(frame, text="온도 (°C):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.temperature_var = tk.StringVar(value="20")
        ttk.Entry(frame, textvariable=self.temperature_var, width=15).grid(row=row, column=1, pady=5)

        # Poisson's ratio
        row += 1
        ttk.Label(frame, text="Poisson 비:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.poisson_var = tk.StringVar(value="0.5")
        ttk.Entry(frame, textvariable=self.poisson_var, width=15).grid(row=row, column=1, pady=5)

    def _create_roughness_inputs(self, parent):
        """Create surface roughness input fields."""
        frame = ttk.LabelFrame(parent, text="표면 거칠기 모델", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # PSD model type
        ttk.Label(frame, text="PSD 모델 유형:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.psd_type_var = tk.StringVar(value="fractal")
        psd_combo = ttk.Combobox(
            frame,
            textvariable=self.psd_type_var,
            values=["fractal", "measured"],
            state="readonly",
            width=12
        )
        psd_combo.grid(row=0, column=1, pady=5)
        psd_combo.bind("<<ComboboxSelected>>", self._on_psd_type_change)

        # Fractal parameters frame
        self.fractal_frame = ttk.Frame(frame)
        self.fractal_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(self.fractal_frame, text="Hurst 지수 H:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.hurst_var = tk.StringVar(value="0.8")
        ttk.Entry(self.fractal_frame, textvariable=self.hurst_var, width=15).grid(row=0, column=1, pady=2)

        ttk.Label(self.fractal_frame, text="RMS 거칠기 (μm):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.rms_roughness_var = tk.StringVar(value="10.0")
        ttk.Entry(self.fractal_frame, textvariable=self.rms_roughness_var, width=15).grid(row=1, column=1, pady=2)

        # Wavenumber range
        ttk.Label(frame, text="파수 범위:").grid(row=2, column=0, sticky=tk.W, pady=5)

        range_frame = ttk.Frame(frame)
        range_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))

        ttk.Label(range_frame, text="q_min (1/m):").grid(row=0, column=0, sticky=tk.W)
        self.q_min_var = tk.StringVar(value="100")
        ttk.Entry(range_frame, textvariable=self.q_min_var, width=12).grid(row=0, column=1, padx=5)

        ttk.Label(range_frame, text="q_max (1/m):").grid(row=1, column=0, sticky=tk.W)
        self.q_max_var = tk.StringVar(value="1e8")
        ttk.Entry(range_frame, textvariable=self.q_max_var, width=12).grid(row=1, column=1, padx=5)

    def _create_material_display(self, parent):
        """Create material properties display."""
        frame = ttk.LabelFrame(parent, text="재료 정보", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.material_info_text = tk.Text(frame, height=15, width=40)
        self.material_info_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, command=self.material_info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.material_info_text.config(yscrollcommand=scrollbar.set)

    def _create_settings_inputs(self, parent):
        """Create calculation settings."""
        frame = ttk.LabelFrame(parent, text="계산 설정", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(frame, text="파수 포인트 수:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.n_q_points_var = tk.StringVar(value="100")
        ttk.Entry(frame, textvariable=self.n_q_points_var, width=15).grid(row=0, column=1, pady=5)

        ttk.Label(frame, text="각도 적분 포인트:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.n_angle_var = tk.StringVar(value="36")
        ttk.Entry(frame, textvariable=self.n_angle_var, width=15).grid(row=1, column=1, pady=5)

        ttk.Label(frame, text="적분 방법:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.integration_method_var = tk.StringVar(value="trapz")
        ttk.Combobox(
            frame,
            textvariable=self.integration_method_var,
            values=["trapz", "simpson", "quad"],
            state="readonly",
            width=12
        ).grid(row=2, column=1, pady=5)

    def _create_results_panel(self, parent):
        """Create results display panel."""
        # Notebook for different result views
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: G(q) plot
        self.g_plot_frame = ttk.Frame(notebook)
        notebook.add(self.g_plot_frame, text="G(q) 그래프")
        self._create_plot_canvas(self.g_plot_frame, "g_plot")

        # Tab 2: Stress distribution
        self.stress_plot_frame = ttk.Frame(notebook)
        notebook.add(self.stress_plot_frame, text="응력 분포")
        self._create_plot_canvas(self.stress_plot_frame, "stress_plot")

        # Tab 3: Contact area
        self.contact_plot_frame = ttk.Frame(notebook)
        notebook.add(self.contact_plot_frame, text="접촉 면적")
        self._create_plot_canvas(self.contact_plot_frame, "contact_plot")

        # Tab 4: Numerical results
        self.results_frame = ttk.Frame(notebook)
        notebook.add(self.results_frame, text="수치 결과")
        self._create_results_display(self.results_frame)

    def _create_plot_canvas(self, parent, plot_name):
        """Create matplotlib canvas for plotting."""
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        # Store references
        setattr(self, f"{plot_name}_fig", fig)
        setattr(self, f"{plot_name}_canvas", canvas)

    def _create_results_display(self, parent):
        """Create numerical results display."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.results_text = tk.Text(frame, font=("Courier", 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, command=self.results_text.yview)
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

    def _on_psd_type_change(self, event=None):
        """Handle PSD type selection change."""
        if self.psd_type_var.get() == "fractal":
            self.fractal_frame.grid()
        else:
            self.fractal_frame.grid_remove()

    def _load_default_material(self):
        """Load default material (SBR)."""
        self._load_example_material("SBR")

    def _load_example_material(self, material_type):
        """Load example material."""
        try:
            if material_type == "SBR":
                self.material = ViscoelasticMaterial.create_example_sbr()
            elif material_type == "PDMS":
                self.material = ViscoelasticMaterial.create_example_pdms()

            self._update_material_display()
            self.status_var.set(f"{material_type} 재료 로드 완료")
        except Exception as e:
            messagebox.showerror("오류", f"재료 로드 실패: {str(e)}")

    def _update_material_display(self):
        """Update material information display."""
        if self.material is None:
            return

        self.material_info_text.delete(1.0, tk.END)

        info = f"재료명: {self.material.name}\n"
        info += f"기준 온도: {self.material.reference_temp}°C\n\n"

        if self.material.C1 is not None:
            info += f"WLF 파라미터:\n"
            info += f"  C1 = {self.material.C1:.2f}\n"
            info += f"  C2 = {self.material.C2:.2f} K\n\n"

        info += "주파수 범위:\n"
        info += f"  {self.material._frequencies[0]:.2e} - {self.material._frequencies[-1]:.2e} rad/s\n"

        self.material_info_text.insert(1.0, info)

    def _run_calculation(self):
        """Run Persson model calculation."""
        try:
            self.status_var.set("계산 중...")
            self.root.update()

            # Get parameters
            sigma_0 = float(self.sigma_0_var.get()) * 1e6  # MPa to Pa
            velocity = float(self.velocity_var.get())
            poisson = float(self.poisson_var.get())
            q_min = float(self.q_min_var.get())
            q_max = float(self.q_max_var.get())
            n_q = int(self.n_q_points_var.get())
            n_angle = int(self.n_angle_var.get())

            # Create wavenumber array
            q_values = np.logspace(np.log10(q_min), np.log10(q_max), n_q)

            # Create PSD model
            if self.psd_type_var.get() == "fractal":
                hurst = float(self.hurst_var.get())
                h_rms = float(self.rms_roughness_var.get()) * 1e-6  # μm to m

                self.psd_model = FractalPSD(
                    hurst_exponent=hurst,
                    rms_roughness=h_rms,
                    q_min=q_min,
                    q_max=q_max
                )

            # Create G calculator
            self.g_calculator = GCalculator(
                psd_func=self.psd_model,
                modulus_func=lambda w: self.material.get_modulus(w),
                sigma_0=sigma_0,
                velocity=velocity,
                poisson_ratio=poisson,
                n_angle_points=n_angle,
                integration_method=self.integration_method_var.get()
            )

            # Calculate G(q) with detailed intermediate values
            self.status_var.set("G(q) 계산 중 (상세 모드)...")
            self.root.update()

            # Calculate detailed results
            detailed_results = self.g_calculator.calculate_G_with_details(q_values, q_min=q_min)
            G_values = detailed_results['G']

            # Create contact mechanics calculator
            self.contact_mechanics = ContactMechanics(
                G_function=lambda q: np.interp(q, q_values, G_values),
                sigma_0=sigma_0,
                q_values=q_values,
                G_values=G_values
            )

            # Store results (including detailed intermediate values)
            self.results = {
                'q_values': q_values,
                'G_values': G_values,
                'sigma_0': sigma_0,
                'velocity': velocity,
                'contact_stats': self.contact_mechanics.contact_statistics(),
                'detailed_results': detailed_results  # Store all intermediate values
            }

            # Update displays
            self._plot_g_function()
            self._plot_stress_distribution()
            self._plot_contact_area()
            self._display_numerical_results()

            self.status_var.set("계산 완료")

        except Exception as e:
            messagebox.showerror("계산 오류", f"계산 중 오류 발생:\n{str(e)}")
            self.status_var.set("오류 발생")
            import traceback
            traceback.print_exc()

    def _plot_g_function(self):
        """Plot G(q) function."""
        fig = self.g_plot_fig
        fig.clear()
        ax = fig.add_subplot(111)

        q_values = self.results['q_values']
        G_values = self.results['G_values']
        zeta = q_values / q_values[0]

        ax.loglog(zeta, G_values, 'b-', linewidth=2, label='G(ζ)')
        ax.set_xlabel('배율 ζ = q/q₀', fontsize=12)
        ax.set_ylabel('G(ζ)', fontsize=12)
        ax.set_title('Persson G 함수', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        self.g_plot_canvas.draw()

    def _plot_stress_distribution(self):
        """Plot stress probability distribution."""
        fig = self.stress_plot_fig
        fig.clear()
        ax = fig.add_subplot(111)

        sigma, P_sigma = self.contact_mechanics.plot_stress_distribution()

        ax.plot(sigma / 1e6, P_sigma * 1e6, 'r-', linewidth=2)
        ax.axvline(
            self.results['sigma_0'] / 1e6,
            color='k',
            linestyle='--',
            label=f'σ₀ = {self.results["sigma_0"]/1e6:.2f} MPa'
        )
        ax.set_xlabel('응력 σ (MPa)', fontsize=12)
        ax.set_ylabel('확률 밀도 P(σ) (1/MPa)', fontsize=12)
        ax.set_title('접촉 응력 분포', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        self.stress_plot_canvas.draw()

    def _plot_contact_area(self):
        """Plot contact area vs magnification."""
        fig = self.contact_plot_fig
        fig.clear()
        ax = fig.add_subplot(111)

        zeta, area_frac = self.contact_mechanics.contact_area_vs_magnification()

        ax.semilogx(zeta, area_frac, 'g-', linewidth=2)
        ax.set_xlabel('배율 ζ', fontsize=12)
        ax.set_ylabel('접촉 면적 비율 A/A₀', fontsize=12)
        ax.set_title('배율에 따른 실 접촉 면적', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])

        fig.tight_layout()
        self.contact_plot_canvas.draw()

    def _display_numerical_results(self):
        """Display numerical calculation results."""
        self.results_text.delete(1.0, tk.END)

        stats = self.results['contact_stats']

        output = "=" * 60 + "\n"
        output += "Persson 마찰 모델 계산 결과\n"
        output += "=" * 60 + "\n\n"

        output += "입력 조건:\n"
        output += "-" * 60 + "\n"
        output += f"  명목 접촉 압력:     {self.results['sigma_0']/1e6:10.3f} MPa\n"
        output += f"  미끄럼 속도:        {self.results['velocity']:10.4f} m/s\n"
        output += f"  Poisson 비:        {float(self.poisson_var.get()):10.3f}\n"
        output += f"  온도:              {float(self.temperature_var.get()):10.1f} °C\n\n"

        output += "거칠기 파라미터:\n"
        output += "-" * 60 + "\n"
        if self.psd_type_var.get() == "fractal":
            output += f"  Hurst 지수:        {float(self.hurst_var.get()):10.3f}\n"
            output += f"  RMS 거칠기:        {float(self.rms_roughness_var.get()):10.3f} μm\n"
        output += f"  파수 범위:          {float(self.q_min_var.get()):.2e} - {float(self.q_max_var.get()):.2e} 1/m\n\n"

        output += "계산 결과:\n"
        output += "=" * 60 + "\n"
        output += f"  G(ζ_max):          {self.results['G_values'][-1]:10.6f}\n"
        output += f"  실 접촉 면적 비율:  {stats['area_fraction']:10.6f} ({stats['area_fraction']*100:.3f}%)\n"
        output += f"  평균 접촉 압력:     {stats['mean_pressure']/1e6:10.3f} MPa\n"
        output += f"  RMS 압력 변동:     {stats['rms_pressure']/1e6:10.3f} MPa\n"
        output += f"  응력 분산:         {np.sqrt(stats['stress_variance'])/1e6:10.3f} MPa\n"
        output += f"  배율 ζ_max:        {stats['magnification']:10.2f}\n\n"

        output += "=" * 60 + "\n"

        self.results_text.insert(1.0, output)

    def _load_material(self):
        """Load material DMA data from file."""
        filename = filedialog.askopenfilename(
            title="DMA 데이터 파일 선택",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                from utils.data_loader import load_dma_from_file, create_material_from_dma

                # Load DMA data
                omega, E_storage, E_loss = load_dma_from_file(
                    filename,
                    skip_header=1,  # Skip first line (header)
                    freq_unit='Hz',
                    modulus_unit='MPa'
                )

                # Create material
                material_name = os.path.splitext(os.path.basename(filename))[0]
                self.material = create_material_from_dma(
                    omega=omega,
                    E_storage=E_storage,
                    E_loss=E_loss,
                    material_name=material_name,
                    reference_temp=float(self.temperature_var.get())
                )

                self._update_material_display()

                messagebox.showinfo("성공",
                    f"재료 데이터 로드 완료:\n{filename}\n\n"
                    f"재료명: {self.material.name}\n"
                    f"데이터 포인트: {len(omega)}개\n"
                    f"주파수 범위: {omega[0]:.2e} ~ {omega[-1]:.2e} rad/s\n"
                    f"E' 범위: {E_storage.min()/1e6:.2f} ~ {E_storage.max()/1e6:.2f} MPa")

            except Exception as e:
                messagebox.showerror("오류", f"재료 파일 로드 실패:\n{str(e)}")
                import traceback
                traceback.print_exc()

    def _load_psd_data(self):
        """Load PSD data from file."""
        filename = filedialog.askopenfilename(
            title="PSD 데이터 파일 선택",
            filetypes=[
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if filename:
            try:
                from utils.data_loader import load_psd_from_file, create_psd_from_data

                # Load PSD data
                q, C_q = load_psd_from_file(
                    filename,
                    skip_header=1  # Skip first line (header)
                )

                # Create PSD model
                self.psd_model = create_psd_from_data(
                    q=q,
                    C_q=C_q,
                    interpolation_kind='log-log'
                )

                # Update q range
                self.q_min_var.set(f"{q[0]:.2e}")
                self.q_max_var.set(f"{q[-1]:.2e}")

                # Switch to measured PSD mode
                self.psd_type_var.set("measured")
                self._on_psd_type_change()

                messagebox.showinfo("성공",
                    f"PSD 데이터 로드 완료:\n{filename}\n\n"
                    f"데이터 포인트: {len(q)}개\n"
                    f"파수 범위: {q[0]:.2e} ~ {q[-1]:.2e} 1/m\n"
                    f"PSD 범위: {C_q.min():.2e} ~ {C_q.max():.2e} m⁴")

            except Exception as e:
                messagebox.showerror("오류", f"PSD 파일 로드 실패:\n{str(e)}")
                import traceback
                traceback.print_exc()

    def _save_results(self):
        """Save calculation results summary to text file."""
        if not self.results:
            messagebox.showwarning("경고", "저장할 결과가 없습니다. 먼저 계산을 실행하세요.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("성공", f"결과 요약이 저장되었습니다:\n{filename}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 실패:\n{str(e)}")

    def _save_detailed_csv(self):
        """Save detailed calculation results to CSV file."""
        if not self.results or 'detailed_results' not in self.results:
            messagebox.showwarning("경고", "저장할 상세 결과가 없습니다. 먼저 계산을 실행하세요.")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Get parameters
                params = format_parameters_dict(
                    sigma_0=self.results['sigma_0'],
                    velocity=self.results['velocity'],
                    temperature=float(self.temperature_var.get()),
                    poisson_ratio=float(self.poisson_var.get()),
                    q_min=float(self.q_min_var.get()),
                    q_max=float(self.q_max_var.get()),
                    material_name=self.material.name if self.material else "Unknown",
                    hurst_exponent=float(self.hurst_var.get()) if self.psd_type_var.get() == "fractal" else "N/A",
                    rms_roughness_um=float(self.rms_roughness_var.get()) if self.psd_type_var.get() == "fractal" else "N/A"
                )

                # Save CSV
                save_calculation_details_csv(
                    self.results['detailed_results'],
                    filename,
                    params
                )

                messagebox.showinfo("성공",
                    f"상세 계산 결과가 CSV로 저장되었습니다:\n{filename}\n\n"
                    "포함된 열:\n"
                    "- Index, log_q, q, C(q)\n"
                    "- Avg_Modulus_Term\n"
                    "- G_Integrand, Delta_G\n"
                    "- G(q), Contact_Area_Ratio")
            except Exception as e:
                messagebox.showerror("오류", f"CSV 파일 저장 실패:\n{str(e)}")
                import traceback
                traceback.print_exc()

    def _export_all_results(self):
        """Export all results (CSV, TXT, plotting data)."""
        if not self.results or 'detailed_results' not in self.results:
            messagebox.showwarning("경고", "내보낼 결과가 없습니다. 먼저 계산을 실행하세요.")
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(title="결과 파일을 저장할 폴더를 선택하세요")

        if output_dir:
            try:
                # Get parameters
                params = format_parameters_dict(
                    sigma_0=self.results['sigma_0'],
                    velocity=self.results['velocity'],
                    temperature=float(self.temperature_var.get()),
                    poisson_ratio=float(self.poisson_var.get()),
                    q_min=float(self.q_min_var.get()),
                    q_max=float(self.q_max_var.get()),
                    material_name=self.material.name if self.material else "Unknown",
                    hurst_exponent=float(self.hurst_var.get()) if self.psd_type_var.get() == "fractal" else "N/A",
                    rms_roughness_um=float(self.rms_roughness_var.get()) if self.psd_type_var.get() == "fractal" else "N/A"
                )

                detailed_results = self.results['detailed_results']

                # Save detailed CSV
                csv_file = os.path.join(output_dir, "persson_detailed_results.csv")
                save_calculation_details_csv(detailed_results, csv_file, params)

                # Save summary TXT
                txt_file = os.path.join(output_dir, "persson_summary.txt")
                save_summary_txt(detailed_results, txt_file, params)

                # Export plotting data
                export_for_plotting(detailed_results, output_dir=output_dir, prefix='persson')

                messagebox.showinfo("성공",
                    f"모든 결과가 저장되었습니다:\n{output_dir}\n\n"
                    "생성된 파일:\n"
                    "- persson_detailed_results.csv (전체 계산 데이터)\n"
                    "- persson_summary.txt (요약)\n"
                    "- persson_G_vs_q.csv (G 함수 플롯)\n"
                    "- persson_contact_area.csv (접촉 면적 플롯)\n"
                    "- persson_PSD.csv (PSD 플롯)")

            except Exception as e:
                messagebox.showerror("오류", f"결과 내보내기 실패:\n{str(e)}")
                import traceback
                traceback.print_exc()

    def _show_help(self):
        """Show help dialog."""
        help_text = """
Persson 마찰 모델 계산기 사용법

1. 재료 선택
   - 메뉴 > 재료에서 예제 재료 선택
   - 또는 파일에서 재료 물성 불러오기

2. 조건 입력
   - 작동 조건 탭: 압력, 속도, 온도 입력
   - 표면 거칠기 탭: PSD 모델 및 파라미터 설정

3. 계산 실행
   - "계산 실행" 버튼 클릭
   - 결과 탭에서 그래프 및 수치 확인

4. 결과 저장
   - 파일 > 결과 저장
        """
        messagebox.showinfo("사용법", help_text)

    def _show_about(self):
        """Show about dialog."""
        about_text = """
Persson 마찰 모델 계산기
버전 1.0.0

거친 표면과 점탄성 재료의 접촉 역학 및
마찰을 계산하는 Persson 이론 구현

주요 기능:
- G(q) 계산 (이중 적분)
- 접촉 응력 분포
- 실 접촉 면적
- 점탄성 재료 물성

개발: Persson Modelling Team
        """
        messagebox.showinfo("정보", about_text)


def main():
    """Run the application."""
    root = tk.Tk()
    app = PerssonModelGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
