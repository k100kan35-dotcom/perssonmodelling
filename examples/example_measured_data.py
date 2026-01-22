"""
측정 데이터 사용 예제 (Using Measured Data)
===========================================

실제 측정된 PSD 및 DMA 데이터를 로드하여 Persson 모델 계산을 수행하는 예제입니다.

데이터 형식:
- PSD: q(1/m) vs C(q)(m^4)
- DMA: Frequency(Hz), E'(MPa), E''(MPa)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persson_model import GCalculator
from persson_model.core import ContactMechanics
from persson_model.utils.data_loader import (
    load_psd_from_file,
    load_dma_from_file,
    create_material_from_dma,
    create_psd_from_data
)
from persson_model.utils.output import (
    save_calculation_details_csv,
    save_summary_txt,
    format_parameters_dict
)


def main():
    print("=" * 80)
    print("Persson 모델 - 측정 데이터 사용 예제")
    print("=" * 80)

    # ========================================================================
    # 1. 측정 데이터 로드 (Load Measured Data)
    # ========================================================================
    print("\n[1] 측정 데이터 로드")
    print("-" * 80)

    # 데이터 파일 경로
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    psd_file = os.path.join(data_dir, 'measured_psd.txt')
    dma_file = os.path.join(data_dir, 'measured_dma.txt')

    # PSD 데이터 로드
    print(f"  PSD 파일: {psd_file}")
    q_measured, C_measured = load_psd_from_file(psd_file, skip_header=1)
    print(f"  ✓ PSD 데이터 로드 완료: {len(q_measured)}개 포인트")
    print(f"    파수 범위: {q_measured[0]:.2e} ~ {q_measured[-1]:.2e} 1/m")
    print(f"    PSD 범위: {C_measured.min():.2e} ~ {C_measured.max():.2e} m⁴")

    # DMA 데이터 로드
    print(f"\n  DMA 파일: {dma_file}")
    omega_measured, E_storage, E_loss = load_dma_from_file(
        dma_file,
        skip_header=1,
        freq_unit='Hz',     # 입력 데이터는 Hz 단위
        modulus_unit='MPa'  # 입력 데이터는 MPa 단위
    )
    print(f"  ✓ DMA 데이터 로드 완료: {len(omega_measured)}개 포인트")
    print(f"    주파수 범위: {omega_measured[0]:.2e} ~ {omega_measured[-1]:.2e} rad/s")
    print(f"    E' 범위: {E_storage.min()/1e6:.2f} ~ {E_storage.max()/1e6:.2f} MPa")
    print(f"    E'' 범위: {E_loss.min()/1e6:.2f} ~ {E_loss.max()/1e6:.2f} MPa")

    # ========================================================================
    # 2. 재료 및 PSD 모델 생성 (Create Models)
    # ========================================================================
    print("\n[2] 재료 및 PSD 모델 생성")
    print("-" * 80)

    # 재료 물성 생성
    material = create_material_from_dma(
        omega=omega_measured,
        E_storage=E_storage,
        E_loss=E_loss,
        material_name="Measured Rubber Material",
        reference_temp=20.0
    )
    print(f"  ✓ 재료 모델 생성: {material.name}")

    # PSD 모델 생성
    psd = create_psd_from_data(
        q=q_measured,
        C_q=C_measured,
        interpolation_kind='log-log'
    )
    print(f"  ✓ PSD 모델 생성 (log-log 보간)")

    # ========================================================================
    # 3. 계산 조건 설정 (Set Calculation Parameters)
    # ========================================================================
    print("\n[3] 계산 조건 설정")
    print("-" * 80)

    # 작동 조건
    sigma_0 = 1.0e6      # 1.0 MPa
    velocity = 0.01      # 0.01 m/s
    temperature = 20.0   # 20°C
    poisson = 0.5        # Poisson's ratio

    print(f"  명목 접촉 압력:     {sigma_0/1e6:.2f} MPa")
    print(f"  미끄럼 속도:       {velocity:.4f} m/s")
    print(f"  온도:             {temperature:.1f}°C")
    print(f"  Poisson 비:       {poisson:.2f}")

    # 파수 범위 설정 (측정 데이터 범위 사용)
    q_min = q_measured[0]
    q_max = q_measured[-1]

    # 계산용 파수 배열 생성 (로그 스케일)
    n_points = 100
    q_values = np.logspace(np.log10(q_min), np.log10(q_max), n_points)

    print(f"\n  계산 설정:")
    print(f"    파수 범위:        {q_min:.2e} ~ {q_max:.2e} 1/m")
    print(f"    계산 포인트 수:    {n_points}")
    print(f"    배율 범위:        1 ~ {q_max/q_min:.2e}")

    # ========================================================================
    # 4. G(q) 계산 (Calculate G(q))
    # ========================================================================
    print("\n[4] G(q) 계산 실행")
    print("-" * 80)

    # G calculator 생성
    g_calc = GCalculator(
        psd_func=psd,
        modulus_func=lambda w: material.get_modulus(w, temperature=temperature),
        sigma_0=sigma_0,
        velocity=velocity,
        poisson_ratio=poisson,
        n_angle_points=36,
        integration_method='trapz'
    )

    print("  계산 중...")

    # 상세 계산 실행
    results = g_calc.calculate_G_with_details(q_values, q_min=q_min)

    print(f"  ✓ 계산 완료!")
    print()
    print(f"  최종 결과:")
    print(f"    G(q_max)         = {results['G'][-1]:.6e}")
    print(f"    P(q_max)         = {results['contact_area_ratio'][-1]:.6f}")
    print(f"    접촉 면적 비율    = {results['contact_area_ratio'][-1]*100:.3f}%")

    # ========================================================================
    # 5. 접촉 역학 분석 (Contact Mechanics)
    # ========================================================================
    print("\n[5] 접촉 역학 분석")
    print("-" * 80)

    contact = ContactMechanics(
        G_function=lambda q: np.interp(q, q_values, results['G']),
        sigma_0=sigma_0,
        q_values=q_values,
        G_values=results['G']
    )

    stats = contact.contact_statistics()

    print(f"  실 접촉 면적 비율:  {stats['area_fraction']:.6f} ({stats['area_fraction']*100:.3f}%)")
    print(f"  평균 접촉 압력:    {stats['mean_pressure']/1e6:.2f} MPa")
    print(f"  RMS 압력 변동:    {stats['rms_pressure']/1e6:.2f} MPa")

    # ========================================================================
    # 6. 결과 저장 (Save Results)
    # ========================================================================
    print("\n[6] 결과 파일 저장")
    print("-" * 80)

    # 파라미터 정리
    params = format_parameters_dict(
        sigma_0=sigma_0,
        velocity=velocity,
        temperature=temperature,
        poisson_ratio=poisson,
        q_min=q_min,
        q_max=q_max,
        material_name=material.name,
        data_source="Measured PSD and DMA data"
    )

    # CSV 저장
    csv_file = 'measured_data_results.csv'
    save_calculation_details_csv(results, csv_file, params)
    print(f"  ✓ {csv_file}")

    # 요약 저장
    summary_file = 'measured_data_summary.txt'
    save_summary_txt(results, summary_file, params)
    print(f"  ✓ {summary_file}")

    # ========================================================================
    # 7. 시각화 (Visualization)
    # ========================================================================
    print("\n[7] 결과 시각화")
    print("-" * 80)

    fig = plt.figure(figsize=(16, 12))

    # Subplot 1: Measured PSD
    ax1 = plt.subplot(3, 3, 1)
    ax1.loglog(q_measured, C_measured, 'o', markersize=3, alpha=0.5, label='Measured data')
    ax1.loglog(q_values, results['C_q'], 'r-', linewidth=2, label='Interpolated')
    ax1.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax1.set_ylabel('PSD C(q) (m⁴)', fontsize=10)
    ax1.set_title('(a) Measured Surface PSD', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: DMA Master Curve
    ax2 = plt.subplot(3, 3, 2)
    ax2.loglog(omega_measured, E_storage/1e6, 'o', markersize=3, alpha=0.5, label="E' (data)")
    ax2.loglog(omega_measured, E_loss/1e6, 's', markersize=3, alpha=0.5, label="E'' (data)")
    # Interpolated
    omega_plot = np.logspace(np.log10(omega_measured[0]), np.log10(omega_measured[-1]), 200)
    E_plot = material.get_storage_modulus(omega_plot)
    E_loss_plot = material.get_loss_modulus(omega_plot)
    ax2.loglog(omega_plot, E_plot/1e6, 'g-', linewidth=2, label="E' (interp)")
    ax2.loglog(omega_plot, E_loss_plot/1e6, 'orange', linewidth=2, label="E'' (interp)")
    ax2.set_xlabel('Angular Frequency ω (rad/s)', fontsize=10)
    ax2.set_ylabel('Modulus (MPa)', fontsize=10)
    ax2.set_title('(b) DMA Master Curve', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: G(q)
    ax3 = plt.subplot(3, 3, 3)
    zeta = q_values / q_min
    ax3.loglog(zeta, results['G'], 'b-', linewidth=2)
    ax3.set_xlabel('Magnification ζ = q/q₀', fontsize=10)
    ax3.set_ylabel('G(ζ)', fontsize=10)
    ax3.set_title('(c) Persson G Function', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Contact Area
    ax4 = plt.subplot(3, 3, 4)
    ax4.semilogx(zeta, results['contact_area_ratio'], 'r-', linewidth=2)
    ax4.set_xlabel('Magnification ζ', fontsize=10)
    ax4.set_ylabel('Contact Area Ratio P(q)', fontsize=10)
    ax4.set_title('(d) Contact Area vs Magnification', fontsize=11, fontweight='bold')
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3)
    ax4.axhline(results['contact_area_ratio'][-1], color='k', linestyle='--',
                label=f'Final: {results["contact_area_ratio"][-1]:.4f}')
    ax4.legend()

    # Subplot 5: G Integrand
    ax5 = plt.subplot(3, 3, 5)
    ax5.loglog(q_values, results['G_integrand'], 'purple', linewidth=2)
    ax5.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax5.set_ylabel('G Integrand', fontsize=10)
    ax5.set_title('(e) G Integrand Distribution', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Delta_G
    ax6 = plt.subplot(3, 3, 6)
    ax6.loglog(q_values[1:], results['delta_G'][1:], 'orange', linewidth=2)
    ax6.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax6.set_ylabel('ΔG (incremental)', fontsize=10)
    ax6.set_title('(f) Incremental G Contribution', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Subplot 7: Avg Modulus Term
    ax7 = plt.subplot(3, 3, 7)
    ax7.loglog(q_values, results['avg_modulus_term'], 'brown', linewidth=2)
    ax7.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax7.set_ylabel('∫dφ |E/(1-ν²)σ₀|²', fontsize=10)
    ax7.set_title('(g) Avg Modulus Term', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Subplot 8: Cumulative G
    ax8 = plt.subplot(3, 3, 8)
    ax8.semilogx(q_values, results['G'], 'b-', linewidth=2)
    ax8.set_xlabel('Wavenumber q (1/m)', fontsize=10)
    ax8.set_ylabel('Cumulative G(q)', fontsize=10)
    ax8.set_title('(h) Cumulative G(q)', fontsize=11, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Subplot 9: Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
측정 데이터 기반 계산 결과
{'='*42}

데이터:
  PSD 포인트:  {len(q_measured)}개
  DMA 포인트:  {len(omega_measured)}개

입력 조건:
  σ₀ = {sigma_0/1e6:.2f} MPa
  v  = {velocity:.4f} m/s
  T  = {temperature:.1f}°C

계산 결과:
  G(q_max) = {results['G'][-1]:.6e}
  P(q_max) = {results['contact_area_ratio'][-1]:.6f}

실 접촉 면적:
  {results['contact_area_ratio'][-1]*100:.3f}%

평균 접촉 압력:
  {stats['mean_pressure']/1e6:.2f} MPa
  (명목 압력의 {stats['mean_pressure']/sigma_0:.2f}배)

물리적 의미:
  거칠기로 인해 실제 접촉 면적은
  명목 면적의 약 {results['contact_area_ratio'][-1]*100:.1f}%
    """
    ax9.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center', transform=ax9.transAxes)

    plt.suptitle('Persson Contact Mechanics - Measured Data Results',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('measured_data_visualization.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ measured_data_visualization.png")

    plt.show()

    print("\n" + "=" * 80)
    print("계산 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()
