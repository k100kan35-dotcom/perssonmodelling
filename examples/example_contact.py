"""
접촉 역학 분석 예제
===================

G(q) 계산 후 접촉 면적, 응력 분포 등을 분석하는 예제입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persson_model import (
    GCalculator,
    FractalPSD,
    ViscoelasticMaterial,
    ContactMechanics
)


def main():
    print("=" * 60)
    print("Persson 접촉 역학 분석 예제")
    print("=" * 60)

    # Setup
    print("\n재료 및 표면 설정...")
    material = ViscoelasticMaterial.create_example_sbr()
    psd = FractalPSD(
        hurst_exponent=0.8,
        rms_roughness=5e-6,
        q_min=1e2,
        q_max=1e8
    )

    sigma_0 = 0.5e6  # 0.5 MPa
    velocity = 0.1    # 0.1 m/s

    # G(q) calculation
    print("G(q) 계산 중...")
    g_calc = GCalculator(
        psd_func=psd,
        modulus_func=lambda w: material.get_modulus(w),
        sigma_0=sigma_0,
        velocity=velocity,
        poisson_ratio=0.5
    )

    q_values = np.logspace(2, 8, 80)
    G_values = g_calc.calculate_G(q_values)

    # Contact mechanics
    print("접촉 역학 분석...")
    contact = ContactMechanics(
        G_function=lambda q: np.interp(q, q_values, G_values),
        sigma_0=sigma_0,
        q_values=q_values,
        G_values=G_values
    )

    # Calculate statistics
    stats = contact.contact_statistics()

    print("\n" + "=" * 60)
    print("접촉 통계 결과")
    print("=" * 60)
    print(f"실 접촉 면적 비율:  {stats['area_fraction']:.6f} ({stats['area_fraction']*100:.3f}%)")
    print(f"평균 접촉 압력:     {stats['mean_pressure']/1e6:.3f} MPa")
    print(f"RMS 압력 변동:     {stats['rms_pressure']/1e6:.3f} MPa")
    print(f"응력 표준편차:      {np.sqrt(stats['stress_variance'])/1e6:.3f} MPa")
    print(f"배율 ζ_max:        {stats['magnification']:.2f}")
    print("=" * 60)

    # Visualization
    print("\n결과 시각화...")
    fig = plt.figure(figsize=(14, 10))

    # Plot 1: Stress distribution
    ax1 = plt.subplot(2, 3, 1)
    sigma, P_sigma = contact.plot_stress_distribution()
    ax1.plot(sigma / 1e6, P_sigma * 1e6, 'r-', linewidth=2)
    ax1.axvline(sigma_0 / 1e6, color='k', linestyle='--', linewidth=1.5,
                label=f'σ₀ = {sigma_0/1e6:.2f} MPa')
    ax1.set_xlabel('응력 σ (MPa)')
    ax1.set_ylabel('확률 밀도 P(σ) (1/MPa)')
    ax1.set_title('접촉 응력 분포')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Contact area vs magnification
    ax2 = plt.subplot(2, 3, 2)
    zeta, area_frac = contact.contact_area_vs_magnification()
    ax2.semilogx(zeta, area_frac, 'g-', linewidth=2)
    ax2.axhline(stats['area_fraction'], color='r', linestyle='--',
                label=f'최종: {stats["area_fraction"]:.4f}')
    ax2.set_xlabel('배율 ζ')
    ax2.set_ylabel('접촉 면적 A/A₀')
    ax2.set_title('배율에 따른 실 접촉 면적')
    ax2.set_ylim([0, 1.1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: G(q)
    ax3 = plt.subplot(2, 3, 3)
    ax3.loglog(zeta, G_values, 'b-', linewidth=2)
    ax3.set_xlabel('배율 ζ')
    ax3.set_ylabel('G(ζ)')
    ax3.set_title('G 함수')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Stress variance vs magnification
    ax4 = plt.subplot(2, 3, 4)
    stress_var = np.array([contact.stress_variance(z) for z in zeta])
    stress_std = np.sqrt(stress_var)
    ax4.semilogx(zeta, stress_std / 1e6, 'purple', linewidth=2)
    ax4.set_xlabel('배율 ζ')
    ax4.set_ylabel('응력 표준편차 (MPa)')
    ax4.set_title('응력 변동 vs 배율')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Mean contact pressure vs magnification
    ax5 = plt.subplot(2, 3, 5)
    mean_pressure = np.array([
        sigma_0 / contact.contact_area_fraction(z) if contact.contact_area_fraction(z) > 0
        else 0 for z in zeta
    ])
    ax5.semilogx(zeta, mean_pressure / 1e6, 'orange', linewidth=2)
    ax5.axhline(stats['mean_pressure'] / 1e6, color='r', linestyle='--',
                label=f'최종: {stats["mean_pressure"]/1e6:.2f} MPa')
    ax5.set_xlabel('배율 ζ')
    ax5.set_ylabel('평균 접촉 압력 (MPa)')
    ax5.set_title('평균 압력 vs 배율')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    접촉 역학 결과 요약

    입력 조건:
    • 명목 압력: {sigma_0/1e6:.2f} MPa
    • 미끄럼 속도: {velocity:.3f} m/s
    • RMS 거칠기: {5:.2f} μm
    • Hurst 지수: 0.8

    계산 결과:
    • 실 접촉 면적: {stats['area_fraction']*100:.3f}%
    • 평균 압력: {stats['mean_pressure']/1e6:.3f} MPa
    • RMS 압력: {stats['rms_pressure']/1e6:.3f} MPa
    • 압력 증폭: {stats['mean_pressure']/sigma_0:.2f}×

    물리적 의미:
    • 거칠기로 인해 실제 접촉 면적은
      명목 면적의 약 {stats['area_fraction']*100:.1f}%
    • 실제 접촉 지점의 압력은
      명목 압력보다 약 {stats['mean_pressure']/sigma_0:.1f}배 높음
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.suptitle('Persson 접촉 역학 분석 결과', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('persson_contact_results.png', dpi=150, bbox_inches='tight')
    print("그래프 저장: persson_contact_results.png")

    plt.show()

    print("\n계산 완료!")


if __name__ == "__main__":
    main()
