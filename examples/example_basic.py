"""
기본 G(q) 계산 예제
===================

Persson 모델의 기본 사용법을 보여주는 간단한 예제입니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persson_model import GCalculator, FractalPSD, ViscoelasticMaterial


def main():
    print("=" * 60)
    print("Persson 모델 기본 예제")
    print("=" * 60)

    # 1. 재료 물성 설정
    print("\n1. 재료 설정 (SBR 고무)...")
    material = ViscoelasticMaterial.create_example_sbr()
    print(f"   재료명: {material.name}")
    print(f"   기준 온도: {material.reference_temp}°C")

    # 2. 표면 거칠기 모델
    print("\n2. 표면 거칠기 모델 생성...")
    hurst = 0.8
    h_rms = 10e-6  # 10 μm
    q_min = 1e2
    q_max = 1e8

    psd = FractalPSD(
        hurst_exponent=hurst,
        rms_roughness=h_rms,
        q_min=q_min,
        q_max=q_max
    )
    print(f"   Hurst 지수: {hurst}")
    print(f"   RMS 거칠기: {h_rms*1e6:.2f} μm")

    # 3. 작동 조건
    print("\n3. 작동 조건 설정...")
    sigma_0 = 1.0e6  # 1 MPa
    velocity = 0.01   # 0.01 m/s
    poisson = 0.5

    print(f"   명목 압력: {sigma_0/1e6:.2f} MPa")
    print(f"   미끄럼 속도: {velocity:.3f} m/s")
    print(f"   Poisson 비: {poisson}")

    # 4. G(q) 계산
    print("\n4. G(q) 계산 중...")
    g_calc = GCalculator(
        psd_func=psd,
        modulus_func=lambda w: material.get_modulus(w),
        sigma_0=sigma_0,
        velocity=velocity,
        poisson_ratio=poisson,
        n_angle_points=36
    )

    q_values = np.logspace(np.log10(q_min), np.log10(q_max), 100)
    G_values = g_calc.calculate_G(q_values)

    print(f"   계산 완료!")
    print(f"   G(q_max) = {G_values[-1]:.6f}")

    # 5. 결과 시각화
    print("\n5. 결과 그래프 생성...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Persson 모델 계산 결과', fontsize=16, fontweight='bold')

    # Plot 1: G(q)
    ax = axes[0, 0]
    zeta = q_values / q_values[0]
    ax.loglog(zeta, G_values, 'b-', linewidth=2)
    ax.set_xlabel('배율 ζ = q/q₀')
    ax.set_ylabel('G(ζ)')
    ax.set_title('Persson G 함수')
    ax.grid(True, alpha=0.3)

    # Plot 2: PSD
    ax = axes[0, 1]
    C_q = psd(q_values)
    ax.loglog(q_values, C_q, 'r-', linewidth=2)
    ax.set_xlabel('파수 q (1/m)')
    ax.set_ylabel('C(q) (m⁴)')
    ax.set_title('표면 거칠기 PSD')
    ax.grid(True, alpha=0.3)

    # Plot 3: Material modulus
    ax = axes[1, 0]
    freq = np.logspace(-2, 6, 100)
    E_storage = material.get_storage_modulus(freq)
    E_loss = material.get_loss_modulus(freq)
    ax.loglog(freq, E_storage, 'g-', linewidth=2, label="E' (저장)")
    ax.loglog(freq, E_loss, 'orange', linewidth=2, label="E'' (손실)")
    ax.set_xlabel('주파수 ω (rad/s)')
    ax.set_ylabel('탄성률 (Pa)')
    ax.set_title('재료 Master Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: dG/dq
    ax = axes[1, 1]
    dG_dq = np.gradient(G_values, q_values)
    ax.loglog(q_values, np.abs(dG_dq), 'purple', linewidth=2)
    ax.set_xlabel('파수 q (1/m)')
    ax.set_ylabel('|dG/dq|')
    ax.set_title('G의 기울기')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('persson_basic_results.png', dpi=150, bbox_inches='tight')
    print("   그래프 저장: persson_basic_results.png")

    plt.show()

    print("\n" + "=" * 60)
    print("계산 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
