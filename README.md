# Persson 마찰 모델 (Persson Friction Model)

Python 기반 Persson 접촉 역학 및 마찰 이론 구현

## 개요

이 프로그램은 거친 표면과 점탄성 재료(고무, 엘라스토머 등) 사이의 접촉 역학과 마찰을 계산하기 위한 Persson 이론의 완전한 구현입니다.

### 주요 기능

- **G(q) 계산**: Persson 이론의 핵심인 탄성 에너지 밀도 함수 계산 (이중 적분)
- **접촉 응력 분포**: 거칠기에 따른 응력 확률 분포 P(σ,ζ)
- **실 접촉 면적**: 배율에 따른 실제 접촉 면적 비율 A/A₀
- **PSD 모델**: 프랙탈, 측정 데이터 등 다양한 표면 거칠기 모델
- **점탄성 물성**: Master curve, WLF 변환, 주파수 의존 탄성률
- **GUI 인터페이스**: 직관적인 그래픽 사용자 인터페이스

## 설치

### 요구사항

- Python 3.7 이상
- NumPy, SciPy, Matplotlib

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/k100kan35-dotcom/perssonmodelling.git
cd perssonmodelling

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용법

### GUI 프로그램 실행

```bash
python main.py
```

### Python 스크립트에서 사용

```python
import numpy as np
from persson_model import GCalculator, FractalPSD, ViscoelasticMaterial, ContactMechanics

# 1. 재료 물성 설정
material = ViscoelasticMaterial.create_example_sbr()

# 2. 표면 거칠기 모델 생성
psd = FractalPSD(
    hurst_exponent=0.8,
    rms_roughness=10e-6,  # 10 μm
    q_min=100,
    q_max=1e8
)

# 3. G(q) 계산기 생성
g_calc = GCalculator(
    psd_func=psd,
    modulus_func=lambda w: material.get_modulus(w),
    sigma_0=1e6,  # 1 MPa
    velocity=0.01,  # 0.01 m/s
    poisson_ratio=0.5
)

# 4. G(q) 계산
q_values = np.logspace(2, 8, 100)
G_values = g_calc.calculate_G(q_values)

# 5. 접촉 역학 분석
contact = ContactMechanics(
    G_function=lambda q: np.interp(q, q_values, G_values),
    sigma_0=1e6,
    q_values=q_values,
    G_values=G_values
)

# 6. 결과 확인
stats = contact.contact_statistics()
print(f"실 접촉 면적 비율: {stats['area_fraction']:.4f}")
print(f"평균 접촉 압력: {stats['mean_pressure']/1e6:.2f} MPa")
```

## 이론 배경

### G(q) 함수

G(q)는 Persson 이론의 핵심 함수로, 접촉 응력 분포의 분산을 결정합니다:

```
G(q) = (1/8) ∫₀^q (1/√q') dq' q'³ C(q') ∫₀^(2π) dφ |E(q'v cosφ) / ((1-ν²)σ₀)|²
```

여기서:
- `q`: 파수 (wavenumber, 1/m)
- `C(q)`: 표면의 파워 스펙트럼 밀도 (PSD)
- `E(ω)`: 주파수 ω에서의 복소 탄성률
- `v`: 미끄럼 속도 (m/s)
- `ν`: Poisson 비
- `σ₀`: 명목 접촉 압력 (Pa)

### 응력 분포

배율 ζ = q/q₀에서의 응력 확률 분포:

```
P(σ,ζ) = (1/√(2πVar)) exp(-(σ-σ₀)²/(2Var))
```

여기서 `Var = σ₀² G(ζ)`

### 실 접촉 면적

```
A/A₀ = (1/2)(1 + erf(σ₀/(√2 σ_std)))
```

## 프로젝트 구조

```
perssonmodelling/
├── persson_model/          # 메인 패키지
│   ├── core/               # 핵심 계산 모듈
│   │   ├── g_calculator.py # G(q) 계산 엔진
│   │   ├── psd_models.py   # PSD 모델들
│   │   ├── viscoelastic.py # 점탄성 재료
│   │   └── contact.py      # 접촉 역학
│   ├── gui/                # GUI 컴포넌트
│   │   └── main_window.py  # 메인 윈도우
│   └── utils/              # 유틸리티 함수
│       └── numerical.py    # 수치 계산
├── examples/               # 예제 스크립트
├── main.py                 # 프로그램 진입점
├── requirements.txt        # 의존성 패키지
└── README.md              # 이 파일
```

## 모듈 설명

### core/g_calculator.py

G(q) 계산의 핵심 구현:
- 이중 적분 (각도 φ, 파수 q)
- 수치 적분 방법 (사다리꼴, Simpson, quad)
- 누적 G(q) 계산

### core/psd_models.py

다양한 표면 거칠기 모델:
- `FractalPSD`: 프랙탈 (자기 유사) 표면
- `MeasuredPSD`: 측정 데이터 기반
- `RollOffPSD`: Roll-off가 있는 프랙탈
- `CombinedPSD`: 복합 PSD 모델

### core/viscoelastic.py

점탄성 재료 물성:
- Master curve (저장/손실 탄성률)
- WLF 시간-온도 중첩
- 주파수 의존 복소 탄성률

### core/contact.py

접촉 역학 계산:
- 응력 분포 P(σ,ζ)
- 실 접촉 면적 A/A₀
- 평균 접촉 압력
- 접촉 통계량

## GUI 사용법

### 1. 재료 선택
- 메뉴 → 재료 → SBR/PDMS 예제 선택
- 또는 파일에서 재료 데이터 로드

### 2. 조건 입력

**작동 조건 탭:**
- 명목 접촉 압력 (MPa)
- 미끄럼 속도 (m/s)
- 온도 (°C)
- Poisson 비

**표면 거칠기 탭:**
- PSD 모델 유형 (프랙탈/측정)
- Hurst 지수 H (일반적으로 0.7-0.9)
- RMS 거칠기 (μm)
- 파수 범위 (q_min, q_max)

**계산 설정 탭:**
- 파수 포인트 수 (정확도)
- 각도 적분 포인트 (정밀도)
- 적분 방법

### 3. 계산 실행
"계산 실행" 버튼 클릭

### 4. 결과 확인

**G(q) 그래프**: G 함수의 배율 의존성
**응력 분포**: 접촉 응력 확률 분포
**접촉 면적**: 배율에 따른 실 접촉 면적 변화
**수치 결과**: 상세 계산 결과

## 예제

### 예제 1: 기본 G(q) 계산

```python
from persson_model import GCalculator, FractalPSD, ViscoelasticMaterial
import numpy as np
import matplotlib.pyplot as plt

# 재료 및 PSD 설정
material = ViscoelasticMaterial.create_example_sbr()
psd = FractalPSD(hurst_exponent=0.8, rms_roughness=5e-6, q_min=100, q_max=1e8)

# G 계산기
g_calc = GCalculator(
    psd_func=psd,
    modulus_func=lambda w: material.get_modulus(w),
    sigma_0=0.5e6,
    velocity=0.1,
    poisson_ratio=0.5
)

# 계산
q = np.logspace(2, 8, 80)
G = g_calc.calculate_G(q)

# 그래프
plt.loglog(q/q[0], G)
plt.xlabel('Magnification ζ')
plt.ylabel('G(ζ)')
plt.title('Persson G Function')
plt.grid(True)
plt.show()
```

### 예제 2: 접촉 면적 분석

```python
from persson_model import ContactMechanics

# ContactMechanics 객체 생성 (위 예제에서 계속)
contact = ContactMechanics(
    G_function=lambda q_val: np.interp(q_val, q, G),
    sigma_0=0.5e6,
    q_values=q,
    G_values=G
)

# 접촉 통계
stats = contact.contact_statistics()

print(f"실 접촉 면적: {stats['area_fraction']*100:.2f}%")
print(f"평균 압력: {stats['mean_pressure']/1e6:.2f} MPa")
print(f"RMS 압력: {stats['rms_pressure']/1e6:.2f} MPa")

# 배율에 따른 접촉 면적
zeta, A = contact.contact_area_vs_magnification()

plt.semilogx(zeta, A)
plt.xlabel('Magnification ζ')
plt.ylabel('Contact Area A/A₀')
plt.title('Contact Area vs Magnification')
plt.grid(True)
plt.show()
```

## 참고문헌

1. Persson, B. N. J. (2001). "Theory of rubber friction and contact mechanics." *Journal of Chemical Physics*, 115(8), 3840-3861.

2. Persson, B. N. J. (2006). "Contact mechanics for randomly rough surfaces." *Surface Science Reports*, 61(4), 201-227.

3. Persson, B. N. J., Albohr, O., Tartaglino, U., Volokitin, A. I., & Tosatti, E. (2005). "On the nature of surface roughness with application to contact mechanics, sealing, rubber friction and adhesion." *Journal of Physics: Condensed Matter*, 17(1), R1.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 제안, Pull Request를 환영합니다.

## 문의

문제가 발생하거나 질문이 있으시면 GitHub Issues를 이용해 주세요.