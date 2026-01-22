# 예제 스크립트

이 디렉토리에는 Persson 모델 사용법을 보여주는 예제 스크립트들이 포함되어 있습니다.

## 예제 목록

### 1. example_basic.py
기본 G(q) 계산 예제

**실행 방법:**
```bash
cd examples
python example_basic.py
```

**내용:**
- SBR 재료 물성 로드
- 프랙탈 PSD 모델 생성
- G(q) 계산
- 결과 시각화 (4개 그래프)

### 2. example_contact.py
접촉 역학 분석 예제

**실행 방법:**
```bash
cd examples
python example_contact.py
```

**내용:**
- G(q) 계산
- 접촉 응력 분포
- 실 접촉 면적 계산
- 다양한 접촉 통계량
- 종합 결과 시각화 (6개 그래프)

### 3. example_detailed_output.py ⭐ 신규
상세 출력 예제 (작업 지시서 기반)

**실행 방법:**
```bash
cd examples
python example_detailed_output.py
```

**내용:**
- 모든 중간 계산값 추적
- CSV 출력 (Index, log_q, q, C(q), Avg_Modulus_Term, G_Integrand, Delta_G, G(q), P(q))
- 요약 텍스트 파일 생성
- 플롯용 개별 데이터 파일 생성
- 종합 시각화 (9개 그래프)

**출력 파일:**
- `persson_detailed_results.csv` - 전체 계산 과정
- `persson_summary.txt` - 결과 요약
- `output/persson_G_vs_q.csv` - G 함수 플롯 데이터
- `output/persson_contact_area.csv` - 접촉 면적 플롯 데이터
- `output/persson_PSD.csv` - PSD 플롯 데이터
- `persson_detailed_visualization.png` - 종합 그래프

### 4. example_measured_data.py ⭐ 신규
실제 측정 데이터 사용 예제

**실행 방법:**
```bash
cd examples
python example_measured_data.py
```

**내용:**
- 측정된 PSD 데이터 로드 (q vs C(q))
- 측정된 DMA 데이터 로드 (주파수, E', E'')
- 단위 자동 변환 (Hz→rad/s, MPa→Pa)
- 실제 데이터 기반 G(q) 계산
- 9개 그래프로 시각화

**출력 파일:**
- `measured_data_results.csv` - 계산 결과
- `measured_data_summary.txt` - 요약
- `measured_data_visualization.png` - 그래프

## 데이터 파일 형식

### PSD 데이터 (`examples/data/measured_psd.txt`)
```
# Measured Surface PSD Data
# Wavenumber q (1/m)	PSD C(q) (m^4)
2.0e+01	3.0e-09
3.0e+01	6.0e-10
...
```

### DMA 데이터 (`examples/data/measured_dma.txt`)
```
# Measured DMA Master Curve Data
# Frequency(Hz)  E'(MPa)  E''(MPa)
0.01      6.7     0.7
0.1       7.8     1.0
...
```

## 출력 파일

각 예제는 실행 후 PNG 이미지 파일을 생성합니다:
- `persson_basic_results.png` - 기본 예제 결과
- `persson_contact_results.png` - 접촉 역학 결과
- `measured_data_visualization.png` - 측정 데이터 결과

## 사용자 정의

예제 스크립트를 수정하여 다른 조건으로 계산할 수 있습니다:

```python
# 압력 변경
sigma_0 = 2.0e6  # 2 MPa

# 속도 변경
velocity = 0.05  # 0.05 m/s

# 거칠기 변경
h_rms = 20e-6  # 20 μm

# Hurst 지수 변경
hurst = 0.7
```

## 측정 데이터 사용

자신의 측정 데이터를 사용하려면:

1. **PSD 데이터 준비**: 위 형식으로 텍스트 파일 생성
2. **DMA 데이터 준비**: 위 형식으로 텍스트 파일 생성
3. **예제 4 수정**: 파일 경로 변경
4. **실행**: `python example_measured_data.py`

또는 **GUI 사용**:
- `파일 → 재료 물성 불러오기` (DMA 데이터)
- `파일 → PSD 데이터 불러오기` (PSD 데이터)
