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

## 출력 파일

각 예제는 실행 후 PNG 이미지 파일을 생성합니다:
- `persson_basic_results.png` - 기본 예제 결과
- `persson_contact_results.png` - 접촉 역학 결과

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
