# 👥 직원 이직 요인 분석 (Employee Attrition Analysis)

IBM HR 직원 데이터를 활용해 **어떤 요인이 직원의 이직에 영향을 주는지** 통계적으로 분석하고, **이직 위험이 높은 직원 집단의 특성**을 도출한 프로젝트입니다.
요인분석 → 군집분석 → 로지스틱 회귀분석 순서로 진행했습니다.

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 형태 | 팀 프로젝트 (팀장) |
| 목적 | 이직에 영향을 미치는 핵심 요인 파악 및 고위험 집단 식별 |
| 데이터 | IBM HR Analytics Employee Attrition & Performance (Kaggle) |
| 규모 | 1,470명 × 35개 변수 (이직률 16.1%) |
| 분석 도구 | Python (pandas, statsmodels, scikit-learn, factor_analyzer, scipy) |
| 분석 기법 | VIF 다중공선성 진단, 요인분석, 계층적/K-means 군집분석, 로지스틱 회귀분석 |

## 📂 데이터

- **출처**: [Kaggle - IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **파일명**: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- IBM 데이터 사이언티스트가 만든 **가상 데이터셋**이며, 실제 개인정보는 포함하지 않습니다.
- 타겟 변수 `Attrition` 비율: 재직(No) 83.9% / 이직(Yes) 16.1% → **클래스 불균형** 존재

## 🔍 사용 변수

35개 변수를 **개인 · 직무 · 보상 · 조직문화** 4개 카테고리로 나누고, 카테고리별 핵심 변수 9개와 직접 만든 파생변수 1개(총 10개)를 독립변수로 선택했습니다.

> 개인 카테고리는 직접 만든 `GrowthPotential2`가 포함된 **성장성 카테고리**로 대체했습니다.

| 구분 | 변수 |
|---|---|
| 보상/경력 | `MonthlyIncome`, `PercentSalaryHike`, `YearsAtCompany`, `Age` |
| 만족도 | `JobSatisfaction`, `WorkLifeBalance`, `RelationshipSatisfaction` |
| 근무환경 | `OverTime`, `DistanceFromHome` |
| **파생변수** | `GrowthPotential2` |
| 종속변수 | `Attrition` (Yes=1, No=0) |

### 파생변수: `GrowthPotential2`

```
GrowthPotential2 = (해당 JobLevel의 평균 YearsInCurrentRole) − (본인의 YearsInCurrentRole)
```

같은 직급의 평균 현 직무 근속연수와 본인의 현 직무 근속연수를 비교한 값입니다.
값이 클수록 **동일 직급 평균보다 현재 역할에 머무른 기간이 짧다**는 뜻입니다.

## 🛠 분석 과정

1. **데이터 확인 및 전처리**: 결측치 없음 확인, 범주형 변수(`Attrition`, `OverTime`) 0/1 인코딩
2. **다중공선성 진단 (VIF)**: 모든 독립변수의 VIF가 3 미만(최대 2.70)으로 다중공선성 문제 없음
3. **탐색적 분석**: 상관관계 히트맵, pairplot, 이직 여부별 박스플롯
4. **요인분석**: 고유값 1 이상 기준(Scree plot)으로 4요인 Varimax 회전 → 직무·보상·조직문화 카테고리 구성의 타당성 확인
5. **군집분석**: 표준화 후 계층적 군집(Ward) 및 K-means(엘보우/실루엣으로 k 탐색), k=4
6. **로지스틱 회귀분석**: 기본 모델 → 교호작용항 추가 → 유의하지 않은 교호작용 제거한 최종 모델
   - 교호작용항은 군집분석에서 근속·연봉 조합에 따라 집단이 갈리는 패턴과 HR 이론(보상 수준×변화, 경력 단계×성장 기회)을 근거로 설정했습니다.
7. **모델 성능 평가**: 혼동행렬, Accuracy, Precision, Recall, F1, ROC-AUC

## 📊 주요 결과

### 1. 요인분석

<img src="images/scree_plot.png" width="480">

Scree plot 기준으로 4개 요인을 추출했습니다. 주요 적재 결과는 다음과 같습니다.

- **요인 1**: `MonthlyIncome`, `Age`, `YearsAtCompany` → 경력·보상 수준
- **요인 2**: `GrowthPotential2` (적재량 0.99) → 직무 내 성장 여지
- **요인 3**: `RelationshipSatisfaction` → 관계 만족도

### 2. 군집분석 (계층적 군집, k=4)

<img src="images/dendrogram.png" width="600">

| 군집 | 월소득 | 근속연수 | 야근 비율 | 특징 | 이직률 |
|:---:|---:|---:|---:|---|---:|
| **0** | 4,811 | 4.6년 | **100%** | 저소득 + 전원 야근 | **36.5%** |
| 1 | 4,890 | 6.1년 | 3.6% | 저소득 + 야근 거의 없음 | 11.4% |
| 2 | 14,408 | 19.5년 | 33.3% | 고소득 + 장기근속 | 12.7% |
| 3 | 15,070 | 3.9년 | 33.7% | 고소득 + 연령대 높음 | 3.2% |

<img src="images/attrition_rate_by_cluster.png" width="480">

**→ 저소득이면서 야근을 하는 집단(군집 0)의 이직률이 36.5%로 전체 평균(16.1%)의 2배 이상**입니다.
같은 저소득 집단이어도 야근이 없는 군집 1은 이직률이 11.4%로, 야근 여부가 이직 위험을 가르는 핵심 변수임을 확인할 수 있습니다.

**K-means (k=4, 엘보우/실루엣으로 k 탐색)**

| 군집 | 월소득 | 근속연수 | 나이 | 특징 | 이직률 |
|:---:|---:|---:|---:|---|---:|
| 0 | 4,350 | 5.3년 | 32.2 | 저소득 · 경력 초기 | 18.9% |
| 2 | 4,619 | 5.0년 | 33.9 | 저소득 · 경력 초기 (높은 인상률) | 18.8% |
| 1 | 9,770 | 4.2년 | 47.4 | 중상위 소득 · 근속 짧음 | 12.8% |
| 3 | 12,270 | 18.5년 | 43.8 | 고연봉 · 장기근속 안정군 | **7.4%** |

**→ 저연봉·경력 초기 군집의 이직률(약 19%)이 고연봉·장기근속 안정군(7.4%)의 약 2.5배**입니다.
(K-means에서는 계층적 군집에서 뚜렷했던 '야근' 기준의 분리는 나타나지 않았습니다.)

### 3. 로지스틱 회귀분석 (최종 모델)

독립변수를 표준화한 뒤 분석했으므로, 오즈비는 **해당 변수가 1 표준편차 증가할 때** 이직 오즈가 몇 배가 되는지를 의미합니다.

| 변수 | 계수 | 오즈비 | p-value | 해석 |
|---|---:|---:|---:|---|
| **OverTime** | 0.673 | **1.96** | <0.001 | 야근하면 이직 오즈 약 2배 ↑ |
| **GrowthPotential2** | 0.430 | **1.54** | 0.001 | 직급 평균보다 빠르게 역할이 이동할수록 ↑ |
| **DistanceFromHome** | 0.219 | 1.25 | 0.004 | 출퇴근 거리가 멀수록 ↑ |
| **MonthlyIncome** | -0.668 | **0.51** | <0.001 | 소득이 높을수록 이직 오즈 약 절반 ↓ |
| **Age** | -0.373 | 0.69 | <0.001 | 나이가 많을수록 ↓ |
| **JobSatisfaction** | -0.347 | 0.71 | <0.001 | 직무 만족도가 높을수록 ↓ |
| **WorkLifeBalance** | -0.187 | 0.83 | 0.014 | 워라밸이 좋을수록 ↓ |
| **RelationshipSatisfaction** | -0.181 | 0.83 | 0.019 | 관계 만족도가 높을수록 ↓ |
| MonthlyIncome × PercentSalaryHike | -0.247 | 0.78 | 0.049 | 교호작용 (유의) |
| YearsAtCompany × GrowthPotential2 | -0.292 | 0.75 | <0.001 | 교호작용 (유의) |
| YearsAtCompany | 0.074 | 1.08 | 0.641 | 유의하지 않음 |
| PercentSalaryHike | -0.174 | 0.84 | 0.068 | 유의하지 않음 (α=0.05) |

- 교호작용항 4개를 추가한 뒤, 유의하지 않았던 2개(`JobSatisfaction×OverTime`, `DistanceFromHome×WorkLifeBalance`)를 제외해 최종 모델을 구성했습니다.
- 기본 모델(Pseudo R² 0.153) 대비 교호작용 모델은 **Pseudo R² 0.165로 설명력이 개선**되었고, 두 교호작용항 모두 통계적으로 유의했습니다.
- LLR p-value < 0.001 (모형 전체는 통계적으로 유의)

### 4. 모델 성능

<img src="images/roc_curve.png" width="400">

| 지표 | 값 |
|---|---:|
| Accuracy | 0.852 |
| Precision | 0.656 |
| Recall | 0.177 |
| F1 Score | 0.279 |
| ROC-AUC | 0.776 |

```
Confusion Matrix
[[1211   22]
 [ 195   42]]
```

## 💡 인사이트 및 제언

**직원 이직은 '보상 × 성장성 × 근무환경'이 결합된 결과이며, 요인별 맞춤형 이직 방지 전략이 필요합니다.**

| 영역 | 근거 | 제언 |
|---|---|---|
| **보상**<br>급여는 이직을 낮추고, 인상률은 고연봉자에게 효과가 더 크다 | `MonthlyIncome` OR 0.51<br>`월급×인상률` 교호작용 OR 0.78 | 핵심 인재 대상 **차등적 보상·인상률 전략** |
| **성장성**<br>근속과 결합될 때 영향이 달라진다 | `근속×성장성` 교호작용 OR 0.75<br>→ 근속이 길수록 역할 이동이 없는 직원의 이직 위험이 커지는 구조 | 근속만으로 충성도를 판단하지 말고 **승진 속도 · 직무 이동 · 경력개발 관리** |
| **근무환경**<br>초과근무 · 먼 통근 · 낮은 직무만족이 위험 요인 | `OverTime` OR 1.96<br>`DistanceFromHome` OR 1.25<br>`JobSatisfaction` OR 0.71 | **근무 방식 유연화** (재택·하이브리드·배치 조정) |

## ⚠️ 한계점 및 개선 방향

- **Recall이 낮습니다 (0.177).** Accuracy는 85%이지만 이직자의 약 18%만 찾아내므로, 이직 예측 목적으로는 부족합니다. 클래스 불균형 때문에 기준 확률(0.5)을 낮추거나, 가중치·SMOTE 등 불균형 처리가 필요합니다.
- **전체 데이터로 학습·평가**해 일반화 성능은 검증하지 못했습니다. 이후 train/test 분할 및 교차검증을 적용할 계획입니다.
- 분석에 사용한 변수가 10개로 제한적입니다. 직무(`JobRole`), 출장(`BusinessTravel`), 주식옵션(`StockOptionLevel`) 등 범주형 변수까지 포함하면 설명력이 개선될 수 있습니다.
- 다른 모델(Random Forest, XGBoost 등 트리 기반)로 확장해 성능과 변수 중요도를 비교해 볼 수 있습니다.
- 가상 데이터를 사용했으므로 결과를 실제 기업에 그대로 일반화하기는 어렵습니다.

## 🙋 프로젝트 역할 및 배운 점

**역할**: 팀장 · PPT 제작 및 분석 전 과정 공동 토의 · **교호작용항 아이디어 제안**으로 모형 확장에 기여

**배운 점**
- 데이터에 없는 '성장 가능성'을 기존 변수(`JobLevel`, `YearsInCurrentRole`)의 조합으로 **정량화**하고, 요인분석으로 타당성을 교차 검증했습니다.
- 요인분석 → 군집분석 → 회귀분석을 **하나의 검증 흐름**으로 설계했으며, HR·고객 이탈 분석 등 실무에도 적용할 수 있는 접근입니다.
- 서로 다른 관점을 나누며 인사이트를 확장하는 협업을 경험했습니다.

## 📁 폴더 구조

```
employee-attrition-analysis/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   └── employee_attrition_analysis.ipynb
├── images/
│   ├── scree_plot.png
│   ├── dendrogram.png
│   ├── attrition_rate_by_cluster.png
│   ├── cluster_mean_heatmap.png
│   ├── correlation_heatmap.png
│   └── roc_curve.png
├── requirements.txt
└── README.md
```

## ▶️ 실행 방법

```bash
# 1. 저장소 클론
git clone https://github.com/jiyunji/employee-attrition-analysis.git
cd employee-attrition-analysis

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 노트북 실행
jupyter notebook notebooks/employee_attrition_analysis.ipynb
```

> 노트북 안의 데이터 경로가 `WA_Fn-UseC_-HR-Employee-Attrition.csv`로 되어 있으므로, `data/` 폴더에 넣었다면 `pd.read_csv("../data/WA_Fn-UseC_-HR-Employee-Attrition.csv")`로 수정해 주세요.

## 🧰 Tech Stack

`Python` `pandas` `NumPy` `matplotlib` `seaborn` `SciPy` `scikit-learn` `statsmodels` `factor_analyzer` `Jupyter Notebook`
