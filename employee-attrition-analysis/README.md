# 다변량 분석을 통한 "이직률 결정 요인" 탐구 연구

> 이직률을 단일 지표가 아닌 **여러 정량 요인의 결합**으로 해석하기 위한 다변량 분석 프로젝트

- **팀 구성**: 이지윤, 이예은 (2인, 2조)
- **과목**: 다변량 분석
- **사용 도구**: Python (pandas, scikit-learn, statsmodels, matplotlib, seaborn)
- **발표 자료**: [다변량_프로젝트_PPT.pdf](docs/project_PPT.pdf)
- **보충 자료**: [다변량_프로젝트_보충자료.pdf](docs/project_details.pdf)

---

## 1. 프로젝트 배경 및 문제 의식

최근 기업의 이직률 상승이 사회적 이슈가 되고 있지만, 많은 기업이 이직률을 급여 수준이나 근속연수 같은 **단일 지표 또는 단순 변수**로만 판단해 정확한 원인을 파악하기 어렵습니다.

**목표: 이직률을 설명하는 다양한 정량 요인을 통합적으로 분석한다.**

## 2. 데이터

- **출처**: IBM HR Analytics Employee Attrition & Performance (기업 인사관리·직원 이직 분석에 사용되는 공개 데이터셋)
- **규모**: 직원 1,470명
- **종속변수**: `Attrition` (이직 여부, 0/1)

## 3. 분석 과정

| 단계 | 내용 |
|---|---|
| Step 1. 변수 선택 | 변수를 4개 카테고리로 나눈 뒤 중요하다고 판단한 9개 변수 선택 |
| Step 2. 파생변수 추가 | 기업 내 성장 가능성을 반영하는 `GrowthPotential2` 생성 |
| Step 3. 탐색적 데이터 분석 | 이직 여부(`Attrition`)와 각 독립변수 간 관계를 박스플롯으로 확인 |
| Step 4. 요인분석 | 직접 만든 4개 카테고리가 논리적으로 타당한지 Factor Analysis로 확인 |
| Step 5. 군집분석 | 계층적 군집분석(Ward)과 K-means(Elbow Method)로 유사 집단 분류 |
| Step 6. 로지스틱 회귀 | 선택 변수로 이직 여부 모델링 |
| Step 7. 교호작용항 추가 | 영향이 있을 것으로 보이는 교호작용항 4개를 추가하고, 유의한 항만 채택 |

### 선택한 변수 (4개 카테고리, 9개)

| 카테고리 | 변수 |
|---|---|
| 개인 특성 | `Age`, `YearsAtCompany`, `DistanceFromHome` |
| 직무 요인 | `OverTime`, `JobSatisfaction`, `YearsInCurrentRole` |
| 보상 요인 | `MonthlyIncome`, `PercentSalaryHike` |
| 조직 문화 | `RelationshipSatisfaction` |

### 파생변수 `GrowthPotential2`

기업 내 성장 가능성의 간접 지표로, 직급(`JobLevel`)별 평균 역할 근속기간에서 개인의 역할 근속기간을 뺀 값입니다.

```python
joblevel_mean = df_original.groupby("JobLevel")["YearsInCurrentRole"].mean()
df_original["MeanYearsByJobLevel"] = df_original["JobLevel"].map(joblevel_mean)
df_original["GrowthPotential2"] = df_original["MeanYearsByJobLevel"] - df_original["YearsInCurrentRole"]
```

이 변수를 쓰기 위해 아래 세 가지 가정이 필요합니다.

1. 승진(직무 전환) 규칙의 일관성
2. `JobLevel` 정의의 안정성
3. 승진 정책의 시간적 안정성

### 군집분석 결과 (K-means, k=4)

| 군집 | 해석 |
|---|---|
| 군집 1 | 젊고 저연봉, 성장 가능성 낮고 이직률 높은 그룹 |
| 군집 2 | 고연령 중급자, 과로 위험, 중간 수준 이직률 |
| 군집 3 | 성장 가능성이 가장 높은 신입/초급, 이직률 높음 |
| 군집 4 | 고연봉·고근속 핵심 인력, 안정적인 그룹 (이직률 가장 낮음) |

## 4. 모델링 결과

### 로지스틱 회귀 (Pseudo R² = 0.1528)

| 변수 | 해석 |
|---|---|
| `MonthlyIncome` | 이직 odds 약 0.58배로 감소 |
| `Age` | 이직 odds 약 0.70배로 감소 |
| `JobSatisfaction` | 이직 odds 약 0.70배로 감소 |
| `OverTime` | 초과근무를 하는 직원은 그렇지 않은 직원보다 이직 odds 1.95배 |

`WorkLifeBalance`, `RelationshipSatisfaction`, `DistanceFromHome`, `GrowthPotential2`도 5% 수준에서 유의했고, `YearsAtCompany`와 `PercentSalaryHike`는 단독으로는 유의하지 않았습니다.

### 교호작용항 추가 (Pseudo R² = 0.1646)

- 검토한 교호작용항: `MonthlyIncome × PercentSalaryHike`, `YearsAtCompany × GrowthPotential2`, `JobSatisfaction × OverTime`, `DistanceFromHome × WorkLifeBalance`
- p-value 확인 후 **유의한 2개만 채택**
  - `MonthlyIncome × PercentSalaryHike`
  - `YearsAtCompany × GrowthPotential2`

## 5. 결론

**직원 이직은 '보상 × 성장성 × 근무환경'의 삼각 구조로 설명된다.**

- **보상**: 급여가 높을수록 이직 확률이 낮아지고, 인상률 효과는 고연봉 직원에게서 더 크게 나타남 → 핵심 인재에게는 차등적 보상·인상률 전략 필요
- **성장성/경력**: 단순 근속 기간이 충성도를 보장하지는 않으며, 근속과 성장 기회의 조합이 이직 위험과 연결됨 → 승진 속도 관리, 직무 이동, 경력개발 프로그램 필요
- **직무·워라밸**: 직무 만족, 워라밸, 통근 거리가 체감 만족을 좌우 → 재택·하이브리드·배치 변경 등 정책 구비
- **조직 문화**: 관계 만족 저하와 초과근무가 이직 증가와 연결됨

**핵심 이탈 위험군**: 성장 정체의 장기 근속자 / 과도한 초과근로자 / 젊고 만족도가 낮은 직원 / 장거리 출퇴근자

## 6. 폴더 구조

```
.
├── README.md
├── notebooks/
│   └── 이직률_결정_요인_탐구.ipynb
├── docs/
│   ├── 다변량_프로젝트_PPT.pdf
│   └── 다변량_프로젝트_보충자료.pdf
└── data/            # 데이터셋 (직접 내려받아 배치)
```

## 7. 역할

<!-- 본인이 맡은 부분을 구체적으로 적어주세요. 예: 변수 선택 및 EDA, 군집분석, 로지스틱 회귀 모델링 등 -->

- 이지윤: (작성 필요)
- 이예은: (작성 필요)

## 8. 한계 및 개선 방향

<!-- 선택 사항: 교수 피드백이나 직접 느낀 한계를 적으면 좋습니다. -->

- 이 분석은 로지스틱 회귀 기반의 **요인 해석**이 목적이며, 예측 성능 평가는 다루지 않았습니다.
- `GrowthPotential2`는 위 세 가지 가정 아래에서 만든 간접 지표입니다.
