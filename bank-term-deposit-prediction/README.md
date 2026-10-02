# 은행 고객 정기예금 가입 예측 (ML Competition, ML2 4조)

은행 마케팅 고객 데이터로 **정기예금 가입 여부를 예측하는 이진 분류 프로젝트**입니다.
가입자가 11.7%뿐인 불균형 데이터라 정확도 대신 **F1-score**로 평가했고, 여러 불균형 처리·앙상블 기법을 실험·비교해 최종 모델을 선정했습니다.

| 항목 | 내용 |
|---|---|
| 형태 | 팀 프로젝트 (3인), Kaggle 형식 ML 대회 |
| 데이터 | Bank Marketing 고객 데이터 31,647행, 가입 비율 11.7% |
| 평가 지표 | F1-score |
| **최종 결과** | **검증 데이터 F1 = 0.618** (train:validation = 8:2, `random_state=42`) |

---

## 최종 모델

**LightGBM + CatBoost를 각각 Bagging(50개)한 뒤 Soft Voting으로 결합**했습니다.

```
LightGBM  ── Bagging(n=50) ─┐
                            ├─ Soft Voting ─→ 정기예금 가입 여부
CatBoost  ── Bagging(n=50) ─┘
```

- 불균형 보정: `scale_pos_weight` = 음성 수 / 양성 수 ≈ 7.55 (두 모델 모두 적용)
- LightGBM: `RandomizedSearchCV`(7-fold, scoring=F1)로 탐색해 선정한 `learning_rate=0.11` 사용
- CatBoost: `iterations=1000, depth=10, learning_rate=0.1`
- 두 가지 다 시도해본 결과, Hard Voting보다 예측 확률을 반영하는 Soft Voting에서 성능이 더 높았습니다. 

## 데이터 처리

1. **변수 탐색**: 수치형은 종속변수와 피어슨 상관분석(통화시간 `duration`이 0.39로 가장 높음), 범주형은 카이제곱 검정으로 가입 여부(종속변수)와의 관련성 확인
2. **전처리**: `month` 문자열을 숫자로 변환
3. **파생변수 3개**: `age_group`(연령대), `balance_group`(잔고 구간), `campaign_duration`(통화시간 × 캠페인 횟수)
4. **인코딩**: 범주형 변수와 파생변수에 Label Encoding

## 실험 로그

최종 모델에 이르기까지 시도한 방법입니다. 성능 수치는 최종 모델만 검증 데이터 F1으로 측정했습니다.

| 방법 | 목적 | 결과 | 파일 위치 |
|---|---|---|---|
| `scale_pos_weight` | 클래스 불균형 보정 | 채택 | `final/` |
| Bagging (LightGBM, CatBoost 각각) | 과적합 완화 | 채택 | `final/` |
| Soft Voting | 두 모델 결합 | 채택 (Hard Voting보다 성능 우수) | `final/` |
| RandomizedSearchCV | LightGBM 하이퍼파라미터 탐색 | 채택 | `final/` |
| SMOTE | 오버샘플링으로 불균형 완화 | 과적합 발생 → 미채택 | `experiments/01_smote_lightgbm.ipynb` |
| OOF 예측 | 일반화 성능 검증 | 검증 F1이 Soft Voting보다 낮음 → 미채택 | `experiments/02_oof_gbm_cat.ipynb` |
| Stacking (+ early stopping) | 메타 모델로 결합 | 과적합 발생 → 미채택 | `experiments/03_stacking_early_stopping.ipynb` |

> **측정 범위**: 최종 모델 외 실험은 노트북에 F1을 따로 출력하지 않고, 제출 예측 결과(양성 예측 수 등)로 1차 판단했습니다. 따라서 위 표의 "과적합", "F1 낮음"은 당시 실험 기록 기준이며, 이 저장소에서 수치로 재현되는 값은 최종 모델의 0.618뿐입니다. Hard Voting 비교 실험은 별도 노트북으로 포함하지 않았습니다.

## 한계와 개선점

- **LightGBM 파라미터명 문제**: 초기 탐색에서 CatBoost식 이름(`depth`, `iterations`, `l2_leaf_reg`)을 LightGBM에 넘겨 일부가 적용되지 않았습니다. 이를 발견해 **실제로 적용된 `learning_rate`와 `scale_pos_weight`만 남겨 정리**했습니다. <!-- 정리 후 재실행해서 F1 0.618이 재현되는지 확인한 뒤 이 주석을 지우세요 --> 정리 후에도 검증 F1은 동일하게 재현됐습니다. 파라미터명을 LightGBM식으로 바꿔 넓은 범위에서 재탐색했을 때는 검증 F1이 0.587로 낮아져(탐색 범위가 넓어 과적합으로 추정), 기존 구성을 유지했습니다.
- **검증의 낙관성**: 하이퍼파라미터 탐색을 전체 학습 데이터로 한 뒤 train/validation을 나눴고 `stratify`도 쓰지 않아, 0.618은 다소 낙관적일 수 있습니다. 다음에는 분할 후 학습 데이터 안에서만 탐색하고 `stratify=y`를 적용할 계획입니다.
- **SMOTE 적용 위치**: SMOTE를 교차검증 이전에 전체 데이터에 적용하면 합성 샘플이 검증 fold에 섞여 성능이 부풀려질 수 있습니다. 과적합 원인으로 추정하며, fold 안에서만 적용하는 방식(`imblearn.pipeline.Pipeline`)으로 재검증해 볼 필요가 있습니다.
- **`duration` 변수**: 통화 종료 후에야 알 수 있는 값이라, 실제 사전 타겟팅 모델에는 쓸 수 없습니다. 대회 환경에서는 가장 중요한 변수였지만 실무 적용 시에는 제외하고 다시 평가해야 합니다.
- **EDA·시각화 부족**: 초기 단계에서 데이터 탐색과 시각화가 부족하다는 피드백을 받았습니다. 이후 프로젝트에서는 모델링 전에 충분한 탐색과 시각화를 먼저 수행합니다.

## 배운 점

- 불균형 데이터에서는 정확도가 아니라 F1 같은 지표를 기준으로 모델을 비교해야 한다는 점
- 성능을 높이는 방법(SMOTE, Stacking 등)이 오히려 과적합을 만들 수 있어, 성능과 일반화 사이의 균형을 확인해야 한다는 점
- 팀장으로서 참여가 저조한 팀원에게 수행 가능한 업무를 구체적으로 나눠 주고 어려운 부분을 함께 풀며 일정을 맞춘 경험

## 팀 구성

| 이름 | 역할 |
|---|---|
| 이지윤 (팀장) | Kaggle 제출·회의 추진·중간 점검, 전처리(month 변환, Label Encoding), 모델링 및 하이퍼파라미터 조정, PPT·코드 정리 |
| 이예은 | 특성 엔지니어링, `scale_pos_weight`, LightGBM·CatBoost·Bagging 앙상블 추진, 발표 대본 |
| 전정인 | 다양한 방향성 제시, 발표 |
팀장 (Kaggle 제출·회의·중간 점검), 전처리, 모델링·하이퍼파라미터 조정, PPT·코드 정리 

## 저장소 구조

```
.
├── final/
│   └── final_softvoting_lgbm_cat.ipynb      # 최종 모델 (검증 F1 0.618)
├── experiments/
│   ├── 01_smote_lightgbm.ipynb              # SMOTE (미채택)
│   ├── 02_oof_gbm_cat.ipynb                 # OOF 예측 (미채택)
│   └── 03_stacking_early_stopping.ipynb     # Stacking (미채택)
└── README.md
```

## 실행 방법

대회 제공 데이터(`train.csv`, `test.csv`, `submission_example.csv`)는 저장소에 포함하지 않았습니다. 노트북과 같은 폴더에 직접 넣고 실행하세요.

```bash
pip install numpy pandas scipy scikit-learn lightgbm catboost imbalanced-learn
```

## 사용 기술

Python, pandas, scikit-learn, LightGBM, CatBoost, imbalanced-learn (SMOTE)
