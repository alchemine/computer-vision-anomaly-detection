# Computer Vision 이상치 탐지 알고리즘 경진대회
[https://dacon.io/competitions/official/235894/overview/description](https://dacon.io/competitions/official/235894/overview/description)


# 1. Goal
## 1.1 Problem
Classification on [*MVtec AD Dataset*](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## 1.2 Metric
Macro f1 score


# 2. Related work
## 2.1 Baseline: `0.666`
[baseline.ipynb](baseline.ipynb)

1. Preprocessing
   1. Resize to (512 x 512)
   2. Data augmentation: flip left-right or up-down
2. Training
   1. Model: `efficientnet_b0`
   2. Optimizer: `Adam` (lr=1e-3)
   3. Loss: `CEE`
   4. Gradient scaler
3. Evaluation
   1. Training metric: `0.96759`
   2. Test metric: `0.66579`

## 2.2 SOTA baseline
### 2.2.1 [Anomaly detection on MVtec AD](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad)
1. [FastFlow](https://github.com/gathierry/FastFlow) (Detection AUROC: `99.4`)
   1. Rank 1
   2. Unsupervised anomaly detection


# 3. Proposed idea
## 3.1 Proposed 1: `0.614`
[proposed1.ipynb](proposed1.ipynb)
1. Early stopping 적용


## 3.2 Proposed 2: `0.701`
[proposed2.ipynb](proposed2.ipynb)
1. TensorFlow porting
2. Validation
   1. validation_on: optimal epochs 결정 -> **더 좋은 성능**
   2. validation_off: train_full로 학습


## 3.3 Proposed 3: `0.745`
[proposed3.ipynb](proposed3.ipynb)
1. Sample weight 적용


## 3.4 Proposed 4: `0.644`
[proposed4.ipynb](proposed4.ipynb)
1. Model: `efficientnet_b0`
2. 2-level classification
   1. Classification(`class`) \
      `model1`: supervised
   2. Classification(`label`) \
      `model2`: supervised

## 3.5 Proposed 5:
[proposed5.ipynb](proposed5.ipynb)
1. 3-level classification
   1. Classification(`class`) \
      `model1`: supervised (`efficientnet_b0`)
   2. Anomaly detection \
      `model2`: unsupervised (`PatchCore`)
   3. Classification(`label`) \
      `model3`: supervised (`efficientnet_b0`)
