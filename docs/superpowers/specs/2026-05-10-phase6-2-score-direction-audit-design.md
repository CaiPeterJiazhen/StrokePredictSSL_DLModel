# Phase 6.2 Score-Direction and Evaluation Audit 中文设计规格

## 1. 背景与目标

Phase 6.1 已在 CUDA 上完成 supervised no-SSL MatrixNet full-mode 诊断，所有模型均未达到 permutation-significant。随后 score-direction audit 发现可疑现象：多个模型的 `AUC(1-score)` 明显高于 `AUC(score)`，并出现 `direction_correct=False`。因此 Phase 6.2 的目标不是启动 SSL，而是先核查标签编码、分数方向、AUC 计算、patient/matrix row 对齐，并加入只使用 inner validation 的 orientation calibration。

本 phase 完成后才允许讨论 Phase 7 SSL-MatrixNet。Phase 6.2 不启动 SSL、自监督预训练、BYOL、SimSiam、MAE 或 masked matrix modeling。

## 2. 当前代码与产物检查

当前 MatrixNet 相关代码位于：

- `src/stroke_predict/matrixnet_data.py`
- `src/stroke_predict/matrixnet_training.py`
- `scripts/09_train_matrixnet.py`
- `configs/matrixnet.yaml`

已确认：

- `src/stroke_predict/matrixnet_data.py` 中 `LABEL_TO_INT = {"Poor": 0, "Good": 1}`。
- `MatrixDataset.__getitem__` 将 `inputs.labels[index]` 转为 float tensor，并供 `nn.BCEWithLogitsLoss` 使用。
- `_eval_epoch` 当前用 `torch.sigmoid(logits)` 生成 score。
- 现有 full-mode prediction CSV 只有 `predicted_score`，没有 `label_int`、`logit`、`sigmoid_score`、`score_orientation`。
- 现有 metrics 汇总包括 seed mean AUC、bootstrap CI 和 permutation p-value，但缺少 pooled AUC、patient-averaged AUC、`AUC(score)` vs `AUC(1-score)`、Good/Poor mean score 的显式审计表。
- `matrix_subject_index.csv`、`outer_folds.json`、fold registry 已用于 row alignment/no-leakage 校验；Phase 6.2 需要把这些检查继续写入 no-leakage report。

## 3. 设计选择

### 推荐方案：在 MatrixNet 训练链路内加入显式 score contract 和审计输出

把 score 语义固定为“Good 的概率”：logit 经 sigmoid 后得到 `sigmoid_score`，若 orientation calibration 不反转，则 `predicted_score == sigmoid_score`；若只由 inner validation 判定需要反转，则 `predicted_score == 1 - sigmoid_score`。输出表同时保留原始 logit、原始 sigmoid score 和最终 predicted score。

优点：

- 能直接定位标签/score 方向错误，而不是只从 aggregate AUC 猜测。
- 保留原始神经网络输出，方便复现 `AUC(score)` 和 `AUC(1-score)`。
- orientation calibration 明确只使用 inner validation predictions，避免 outer test 泄漏。

风险：

- 如果 inner validation 样本太小或单类，反转决策可能不稳定。因此单类 inner validation 必须不反转，并记录 `normal_insufficient_inner_classes`。

### 备选方案 A：只在报告后处理阶段反转分数

该方案实现较少，但会把模型输出和最终评估分数混在一起，且容易误用 outer test 全量 AUC 决定方向。Phase 6.2 不采用。

### 备选方案 B：强制统一反转所有低于 0.5 的模型

该方案会使用 outer test 或 pooled predictions 的信息决定方向，构成 evaluation leakage。Phase 6.2 明确禁止。

## 4. Label 和 score contract

Phase 6.2 固定以下 contract：

- `LABEL_TO_INT` 必须严格等于 `{"Poor": 0, "Good": 1}`。
- `INT_TO_LABEL` 必须严格等于 `{0: "Poor", 1: "Good"}`。
- `BCEWithLogitsLoss` 的 target `1.0` 表示 Good，target `0.0` 表示 Poor。
- 模型 raw output 是 Good-vs-Poor logit。
- `torch.sigmoid(logit)` 是 Good 的原始概率，字段名为 `sigmoid_score`。
- `predicted_score` 是用于阈值、AUC、bootstrap、permutation 的最终 Good 概率。
- 若 `score_orientation == "normal"` 或 `normal_insufficient_inner_classes`，`predicted_score == sigmoid_score`。
- 若 `score_orientation == "inverted_by_inner_val"`，`predicted_score == 1 - sigmoid_score`。
- `predicted_label` 由 `predicted_score >= threshold` 得到。

新增测试必须覆盖这些语义，避免未来把 Poor 当成 positive class。

## 5. Prediction table 输出

Phase 6.2 的主 prediction CSV 为：

- `outputs/predictions/matrixnet_patient_predictions_phase6_2.csv`

每个 model/seed/fold 必须输出以下字段：

- `model_name`
- `seed`
- `outer_fold`
- `patient_id`
- `true_label`
- `label_int`
- `logit`
- `sigmoid_score`
- `predicted_score`
- `predicted_label`
- `threshold`
- `threshold_source`
- `score_orientation`
- `run_mode`

可继续保留已有辅助字段，如 `input_family`、`best_epoch`、`best_inner_metric`、`train_loss_final`、`val_loss_best`。输出排序为 `model_name, seed, outer_fold`。

## 6. Metric consistency audit

Phase 6.2 需要生成三类指标文件：

- `outputs/evaluation/matrixnet_metrics_phase6_2.csv`
- `outputs/evaluation/seed_wise_metrics_phase6_2.csv`
- `outputs/evaluation/patient_averaged_metrics_phase6_2.csv`

每个模型至少计算：

- 每 seed AUC，再计算 mean/std。
- pooled AUC：把所有 seed 的 prediction row 一起计算 ROC-AUC。由于同一 patient 出现多次，该指标的统计单位是 model-seed-patient row，不是独立 patient。
- patient-averaged AUC：先按 patient 平均各 seed 的 `predicted_score`，再计算 ROC-AUC。该指标统计单位回到 patient。
- `auc_score`：使用最终 `predicted_score`。
- `auc_one_minus_score`：使用 `1 - predicted_score`。
- `mean_score_good`。
- `mean_score_poor`。
- `direction_correct`：当 `auc_score >= auc_one_minus_score` 且 `mean_score_good >= mean_score_poor` 时为 True；否则 False。

报告必须解释：

- mean seed AUC：每个 seed 单独在 19 个 patient 上算 AUC，再对 seed 做均值和标准差，反映随机初始化稳定性。
- pooled AUC：把 5 seeds 的 95 行一起算 AUC，同一 patient 被重复计入，适合检查整体排序方向，但不应当替代 patient-level inference。
- patient-averaged AUC：先把每个 patient 的 5 个 seed score 平均再算 AUC，统计单位是 patient，更适合最终 no-SSL 稳定性判断。

## 7. Inner-validation-only orientation calibration

新增 optional fold-level orientation calibration，默认在 Phase 6.2 full mode 启用。规则如下：

- 每个 outer fold、model、seed 训练完成并加载 best state 后，先在 inner validation set 上得到 raw logits 和 sigmoid scores。
- 如果 inner validation 同时包含 Good 和 Poor，且 `roc_auc_score(inner_y_true, inner_sigmoid_score) < 0.5`：
  - `score_orientation = "inverted_by_inner_val"`
  - inner validation threshold selection 使用 `1 - inner_sigmoid_score`
  - outer test prediction 使用 `1 - sigmoid_score`
- 如果 inner validation 同时包含两类，且 AUC 不小于 0.5：
  - `score_orientation = "normal"`
  - threshold 和 outer test 使用原始 sigmoid score
- 如果 inner validation 只有单类或为空：
  - `score_orientation = "normal_insufficient_inner_classes"`
  - 不反转
  - threshold 使用现有 fallback 或 inner validation balanced accuracy 逻辑

禁止事项：

- 不使用 outer test prediction 决定 orientation。
- 不使用所有 outer predictions、pooled AUC、patient-averaged AUC 决定 orientation。
- 不按模型整体 full run 后处理反转。

## 8. Row alignment 和 no-leakage audit

Phase 6.2 继续要求：

- `matrix_subject_index.csv` row order 必须与 sorted supervised_main subject order 一致，否则 fail loudly。
- 每个矩阵第一维必须等于 supervised_main patient 数。
- 每个 LOPO outer test patient 必须只出现一次。
- outer test patient 不得出现在 train、inner validation、normalization fit、feature selection fit、threshold selection subjects。
- no-leakage report 写入 `outputs/reports/no_leakage_report_phase6_2.txt`。

不读取 post-treatment EEG，不读取 raw `.set`/`.fdt`，只消费已生成 baseline artifacts。

## 9. Phase 6.2 输出与报告

必须生成：

- `outputs/reports/phase6_2_score_direction_audit_report.md`
- `outputs/predictions/matrixnet_patient_predictions_phase6_2.csv`
- `outputs/evaluation/matrixnet_metrics_phase6_2.csv`
- `outputs/evaluation/seed_wise_metrics_phase6_2.csv`
- `outputs/evaluation/patient_averaged_metrics_phase6_2.csv`
- `outputs/reports/no_leakage_report_phase6_2.txt`

报告必须包含：

- label encoding 和 BCE/sigmoid score contract 结论。
- prediction columns audit。
- AUC consistency audit。
- orientation calibration 的规则和每个模型的 orientation 分布。
- mean seed AUC、pooled AUC、patient-averaged AUC 的差异解释。
- no-leakage/row alignment 结果。
- 明确声明 Phase 6.2 未启动 SSL。

## 10. Phase 7 决策规则

报告必须原文表达以下判断逻辑：

- If label/score direction bug is found, fix it before SSL.
- If orientation-calibrated no-SSL MatrixNet remains unstable and non-significant, SSL may proceed only as an exploratory representation-learning experiment, not as a confirmed model improvement stage.
- If orientation-calibrated MatrixNet improves and becomes stable, then Phase 7 SSL can proceed as planned.

## 11. 真实数据验收

Phase 6.2 真实数据 full mode 使用：

- device cuda
- require_cuda
- seeds `[0, 1, 2, 3, 4]`
- patient-level 19-fold LOPO
- bootstrap `1000`
- permutation `1000`
- 模型：
  - `M8a_matrixnet_psd_only`
  - `M8b_matrixnet_fc_only`
  - `M8c_matrixnet_psd_fc`
  - `M8d_matrixnet_psd_fc_tacs`
  - `M12_matrixnet_clinical_eeg`

若当前脚本尚未暴露 device/require_cuda 参数，Phase 6.2 需要补齐 CLI/config 字段并测试：

- CPU smoke tests 仍可在 unit tests 中运行。
- 真实数据验收命令必须拒绝无 CUDA 环境。

## 12. 禁止事项

- 不启动 SSL。
- 不做未计划模型训练。
- 不读取 post-treatment EEG 作为 supervised input。
- 不读取 raw EEG 文件。
- 不提交 `outputs/`、raw EEG、`.xlsx`、`.set`、`.fdt`。
- 不用 outer test 或全量 outer predictions 决定 score orientation。

## 13. 自审结果

- Placeholder scan：未保留 TBD/TODO。
- Scope check：只覆盖 Phase 6.2 score-direction/evaluation audit 和 no-SSL MatrixNet rerun，不包含 Phase 7 SSL 实现。
- Consistency check：label contract、prediction columns、metric files、orientation calibration、真实数据验收与用户要求一致。
- Ambiguity handling：如果 inner validation 单类，则明确不反转并记录 `normal_insufficient_inner_classes`；如果真实 CUDA 不可用，则验收必须失败而不是静默降级。
