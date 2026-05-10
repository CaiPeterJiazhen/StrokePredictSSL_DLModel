# Phase 6 Lin-style MatrixNet 中文设计规格

## 1. 背景与目标

Phase 6 只实现监督式、无自监督预训练的 Lin-style EEG MatrixNet。输入只来自治疗前 baseline resting-state EEG：

- baseline eyes-open PSD matrix
- baseline eyes-closed PSD matrix
- baseline eyes-open ROI functional connectivity matrix
- baseline eyes-closed ROI functional connectivity matrix
- 可选 tACS target summary vector
- 可选 baseline clinical vector

Phase 6 的核心问题是：在严格 patient-level LOPO 下，Lin-style 多分支深度模型能否优于 Phase 5.2 的 classical ML 与 flattened-matrix ML controls。Phase 6 不训练 BYOL、SimSiam、MAE、masked matrix modeling，也不启动任何自监督流程。Phase 7 才考虑 SSL-MatrixNet。

## 2. 已检查的仓库与输入产物

本设计基于当前仓库结构和只读数据检查：

- Python 包位于 `src/stroke_predict/`。
- 现有脚本位于 `scripts/00_validate_environment.py` 到 `scripts/08_train_ml_baselines.py`。
- Phase 6 新脚本计划为 `scripts/09_train_matrixnet.py`。
- 配置文件位于 `configs/`，Phase 6 新配置计划为 `configs/matrixnet.yaml`。
- 真实输入产物位于 `outputs/`，该目录不提交。

已检查到的关键输入：

- `outputs/cohort/cohort_master.csv`
  - `supervised_main=19`
  - Good=9, Poor=10
  - 每个 supervised_main 病人有 baseline EO/EC 标记
- `outputs/folds/outer_folds.json`
  - LOPO folds=19
  - 每个 test patient 出现一次
  - registry 文件为 `outputs/folds/fold_XX_registry.json`
  - registry 中已有 `inner_splits`、`normalization_fit_subjects`、`threshold_selection_subjects`
- `outputs/matrices/psd_eo.npy`
  - shape `(19, 2, 62, 90)`
  - 无 NaN/Inf
- `outputs/matrices/psd_ec.npy`
  - shape `(19, 2, 62, 90)`
  - 无 NaN/Inf
- `outputs/matrices/fc_roi_eo.npy`
  - shape `(19, 2, 36, 6, 2)`
  - 无 NaN/Inf
- `outputs/matrices/fc_roi_ec.npy`
  - shape `(19, 2, 36, 6, 2)`
  - 无 NaN/Inf
- `outputs/features/feature_dictionary.csv`
  - 包含 PSD/FC matrix 轴索引元数据
  - `axis0_subject_index` 为空，但上游脚本按 `supervised_main` 的 `subject_id` 排序堆叠矩阵
- `outputs/features/features_tacs_target_summary.csv`
  - shape `(19, 731)`
  - tACS connectivity 数值列 384 个，未发现全 NaN connectivity 列
- Phase 5.2 comparison：
  - `outputs/evaluation/ml_metrics_all.csv`
  - `outputs/reports/phase5_2_ml_baseline_report.md`
  - fast mode 最好 ROC-AUC 为 `M3b_psd_matrix_flatten_ml = 0.5777777777777777`
  - 无 permutation-significant model

隔离工作区说明：

- 代码在 `.worktrees/phase6-matrixnet` 的 `codex/phase-6-matrixnet` 分支开发。
- 真实数据验收会使用 ignored `outputs/` 产物；这些产物和任何 `.xlsx`、`.set`、`.fdt` 都不提交。

## 3. 设计选择

### 推荐方案：小型多分支 PyTorch MatrixNet

采用 PyTorch 实现小型 Lin-style MatrixNet，新增专用数据加载、fold-safe preprocessing、训练与输出模块。这个方案最贴近 Lin et al. 2022 的 PSD matrix + FC matrix + optional scale/clinical fusion 思路，同时控制参数量，适合 `N=19` 的监督学习设置。

优点：

- 直接建模 PSD/FC 矩阵结构，而非只用 flattened controls。
- 可以清楚拆分 PSD、FC、tACS、clinical 分支。
- 每个 preprocessing fit 都能绑定 outer fold train subjects，利于无泄漏审计。

风险：

- 样本量很小，深度模型容易不稳定。
- fast mode 只能做工程 smoke 与初步比较，不能用于疗效结论。

### 备选方案 A：只实现 flattened MLP control

将 PSD/FC flatten 后送入小型 MLP。实现简单，但不符合 Phase 6 的 Lin-style matrix branch 目标，也无法回答卷积分支是否优于 flattened controls。

### 备选方案 B：较大 CNN 或 Transformer

参数量更大，可能提升表达能力，但 `N=19` 下过拟合风险高，不符合本 phase 的模型大小约束。Phase 6 不采用。

## 4. 模型家族

Phase 6 必跑模型：

- `M8a_matrixnet_psd_only`
  - 输入：PSD_EO、PSD_EC
- `M8b_matrixnet_fc_only`
  - 输入：FC_ROI_EO、FC_ROI_EC
- `M8c_matrixnet_psd_fc`
  - 输入：PSD_EO、PSD_EC、FC_ROI_EO、FC_ROI_EC
- `M8d_matrixnet_psd_fc_tacs`
  - 输入：PSD_EO、PSD_EC、FC_ROI_EO、FC_ROI_EC、tACS target summary vector

Secondary model：

- `M12_matrixnet_clinical_eeg`
  - 输入：PSD_EO、PSD_EC、FC_ROI_EO、FC_ROI_EC、可选 tACS、clinical baseline vector
  - 仅在 clinical columns 存在时运行

可选 ablation 暂不列为 Phase 6 完成条件，除非核心实现完成后成本很低：

- `M8e_matrixnet_eo_only`
- `M8f_matrixnet_ec_only`
- `M8h_matrixnet_tacs_only_mlp`

## 5. 数据加载与 subject alignment

新增 `src/stroke_predict/matrixnet_data.py`，负责：

- 读取 cohort、folds、fold registry、PSD/FC matrices、tACS summary、clinical columns、Phase 5.2 metrics。
- 优先使用 `outputs/matrices/*.npy`，如缺失再尝试 legacy `outputs/features/matrices/*.npy`。
- 从 `cohort_master.csv` 推断矩阵行顺序：`role == supervised_main` 后按 `subject_id` 排序。
- 验证矩阵第一维等于 supervised_main 病人数。
- 验证 `outer_folds.json` 的 `supervised_subjects` 与矩阵 subject order 集合一致。
- 验证每个 fold 的 test patient 不在 train/inner-val/fit/threshold subjects 中。
- 如果将来出现显式 matrix metadata，则优先使用显式 metadata；若 metadata 与 cohort 推断不一致则 fail loudly。

不读取原始 `.set` 或 `.fdt`。Phase 6 只消费已生成的 public/去标识化 baseline artifacts。

## 6. Fold-safe preprocessing

新增 `src/stroke_predict/matrixnet_preprocessing.py`，负责：

- Matrix z-score normalization：
  - 每个 outer fold 单独 fit。
  - fit subjects 只能来自 outer train subjects。
  - 对每个 matrix family 独立计算 mean/std。
  - std 为 0 时用 1 替代，避免除零。
  - test patient 不参与 fit。
- PSD log transform：
  - 已检查 PSD 值包含负值，说明上游大概率已 log/standardized。
  - Phase 6 默认不再 log transform，只做 fold-safe z-score。
  - 配置中保留 `psd_transform: none` 并写入 report。
- FC：
  - FC 值在 `[0, 1]`，默认只做 fold-safe z-score。
- tACS/clinical vector：
  - 仅在对应模型使用时加载。
  - outer train median imputation。
  - outer train standard scaling。
  - 对 categorical clinical columns 做简单数值编码或 one-hot，编码类别只能从 outer train 拟合。
- Missing values：
  - Matrix 若出现 NaN，默认 fail loudly；可配置为 fold-safe median impute，但 fast/full 默认不启用 matrix impute。
  - Vector NaN 使用 outer train median。

每 fold 输出 normalization audit rows，不输出原始路径。

## 7. 模型结构

新增 `src/stroke_predict/matrixnet.py`。

核心类：

- `MatrixBranch`
  - 输入可为 `[batch, H, W]` 或 `[batch, 1, H, W]` 或更高维矩阵
  - 高维矩阵按最后两个维度作为 H/W，其余非 batch 维展平为 channel-like spatial 维前的组合，最终规范到 `[batch, 1, H, W]` 或 `[batch, 1, H, W_flat]`
  - Conv2d -> BatchNorm2d -> GELU -> Dropout
  - Conv2d -> BatchNorm2d -> GELU -> Dropout
  - Conv2d -> BatchNorm2d -> GELU
  - AdaptiveAvgPool2d
  - Flatten
  - Linear to embedding_dim
- `VectorBranch`
  - Linear -> LayerNorm -> GELU -> Dropout
  - Linear -> embedding_dim
- `MatrixNet`
  - 共享 PSD branch 处理 EO/EC
  - 共享 FC branch 处理 EO/EC
  - optional tACS branch
  - optional clinical branch
  - concatenate embeddings
  - Linear -> LayerNorm -> GELU -> Dropout
  - Linear -> GELU -> Dropout
  - Output Linear -> 1 logit

默认参数：

- `embedding_dim=32`
- `hidden_dim=64`
- `dropout=0.5`
- `activation=gelu`
- `optimizer=AdamW`
- `loss=BCEWithLogitsLoss`
- `weight_decay=1e-2`
- optional class weighting 使用 outer training labels 计算

## 8. 训练策略

新增 `src/stroke_predict/matrixnet_training.py` 和 `scripts/09_train_matrixnet.py`。

Run modes：

- `fast`
  - seeds: `[0]`
  - max_epochs: 30 到 50
  - patience: 5 到 10
  - hyperparameter grid 只保留 1 个小配置
  - 目标：调通 19-fold LOPO 和输出链路
- `full`
  - seeds: `[0, 1, 2, 3, 4]`
  - max_epochs: 200
  - patience: 25
  - grid: learning rate `[1e-3, 3e-4, 1e-4]`，weight decay `[1e-2, 1e-3, 1e-4]`，dropout `[0.3, 0.5]`，embedding_dim `[32, 64]`

Inner validation：

- 优先读取 fold registry 中的 `inner_splits`。
- 默认使用 registry 的 inner split 作为 validation来源；fast mode 可用第一 split 训练和 early stopping，full mode 可在配置中启用多 split selection。
- early stopping metric：
  - validation ROC-AUC 若 validation 两类齐全
  - 否则 validation BCE loss
- threshold：
  - 从 inner validation predictions 选择 balanced accuracy 最优 threshold。
  - 如果 validation 只有单类或预测不可用，使用 `0.5`，并记录 `threshold_source="fixed_0.5"`。
  - 不使用 outer test predictions 调 threshold。

Evaluation：

- 每个 model/seed/fold 输出一个 patient-level prediction。
- 不做 window-level evaluation。
- 不调 all-outer threshold。

## 9. 输出与报告

Phase 6 运行写入 ignored outputs：

- `outputs/matrixnet/checkpoints/`
- `outputs/matrixnet/training_log.csv`
- `outputs/matrixnet/config_used.yaml`
- `outputs/predictions/matrixnet_patient_predictions.csv`
- `outputs/evaluation/matrixnet_metrics.csv`
- `outputs/reports/matrixnet_no_leakage_report.txt`
- `outputs/reports/matrixnet_fold_audit.csv`
- `outputs/reports/phase6_matrixnet_report.md`
- optional figures under `outputs/figures/`

Prediction required columns：

- `model_name`
- `outer_fold`
- `patient_id`
- `true_label`
- `predicted_score`
- `predicted_label`
- `threshold`
- `threshold_source`
- `seed`
- `run_mode`
- `input_family`
- `best_epoch`
- `best_inner_metric`
- `train_loss_final`
- `val_loss_best`

Metrics required columns：

- `model_name`
- `input_family`
- `run_mode`
- `n_patients`
- `n_good`
- `n_poor`
- `n_seeds`
- `roc_auc_mean`
- `roc_auc_std_across_seeds`
- `roc_auc_ci_low`
- `roc_auc_ci_high`
- `pr_auc`
- `balanced_accuracy`
- `sensitivity`
- `specificity`
- `f1`
- `brier_score`
- `permutation_p_value`
- `comparison_to_best_ml_auc`
- `comparison_to_fma_only_auc`
- `comparison_to_clinical_only_auc`

Report 必须明确：

- Phase 6 是 supervised no-SSL。
- fast mode 结果只用于开发，不做疗效结论。
- 若 MatrixNet 未优于 Phase 5.2 baselines，必须如实报告。
- 推荐 Phase 7 再做 SSL-MatrixNet。

## 10. 测试策略

新增测试：

- `tests/test_matrixnet_shapes.py`
  - 验证 PSD/FC/tACS/clinical 输入组合
  - 验证 logits shape
- `tests/test_matrixnet_no_leakage.py`
  - 验证 outer test patient 不在 train/val/scaler/threshold fit subjects
  - 验证 scaler 只由 outer train fit
  - 验证 matrix row mapping 不混病人
- `tests/test_matrixnet_training_smoke.py`
  - synthetic tiny data 跑 1-2 folds
  - training loss finite
  - sigmoid score 在 `[0, 1]`
- `tests/test_matrixnet_outputs.py`
  - 验证 predictions/metrics/report/audit 输出存在
  - 验证 required columns
  - 验证无 duplicated model-patient-seed
  - 验证每个 completed model 每 seed 有 19 patient-level predictions

TDD 顺序：

1. 写 failing tests。
2. 运行对应 pytest，确认因缺失实现失败。
3. 实现最小代码。
4. 重跑测试到通过。
5. 全量 `python -m pytest tests`。
6. 真实数据 fast mode 验收。

## 11. 禁止事项

- 不启动任何 SSL/self-supervised training。
- 不读取 post-treatment EEG。
- 不读取 raw `.set`、`.fdt`。
- 不提交 `outputs/`、`.xlsx`、`.set`、`.fdt`。
- 不在公开输出中写入本机路径或直接个人标识字段。
- 不对 fast mode 做临床疗效结论。

## 12. 自审结果

- Placeholder scan：未保留 TBD/TODO。
- Scope check：只覆盖 Phase 6 supervised MatrixNet，不包含 Phase 7 SSL。
- Consistency check：模型、输入、输出、测试与用户给定验收标准一致。
- Ambiguity handling：矩阵 row metadata 当前由 cohort sorted supervised_main 推断；若后续发现显式 metadata 且不一致，设计要求 fail loudly。
