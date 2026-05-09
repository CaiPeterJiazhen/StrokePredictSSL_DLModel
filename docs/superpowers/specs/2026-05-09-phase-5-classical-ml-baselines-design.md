# Phase 5 传统机器学习 Baselines 设计

## 目标

Phase 5 在 Phase 4 已生成的患者级 LOPO folds 上训练传统机器学习 baseline。该阶段只使用 Phase 3 已产出的 patient-level EEG 特征矩阵、handcrafted/tACS 特征，以及 Phase 1 cohort 中的临床变量，输出患者级 outer test predictions、统一指标、bootstrap CI、permutation test 和 classical feature importance。

该阶段不实现 MatrixNet、SSL 预训练、深度学习模型、解释图表或论文初稿。

## 输入

1. `outputs/cohort/cohort_master.csv`
   - 使用 `role == "supervised_main"` 定义监督主队列。
   - 使用 `label_primary` 作为 Good/Poor 二分类标签，编码为 `Good=1`、`Poor=0`。
   - 临床特征仅来自去标识化字段：`age`、`sex`、`affected_hand`、`treated_hand`、`baseline_fma`、`baseline_mbi`、`mmse`、`outcome_delta_fma` 不作为输入。
2. `outputs/features/handcrafted_features.csv`
   - 使用其中 numeric EEG summary、tACS target 和 EO/EC reactivity 字段。
   - 非数值定位字段如 `target_channel`、`homologous_channel` 不进入模型。
3. `outputs/features/tacs_target_features.csv`
   - 用于 `M5_tacs_target_ml` 的 tACS-informed feature group。
4. `outputs/features/matrices/psd_eo.npy` 和 `outputs/features/matrices/psd_ec.npy`
   - 按 subject、view、channel、frequency flatten 为 patient-level PSD feature 表。
   - subject 顺序与 `cohort_master.csv` 中排序后的 `supervised_main` 一致。
5. `outputs/features/matrices/fc_roi_eo.npy` 和 `outputs/features/matrices/fc_roi_ec.npy`
   - 按 subject、view、ROI-edge、band、method flatten 为 patient-level FC feature 表。
6. `outputs/features/feature_dictionary.csv`
   - 用于给 flattened PSD/FC feature 命名和标注 `feature_group`。
7. `outputs/folds/outer_folds.json` 和 `outputs/folds/fold_XX_registry.json`
   - 所有训练、inner CV、threshold selection 和 outer test prediction 必须以这些 registry 为准。

## 输出

1. `outputs/predictions/classical_patient_predictions.csv`
   - 每个 `model_id` 对每名 `supervised_main` 患者恰好输出一次 outer test prediction。
   - 核心字段：`model_id`、`outer_fold`、`subject_id`、`label_true`、`y_true`、`prob_good`、`pred_label`、`threshold`、`n_train_subjects`。
2. `outputs/evaluation/classical_metrics.csv`
   - 每个 `model_id` 一行。
   - 至少包含 ROC-AUC、balanced accuracy、sensitivity、specificity、PR-AUC、Brier score、`n_subjects`。
3. `outputs/evaluation/classical_bootstrap_ci.csv`
   - 每个 `model_id` × metric 一行。
   - 包含 `observed_value`、`ci_lower`、`ci_upper`、`n_bootstrap`、`random_seed`。
4. `outputs/evaluation/classical_permutation.csv`
   - 每个 `model_id` 至少对 `roc_auc` 输出 permutation null summary 和 p-value。
   - 字段包含 `observed_value`、`null_mean`、`null_std`、`p_value`、`n_permutations`、`random_seed`。
5. `outputs/explain/classical_feature_importance.csv`
   - 只记录 fold 内训练得到的线性系数。
   - 字段包含 `model_id`、`outer_fold`、`feature_name`、`feature_group`、`coefficient`、`abs_coefficient`、`rank_in_fold`。
   - `M0_majority` 无可解释线性系数，不写入 feature importance。

## 必做模型

| model_id | 输入 | 默认估计器 |
|---|---|---|
| `M0_majority` | 无特征，只用 outer train 标签分布 | train majority probability |
| `M1_fma_only` | `baseline_fma` | ridge logistic |
| `M2_clinical_only` | `age`、`sex`、`affected_hand`、`treated_hand`、`baseline_fma`、`baseline_mbi`、`mmse` | elastic-net logistic |
| `M3_psd_ml` | flattened PSD EO/EC matrix features | ridge logistic |
| `M4_fc_ml` | flattened FC ROI EO/EC matrix features | ridge logistic |
| `M5_tacs_target_ml` | tACS target numeric features | ridge logistic |
| `M6_all_handcrafted_eeg_ml` | PSD + FC + handcrafted/tACS numeric EEG features | elastic-net logistic |
| `M12_clinical_plus_eeg_ml` | clinical + M6 EEG features | elastic-net logistic |

## 防泄漏规则

1. 所有模型的 outer test subject 只在预测时进入 pipeline，不参与 imputer、scaler、feature selection、PCA、classifier 或 threshold 的 fit。
2. inner CV 只使用当前 `fold_XX_registry.json` 的 `supervised_train_subjects`。
3. threshold selection 只在当前 outer train subjects 的 inner validation predictions 上完成，默认使用 Youden J；若 inner predictions 只有单一类别或阈值不可定义，则回退到 `0.5`。
4. imputer、scaler、variance filter 和 classifier 均作为 sklearn pipeline 步骤在当前 fold 的 train subjects 上 fit。
5. 不在全体患者上做 preprocessing、feature selection、PCA 或阈值选择。
6. `healthy_ssl` 和 `ssl_only_stroke` 不进入任何 Good/Poor 监督训练或评估。
7. patient-level 评估单位固定为 `subject_id`，禁止 window-level 或 record-level 预测。

## 训练流程

对每个 `model_id`：

1. 读取模型对应的 patient-level feature table。
2. 对每个 outer fold：
   - 从 registry 读取 `test_subject` 和 `supervised_train_subjects`。
   - 构建 outer train / outer test matrix。
   - 对非 M0 模型，在 outer train 内运行 registry 提供的 inner splits。
   - inner splits 中每次 fit 独立 pipeline，并只对 inner validation subjects 预测。
   - 根据 inner validation ROC-AUC 选择超参数；并从 inner validation probabilities 选择 threshold。
   - 用选定超参数在完整 outer train subjects 上重新 fit pipeline。
   - 对唯一 outer test subject 输出 `prob_good` 和 thresholded label。
   - 从 fold 内最终 fitted pipeline 的线性分类器系数生成 feature importance。
3. 汇总 19 个 outer test predictions 后计算该模型的统一评估指标。

## 评估设计

1. ROC-AUC 和 PR-AUC 基于 `prob_good`。
2. balanced accuracy、sensitivity、specificity 基于 fold 内 thresholded `pred_label`。
3. Brier score 基于 `prob_good` 与 `y_true`。
4. Bootstrap CI 在每个 model 的 19 个 patient-level outer predictions 上 resample subjects with replacement。
5. Permutation test 固定 `prob_good`，在 patient-level 打乱 `y_true`，计算 null distribution。p-value 使用包含 observed 的保守估计：`(count(null >= observed) + 1) / (n_permutations + 1)`。

## 文件边界

1. 新增 `configs/models_ml.yaml`
   - 定义输入路径、模型列表、超参数候选、bootstrap/permutation 次数、random seed 和输出路径。
2. 新增 `src/stroke_predict/ml_models.py`
   - 负责 feature assembly、fold 内 pipeline 训练、inner CV、threshold selection、prediction 和 feature importance。
3. 新增 `src/stroke_predict/evaluation.py`
   - 负责 patient-level metrics、bootstrap CI 和 permutation test。
4. 新增 `scripts/08_train_ml_baselines.py`
   - 作为 Phase 5 命令入口，只编排配置读取、训练和写输出。
5. 新增测试
   - `tests/test_feature_pipeline_no_leakage.py`
   - `tests/test_classical_ml_outputs.py`

## 验收命令

```bash
python scripts/08_train_ml_baselines.py --config configs/models_ml.yaml
python -m pytest tests/test_feature_pipeline_no_leakage.py tests/test_classical_ml_outputs.py tests/test_splits_no_leakage.py tests/test_ssl_no_leakage.py -q
python -m pytest
```

真实数据验收还需要检查：

1. `classical_patient_predictions.csv` 中每个 `model_id` 有 19 条 prediction。
2. 每个 `model_id` × `subject_id` 组合唯一。
3. metrics、bootstrap CI、permutation、feature importance 输出存在且无姓名、原始路径、`.set` 或 `.fdt` 泄露。
4. `git status` 不出现 `outputs/`、Excel、`.set` 或 `.fdt`。

## 范围外

1. 不新增深度学习模型。
2. 不实现 MatrixNet、SSL、EEGNet、消融图、ROC/PR 图或 manuscript。
3. 不改写 Phase 0+1、Phase 2、Phase 3 或 Phase 4 产物生成逻辑，除非 Phase 5 验收发现必要缺口。
4. 不提交 `outputs/`、Excel、`.set`、`.fdt` 或包含姓名/原始路径的文件。

## 自检

该设计覆盖 PRD Phase 5 的 8 个 classical model、fold 内 preprocessing、inner CV hyperparameter/threshold selection、patient-level predictions、统一评估指标、bootstrap CI、permutation test、fold-derived feature importance 和 no-leakage 测试要求。设计没有跳到 MatrixNet、SSL、深度学习或论文图表阶段，也没有计划提交任何运行产物。
