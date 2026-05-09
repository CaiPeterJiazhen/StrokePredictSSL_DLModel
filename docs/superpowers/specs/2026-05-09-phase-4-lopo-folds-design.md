# Phase 4 LOPO Folds 与防泄漏设计

## 目标

Phase 4 生成患者级 leave-one-patient-out (LOPO) outer folds，并为每个 outer fold 生成 fold-specific SSL pool registry。该阶段只定义数据划分和 registry，不训练模型、不做阈值选择、不计算任何模型指标。

## 输入

1. `outputs/cohort/cohort_master.csv`
   - 使用 `role == "supervised_main"` 定义监督主队列。
   - 使用 `source` 和 `role` 区分卒中监督患者、卒中 SSL-only 受试者和健康 SSL 受试者。
   - 使用 `label_primary` 仅用于 inner split 分层，不用于 outer fold 排序或模型选择。
2. `outputs/qc/eeg_qc_summary.csv`
   - 使用 `passes_qc == True` 的 EEG 记录构建 SSL registry。
   - 使用 `subject_id`、`source`、`stage`、`condition`、`record_id` 描述可进入 SSL pool 的匿名记录。
3. Phase 3 feature 输出
   - 使用 `outputs/features/handcrafted_features.csv` 的 `subject_id` 覆盖范围确认每名 `supervised_main` 患者有可用特征。
   - 不读取、不排序、不比较任何特征值；特征值不能参与 fold 决策。

## 输出

1. `outputs/folds/outer_folds.json`
   - 顶层字段：
     - `schema_version`
     - `outer_cv`
     - `unit`
     - `inner_cv`
     - `inner_k`
     - `n_supervised_main`
     - `supervised_subjects`
     - `folds`
   - `folds` 中每项包含 `outer_fold`、`test_subject`、`supervised_train_subjects` 和 `registry_path`。
2. `outputs/folds/fold_XX_registry.json`
   - 字段遵循 PRD 15.3，并补充匿名 SSL 记录清单：
     - `outer_fold`
     - `test_subject`
     - `supervised_train_subjects`
     - `inner_splits`
     - `ssl_train_subjects`
     - `ssl_train_records`
     - `ssl_excluded_subjects`
     - `ssl_excluded_records`
     - `healthy_ssl_subjects`
     - `normalization_fit_subjects`
     - `feature_selection_fit_subjects`
     - `threshold_selection_subjects`
     - `stages_used`
     - `conditions_used`
   - 所有 subject 列表按 `subject_id` 排序，保证可复现。

## 划分规则

1. outer folds 只由 `supervised_main` 患者生成。
2. 每个 `supervised_main` 患者恰好作为一次 `test_subject`。
3. 当前 outer fold 的 `supervised_train_subjects` 等于全部 `supervised_main` 减去当前 `test_subject`。
4. inner folds 只在当前 outer fold 的 `supervised_train_subjects` 内生成。
5. inner folds 使用 3-fold 分层划分，分层标签来自 `label_primary`。在样本量很小时，若某类样本数低于 3，则仍保持患者级分组，并以确定性轮转方式尽量平衡各 fold 的 Good/Poor 数量。
6. `normalization_fit_subjects`、`feature_selection_fit_subjects`、`threshold_selection_subjects` 全部等于当前 outer train subjects，后续阶段只能在这些 subject 上 fit scaler、feature selector 和阈值。

## SSL pool 规则

1. SSL pool 以 `eeg_qc_summary.csv` 中 QC pass 的匿名 EEG 记录为候选。
2. 每个 fold 必须排除当前 `test_subject` 的所有阶段 EEG 记录，包括 baseline、immediate、mid、final。
3. `ssl_train_subjects` 可包含当前 outer train 的监督卒中患者、`ssl_only_stroke`、`healthy_ssl`，但不能包含当前 `test_subject`。
4. 健康受试者只出现在 SSL pool，不出现在 `supervised_train_subjects` 或任何 inner split。
5. `ssl_train_records` 只包含匿名字段，不包含姓名、原始路径、`.set` 文件名或 `.fdt` 文件名。
6. `ssl_excluded_subjects` 至少包含当前 `test_subject`；若未来配置增加额外排除规则，也必须在该字段显式记录。

## 防泄漏测试

1. `tests/test_splits_no_leakage.py`
   - 验证每名 `supervised_main` 患者恰好作为 outer test 一次。
   - 验证 outer test subject 不在 train、inner train、inner val。
   - 验证 inner split subject 都来自当前 outer train subjects。
   - 验证 normalization、feature selection、threshold selection fit subjects 都是 train subject 子集。
2. `tests/test_ssl_no_leakage.py`
   - 验证 outer test subject 不在 `ssl_train_subjects`。
   - 验证 outer test subject 在 `ssl_excluded_subjects`。
   - 验证 `ssl_train_records` 不包含 outer test subject 的任何阶段记录。
   - 验证 SSL registry 只包含匿名 record metadata。

## 命令

```bash
python scripts/07_make_folds.py --config configs/cv.yaml
pytest tests/test_splits_no_leakage.py tests/test_ssl_no_leakage.py --basetemp=.codex_pytest_tmp -q
```

## 范围外

1. 不实现 ML baseline、MatrixNet、SSL 预训练或模型评估。
2. 不生成 `outputs/ssl/`、`outputs/predictions/`、`outputs/evaluation/`。
3. 不改写原始 `.set` / `.fdt` 文件。
4. 不提交 `outputs/folds/` 或任何其他 `outputs/` 运行产物。

## 自检

该设计覆盖 PRD Phase 4 的 LOPO folds、inner 3-fold、fold-specific SSL registry、no-leakage tests 和真实数据验收要求。设计中没有使用特征值决定 fold，也没有引入模型训练或后续 phase 范围。
