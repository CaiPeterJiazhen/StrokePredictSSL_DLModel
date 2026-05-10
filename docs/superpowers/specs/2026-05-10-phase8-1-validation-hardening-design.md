# Phase 8.1：比例恢复全边 FC 验证与加固设计说明

## 背景

Phase 8 将主标签改为 `ProportionalRecovery` vs `PoorRecovery`，使用 FMA-UE 比例恢复残差的中位数切分。探索性结果显示 reduced32 full-edge FC 的 M14b elastic-net 有正向信号，但同时报告 `source_mode = psd_artifact_proxy`，且 ROI-FC 与 summary EEG 对照出现完全相同指标。因此 Phase 8.1 的目标不是继续深度学习，而是验证数据来源、模型比较、统计校正、阈值选择和患者级错误是否可靠。

本阶段禁止生成手稿图，禁止启动 MatrixNet，禁止启动 SSL。所有运行产物仍写入被忽略的 `outputs/`，不提交 Excel、`.set`、`.fdt`、checkpoint、原始 EEG、私有路径或个人标识。

## 目标

Phase 8.1 只做验证与硬化：

1. 审计 full-edge FC 的来源模式，区分真实 baseline EO/EC time-series FC 与 PSD/artifact proxy。
2. 如果没有真实 time-series 输入，报告必须明确说明不能称为真实 time-series full-edge FC。
3. 对真实 time-series extraction 的可用路径做测试覆盖；用户请求真实 FC 但未提供 time-series 输入时必须显式失败。
4. 重新运行或验证 M14a、M14b、M14c、M14d、M15a、M15b 的 Phase 8 classical models。
5. 审计 M15a 与 M15b 特征矩阵和预测是否被复用。
6. 增加 raw permutation p、Bonferroni p、Benjamini-Hochberg FDR q 和显著性标志。
7. 对最佳模型输出固定阈值、fold-safe inner-CV 阈值、fold-safe inner-CV Youden 阈值、校准曲线、Brier score 和分组分数分布。
8. 输出患者级错误审计，包含旧标签、新标签、FMA、残差、预测分数、预测类别、正确性、排名和边界邻近性。
9. 生成 Phase 8.1 验证报告、多重比较表、阈值校准表、患者错误审计表和 no-leakage 报告。

## Source Mode 审计

Phase 8 的 full-edge FC extraction 必须写出来源模式：

- `time_series`：由 baseline EO/EC 预处理 EEG time-series 直接计算 coherence、imaginary coherence、wPLI。
- `psd_artifact_proxy`：由 PSD 或 artifact-like summary/matrix 代理构造 edge feature，不能声称为真实 time-series full-edge FC。
- `not_available`：没有可审计产物。

报告必须回答：当前 reduced32 full-edge FC 是否来自真实 time-series。若为 proxy，结论只允许写成“PSD/artifact proxy derived full-edge-like features”，并说明在宣称 time-series FC evidence 之前必须接入 baseline EO/EC preprocessed time-series、固定 reduced32 通道、生成 496 无向边、计算规定 FC 指标和频段，并通过 no-leakage 与 real-data acceptance。

## 真实 Time-Series FC 要求

真实 reduced32 full-edge FC 的输入是 baseline EO/EC 预处理 EEG time-series。固定 reduced32 channel list 优先选择 Phase 8 已定义通道；通道不足时按既有 deterministic fallback 补足，少于 24 个有效通道时失败。reduced32 的 canonical edge count 为 496。

必须支持的 FC 指标：

- coherence
- imaginary coherence
- wPLI

必须支持的频段：

- delta 1-4
- theta 4-8
- alpha_mu 8-13
- low_beta 13-20
- high_beta 20-30
- broad_beta 13-30

canonical matrix 形状为 `[N, metrics, edges, bands]`。EO 和 EC 可以分别保存，再在模型阶段展平合并。edge metadata 和 channel selection audit 不得包含本地路径。

如果运行配置声明需要真实 time-series full-edge FC，但没有提供 time-series input key，脚本必须失败并说明缺少 baseline EO/EC time-series input，不能自动回退到 proxy。

## M15a/M15b 重复指标审计

M15a 和 M15b 是不同对照：

- M15a 使用 ROI-FC baseline feature columns。
- M15b 使用 summary EEG baseline feature columns。

Phase 8.1 必须输出 feature signature 和 prediction signature 审计。feature signature 至少包含列名 hash、数值矩阵 hash、列数和非缺失计数；prediction signature 至少包含每个模型患者顺序下的预测分数 hash 和预测标签 hash。

如果 M15a/M15b 特征不同但预测完全相同，报告解释为小样本 LOPO 与模型选择可能导致相同输出。如果特征相同或预测被同一对象静默复用，管线必须失败，除非配置显式声明该复用是有意的。默认不允许 comparison models silently sharing predictions。

## 多重比较校正

Phase 8.1 对所有 Phase 8 模型输出：

- `raw_permutation_p_value`
- `bonferroni_p_value`
- `fdr_q_value`
- `nominal_p_lt_0_05`
- `fdr_q_lt_0_05`
- `bonferroni_p_lt_0_05`

Bonferroni 使用本次 Phase 8 模型数量作为校正因子。FDR 使用 Benjamini-Hochberg。缺失 p-value 保留为缺失且不标记显著。

## 阈值与校准分析

最佳模型默认按 ROC-AUC 选择，若 ROC-AUC 缺失则按 PR-AUC，再按 raw permutation p-value。阈值分析只使用外层训练侧的 inner-CV 或训练预测选择阈值，不能用外层测试预测挑阈值。

输出必须包含：

- fixed 0.5 threshold 的患者级分类指标。
- inner-CV threshold 的患者级分类指标。
- inner-CV Youden threshold 的患者级分类指标。
- calibration curve data，每个 bin 输出 count、mean_score、observed_rate。
- Brier score。
- 按真实组别的 score distribution summary。

如果历史 Phase 8 predictions 只有固定 0.5 阈值且没有 inner-CV threshold columns，Phase 8.1 可以报告 inner-CV 阈值不可用，但不能用外层测试预测补算阈值。

## 患者级错误审计

最佳模型必须输出每名患者一行：

- `patient_id`
- `old_label`
- `proportional_label`
- `baseline_fma`
- `post_fma`
- `observed_delta`
- `expected_delta`
- `residual`
- `predicted_score`
- `predicted_label`
- `correct`
- `rank`
- `near_median_threshold`
- `old_new_label_disagree`

`near_median_threshold` 使用配置化容差，默认按 analyzable residual 的四分位距乘 0.10；样本太少或 IQR 为 0 时使用 residual 与 median 的绝对差排序，最接近中位数的 25% 标为 true。

## 报告

Phase 8.1 必须生成：

- `outputs/reports/phase8_1_validation_report.md`
- `outputs/evaluation/phase8_1_multiple_comparison_correction.csv`
- `outputs/evaluation/phase8_1_threshold_calibration.csv`
- `outputs/evaluation/phase8_1_patient_error_audit.csv`
- `outputs/reports/phase8_1_no_leakage_report.txt`

验证报告必须逐项回答：

1. full-edge FC 是真实 time-series FC 还是 proxy。
2. 如果是 proxy，宣称 time-series FC evidence 前必须做什么。
3. 真实 time-series FC 是否复现 Phase 8 正向信号。
4. ROI-FC 与 summary EEG 指标为什么相同。
5. 最佳模型是否通过 FDR 或 Bonferroni。
6. 分类性能是否受阈值选择限制。
7. 错误是否集中在 residual median 边界附近。
8. 结果是否足以支持下一步 no-SSL MatrixNet。
9. SSL 是否应继续阻塞，直到真实 FC 与 MatrixNet 验证完成。

## Acceptance

完成标准：

1. 修改前 baseline tests 通过。
2. 中文 spec 和 TDD implementation plan 已写入 docs。
3. 每个新行为先有失败测试，再有实现，再跑绿。
4. 全量 tests 通过。
5. Phase 8.1 real-data acceptance 运行完成，且不启动 MatrixNet/SSL。
6. Source mode 审计清晰。
7. M15a/M15b 重复指标被解释或修复。
8. 多重比较校正、阈值/校准分析、患者错误审计和 no-leakage report 都生成。
9. `outputs/`、Excel、`.set`、`.fdt`、checkpoint、原始 EEG、私有路径和个人标识均不 staged。
10. 所有验证通过后才合并回 main 并推送。

## 自查

- Placeholder scan：本文没有未完成占位项。
- Scope check：范围仅限 Phase 8.1 验证与硬化，不启动 MatrixNet、SSL 或手稿作图。
- Leakage check：阈值选择不能使用外层测试预测；报告必须保留 no-leakage 审计。
- Privacy check：提交范围只包含代码、配置、测试和 docs；运行产物留在 ignored outputs。
