# Phase 8：比例恢复标签与全边功能连接分类设计说明

## 背景与动机

本项目的长期目标是使用治疗前基线静息态 EEG，预测中风患者接受 tACS 后的上肢运动恢复反应。最终应用场景是闭环电刺激参数决策：设备或后续试验设计需要判断当前刺激参数是否更可能产生接近比例恢复的反应，还是更可能出现恢复不足，从而为后续参数调整提供依据。

前序阶段已经完成 EEG 摘要特征修复、传统机器学习基线、无 SSL MatrixNet、标签方向与无泄漏审计，以及探索性 SSL-MatrixNet。既有 MatrixNet 与 SSL 模型没有形成稳定的 permutation 显著证据。因此 Phase 8 不继续扩大深度学习模型，而是回到小样本更稳健的传统分类框架，集中评估两个科学变化：结局标签改为 FMA-UE 比例恢复残差标签，EEG 主特征改为全边功能连接。

## 为什么主任务改为比例恢复组 vs 恢复不足组

既有 Good/Poor 标签强调临床意义改善和天花板修正，适合描述是否达到某个经验改善阈值，但它没有直接表达“患者相对于其基线剩余恢复空间，是否接近预期比例恢复”。Phase 8 的主问题改为：

ProportionalRecovery vs PoorRecovery

该标签更贴近闭环刺激参数优化：如果某个基线 EEG 模式对应更接近比例恢复的患者群体，则它可能为后续刺激参数筛选提供更直接的预后分层依据。该标签仍然只是 cohort-relative outcome definition，不是绝对证明达到 70% 恢复，也不能单独证明 EEG 或 tACS 疗效。

报告中必须称为“proportional-residual median split”或“above-median proportional recovery group vs poor recovery group”。如果结果不稳定或 permutation 不显著，必须明确写明不能宣称 EEG 疗效。

## FMA-UE 比例恢复残差公式

对每个可分析患者使用 FMA-UE 满分：

```text
MAX_FMA_UE = 66
expected_delta_i = 0.7 * (MAX_FMA_UE - baseline_FMA_UE_i)
observed_delta_i = post_FMA_UE_i - baseline_FMA_UE_i
residual_i = expected_delta_i - observed_delta_i
```

解释：

- residual 越小，表示实际恢复越接近或超过 70% 预期恢复。
- residual 越大，表示实际恢复低于 70% 预期恢复。

主阈值为可分析且非天花板患者 residual 的 cohort median：

```text
median_residual = median(residual_i among analyzable non-ceiling patients)
```

主标签：

- `residual_i <= median_residual`：`ProportionalRecovery`
- `residual_i > median_residual`：`PoorRecovery`

整数编码：

- `PoorRecovery = 0`
- `ProportionalRecovery = 1`

必须使用显式列名：

- `primary_label_prop_residual`
- `primary_label_int_prop_residual`

不得把旧 Good/Poor 列名复用为 Phase 8 主标签。

## 排除与并列规则

天花板规则：

- `baseline_FMA_UE == 66`：从主比例恢复分类中排除，标记为 `ceiling_exclude`。
- `baseline_FMA_UE < 66` 且 post FMA-UE 可用：纳入主标签计算。
- baseline 或 post FMA-UE 缺失：从监督主分类中排除，标记为 `excluded_missing`。

并列规则：

- residual 等于 median 时归入 `ProportionalRecovery`。
- residual 大于 median 时归入 `PoorRecovery`。

该规则必须确定性实现并由测试覆盖。

## 敏感性标签

Phase 8 同时生成三个敏感性标签，但主训练默认只使用 proportional-residual median split。

1. `absolute_70_achieved`
   - `residual <= 0`：`ProportionalRecoveryAchieved`
   - `residual > 0`：`NotAchieved`
   - 该标签可能类别不平衡。

2. `current_clinically_meaningful`
   - 复用前序阶段 clinically meaningful / ceiling-adjusted Good/Poor 标签。
   - 仅用于对照，不作为 Phase 8 主标签。

3. `clear_residual_tertile`
   - residual 最低三分位：`ClearProportionalRecovery`
   - residual 最高三分位：`ClearPoorRecovery`
   - 中间三分位：排除出该敏感性训练集。
   - 该标签用于诊断标签噪声。

标签审计必须生成：

- `outputs/reports/phase8_label_audit.md`
- `outputs/evaluation/phase8_label_audit.csv`
- `outputs/evaluation/phase8_label_transition_table.csv`

审计内容包括可分析人数、天花板排除人数、缺失排除人数、两类主标签计数、median residual、按组的 baseline FMA 分布、observed delta 分布、residual 分布、旧标签与新标签的一致性，以及发生标签变化的去标识化 subject ID 列表。

## 全边 FC 特征定义

Phase 8 的主 EEG 特征族是 baseline 静息态全边功能连接。全边表示对选定通道集合中所有无向通道对计算 FC，边数为：

```text
n_edges = n_channels * (n_channels - 1) / 2
```

最小必需 FC 指标：

- coherence
- imaginary coherence
- wPLI

可选指标：

- PLV

频段：

- delta：1-4 Hz
- theta：4-8 Hz
- alpha_mu：8-13 Hz
- low_beta：13-20 Hz
- high_beta：20-30 Hz
- broad_beta：13-30 Hz

条件：

- baseline eyes-open
- baseline eyes-closed

标准矩阵形状：

```text
[N, C, edges, bands]
```

其中 `N` 是当前标签可监督分析患者数，`C` 是 condition 与 metric 的组合维度，`edges` 是全边通道对，`bands` 是频段。

## 为什么 reduced 32-channel full-edge FC 是主特征

主特征为 reduced 32-channel full-edge FC。它保留接近全头覆盖的 10-20 / 10-10 通道，同时把边数控制在小样本传统模型可承受范围内。32 通道时边数为：

```text
32 * 31 / 2 = 496
```

该维度与 Lin-style 小样本分析更兼容，也降低了相对 62 通道全边 FC 的多重比较和高维过拟合风险。

固定优先通道包含：

Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, POz, O1, Oz, O2。

如果精确标签不可用，选择器必须使用确定性 fallback，从可用 10-20 / 10-10 通道中补足最接近的全头覆盖通道，并把选择结果写到：

- `outputs/features/phase8_reduced32_channel_selection.csv`

如果有效通道少于 24 个，必须显式失败，不得静默降级。

## 为什么 62-channel full-edge FC 是次要探索特征

62-channel full-edge FC 使用可用 62 通道 baseline EEG montage 的全部通道，预期边数为：

```text
62 * 61 / 2 = 1891
```

该特征保留更多空间信息，但在样本量很小的 LOPO 设置中维度明显更高，过拟合风险和计算成本更大。因此 Phase 8 要实现并 smoke-test 62 通道全边 FC，但 full-mode 模型训练默认不启动。只有后续书面计划明确调度时，才允许把 M16a/M16b 升级到 full-mode。

必需边元数据：

- `outputs/features/phase8_reduced32_full_edge_index.csv`
- `outputs/features/phase8_full62_full_edge_index.csv`

每行必须包含 `edge_index`、`ch_i`、`ch_j`、可用时的 ROI 和半球字段、`edge_type`，且不得写入本地路径。

必需矩阵产物：

- `outputs/matrices/phase8_fc_full_reduced32_eo.npy`
- `outputs/matrices/phase8_fc_full_reduced32_ec.npy`
- `outputs/matrices/phase8_fc_full62_eo.npy`
- `outputs/matrices/phase8_fc_full62_ec.npy`

如果 62 通道矩阵生成在当前环境过慢，只允许 fast/smoke 模式生成并在报告中说明 full-mode 推迟。

## 模型族

Phase 8 主模型优先使用小样本友好的传统分类器。

主模型：

- `M14a_prop_reduced32_fullfc_ridge_logistic`
- `M14b_prop_reduced32_fullfc_elasticnet`
- `M14c_prop_reduced32_fullfc_linear_svm`
- `M14d_prop_reduced32_fullfc_pls_da`

主输入为 reduced 32-channel full-edge FC 的 EO+EC 合并特征。所有模型必须使用外层 fold-safe imputer、scaler 和可选 variance filter。PLS 或 PCA 只能在外层训练集内拟合，PLS 组件数只能在内层 CV 选择。

对照模型：

- `M15a_prop_roi_fc_best_ml`：既有 ROI-FC 特征。
- `M15b_prop_summary_eeg_best_ml`：既有 EEG 摘要特征。

探索模型：

- `M16a_prop_full62_fullfc_ridge_logistic`
- `M16b_prop_full62_fullfc_pls_da`

M16 仅允许 fast-mode smoke，除非中文 spec 和 TDD plan 后续明确把 full-mode M16 纳入计划。本设计不纳入 M16 full-mode。

## 交叉验证与泄漏规则

评估单位为患者级 LOPO。每个外层折只留出一个患者，且不得产生窗口级预测。

每个外层折必须满足：

1. 外层测试患者不得进入训练、内层验证、imputation fitting、scaling、variance selection、feature selection、PCA/PLS fitting、hyperparameter selection、threshold selection。
2. 主标签的 cohort median residual 可在可分析开发队列上计算一次，因为本阶段把它定义为 cohort-relative outcome definition。
3. 不得在每个外层折重算主 median，除非运行单独的 `train_median_threshold` 敏感性分析。
4. 外层测试 EEG 不得参与任何 preprocessing fit。
5. 标准化、缺失填补、特征过滤、PLS/PCA 和阈值选择均只在外层训练侧完成。
6. 超参数只能在外层训练患者的内层 CV 中选择。
7. 默认阈值为 0.5 或模型决策边界；任何学习阈值必须来自内层 CV。
8. 预测输出必须是一名患者一行，不得重复 model-patient 组合。

必须实现 `train_median_threshold` 敏感性模式：对每个外层折只用外层训练患者 residual median 标记外层测试患者，用于评估 full-cohort median outcome definition 是否影响结论。

## 评估计划

每个模型输出以下患者级指标：

- ROC-AUC
- PR-AUC
- balanced accuracy
- sensitivity
- specificity
- F1
- Brier score
- bootstrap 95% CI
- permutation p-value
- confusion matrix
- score direction audit：ProportionalRecovery 平均分、PoorRecovery 平均分、`AUC(score)`、`AUC(1-score)`

Permutation 必须在患者标签上置换，不得把窗口或种子当作独立样本。Full-mode 默认 bootstrap 1000 次、permutation 1000 次。Fast-mode 可减少重采样次数并允许 fold limit，但报告不得进行科学解释。

模型输出：

- `outputs/predictions/phase8_prop_full_edge_patient_predictions.csv`
- `outputs/evaluation/phase8_prop_full_edge_metrics.csv`
- `outputs/evaluation/phase8_prop_full_edge_seed_wise_metrics.csv`，如果使用多 seed
- `outputs/evaluation/phase8_prop_full_edge_patient_averaged_metrics.csv`，如果使用多 seed
- `outputs/reports/phase8_proportional_full_edge_fc_report.md`

## 隐私与提交规则

所有公开表格、报告、索引和日志只能包含去标识化 subject ID、模型名、折号、指标和相对产物说明。不得提交电子表格、原始 EEG、预处理 EEG、EEG 二进制或头文件、模型 checkpoint、`outputs/` 目录、个人姓名、本地绝对路径或任何可反推个人身份的信息。

代码中的隐私扫描必须覆盖待提交文件和输出元数据。报告中的文件清单只写项目相对路径，不写本机路径。

允许提交的内容仅限：

- `configs/phase8.yaml`
- Phase 8 源码模块
- Phase 8 脚本
- Phase 8 测试
- 本中文 spec
- Phase 8 TDD plan

## Real-data Acceptance

Phase 8 完成标准：

1. 使用 `codex/phase8-proportional-full-edge-fc` 分支和 `.worktrees/phase8-proportional-full-edge-fc` 隔离 worktree。
2. 修改前 baseline tests 通过。
3. 中文 spec 和 TDD plan 均已创建并完成自查。
4. 所有实现遵守 TDD：先写失败测试，确认失败，再写最小实现并确认通过。
5. fast-mode 标签、reduced32 全边 FC、模型训练和报告生成通过。
6. full-mode 完成 M14a/M14b/M14c/M14d/M15a/M15b。
7. 62 通道全边 FC 至少完成 fast-mode smoke；full-mode 不在本阶段启动。
8. 全量测试通过。
9. no-leakage 检查通过。
10. privacy scan 通过，且没有 outputs、原始数据、checkpoint 或私有产物 staged。
11. 只有在测试、real-data acceptance、no-leakage 和隐私扫描全部通过后，才合并回 `main` 并推送远端。

## 阶段边界

Phase 8 不启动 SSL，不启动 BYOL、SimSiam、MAE、masked matrix modeling 或 Phase 7 扩展。不训练未计划的深度学习模型，不启动 MatrixNet，不使用 post-treatment EEG 作为监督分类输入，不使用 immediate、mid 或 final EEG 作为监督分类输入。治疗后临床 FMA-UE 仅用于结局标签定义。

## 自查结论

- Placeholder scan：本文不包含待补标记、未决占位项或空白承诺。
- Scope check：范围仅限比例恢复标签、sensitivity labels、reduced32/full62 全边 FC 产物、小样本传统模型、M14/M15 full-mode 与 M16 fast smoke。
- Leakage check：外层测试患者不进入任何训练侧拟合、选择、阈值或调参流程；主 median residual 作为 cohort-relative outcome definition 单独披露，并提供 train-median sensitivity。
- Privacy check：提交范围只包含代码、配置、脚本、测试和文档；运行产物、原始数据、私有路径、个人标识和模型权重均不得提交。
- Phase boundary：本阶段不启动 SSL、不训练 MatrixNet、不训练未计划深度模型、不把治疗后 EEG 作为监督输入。
