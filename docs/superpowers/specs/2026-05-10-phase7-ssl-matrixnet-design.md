# Phase 7：SSL-MatrixNet 预训练设计说明

## 背景与目标

本阶段面向中风患者静息态 EEG 预后建模，在既有 Phase 6/6.2 Lin-style MatrixNet 监督模型基础上加入自监督预训练。项目最终监督任务保持不变：仅使用治疗前基线静息态 EEG 的睁眼与闭眼 PSD/ROI-FC 矩阵，预测 post-tACS 上肢 FMA-UE 恢复结局 Good/Poor。

Phase 7 的目标是实现并审计 SSL-MatrixNet 基础设施：构建逐外层折安全的未标注矩阵池，对 PSD 与 ROI-FC 编码器执行 masked matrix modeling 预训练，再把预训练编码器迁移到既有患者级 Good/Poor 监督微调流程中。所有结果必须以患者为评估单位，不得把窗口、记录或种子重复行误当独立患者证据。

## 为什么 Phase 7 只是探索性实验

Phase 6.2 已确认 Poor=0、Good=1、`BCEWithLogitsLoss` 目标方向和 `sigmoid(logit)` 的 Good 概率方向没有编码错误，也完成了外层测试泄漏审计。与此同时，Phase 6.2 的无 SSL MatrixNet 仍表现不稳定且未形成稳健显著证据。因此 Phase 7 只能作为表示学习探索，不得预设或宣称 EEG 疗效。

只有当 SSL-A 在患者平均 ROC-AUC 上相对匹配的无 SSL 基线有明确提升、跨种子稳定、bootstrap 置信区间不过宽、permutation p<0.05、方向审计通过且无泄漏报告通过时，才可描述为探索性改善。否则报告必须写明：不能宣称 EEG 疗效，SSL-MatrixNet 未在本阶段产生稳定且 permutation 显著的提升。

## SSL 数据来源

优先复用仓库已有去标识化产物：

- `outputs/cohort/cohort_master.csv`
- `outputs/folds/outer_folds.json`
- `outputs/folds/fold_XX_registry.json`
- `outputs/matrices/psd_eo.npy`
- `outputs/matrices/psd_ec.npy`
- `outputs/matrices/fc_roi_eo.npy`
- `outputs/matrices/fc_roi_ec.npy`
- `outputs/matrices/matrix_subject_index.csv`
- `outputs/features/features_tacs_target_summary.csv`
- Phase 6.2 指标、预测与方向审计报告

若已有 SSL-ready 的未标注 PSD/FC 矩阵和元数据，则加载并验证其字段、行数、subject 对齐、stage/condition 合法性以及路径隐私，不重复生成。若不存在，则实现受测试覆盖的矩阵索引构建路径：只读取项目既有工作流中的忽略 EEG 派生产物，输出去标识化 SSL 矩阵索引和矩阵文件，不提交原始 EEG、预处理 EEG、`.set`、`.fdt`、Excel、checkpoint 或任何本地路径。

## 泄漏规则

每个外层 LOPO 折必须独立构造 SSL 池：

1. 外层测试患者完全排除于监督训练、内层验证、SSL 预训练、归一化拟合、阈值选择、方向校准、超参数选择和 checkpoint 选择之外。
2. 外层测试患者的全部 EEG 阶段均排除于 SSL：baseline、immediate、mid、final，且同时覆盖 eyes-open 与 eyes-closed。
3. 健康对照 EEG 可进入 SSL 池，因为健康对照不是 Good/Poor 监督测试患者；输出中仍不得包含姓名、原始路径或可识别信息。
4. 临床结局不完整的中风患者可作为未标注 EEG 进入 SSL 池。
5. 治疗阶段 EEG 只允许用于 SSL 预训练，绝不作为监督分类器输入。
6. 监督微调和外层评估只使用 19 名 `supervised_main` 患者的基线 EO/EC PSD/FC 矩阵。
7. 监督矩阵归一化只在外层训练患者上拟合。
8. SSL 归一化只在该折 SSL 训练池上拟合，并排除外层测试患者。
9. 外层测试预测不得用于分数方向、阈值、超参数、checkpoint、SSL variant 或模型族选择。

## SSL 方法

本阶段只实现 PSD/FC 矩阵 masked matrix modeling，不实现 raw EEG BYOL、SimSiam、EEGNet 主模型或 full-channel FC。

PSD 输入为 EEG-derived PSD 矩阵，随机遮盖 channel-frequency patch，编码器重建被遮盖值，损失仅在 mask 元素上计算。FC 输入为 ROI-FC canonical edge × band 结构，随机遮盖 edge-band patch，同样只对被遮盖值计算重建损失。PSD 与 FC 同时可用时联合预训练 PSD 与 FC 编码器，默认损失为 `1.0 * PSD masked MSE + 1.0 * FC masked MSE`，权重、mask ratio 和 seed 均由配置控制。

默认 `mask_ratio=0.25`，支持 `0.15`、`0.25`、`0.40`。不把 EO/EC 作为 contrastive positive pair，本阶段没有 contrastive loss。checkpoint 只保存到忽略目录 `outputs/ssl_matrixnet/checkpoints/<run_id>/<ssl_variant>/fold_XX/`，并在公开预测表中仅写 redacted 相对描述，不写绝对路径。

## SSL Variant 与模型族

基础设施必须覆盖四个 SSL variant：

- SSL-A `stroke_baseline`：外层训练中风 baseline EO/EC，加 SSL-only 中风 baseline EO/EC；排除外层测试患者所有阶段。
- SSL-B `stroke_healthy_baseline`：SSL-A 加健康 baseline EO/EC。
- SSL-C `stroke_all_stage`：外层训练中风 baseline/immediate/mid/final EO/EC，加 SSL-only 中风 all-stage；排除外层测试患者所有阶段。
- SSL-D `stroke_all_stage_healthy`：SSL-C 加健康 baseline EO/EC。

本阶段 full-mode 科学训练仅规划 SSL-A。B/C/D 只做 fast-mode 数据构建、泄漏逻辑和 CLI 验收，除非后续计划明确扩展资源评估与 full-mode 训练。

Primary full-mode SSL-A 模型：

- `M9a_sslA_fc_only`：FC_ROI_EO + FC_ROI_EC，加载 SSL-A FC encoder。
- `M9b_sslA_psd_fc`：PSD_EO + PSD_EC + FC_ROI_EO + FC_ROI_EC，加载 SSL-A PSD/FC encoder。
- `M9c_sslA_psd_fc_tacs`：PSD/FC 加 tACS target summary vector，加载 SSL-A PSD/FC encoder。

Secondary 模型：

- `M13_sslA_clinical_eeg`：PSD/FC 加可用 tACS 与 clinical vector，加载 SSL-A PSD/FC encoder；仅作为次要融合分析，不替代 EEG-only 主分析。

Fast-mode 只为基础设施验收运行 `M9a_sslB_fc_only_fast`、`M9a_sslC_fc_only_fast`、`M9a_sslD_fc_only_fast` 对应的数据和泄漏路径。

## 训练、微调与评估设计

`fast` 模式用于测试和 smoke validation：允许 `fold_limit`，仅用 seed 0，SSL epochs 与 fine-tune epochs 取 2-5，不产生科学解释。

`full` 模式用于 SSL-A 探索性结果：必须使用 CUDA 且 `require_cuda=true`，不得静默回退 CPU；seeds 为 `[0,1,2,3,4]`；执行 19-fold LOPO；默认 SSL 预训练 100 epochs，监督微调 `max_epochs=120`、`patience=20`；bootstrap 与 permutation 默认各 1000 次；方向校准沿用 Phase 6.2 的 `inner_val_auc` 合同，不使用外层测试预测。

监督输入始终是基线 EO/EC PSD/FC 与可选 tACS/clinical 向量。Phase 7 明确不把 post-treatment EEG 作为监督分类器输入；治疗阶段 EEG 仅可在 SSL-C/D 的未标注预训练池中出现。

评估指标包括 mean seed ROC-AUC、seed std ROC-AUC、pooled AUC、patient-averaged AUC、patient-averaged bootstrap 95% CI、patient-averaged permutation p-value、PR-AUC、balanced accuracy、sensitivity、specificity、F1、Brier score、`AUC(score)`、`AUC(1-score)`、Good/Poor 平均分、`direction_correct` 与 `score_orientation` 计数。报告必须比较 Phase 6.2 匹配无 SSL 模型：M9a 对 M8b，M9b 对 M8c，M9c 对 M8d，并在可用时纳入 Phase 5.2 ML baseline。

## 输出文件

所有运行产物写入被忽略的 `outputs/`，不得提交。必须生成：

- `outputs/ssl_matrixnet/ssl_matrix_index.csv`
- `outputs/ssl_matrixnet/ssl_fold_pool_audit_phase7.csv`
- `outputs/reports/no_leakage_report_phase7.txt`
- `outputs/ssl_matrixnet/pretrain_log_phase7.csv`
- `outputs/ssl_matrixnet/checkpoints/<run_id>/...`
- `outputs/predictions/ssl_matrixnet_patient_predictions_phase7.csv`
- `outputs/evaluation/ssl_matrixnet_metrics_phase7.csv`
- `outputs/evaluation/ssl_matrixnet_seed_wise_metrics_phase7.csv`
- `outputs/evaluation/ssl_matrixnet_patient_averaged_metrics_phase7.csv`
- `outputs/reports/phase7_ssl_matrixnet_report.md`
- `outputs/ssl_matrixnet/config_used_phase7.yaml`

预测表必须包含：`model_name`、`ssl_variant`、`seed`、`outer_fold`、`patient_id`、`true_label`、`label_int`、`logit`、`sigmoid_score`、`predicted_score`、`predicted_label`、`threshold`、`threshold_source`、`score_orientation`、`run_mode`、`ssl_checkpoint_path_redacted`、`device`、`best_epoch`、`train_loss_final`、`val_loss_best`。

## 隐私与提交规则

公开索引和报告只允许去标识化 subject ID，例如 `STK-xxx` 或健康对照编码。不得写入绝对本地路径、原始 EEG 路径、`.set`、`.fdt`、`.xlsx` 字符串、姓名或其他直接个人标识。不得提交 `outputs/`、checkpoint、CSV 运行输出、Excel、raw EEG、preprocessed EEG 或任何私有路径文件。

允许提交的文件仅限 Phase 7 源码、脚本、配置、测试、中文 spec、TDD plan，以及为了加载 checkpoint 所需的小范围 MatrixNet 代码改动。

## Real-data Acceptance

Phase 7 完成标准：

1. 使用隔离 worktree 和 `codex/phase7-ssl-matrixnet` 分支。
2. 修改前基线测试通过。
3. 中文 spec 和 TDD plan 存在并通过自查。
4. 所有新增实现遵守 TDD：先写失败测试，确认失败，再写最小实现并确认通过。
5. `python -m pytest tests/test_ssl_matrixnet_* -q` 通过。
6. `python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider` 通过。
7. CUDA fast-mode SSL-A 预训练和微调通过。
8. CUDA fast-mode SSL-B/C/D 预训练泄漏检查通过。
9. SSL-A full-mode 完成；若因可复现环境或数据错误失败，则不得声称 Phase 7 完成，也不得 merge/push。
10. SSL pool audit 确认每折外层测试患者所有阶段均排除。
11. no-leakage report 通过。
12. Phase 7 report 存在并回答所有决策问题。
13. 预测输出为患者级。
14. 隐私扫描通过，且没有 forbidden artifact 被 staged 或 committed。
15. 只有在测试、real-data acceptance 和隐私扫描全部通过后，才合并回 `main` 并推送远端。

## 自查结论

- Placeholder scan：本文不包含待补标记、未决占位项或空白实现承诺。
- Scope check：范围仅限 Phase 7 SSL-MatrixNet、四个既定 SSL variant、SSL-A full-mode 与 B/C/D fast-mode 验收。
- Privacy check：输出与提交规则明确禁止 Excel、`.set`、`.fdt`、raw/preprocessed EEG、checkpoint、`outputs/`、绝对路径和姓名。
- No unplanned training：本阶段不启动 raw EEG BYOL、EEGNet、full-channel FC 或未列出的模型族；full-mode 只规划 SSL-A。
