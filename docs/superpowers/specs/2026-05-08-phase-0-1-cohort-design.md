# Phase 0+1 队列与标签设计

## 目标

完成 tACS 脑卒中恢复预测项目的第一段可复现工作：仓库整理、配置、环境验证、队列构建、去标识化 `subject_id`，以及 FMA-UE 恢复标签。

本阶段不训练模型，不提取 EEG 矩阵，也不做完整 EEG 信号质控。它只交付一个可信的患者级队列和标签审计结果，供后续阶段使用。

## 输入

主要结构化输入文件：

`F:\CJZProjectFile\StrokePredictSSL-DLModel\current_data_status_overview_data_only.xlsx`

相关工作表：

- `01_患者数据总览`：患者级临床信息与数据可用性总览。
- `03_临床量表原始`：原始临床量表数值和完整性字段。
- `06_预处理静息态阶段汇总`：预处理后静息态 EEG 汇总。
- `07_预处理静息态文件明细`：预处理后静息态 EEG 文件索引。
- `02_统计汇总`：源数据审计中的汇总计数，用于 sanity check。

后续阶段会使用的外部 EEG 根目录：

- 患者 EEG：`F:\CJZFile\EEG_M1\Patient_tACS_M1_RestingStateEEG_afterProcess`
- 健康 EEG：`F:\CJZFile\EEG_M1\Health_tACS_M1_RestingStateEEG_afterProcess`

在 Phase 0+1 中，这些路径只写入配置。原始 `.set` 和 `.fdt` 文件不复制到项目仓库。

## 隐私边界

工作簿和 EEG 文件夹中包含姓名、类似姓名的文件夹标签等直接身份信息。Phase 0+1 只把这些字段当作私有输入，不写入公开输出。

提交到 Git 的代码和生成的公开输出中不能包含：

- `姓名`
- `姓名写法`
- `EEG文件夹`
- `subject_name`
- 包含真实姓名的原始文件路径
- 未来输入中如果出现病历号，也不能输出

唯一稳定的公开标识符是确定性生成的去标识化 `subject_id`。如确实需要生成私有映射文件，该文件必须被 Git 忽略。

## 仓库结构

Phase 0+1 会创建以下文件：

- `configs/paths.yaml`：本地输入和输出路径。
- `configs/project.yaml`：队列与标签设置。
- `src/stroke_predict/config.py`：配置读取和路径解析。
- `src/stroke_predict/io/excel_status.py`：读取工作簿中的指定工作表，并检查显式 schema。
- `src/stroke_predict/cohort/labels.py`：FMA 标签规则。
- `src/stroke_predict/cohort/ids.py`：确定性去标识化 ID 生成。
- `src/stroke_predict/cohort/build.py`：队列组装和审计表生成。
- `scripts/00_validate_environment.py`：环境与路径验证脚本。
- `scripts/01_build_cohort.py`：命令行队列构建脚本。
- `tests/`：TDD 测试，覆盖标签规则、PII 字段过滤、配置读取、输出 schema 和队列角色分配。

运行时生成的输出：

- `outputs/cohort/cohort_master.csv`
- `outputs/cohort/label_audit.csv`
- `outputs/cohort/label_distribution.json`
- `outputs/cohort/cohort_summary.json`
- `outputs/figures/fig_label_distribution.png`

`outputs/` 已被 Git 忽略。

## 标签规则

主标签采用 PRD 中的 ceiling-adjusted clinically meaningful FMA-UE response 规则：

- 缺少治疗前或治疗后 FMA：`missing`
- 治疗前 FMA 等于 66：`ceiling_exclude`
- 治疗前 FMA <= 61：若 delta FMA >= 5，则为 `Good`，否则为 `Poor`
- 治疗前 FMA 为 62 到 65：若 delta FMA >= `min(3, 66 - baseline_fma)`，则为 `Good`，否则为 `Poor`

敏感性标签字段：

- `label_delta5_all`
- `label_prop70`
- `label_low_baseline_only`
- `outcome_delta_fma`
- `outcome_post_fma`

审计表记录每个标签的数值依据：

- `subject_id`
- `baseline_fma`
- `post_fma`
- `delta_fma`
- `possible_recovery`
- `recovery_ratio`
- `label_primary`
- `label_delta5_all`
- `label_prop70`
- `label_low_baseline_only`
- `label_reason`

## 角色分配

Phase 0+1 基于当前已有元数据分配角色：

- `supervised_main`：卒中患者，FMA 治疗前/后完整，有 baseline eyes-open EEG 和 baseline eyes-closed EEG，主标签为 `Good` 或 `Poor`，且不是天花板排除病例。
- `ceiling_exclude`：卒中患者，治疗前 FMA 等于 66。
- `ssl_only_stroke`：卒中受试者，至少有一条可用的预处理静息态 EEG 记录，但不满足 `supervised_main` 条件。
- `healthy_ssl`：健康受试者，至少有一条可用的预处理静息态 EEG 记录。
- `excluded_no_eeg`：临床表中存在，但预处理索引中没有可用 baseline EO/EC 元数据的患者。
- `excluded_bad_qc`：为 Phase 2 的信号级 EEG 质控预留；Phase 0+1 不根据 EEG 信号质量标记 bad QC。

如果一个受试者满足多个角色，导出的 `role` 字段按以下优先级选择最适合分析的角色：

`supervised_main`, `ceiling_exclude`, `ssl_only_stroke`, `healthy_ssl`, `excluded_no_eeg`, `excluded_bad_qc`。

## 数据流程

1. 验证 Python 能导入所需包，并确认工作簿存在。
2. 读取相关工作表，使用稳定列名。
3. 标准化工作簿中已经存在的 stage 和 condition 标签。
4. 分别为卒中和健康来源生成确定性的去标识化 ID。
5. 应用 FMA 主标签和敏感性标签规则。
6. 按源数据 subject key 将临床行与预处理 baseline EEG 元数据连接。
7. 分配队列角色。
8. 写出匿名队列表、标签审计表、标签分布 JSON 和队列 summary JSON。
9. 只使用去标识化数据生成标签分布图。

## 错误处理

以下情况脚本应直接失败，并给出清楚错误信息：

- 工作簿路径不存在。
- 必需工作表缺失。
- 必需列缺失。
- 完整性字段显示 FMA 完整，但 FMA 数值无法解析为数字。
- 公开输出中包含被禁止的 PII 字段。
- 同一来源内生成了重复的 `subject_id`。

以下情况可以给出 warning：

- MBI、BBT、MMSE 等可选临床字段缺失。
- 某名患者有临床数据，但没有匹配的预处理 EEG 元数据。
- 健康受试者没有临床字段，这属于预期情况。

## 测试策略

本阶段所有生产代码都按 test-first 实现。

最低测试覆盖：

- 标签规则测试：缺失值、baseline 66、baseline <= 61 且 delta 为 4/5、baseline 64/65 的天花板修正规则、proportional recovery 标签。
- ID 测试：确认去标识化 ID 确定、唯一，并带有来源前缀。
- PII 测试：确认公开输出 schema 不包含被禁止字段。
- 配置测试：确认路径可读取和解析。
- 队列构建测试：用小型 synthetic data frame 覆盖 `supervised_main`、`ceiling_exclude`、`ssl_only_stroke`、`healthy_ssl` 和 `excluded_no_eeg`。
- 脚本 smoke test：用 synthetic fixture 测试 `00_validate_environment.py` 和 `01_build_cohort.py`。

验收命令：

```bash
python scripts/00_validate_environment.py
python scripts/01_build_cohort.py --config configs/project.yaml
pytest tests -q
```

## 验收标准

- `current_data_status_overview_data_only.xlsx` 不被 Git 跟踪。
- 原始 EEG 文件不被 Git 跟踪。
- `cohort_master.csv` 和 `label_audit.csv` 只包含去标识化 `subject_id`。
- `label_primary` 只允许 `Good`、`Poor`、`ceiling_exclude` 或 `missing`。
- baseline FMA 为 64/65 的病例在满足修正阈值时可以标为 `Good`。
- baseline FMA 为 66 的病例不进入主二分类监督队列。
- `cohort_summary.json` 中报告的监督队列数量来自实际数据。
- 进入 Phase 2 前，相关测试必须通过。

## 不在本阶段范围内

- 读取原始 EEG 信号。
- 信号级 EEG 质控。
- PSD、FC、tACS-informed 特征提取。
- LOPO fold 生成。
- 传统机器学习、MatrixNet、SSL、解释性分析和 manuscript 生成。

这些内容会在后续阶段分别写 spec 和 plan。
