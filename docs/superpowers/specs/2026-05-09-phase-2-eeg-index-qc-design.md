# Phase 2 EEG 索引与 QC 设计

## 目标

完成 PRD 路线图中的 Phase 2：为 afterProcess 静息态 EEG 建立可复现、去标识化的记录索引和信号头信息 QC 汇总。

本阶段只确认 baseline 及其他阶段静息态 EEG 文件是否存在、是否可读、通道数、采样率、时长、有效窗口数和通道顺序一致性。它不提取 PSD、functional connectivity、tACS target features，不生成 LOPO folds，也不训练任何模型。

## 输入

Phase 2 继续使用 Phase 0+1 的配置入口：

- `configs/project.yaml`
- `configs/paths.yaml`
- `current_data_status_overview_data_only.xlsx`

主要工作表：

- `01_患者数据总览`：用于重建去标识化 stroke `subject_id` 映射。
- `07_预处理静息态文件明细`：用于读取 EEG 记录的私有源 key、阶段、条件、`.set` 和 `.fdt` 私有路径，以及已有头信息字段。

原始 EEG 根目录仍只作为私有输入使用，不复制到仓库。

## 隐私边界

公开输出和 Git 提交中禁止出现：

- 姓名或类似姓名字段；
- 原始 EEG 文件夹名；
- `.set` / `.fdt` 原始文件名；
- 任意本机绝对路径或原始路径片段；
- 工作簿中的 `subject_name`、`set_path`、`fdt_path`。

代码可以在内存中使用私有路径检查文件和读取 EEGLAB 头信息，但写出的 CSV/JSON 只能包含去标识化 `subject_id`、稳定生成的 `record_id` 和 QC 指标。PRD 中的 `file_path` 字段在本项目公开输出中不落盘，以满足“不允许路径泄露”的更高优先级要求。

## 配置

新增 `configs/eeg.yaml`，只保存 Phase 2 的非隐私参数：

- `project_config`: 指向 `configs/project.yaml`
- `qc.min_duration_sec_main`: 主监督 baseline EO/EC 最短有效时长，默认 60 秒
- `qc.min_duration_sec_ssl`: SSL 记录最短有效时长，默认 30 秒
- `qc.allowed_sampling_rate_hz`: 250
- `qc.required_channels`: 62
- `window.length_sec`: 4
- `window.overlap`: 0.5
- `window.min_valid_windows_per_condition`: 10

## 输出

运行时输出到被 Git 忽略的 `outputs/qc/`：

- `eeg_record_index.csv`
- `eeg_qc_summary.csv`
- `channel_order_report.csv`

`eeg_record_index.csv` 字段：

- `record_id`
- `subject_id`
- `source`
- `stage`
- `condition`
- `record_index`
- `set_exists`
- `fdt_exists`

`eeg_qc_summary.csv` 字段：

- `record_id`
- `subject_id`
- `source`
- `stage`
- `condition`
- `exists`
- `readable`
- `n_channels`
- `sfreq`
- `channel_order_hash`
- `duration_sec`
- `n_valid_samples`
- `n_valid_windows_2s`
- `n_valid_windows_4s`
- `n_valid_windows_8s`
- `bad_channel_count`
- `artifact_ratio_if_available`
- `passes_qc`
- `qc_reason`

`channel_order_report.csv` 字段：

- `channel_order_hash`
- `n_records`
- `n_channels`
- `example_subject_id`
- `example_record_id`

通道名本身不写入公开 CSV，避免输出过宽且便于后续只通过 hash 判断是否需要重排。

## 数据流程

1. 读取 `configs/eeg.yaml`，再加载 `configs/project.yaml` 和 `configs/paths.yaml`。
2. 读取工作簿中 Phase 0+1 已使用的 clinical/preprocessed sheets。
3. 复用 Phase 0+1 的确定性 ID 逻辑，按 source key 生成 stroke `STK-xxx` 和 healthy `HC-xxx`。
4. 从 `07_预处理静息态文件明细` 逐行生成匿名 `record_id`。
5. 用私有 `set_path` / `fdt_path` 检查文件是否存在。
6. 优先从 `.set` EEGLAB MATLAB 头文件读取 `nbchan`、`srate`、`pnts`、`trials`、`chanlocs` 和外部 `datfile`。
7. 如果 `.set` 读取失败，但工作簿已有 `nbchan/pnts/trials/srate/chan_labels`，输出 `readable=False` 并保留可审计原因；不把失败路径写入公开输出。
8. 根据采样率、通道数、时长、有效 4 秒窗口数判断 `passes_qc`。
9. 写出索引、QC 汇总和通道顺序报告。

## QC 规则

一条记录通过 QC 需要同时满足：

- `.set` 存在；
- 对外部 `.fdt` 数据的引用存在，或工作簿中的 `fdt_path` 存在；
- `.set` 头信息可读；
- `n_channels == 62`；
- `sfreq == 250`；
- 对 `supervised_main` 的 baseline EO/EC，`duration_sec >= 60`；
- 对其他可进入 SSL 的记录，`duration_sec >= 30`；
- `n_valid_windows_4s >= 10`。

本阶段没有 artifact rejection，不估计坏导和伪迹比例：

- `bad_channel_count` 固定为 0；
- `artifact_ratio_if_available` 为空；
- `n_valid_samples` 等于头信息中的 `pnts * trials`。

## 错误处理

脚本应在以下情况下失败：

- 配置文件缺失或格式错误；
- 工作簿缺失；
- 必需工作表或列缺失；
- 公开输出 schema 包含被禁止的 PII 或路径字段。

以下情况写入 QC 行但不使脚本失败：

- 某条 `.set` 或 `.fdt` 缺失；
- 某条 `.set` 无法读取；
- 单条记录通道数、采样率或时长不达标。

## 测试策略

本阶段所有生产代码按 TDD 实现。最低测试覆盖：

- EEG 配置读取和默认参数。
- 从 synthetic workbook frame 生成匿名 EEG record index。
- 输出 schema 不包含 `subject_name`、`set_path`、`fdt_path`、`file_path` 或本机路径。
- QC 窗口数计算。
- QC 规则对通道数、采样率、时长和读取失败给出明确 `qc_reason`。
- 通道顺序 hash 对大小写和空格稳定。
- 脚本 smoke test 可以在 synthetic workbook 和临时 `.set` 头文件 fixture 上写出 `outputs/qc` 文件。

## 验收命令

```bash
python scripts/00_validate_environment.py
python scripts/01_build_cohort.py --config configs/project.yaml
python scripts/02_index_eeg.py --config configs/eeg.yaml
python scripts/03_run_eeg_qc.py --config configs/eeg.yaml
pytest tests -q
```

## 真实数据验收标准

- `eeg_record_index.csv` 和 `eeg_qc_summary.csv` 记录数与当前 afterProcess 明细一致。
- `supervised_main` 的 19 名患者均有 baseline eyes_open 和 eyes_closed 记录。
- `supervised_main` baseline EO/EC 记录均 `passes_qc=True`。
- 主分析记录的 `n_channels=62`、`sfreq=250`。
- 每条可读记录都有非空 `duration_sec` 和 `n_valid_windows_4s`。
- 公开输出不包含姓名、原始路径、`.set` 文件名或 `.fdt` 文件名。

## 不在本阶段范围内

- 读取或保存原始 EEG 数组。
- artifact rejection、坏导检测或通道重排。
- PSD matrix、FC matrix、tACS-informed features、EO/EC reactivity。
- LOPO fold registry。
- 任何传统机器学习、MatrixNet、SSL 或模型评估。
