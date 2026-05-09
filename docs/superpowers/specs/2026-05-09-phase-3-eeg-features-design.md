# Phase 3 PSD / FC / tACS-informed 特征设计

## 目标

完成 PRD 路线图中的 Phase 3：只使用 baseline eyes-open / eyes-closed 静息态 EEG，为 `supervised_main` 患者生成患者级 PSD matrix、ROI-level functional connectivity matrix、手工 EEG 特征、tACS-informed target features 和 feature dictionary。

本阶段不生成 LOPO folds，不做 fold 内标准化，不训练 ML baseline、MatrixNet 或 SSL。Phase 4 以后再处理数据划分和防泄漏归一化。

## 输入

本阶段依赖 Phase 0+1 和 Phase 2 的已完成输出：

- `outputs/cohort/cohort_master.csv`
- `outputs/qc/eeg_record_index.csv`
- `outputs/qc/eeg_qc_summary.csv`
- `configs/project.yaml`
- `configs/eeg.yaml`
- 新增 `configs/features.yaml`

原始 `.set` / `.fdt` 只作为私有输入读取，不复制、不改写、不输出文件名或路径。

## 隐私边界

公开输出和 Git 提交中禁止出现：

- 姓名或类似姓名字段；
- 原始 EEG 路径；
- `.set` / `.fdt` 文件名；
- 本机绝对路径；
- Excel、原始 EEG 文件、`outputs/` 内容。

矩阵 `.npy` 和 CSV 输出只能包含去标识化 `subject_id`、匿名 feature 名称和数值。脚本可以在内存中通过 Phase 2 的 private record 读取真实 EEG，但不得把 private path 写入公开输出。

## 配置

新增 `configs/features.yaml`，包含：

- `project_config`: 指向 `configs/project.yaml`
- `eeg_config`: 指向 `configs/eeg.yaml`
- `psd`: Welch 参数、频率范围 0.5-45 Hz、0.5 Hz resolution、log transform
- `bands`: delta、theta、alpha_mu、low_beta、high_beta、low_gamma_optional
- `connectivity`: ROI 列表、coherence / wpli、ROI-level matrix
- `lesion_normalization`: 是否输出 `native` 与 `lesion_normalized` 两个 view、目标统一到 C3
- `channel_pair_map`: 左右同源通道交换表

## 输出

运行 Phase 3 后输出到被 Git 忽略的 `outputs/features/`：

- `matrices/psd_eo.npy`
- `matrices/psd_ec.npy`
- `matrices/fc_roi_eo.npy`
- `matrices/fc_roi_ec.npy`
- `handcrafted_features.csv`
- `tacs_target_features.csv`
- `feature_dictionary.csv`

矩阵均为患者级平均结果：

- PSD shape: `[n_subjects, n_views, n_channels, n_freq_bins]`
- FC ROI shape: `[n_subjects, n_views, n_roi_edges, n_bands, n_methods]`

`n_views=2`，顺序固定为 `native`, `lesion_normalized`。`feature_dictionary.csv` 必须记录 `feature_name`、`feature_group`、`condition`、`band`、`channel`、`roi`、`metric`、`hemisphere_space`、`matrix_file` 和索引字段。

## Lesion-normalized EEG feature view

本阶段保留两个空间：

- `native`: 原始标准 EEG 通道顺序，不翻转。
- `lesion_normalized`: 将左患手 / 左治疗手患者的左右同源通道交换，使治疗靶点对齐到 C3 空间。

翻转规则：

- 右患手或右治疗手：native target 为 C3，不翻转，lesion_normalized 与 native 相同。
- 左患手或左治疗手：native target 为 C4，使用同源通道交换表翻转，lesion_normalized target 对齐为 C3。
- 中线通道如 `Fz`、`FCz`、`Cz`、`CPz`、`Pz`、`POz`、`Oz` 不动。

最小同源交换表包括：

- `Fp1 <-> Fp2`
- `AF3 <-> AF4`
- `F7 <-> F8`
- `F5 <-> F6`
- `F3 <-> F4`
- `F1 <-> F2`
- `FT7 <-> FT8`
- `FC5 <-> FC6`
- `FC3 <-> FC4`
- `FC1 <-> FC2`
- `T7 <-> T8`
- `C5 <-> C6`
- `C3 <-> C4`
- `C1 <-> C2`
- `TP7 <-> TP8`
- `CP5 <-> CP6`
- `CP3 <-> CP4`
- `CP1 <-> CP2`
- `P7 <-> P8`
- `P5 <-> P6`
- `P3 <-> P4`
- `P1 <-> P2`
- `PO7 <-> PO8`
- `PO5 <-> PO6`
- `PO3 <-> PO4`
- `O1 <-> O2`

PSD 翻转通过 channel axis 重排实现。FC 翻转通过 edge 两端的 channel/ROI 标签映射后重排实现。tACS target features 同时输出 native target/homologous 和 lesion-normalized target/homologous 版本。

## PSD 设计

PSD 使用 `scipy.signal.welch`。每条 baseline EO/EC EEG：

1. 按 Phase 2 的窗口长度和 overlap 切 4 秒窗口；
2. 每个窗口计算 0.5-45 Hz PSD；
3. 插值或选择到 0.5 Hz frequency grid；
4. 对 window 求平均得到 patient-condition PSD；
5. 取 log10 PSD。

如果真实 `.set` 头信息可读但原始 `.fdt` 读取失败，脚本必须明确失败并说明 record_id，不写出半成品。

## FC ROI 设计

FC 只实现 ROI-level 主分析版本，不在本阶段生成 full-edge 补充矩阵。ROI 对为配置中 ROI 名称的无向组合，包含 ROI 自身内部连接。每个 ROI pair、band、method 输出一个患者级值：

- coherence: 基于 `scipy.signal.coherence`，在 band 内平均。
- wPLI: 基于窗口 FFT 的 imaginary cross-spectrum，按 band 聚合。

ROI 中缺失的通道写入 feature dictionary warning 字段；如果某个 ROI pair 因通道不足无法计算，对应值为 `NaN`，不得静默删除矩阵位置。

## tACS-informed features

根据 `treated_hand` 优先、`affected_hand` 兜底映射 target：

- `right` -> native target `C3`, homologous `C4`
- `left` -> native target `C4`, homologous `C3`

输出特征包括：

- target / homologous channel band power
- target / homologous ROI mean band power
- target-minus-homologous、target-div-homologous、log-target-minus-log-homologous
- target-homologous coherence / wPLI
- target-to-midline / frontal / parietal connectivity
- EC-minus-EO 和 EC-div-EO target power

所有 tACS 特征至少有 `native` 和 `lesion_normalized` 两套字段。

## 手工特征

`handcrafted_features.csv` 面向后续 ML baseline，但本阶段不训练模型。字段包括：

- subject metadata: `subject_id`, `label_primary`, `treated_hand`, `affected_hand`
- band power summary
- EO/EC reactivity
- ROI connectivity summary
- tACS feature 合并或引用字段

## 错误处理

脚本应在以下情况下失败：

- 缺少 Phase 0+1 或 Phase 2 输出；
- supervised_main baseline EO/EC 缺失或未通过 QC；
- 通道顺序 hash 不唯一且未提供重排策略；
- EEG 原始数据无法读取；
- 输出 schema 包含 PII 或 path-like 值。

以下情况不使脚本失败，但必须在 dictionary 或 warning 字段中记录：

- 某个非核心 ROI 缺少配置通道；
- 某个 ROI pair 因有效通道不足得到 NaN。

## 测试策略

本阶段生产代码按 TDD 实现。最低测试覆盖：

- channel pair map 对 `C3/C4`、`FC3/FC4`、`CP3/CP4` 翻转正确；
- 中线通道保持不动；
- PSD matrix 翻转 channel axis 正确；
- FC edge mapping 对端点索引翻转正确；
- tACS target mapping 对 left/right 手正确；
- Welch PSD 输出频率 grid 和 shape 正确；
- ROI FC 输出 shape、band、method 和 ROI edge 顺序稳定；
- feature dictionary 包含 view / feature_group / condition / band / channel / ROI / metric；
- public outputs 不含 PII、路径、`.set` 或 `.fdt`。

## 验收命令

```bash
python scripts/04_extract_psd_matrices.py --config configs/features.yaml
python scripts/05_extract_fc_matrices.py --config configs/features.yaml
python scripts/06_build_handcrafted_features.py --config configs/features.yaml
pytest tests -q
```

## 真实数据验收标准

- `supervised_main` 数量为 19；
- `psd_eo.npy` 和 `psd_ec.npy` 第一维为 19；
- `fc_roi_eo.npy` 和 `fc_roi_ec.npy` 第一维为 19；
- 所有矩阵包含 `native` 与 `lesion_normalized` 两个 view；
- channel_order_hash 数量为 1，或通道重排有明确记录；
- `handcrafted_features.csv` 和 `tacs_target_features.csv` 行数为 19；
- `feature_dictionary.csv` 覆盖所有矩阵与 CSV 特征；
- 公开输出不包含姓名、原始路径、`.set` 或 `.fdt` 文件名。

## 不在本阶段范围内

- LOPO fold registry；
- fold 内 z-score 标准化；
- ML baseline；
- MatrixNet；
- SSL；
- 模型评估、bootstrap、permutation 或解释性分析。
