# Attention / 使用前必读

本文件用于集中说明：**数据准备、配置文件调整、运行步骤、输出文件位置**以及**交流与反馈方式**。  
（建议你把它放在项目根目录：`attention.md`）

---

## 数据准备（本地）

### 1）数据文件位置（默认）
请将你清洗后的数据放到（路径可在 config 中修改）：
- `data/processed/train.csv`
- `data/processed/test.csv`

> 注意：出于数据授权与隐私合规要求，**不要把 CEPS 原始数据或清洗后 CSV 上传到 GitHub**。

### 2）数据格式建议
- 每一行代表一个样本（学生/个案/服务对象等）
- 至少包含：
  - **特征列**（可为数值/类别/布尔）
  - **标签列**（0/1 二分类），标签列名由 `target_col` 指定
- 训练集与测试集应尽量保持：
  - 特征列一致（列名一致）
  - 数据类型尽量稳定（同一列不要混合数字与文本）

---

## 配置文件说明（config.yaml 为示例）

本仓库的 `configs/config.yaml` 仅提供**示例配置**，你必须根据自己的数据结构进行调整，重点包括：

- `train_path` / `test_path`：训练/测试数据路径
- `target_col`：标签列名（如 `dropout` / `risk_label`）
- `id_col`：个体ID列（如有则填；没有可设为 `null`）
- `categorical_cols` / `numerical_cols`：
  - 可留空，让程序自动推断
  - 但建议你在正式复现/论文中**明确列出**（增强可复现性与一致性）
- `model.params`：LightGBM 参数（如 `learning_rate`, `num_leaves` 等）
- `early_stopping.stopping_rounds` 与 `early_stopping.log_period`：早停与日志输出频率

> 总结：**代码流程是通用的，但 config 必须与数据一一对应**，否则会出现“找不到列名/维度不一致”等报错。

---



