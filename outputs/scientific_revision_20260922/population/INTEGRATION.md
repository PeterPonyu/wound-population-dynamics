# 群体协议修复整合说明

这是既有三供者队列上的新受控计算实验。没有新增生物重复，未修改旧报告或论文。

历史六方法 benchmark 的目标边际基线：严格同目标抽样/同预测粒子数/供者等质量，50 draws 宏平均 ED 0.181777，范围 0.147906–0.214953；历史共享流 0.394701。这是对原方法集合的正式补充；shared 原数值沿用历史工件，不能与新模型点值混称同一次拟合。

统一评分空间的新 Shared CFM 曲线：k=1 ED 0.546634，k=2 ED 0.398307，平均下降 0.148327；三个 seed 的下降范围 0.143029–0.155782；3/3 个供者块下降。三块描述重采样范围 [0.073221, 0.272683]。

训练 scaler 只看各自 k 个供者。评分 reference scaler 看该 fold 的两位可用训练供者全部时点，绝不进入 k=1 模型拟合。模型先逆变换回 μ，再进入共同参考空间；这种评分定义是回顾性审计，不把第二位供者的信息伪称为 k=1 训练信息。

每个方法统一尺度结果：

- Shared CFM: k1=0.546634, k2=0.398307, reduction=0.148327; relative improvement=65.35%/74.77%.
- Mean displacement: k1=0.509492, k2=0.418588, reduction=0.090903; relative improvement=64.80%/71.48%.
- Training-target marginal (exact): k1=0.222199, k2=0.167598, reduction=0.054601; relative improvement=83.72%/87.39%.
- Training-target marginal (matched): k1=0.233165, k2=0.181777, reduction=0.051388; relative improvement=82.87%/86.37%.
- Unchanged source: k1=1.475999, k2=1.475999, reduction=0.000000; relative improvement=0.00%/0.00%.

应更新的主张与图表：

- 摘要、Results 共享流最低、Discussion source 增量解释、图5及其逐供者表：加入合法目标边际基线；保留供者异质性，不声称新的总体排名。
- 图6、Results 第二供者、种子表、donor-block 表：使用 curve_summary.csv、curve_by_seed.csv 和 curve_donor_blocks.csv 的共同尺度结果；旧0.165不再作统一误差降低。
- Methods：明确训练/评分 scaler 的信息隔离、每配置 NumPy/Torch 重置、原 8000步/256批次/50 RK4不变，以及完整预测工件。
- exact marginal 是训练供者目标经验分布的精确等质量混合，粒子数多于 source；matched marginal 是同source粒子数、50 draws误差的平均。两者必须分开标记。
- 所有重采样范围均描述已有三供者；seed/subset/draw不是新增生物重复。此修复不解决时钟实验、conditioned/OT多seed或临床泛化。

主要文件：

- report.json：全部协议、结果、来源哈希及验证。
- historical_benchmark/report.json、donor_summary.csv、draw_scores.csv、macro_draw_scores.csv、draw_indices.json：正式历史基线补充。
- curve_scores.csv：27配置×5方法逐行统一尺度误差；curve_summary.csv、curve_by_seed.csv、curve_donor_blocks.csv：整合论文的主要表。
- curve_marginal_draw_scores.csv：所有9训练配置的50抽样；paired_donor_seed_scores.csv：每donor/seed配对。
- configuration_stability.csv：逐方法/held donor/训练子集的统一尺度seed均值、SD、范围与split-half参考；独立seed数仍为3。
- historical_curve_comparison.csv：只供历史对照，不混合进新结果。
- cell_identity.csv：原始barcode、GSM、矩阵行与latent行；fits/**：训练/参考scaler、检查点、training-z/μ/reference-z预测、源/目标身份。
- verification.json：全部模型检查点重新载入积分、评分与信息排除验证。

复现：`OPENBLAS_NUM_THREADS=1 python scripts/repair_population_protocol.py --output-dir outputs/scientific_revision_20260922/population`。已有完整运行用 `--verify-only` 验证；同一协议中断可用 `--resume`，不得无意覆盖原实验。
