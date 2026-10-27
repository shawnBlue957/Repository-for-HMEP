# 关系范畴建模的视觉关系预测（VRP-Cat）

视觉关系预测（PredCls 协议）的参考实现。核心主张：**关系范畴的 hom-集允许为空（$\varnothing$）**，而主流实现把它约束成 $K$ 类非空谓词的概率单纯形，导致「这两个物体之间没有关系」在结构上无法表达。本仓库补入 $\varnothing$ 并把 pair 内归一化替换为非归一化强度。

## 主要结果（PredCls，logits 口径，3 seed 均值 ± 标准差）

| 数据集 / 口径 | $R@50$ | $R@100$ | $mR@50$ | $top1$ |
|---|---|---|---|---|
| VG-150 全量 | **0.6401 ± 0.0006** | 0.7687 ± 0.0003 | 0.2411 ± 0.0005 | 0.6765 ± 0.0003 |
| GQA 判别视图（$\beta{=}0.5$） | **0.4793 ± 0.0019** | 0.5851 ± 0.0023 | 0.2080 ± 0.0009 | 0.6068 ± 0.0005 |
| GQA 全量口径（$\beta{=}1$） | **0.5977 ± 0.0001** | 0.8070 ± 0.0000 | 0.1162 ± 0.0018 | 0.9525 ± 0.0000 |

对照基线（同冻结特征、同评测函数）：VG 上最强基线 MLP 为 $R@50=0.5433$（+17.8%）；GQA 判别视图旧最好为 0.4597（+4.3%）。

**机制验证**：$\beta$ 从 0 扫到 100 时 $top1$ 逐位不变（VG 恒 0.6751、GQA 恒 0.9338–0.9341），说明 $\varnothing$ 校准只改变 pair 之间的排序，不改变 pair 内的 $\arg\max$。

## 目录结构

```
vrp/                     核心库
  config.py              HyperParams（结构超参、损失权重、调度）
  data.py / gqa_data.py  数据集装配
  backbone.py            Chain_3 编码 + 双流（r_exp / r_imp）+ 门控融合 + DualHead
  closure.py             复合表 P_conv、传递闭包、混合学生分布
  enrich.py              低秩富化度量（式 14）
  losses.py              cls / inv / dag / sym / coh / ord / PI 等约束
  model.py               VRPCatModel、PairResidualHead（逐对旁路 + ∅ 判定）
  rerank.py              Stage-2 pair 内重排序（可选模块）
  pathdiag.py            三角路径一致性的判别力诊断（AUC）
tools/
  small_sample_experiment.py   主实验入口（训练 + 评测 + 消融 + β 扫描）
  sota_baselines.py            MLP / GPS-Net / VCTree / SGTR / Transformer / LoGAN
  roi_features.py              ROI 特征预抽取（ResNeXt-101 + FPN）
  clip_cpu.py / clip_text_cpu.py   CLIP 权重转换与文本塔
  build_inv_pairs_from_data.py 由标注报告逆谓词表
  build_semantic_path.py       谓词语义相似度矩阵
  prepare_dataset.py / prepare_gqa.py   数据装配
  count_twohop_empty.py        统计「二跳成立时三跳为空」的经验比例
  eval_geom_baseline.py        几何规则基线
  _paths.py                    跨机器路径解析（环境变量优先）
tests/                   单元 / 契约 / 端到端 / 不变量测试
```

## 环境

```
Python >= 3.9
PyTorch >= 2.0（CUDA）
numpy
```

```bash
pip install -r requirements.txt
```

## 数据准备

1. **Visual Genome**：下载 `objects.json` / `relationships.json` / `image_data.json` 与图像，放入 `$VG_DATA_ROOT/vg150/`（需含 `annotations_train.json`、`annotations_test.json`）。
2. **GQA**：下载 scene graph 标注，放入 `$GQA_ROOT/`，用 `tools/prepare_gqa.py` 转成统一格式。
3. **CLIP**：`ViT-B-32.pt` 与 `bpe_simple_vocab_16e6.txt.gz` 放入 `$VG_DATA_ROOT/clip/`。
4. **ROI 特征**：`tools/roi_features.py` 预抽取并缓存（VG 约 2.0 GB / GQA 约 2.4 GB）。
5. **逆谓词表**：`tools/build_inv_pairs_from_data.py` 由训练集统计生成 `inv_pairs_*.json`。

所需环境变量（未设置时回退到仓库内 `../data` 与占位路径，建议显式设置）：

```bash
export VG_DATA_ROOT=/path/to/data        # 含 vg150/、gqa/、clip/
export GQA_ROOT=/path/to/data/gqa
export CLIP_WEIGHTS=/path/to/clip/ViT-B-32.pt
python tools/_paths.py                    # 打印解析结果，先自查
```

## 复现

**VG-150 全量（3 seed）**

```bash
python tools/small_sample_experiment.py \
  --dataset vg150 --train-n 57723 --test-n 26446 --test-split test \
  --epochs 30 --seeds 0,1,2 --batch-size 32 --eval-mode predcls \
  --use-clip-text --device cuda --nmax 64 --pmax 288 \
  --feat-backbone roi --roi-arch resnext101_vg \
  --arms A --calibrate-loss --use-adjoint --inv-source data --rev-mode sym_check \
  --pair-res --pair-rel --pair-rel-lambda 1.0 --pair-rel-neg-ratio 3 \
  --pair-rel-beta 1.0 \
  --out results/vg.json
```

**GQA 全量（判别视图，$\beta{=}0.5$）**

```bash
python tools/small_sample_experiment.py \
  --dataset gqa --train-n 72581 --test-n 10297 --test-split val \
  --epochs 30 --seeds 0,1,2 --batch-size 32 --eval-mode predcls \
  --use-clip-text --device cuda --nmax 64 --pmax 288 \
  --feat-backbone roi --roi-arch resnext101_vg --allow-trunc \
  --exclude-preds "to the left of,to the right of" \
  --arms A --calibrate-loss --use-adjoint --inv-source data --rev-mode sym_check \
  --pair-res --pair-rel --pair-rel-lambda 1.0 --pair-rel-neg-ratio 3 \
  --pair-rel-beta 0.5 \
  --out results/gqa.json
```

**对照 SOTA 基线**

```bash
python tools/sota_baselines.py --dataset vg150 --train-n 57723 --test-n 26446 \
  --test-split test --epochs 30 --seeds 0,1,2 --batch-size 32 --eval-mode predcls \
  --use-clip-text --device cuda --nmax 64 --pmax 288 \
  --feat-backbone roi --roi-arch resnext101_vg \
  --models mlp,gpsnet,vctree,sgtr,transformer,login --out results/sota.json
```

## 关键开关

| 开关 | 作用 |
|---|---|
| `--pair-res` | 启用 MLP 逐对旁路（只吃端点原始特征，不含图级上下文） |
| `--pair-rel` | 启用 pair 存在性监督（同图非 GT 有序对 1:3 负采样 + BCE） |
| `--pair-rel-beta` | 推理期 $\varnothing$ 校准强度，`z[k] += β·log σ(s)` |
| `--pair-rel-beta-scan` | 同一权重上多 $\beta$ 评测，如 `"0,0.5,1,2,4"` |
| `--score-softmax` | 回退到 per-pair softmax 打分（历史口径，仅用于对照） |
| `--rank-scope` | `global`（默认，Motifs 口径）/ `pair`（pair 内排序） |

## 诊断脚本

```bash
# 二跳成立时三跳为空的经验比例（对应论文中 κ 的实测）
python tools/count_twohop_empty.py

# pair 判别力 AUC（判断 ∅ 判定是否学到信号）
python tools/small_sample_experiment.py ... --diag-path-auc
```

## 引用

若本仓库对研究有帮助，请引用对应论文（标题与作者信息待补充）。

## 许可

MIT，见 `LICENSE`。
