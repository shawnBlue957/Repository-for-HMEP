# Visual Relationship Prediction with Relational Category Modeling (VRP-Cat)

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22231950-blue.svg)](https://doi.org/10.5281/zenodo.22231950)

A reference implementation for visual relationship prediction under the PredCls protocol. The core premise is that **hom-sets in a relational category may be empty ($\varnothing$)**. Mainstream implementations instead constrain predictions to a probability simplex over $K$ nonempty predicate classes, making it structurally impossible to express that two objects have no relationship. This repository introduces $\varnothing$ and replaces within-pair normalization with unnormalized intensities.

## Main Results (PredCls, Logit Scoring, Mean ± Standard Deviation over 3 Seeds)

| Dataset / Evaluation Setting | $R@50$ | $R@100$ | $mR@50$ | $top1$ |
|---|---|---|---|---|
| Full VG-150 | **0.6401 ± 0.0006** | 0.7687 ± 0.0003 | 0.2411 ± 0.0005 | 0.6765 ± 0.0003 |
| GQA Discriminative View ($\beta{=}0.5$) | **0.4793 ± 0.0019** | 0.5851 ± 0.0023 | 0.2080 ± 0.0009 | 0.6068 ± 0.0005 |
| GQA Full Evaluation ($\beta{=}1$) | **0.5977 ± 0.0001** | 0.8070 ± 0.0000 | 0.1162 ± 0.0018 | 0.9525 ± 0.0000 |

Baseline comparisons use the same frozen features and evaluation function. The strongest VG baseline, MLP, achieves $R@50=0.5433$ (+17.8% relative improvement for our method); the previous best result on the GQA discriminative view is 0.4597 (+4.3%).

**Mechanism verification**: Sweeping $\beta$ from 0 to 100 leaves $top1$ bitwise unchanged (0.6751 on VG and 0.9338–0.9341 on GQA). This indicates that $\varnothing$ calibration changes rankings across pairs while preserving the within-pair $\arg\max$.

## Directory Structure

```
vrp/                     Core library
  config.py              HyperParams (structural hyperparameters, loss weights, schedules)
  data.py / gqa_data.py  Dataset assembly
  backbone.py            Chain_3 encoding + dual streams (r_exp / r_imp) + gated fusion + DualHead
  closure.py             Composition table P_conv, transitive closure, mixed student distribution
  enrich.py              Low-rank enrichment metric (Eq. 14)
  losses.py              Constraints: cls / inv / dag / sym / coh / ord / PI, etc.
  model.py               VRPCatModel, PairResidualHead (per-pair bypass + ∅ prediction)
  rerank.py              Stage-2 within-pair reranking (optional module)
  pathdiag.py            Discriminative power of triangular path consistency (AUC)
tools/
  small_sample_experiment.py   Main experiment entry point (training + evaluation + ablations + β sweep)
  sota_baselines.py            MLP / GPS-Net / VCTree / SGTR / Transformer / LoGAN
  roi_features.py              ROI feature pre-extraction (ResNeXt-101 + FPN)
  clip_cpu.py / clip_text_cpu.py   CLIP weight conversion and text tower
  build_inv_pairs_from_data.py Inverse-predicate table derived from annotations
  build_semantic_path.py       Predicate semantic similarity matrix
  prepare_dataset.py / prepare_gqa.py   Dataset assembly
  count_twohop_empty.py        Empirical fraction of empty three-hop relations given valid two-hop relations
  eval_geom_baseline.py        Geometric rule baseline
  _paths.py                    Cross-machine path resolution (environment variables take priority)
tests/                   Unit / contract / end-to-end / invariant tests
```

## Environment

```
Python >= 3.9
PyTorch >= 2.0 (CUDA)
numpy
```

```bash
pip install -r requirements.txt
```

## Dataset Information
- **Primary Dataset**: The Visual Genome dataset (Version 1.4) is used for model training and evaluation.
  - **DOI**: 10.5281/zenodo.4004776
  - **URL**: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html

- **Additional Dataset**: VRD (Visual Relationship Detection) dataset.
  - **Introduction**: The VRD dataset is used for visual relation detection/prediction tasks. It provides triplet annotations of subject, predicate, and object in images and is one of the classic benchmark datasets in this field.
  - **Citation**:Lu, C., Krishna, R., Bernstein, M., & Fei-Fei, L. (2016). *Visual Relationship Detection with Language Priors*. In *ECCV 2016*.
  - **URL**:https://cs.stanford.edu/people/ranjaykrishna/vrd/

- **Other Data Sources**: See respective sections or code comments for details.

### Original Processed Data Downloads

The download links and extraction codes supplied with the original repository are retained:

- [Processed VG annotations](https://pan.baidu.com/s/13TK6c1QN_p9loLy6dGHD-w), extraction code: `2rpj`; see [annotation download instructions](dataset/vg/annotations/readme.md).
- [Processed scene graph annotations](https://pan.baidu.com/s/1ip_AaTaLLd1O29_qS2l10Q), extraction code: `kqsv`; see [scene graph download instructions](dataset/vg_scene_graph_annot/readme.md).
- [Visual Genome images](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html); see [image download instructions](dataset/vg/images/readme.md).

For the directory layout and preprocessing required by the current implementation, follow [Data Preparation](#data-preparation).

### Dataset Specifications (Visual Genome)
- 108,077 images with annotated objects and relationships
- 150 object categories and 50 relationship types used in experiments
- Average of 21 objects and 18 relationships per image

## Data Preparation

1. **Visual Genome**: Download `objects.json` / `relationships.json` / `image_data.json` and the images, and place them under `$VG_DATA_ROOT/vg150/` (which must contain `annotations_train.json` and `annotations_test.json`).
2. **GQA**: Download the scene graph annotations into `$GQA_ROOT/`, then use `tools/prepare_gqa.py` to convert them into the unified format.
3. **CLIP**: Place `ViT-B-32.pt` and `bpe_simple_vocab_16e6.txt.gz` under `$VG_DATA_ROOT/clip/`.
4. **ROI features**: Pre-extract and cache features with `tools/roi_features.py` (approximately 2.0 GB for VG / 2.4 GB for GQA).
5. **Inverse-predicate table**: Use `tools/build_inv_pairs_from_data.py` to generate `inv_pairs_*.json` from training-set statistics.

Required environment variables (when unset, paths fall back to `../data` relative to the repository and placeholder paths; explicit configuration is recommended):

```bash
export VG_DATA_ROOT=/path/to/data        # Contains vg150/, gqa/, and clip/
export GQA_ROOT=/path/to/data/gqa
export CLIP_WEIGHTS=/path/to/clip/ViT-B-32.pt
python tools/_paths.py                    # Print resolved paths and check them first
```

## Reproduction

**Full VG-150 (3 seeds)**

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

**Full GQA (discriminative view, $\beta{=}0.5$)**

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

**SOTA Baseline Comparisons**

```bash
python tools/sota_baselines.py --dataset vg150 --train-n 57723 --test-n 26446 \
  --test-split test --epochs 30 --seeds 0,1,2 --batch-size 32 --eval-mode predcls \
  --use-clip-text --device cuda --nmax 64 --pmax 288 \
  --feat-backbone roi --roi-arch resnext101_vg \
  --models mlp,gpsnet,vctree,sgtr,transformer,login --out results/sota.json
```

## Key Options

| Option | Purpose |
|---|---|
| `--pair-res` | Enable the per-pair MLP bypass (uses only raw endpoint features, without graph-level context) |
| `--pair-rel` | Enable pair-existence supervision (negative sampling of non-GT ordered pairs within the same image at a 1:3 ratio + BCE) |
| `--pair-rel-beta` | Inference-time $\varnothing$ calibration strength: `z[k] += β·log σ(s)` |
| `--pair-rel-beta-scan` | Evaluate multiple $\beta$ values with the same model weights, e.g., `"0,0.5,1,2,4"` |
| `--score-softmax` | Revert to per-pair softmax scoring (legacy evaluation setting, for comparison only) |
| `--rank-scope` | `global` (default, Motifs convention) / `pair` (within-pair ranking) |

## Diagnostic Scripts

```bash
# Empirical fraction of empty three-hop relations given valid two-hop relations (measured κ in the paper)
python tools/count_twohop_empty.py

# Pair discrimination AUC (check whether ∅ prediction learns a signal)
python tools/small_sample_experiment.py ... --diag-path-auc
```

## Citation

The original repository identifies the associated work as *Hierarchical Embedding-Based Visual Relationship Prediction Method with Multimodal Fusion*, by Yunhao Sun, Xiaoao Chen, Heng Chen, and Ruihua Qi. Its original citation labels it as a 2024 preprint.

The repository/archive DOI recorded in the original README and citation metadata is [10.5281/zenodo.22231950](https://doi.org/10.5281/zenodo.22231950). The original `CITATION.cff` also records version `1.0.1` and the URL [10.5281/zenodo.22147896](https://doi.org/10.5281/zenodo.22147896). Both identifiers are retained as originally provided; they should not be interpreted as a newly verified journal-article DOI or a release identifier for the current code.

If you use this repository or the associated datasets in your research, please cite:

```bibtex
@misc{visualgenome,
  title     = {Visual Genome Dataset},
  author    = {Krishna, Ranjay and et al.},
  year      = {2017},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.4004776},
  url       = {https://visualgenome.org/}
}

@inproceedings{lu2016vrd,
  title     = {Visual Relationship Detection with Language Priors},
  author    = {Lu, Cewu and Krishna, Ranjay and Bernstein, Michael and Fei-Fei, Li},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2016}
}

@article{hmep2024,
  title   = {Hierarchical Embedding-Based Visual Relationship Prediction Method with Multimodal Fusion},
  author  = {Sun, Yunhao and Chen, Xiaoao and Chen, Heng and Qi, Ruihua},
  journal = {Preprint},
  year    = {2024}
}
```

## License

MIT; see `LICENSE`.
