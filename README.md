# Hierarchical embedding-based approach for accurate visual relationship prediction

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22231950-blue.svg)](https://doi.org/10.5281/zenodo.22231950)

Implementation and experiment tools for **HMEP (Hierarchical Multimodal Embedding for Prediction)**, corresponding to the manuscript *Hierarchical embedding-based approach for accurate visual relationship prediction* by **Yunhao Sun, Xiaoao Chen, Heng Chen, and Ruihua Qi**.

The framework combines multimodal feature fusion, dynamically weighted relationship prediction, and consistency and reversibility regularization. A **pair-existence module (PEM)** learns which ordered object pairs carry an annotated relation and calibrates their scores for image-level ranking. The main improvement is in ranking pairs; it does not establish uniformly better predicate classification within a known pair.

The source package retains the implementation names `vrp-cat`, `VRPCatModel`, and `PairResidualHead`. These names identify the current code layout. The paper-facing model name used below is **HMEP + PEM**.

## Main Results

Results below follow Tables 3-5 of the supplied manuscript. All values are **percentages**; ± denotes the standard deviation over three runs for the full model.

| Dataset / Evaluation Setting | R@20 | R@50 | R@100 | mR@50 | top1 | BRC@1 |
|---|---|---|---|---|---|---|
| VG-150, HMEP + PEM | 43.77 ± 0.05 | **64.01 ± 0.06** | 76.87 ± 0.03 | 24.11 ± 0.05 | 67.65 ± 0.03 | 42.47 ± 0.23 |
| GQA discriminative view, HMEP + PEM ($\gamma=0.5$) | — | **47.93 ± 0.19** | 58.51 ± 0.23 | 20.80 ± 0.09 | 60.68 ± 0.05 | 31.16 ± 0.29 |
| GQA full view, HMEP + PEM ($\gamma=1$) | — | **59.77 ± 0.01** | 80.70 ± 0.00 | 11.62 ± 0.18 | 95.25 ± 0.00 | — |

- **VG-150:** The strongest re-implemented baseline, MLP, achieves R@50 = 54.33%, versus 64.01% for HMEP + PEM (+17.8% relative). Its top1 (68.77%) and BRC@1 (43.00%) remain slightly higher than those of HMEP + PEM. The relation modules contain 1.34M and 8.39M parameters, respectively; PEM accounts for 1.38M of the latter.
- **GQA discriminative view:** HMEP + PEM reaches R@50 = 47.93%, versus 42.87% for MLP. The earlier softmax-scored OURS-F result is 45.97%; its mR@50 of 15.93% increases to 20.80% (+30.6% relative). These are different reference comparisons.
- **GQA full view:** R@50 is 59.77%, versus 48.67% for VCTree (+22.8% relative). The manuscript treats this view as supplementary because its two spatial predicates can be recovered from box coordinates.

### Evaluation Protocol

The task is **predicate classification (PredCls)** with ground-truth object boxes and class labels. Candidate ordered object pairs and predicate labels are ranked **globally within each image**. Results under this protocol should not be compared directly with per-pair or per-subject ranking results.

R@K is reported at K = 20, 50, and 100. mR@50 averages recall across predicate classes; top1 measures predicate accuracy within annotated pairs; BRC@1 requires correct predictions in both directions for pairs annotated in both directions. Baselines use the same cached, frozen region features. They are re-implementations under this shared protocol, rather than their original published benchmark scores.

### Pair Existence and Calibration

PEM uses binary cross-entropy supervision with three sampled non-ground-truth ordered pairs per positive pair from the same image. The manuscript writes inference calibration as:

$$
\widetilde z_k(i,j)=z_k(i,j)+\gamma s(i,j).
$$

The same scalar offset is applied to every predicate of a pair, preserving within-pair ordering while changing the ranking across pairs. The manuscript reports that 93.46% of two-hop-connected pairs in VG-150 and 62.11% in GQA have no annotated relation. These are annotation statistics, not proof that the corresponding objects have no real-world relationship.

**Implementation correspondence:** The current code exposes calibration through `--pair-rel-beta` and computes `z[k] += beta * log_sigmoid(rel_logit)` in `vrp/model.py`. The manuscript uses $\gamma$ for calibration strength and $\alpha,\beta$ for dynamic weights. The code's CLI name is retained, and its log-sigmoid transform should be distinguished from the raw-score expression in the manuscript.

## Directory Structure

```
vrp/                     Core library
  config.py              HyperParams (structural hyperparameters, loss weights, schedules)
  data.py / gqa_data.py  Dataset assembly
  backbone.py            Chain_3 encoding + dual streams (r_exp / r_imp) + gated fusion + DualHead
  closure.py             Composition table P_conv, transitive closure, mixed student distribution
  enrich.py              Low-rank enrichment metric (implementation module)
  losses.py              Constraints: cls / inv / dag / sym / coh / ord / PI, etc.
  model.py               VRPCatModel, PairResidualHead (per-pair bypass + ∅ prediction)
  rerank.py              Stage-2 within-pair reranking (optional module)
  pathdiag.py            Discriminative power of triangular path consistency (AUC)
tools/
  small_sample_experiment.py   Main experiment entry point (training + evaluation + ablations + β sweep)
  sota_baselines.py            MLP / GPS-Net / VCTree / SGTR / Transformer / LOGIN
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

- **Second evaluation dataset: GQA** (Hudson and Manning, 2019), using 72,581 training and 10,297 evaluation items. Its scene graphs derive from Visual Genome scenes, so the two benchmarks are not independent image domains. The primary discriminative view removes `to the left of` and `to the right of` from both ground truth and candidates, leaving 56,027 ground-truth triplets. The full view retains both predicates.
  - **Annotation preparation**: See `tools/prepare_gqa.py` and [Data Preparation](#data-preparation).
  - **Citation**: Hudson, D. A., and Manning, C. D. (2019). *GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering*.

- **Legacy dataset resource: VRD** (Visual Relationship Detection). The original repository's links and citation are retained below; the supplied manuscript reports its main experiments on VG-150 and GQA.
  - **Introduction**: The VRD dataset is used for visual relation detection/prediction tasks. It provides triplet annotations of subject, predicate, and object in images and is one of the classic benchmark datasets in this field.
  - **Citation**: Lu, C., Krishna, R., Bernstein, M., & Fei-Fei, L. (2016). *Visual Relationship Detection with Language Priors*. In *ECCV 2016*.
  - **URL**: https://cs.stanford.edu/people/ranjaykrishna/vrd/

- **Other Data Sources**: See respective sections or code comments for details.

### Original Processed Data Downloads

The download links and extraction codes supplied with the original repository are retained:

- [Processed VG annotations](https://pan.baidu.com/s/13TK6c1QN_p9loLy6dGHD-w), extraction code: `2rpj`; see [annotation download instructions](dataset/vg/annotations/readme.md).
- [Processed scene graph annotations](https://pan.baidu.com/s/1ip_AaTaLLd1O29_qS2l10Q), extraction code: `kqsv`; see [scene graph download instructions](dataset/vg_scene_graph_annot/readme.md).
- [Visual Genome images](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html); see [image download instructions](dataset/vg/images/readme.md).

For the directory layout and preprocessing required by the current implementation, follow [Data Preparation](#data-preparation).

### Dataset Specifications (Visual Genome)
- 108,077 images with annotated objects and relationships
- VG-150 evaluation uses 150 object categories and 50 predicates
- 57,723 training images and 26,446 test images
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

The commands below retain the existing implementation examples, including `--batch-size 32`. The manuscript reports 30 epochs on NVIDIA RTX 4080 GPUs, Adam with an initial learning rate of $10^{-4}$ reduced to $10^{-6}$ by a cosine schedule, and a batch size of **128**. Set `--batch-size 128 --lr 1e-4` when matching those reported settings, and verify the scheduler and remaining configuration in the training code. The examples are not a claim that every manuscript setting has been reconciled with this code revision.

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
# Measure unannotated relations among pairs connected by two-hop paths
python tools/count_twohop_empty.py

# Pair discrimination AUC (check whether ∅ prediction learns a signal)
python tools/small_sample_experiment.py ... --diag-path-auc
```

## Citation

The supplied manuscript is **Hierarchical embedding-based approach for accurate visual relationship prediction**, by **Yunhao Sun, Xiaoao Chen, Heng Chen, and Ruihua Qi**. The authors are affiliated with the School of Software and the Research Center for Language Intelligence at Dalian University of Foreign Languages. Heng Chen and Ruihua Qi are the corresponding authors.

The supplied PDF does not specify an article DOI, journal, or publication year. The manuscript citation below therefore does not assign these fields. The original repository used the earlier title *Hierarchical Embedding-Based Visual Relationship Prediction Method with Multimodal Fusion* and labeled it as a 2024 preprint; that historical metadata is distinct from the title of the supplied manuscript.

The original repository/archive DOI is [10.5281/zenodo.22231950](https://doi.org/10.5281/zenodo.22231950). The retained `CITATION.cff` also records version `1.0.1` and the URL [10.5281/zenodo.22147896](https://doi.org/10.5281/zenodo.22147896). These original archive identifiers are preserved separately from the manuscript citation and the Visual Genome dataset DOI [10.5281/zenodo.4004776](https://doi.org/10.5281/zenodo.4004776).

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

@unpublished{hmep,
  title  = {Hierarchical embedding-based approach for accurate visual relationship prediction},
  author = {Sun, Yunhao and Chen, Xiaoao and Chen, Heng and Qi, Ruihua},
  note   = {Manuscript supplied by the authors}
}

@inproceedings{hudson2019gqa,
  title     = {GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering},
  author    = {Hudson, Drew A. and Manning, Christopher D.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2019}
}
```

## License

MIT; see `LICENSE`.
