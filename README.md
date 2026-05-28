# Hierarchical embedding-based visual relationship prediction method with multimodal fusion（HMEP Repository）

## Description
This repository contains the official implementation of the Hierarchical Multimodal Embedding for Prediction (HMEP) framework, a novel approach for visual relationship prediction through multimodal fusion.

## Dataset Information
- **Primary Dataset**: The Visual Genome dataset (Version 1.4) is used for model training and evaluation.
  - **DOI**: 10.5281/zenodo.4004776
  - **URL**: https://visualgenome.org/
  - **Download Link**: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html

- **Additional Dataset**: VRD (Visual Relationship Detection) dataset.
  - **简介**：VRD 数据集用于视觉关系检测/预测任务，提供图像中主体（subject）、谓词（predicate）与客体（object）三元组标注，是视觉关系理解领域的经典基准之一。
  - **引用**：Lu, C., Krishna, R., Bernstein, M., & Fei-Fei, L. (2016). *Visual Relationship Detection with Language Priors*. In *ECCV 2016*.
  - **下载链接**：https://cs.stanford.edu/people/ranjaykrishna/vrd/

- **Other Data Sources**: See respective sections or code comments for details.

### Dataset Specifications (Visual Genome)
- 108,077 images with annotated objects and relationships
- 150 object categories and 50 relationship types used in experiments
- Average of 21 objects and 18 relationships per image

## Code Information
- Language: Python
- Framework: PyTorch
- Main Functionalities:
  - Data preprocessing and feature extraction
  - Multimodal feature fusion using multi-head self-attention
  - Dynamic relationship prediction with adaptive weighting
  - Consistency and reversibility verification
  - Model training and evaluation

## Usage Instructions
1. **Clone this repository**:
    ```bash
    git clone https://github.com/shawnBlue957/Repository-for-HMEP.git
    cd Repository-for-HMEP
    ```
2. **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```
3. **Prepare the dataset**:
    - Download the Visual Genome dataset from the official website. We provide links to download processed data in the corresponding directories.
    - Place the downloaded files in the `dataset/` directory.
    - Run preprocessing scripts to extract and format data:
      ```bash
      python preprocess.py
      ```

4. **Train the model**:
    ```bash
    python train.py
    ```

5. **Evaluate the model**:
    ```bash
    python evaluate.py
    ```

## Requirements
- Python >= 3.7
- Required libraries (see `requirements.txt`):
  - torch >= 1.9.0
  - torchvision >= 0.10.0
  - transformers >= 4.20.0
  - numpy >= 1.21.0
  - pandas >= 1.3.0
  - scikit-learn >= 0.24.0
  - tqdm >= 4.62.0
  - Pillow >= 8.3.0
  - opencv-python >= 4.5.0

## Methodology
- **Data Preprocessing**:
  - Object detection using pre-trained Faster R-CNN with frozen parameters
  - Bounding box extraction and union box calculation (u = s ∪ o)
  - Feature extraction using CLIP image and text encoders (ViT-B/32)
  - Spatial coordinate normalization and projection

- **Feature Extraction**:
  - Visual features from image regions
  - Text features from category labels
  - Spatial features from bounding box coordinates

- **Multimodal Fusion**:
  - Multi-head self-attention mechanism for deep feature fusion
  - Dynamic weighting of modality contributions

- **Modeling**:
  - Hierarchical embedding architecture
  - Dynamic relationship prediction operator
  - Consistency and reversibility regularization

- **Consistency Verification**:
  - Reversibility constraint: f_predicate_i + f_predicate_j ≈ (f_union_i + f_union_j) - (α_i·f_sub_i + α_j·f_sub_j + β_i·f_obj_i + β_j·f_obj_j)
  - Additional regularization loss L_inv for structural consistency

- **Evaluation**:
  - Recall@K metrics for predicate classification
  - Ablation studies for component analysis

## Citations
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
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contribution Guidelines
Contributions are welcome! Please submit pull requests or open issues for suggestions or bug reports. Follow standard Python coding conventions and document your code clearly.

## Materials and Methods
- **Materials**: Visual Genome dataset (DOI: 10.5281/zenodo.4004776), VRD dataset, Python scripts, CLIP model weights, Faster R-CNN detector
- **Methods**: Detailed in the [Methodology](#methodology) section above

---

For more information, refer to the code comments and individual module documentation.
