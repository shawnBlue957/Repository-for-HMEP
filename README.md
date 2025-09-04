# Hierarchical embedding-based visual relationship prediction method with multimodal fusion（HMEP Repository）

## Description
This repository contains code and resources for the Hierarchical Multimodal Embedding for Prediction (HMEP) project. It is designed for visual relationship prediction using multimodal fusion and hierarchical embedding techniques.

## Dataset Information
- **Primary Dataset**: The Visual Genome dataset (Version 1.4) is used for model training and evaluation.
  - **DOI**: 10.5281/zenodo.4004776
  - **URL**: https://visualgenome.org/
  - **Download Link**: https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- **Other Data Sources**: See respective sections or code comments for details.

## Code Information
- All scripts are written in Python.
- Main functionalities include data preprocessing, feature extraction, multimodal fusion, and model training.
- The code implements the HMEP framework with dynamic weighting and consistency regularization.

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
    - Download the Visual Genome dataset from the official website.
    - Place the downloaded files in the `data/` directory.
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
  - numpy
  - pandas
  - scikit-learn
  - torch
  - torchvision
  - transformers
  - tqdm
  - Pillow

## Methodology
- **Data Preprocessing**: 
  - Object detection using Faster R-CNN
  - Bounding box extraction and union box calculation
  - Feature extraction using CLIP image and text encoders
  - Normalization and formatting of spatial coordinates
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
- **Evaluation**: 
  - Recall@K metrics for predicate classification
  - Ablation studies for component analysis

## Citations
If you use this repository or the associated datasets in your research, please cite:
@misc{visualgenome,
title = {Visual Genome Dataset},
author = {Krishna, Ranjay and et al.},
year = {2017},
publisher = {Zenodo},
doi = {10.5281/zenodo.4004776},
url = {https://visualgenome.org/}
}

@article{hmep2024,
title = {Hierarchical Embedding-Based Visual Relationship Prediction Method with Multimodal Fusion},
author = {Sun, Yunhao and Chen, Xiaoao and Chen, Heng and Qi, Ruihua},
journal = {Preprint},
year = {2024}
}

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contribution Guidelines
Contributions are welcome! Please submit pull requests or open issues for suggestions or bug reports. Follow standard Python coding conventions and document your code clearly.

## Materials and Methods
- **Materials**: Visual Genome dataset (DOI: 10.5281/zenodo.4004776), Python scripts, CLIP model weights, Faster R-CNN detector
- **Methods**: Detailed in the [Methodology](#methodology) section above

---

For more information, refer to the code comments and individual module documentation.
