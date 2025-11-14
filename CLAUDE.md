# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT GUIDELINES

### Communication Language
**ALL communication must be in SPANISH.** This is a Spanish-language academic project from Universidad de San Andrés.

### Academic Integrity Policy
**Claude is STRICTLY PROHIBITED from making ANY code modifications.** This is an ACADEMIC project and the student must write all code themselves. Claude's role is limited to:
- Explaining concepts and providing guidance
- Reading and interpreting the assignment requirements
- Answering questions about the codebase structure
- Providing technical advice and suggestions

**Claude MUST NEVER:**
- Write implementation code
- Modify existing files
- Create new code files
- Complete assignments or exercises

### Assignment Reference
The file `consigna.pdf` contains the complete assignment specifications. When the user requests "dame la consigna del punto X" (give me the instructions for section X), Claude must:
1. Read the `consigna.pdf` file
2. Extract the specific section requested
3. Present it in markdown format for easy copy-paste

## Project Overview

This is an academic machine learning project for Universidad de San Andrés (UdeSA), course I302 - Machine Learning and Deep Learning, focusing on **unsupervised learning methods**. The assignment (TP4) involves implementing dimensionality reduction and clustering algorithms from scratch on a facial image dataset.

**Key Constraints:**
- NO scikit-learn allowed for core algorithms (PCA, k-Means, GMM) - must implement from scratch
- PyTorch is ONLY allowed for autoencoder implementation
- Due date: November 17, 2025

## Dataset Information

- **File**: `data/caras.csv`
- **Content**: 400 grayscale face images (68×68 pixels) + header row = 401 total rows
- **Dimensions**: 4,096 pixel features (pixel_0 through pixel_4095) + class labels
- **Structure**: Each row represents one flattened 68×68 image with corresponding class

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install required packages (need to create requirements.txt)
pip install numpy pandas matplotlib seaborn torch torchvision jupyter
```

### Run Analysis
```bash
# Start Jupyter notebook for main work
jupyter notebook Zabaleta_Gonzalo_Notebook_TP4.ipynb

# Run modularized code (when created)
python src/preprocessing.py
python src/dimensionality_reduction.py
```

## Architecture and Implementation Structure

The project follows a 3-phase pipeline that must be implemented in the Jupyter notebook:

### Phase 1: Data Inspection (Section 1)
- **Location**: Early cells in main notebook
- **Components**:
  - Image visualization function (reusable for 15 random images)
  - Exploratory data analysis with class distribution
  - Stratified train/test split (80/20)

### Phase 2: Dimensionality Reduction (Section 2)
- **PCA Implementation** (`src/dimensionality_reduction.py` recommended):
  - Data standardization function
  - Custom PCA implementation (NO sklearn)
  - Find components for 90% variance explanation
  - Image reconstruction comparison

- **Autoencoder Implementation** (`src/autoencoder.py` recommended):
  - PyTorch-based encoder/decoder networks
  - Matching latent dimension to PCA components
  - Train/validation split for hyperparameter tuning
  - Reconstruction quality comparison with PCA

### Phase 3: Clustering (Section 3)
- **k-Means Implementation** (`src/clustering.py` recommended):
  - Custom k-Means algorithm (NO sklearn)
  - Test K values in range [5, 20]

- **GMM Implementation** (same file):
  - Custom Gaussian Mixture Model (NO sklearn)
  - Test K values in range [5, 20]

- **Evaluation Methods**:
  - Elbow method (diminishing returns analysis)
  - Silhouette score calculation
  - 2D cluster visualization
  - Class homogeneity analysis

## Recommended File Structure

```
src/
├── preprocessing.py       # Data loading, splitting, standardization
├── dimensionality_reduction.py  # Custom PCA implementation
├── autoencoder.py         # PyTorch autoencoder model & training
├── clustering.py          # Custom k-Means and GMM implementations
├── evaluation.py          # Silhouette score, elbow method metrics
└── visualization.py       # Image display, 2D plots, comparisons
```

## Key Technical Requirements

### Custom Implementations Required
- **PCA**: Eigenvalue decomposition, variance explained calculation
- **k-Means**: Centroid initialization, assignment, update steps
- **GMM**: EM algorithm with Gaussian components
- **Data standardization**: Mean centering and scaling

### PyTorch Components (Allowed)
- Autoencoder architecture (encoder + decoder networks)
- Neural network training loops
- Loss functions and optimizers

### Data Flow
1. Raw 4,096-dimensional pixel data → Standardized data
2. Standardized data → PCA/Autoencoder latent space (reduced dimensions)
3. Reduced data → k-Means/GMM clustering
4. Clustering results → Evaluation and visualization

## Critical Implementation Notes

- **Dimensionality matching**: Autoencoder latent dimension must equal PCA components for fair comparison
- **Data consistency**: Always use training-learned transformations on evaluation data
- **Performance comparison**: Compare PCA vs Autoencoder reconstruction quality on 10 random validation images
- **Clustering input**: Use reduced-dimension data from Phase 2, not original 4,096 features
- **K selection**: Analyze relationship between optimal K and actual number of classes in dataset

## Deliverables Structure

The final submission requires:
1. **`Zabaleta_Gonzalo_Notebook_TP4.ipynb`** - Complete technical implementation
2. **`Zabaleta_Gonzalo_Informe_TP4.pdf`** - Academic report (max 10 pages, using provided LaTeX template)
3. **`data/`** directory - Dataset files

## Common Development Patterns

- Modularize reusable functions in `src/` directory
- Use descriptive variable names in Spanish (project language)
- Create visualization functions that can be reused throughout analysis
- Implement proper train/test/validation splits consistently
- Document hyperparameter choices and architectural decisions