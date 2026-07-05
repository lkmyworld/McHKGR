# Multimodal Contrastive Learning with Hyperbolic Geometry for KG-based Game Recommendation

## Introduction

We propose a multimodal contrastive learning method with hyperbolic geometry for KG-based recommendation (McHKGR), which improves recommendations by leveraging multimodal data (e.g., images, text) and Knowledge Graphs (KGs). By representing the KG in hyperbolic space, we better capture the complex relationships within the data. Additionally, we enhance user representations through a user co-occurrence graph and employ cross-modal contrastive learning to uncover latent connections between modalities.

## Dataset

The dataset used in this project includes:

- **Steam**: A real-world dataset that includes user-game interactions and multimodal features such as game images and descriptions (constructed from https://steamcommunity.com/).
- **MovieLens**: A public dataset commonly used for recommendation system benchmarks, we further integrate multimodal information (constructed from https://movielens.org/).

## Environment Requirement

The project requires the following environment:

- **Python Version**: `>=3.8`
- **PyTorch Version**: `>=1.12`
- **Hardware**:
    - Recommended GPU: NVIDIA 3090 (24GB)
    - System memory: ≥64GB
