# Probabilistic Data Augmentation (PDA)

**Compositional Similarity Estimation for Open-Set Recognition via Augmentation-Expanded Feature Manifolds**

![PDA Overview](https://via.placeholder.com/800x400?text=Figure+1:+PDA+Manifold+vs+Standard+Prototypes)  
*Figure 1: Standard prototype methods use single points per class (left), leading to misclassification of novel inputs. PDA expands each class into a dense augmented manifold and models it with kernel density estimation (right), enabling meaningful similarity profiles.*

Official repository for the paper:  
**Probabilistic Data Augmentation: Compositional Similarity Estimation for Open-Set Recognition via Augmentation-Expanded Feature Manifolds**  
Yeabsira Teshome (Aule Technologies)

## Abstract

Deep learning classifiers excel on closed-set benchmarks but fail on novel classes, either misclassifying them with high confidence or providing uninformative "unknown" outputs. Humans, by contrast, naturally decompose novel inputs into familiar components: a child seeing a tiger for the first time recognizes it as "a big cat with stripes." We introduce Probabilistic Data Augmentation (PDA), an open-set recognition framework that computes calibrated similarity profiles over known classes for any input, including classes never seen during training. PDA operates in three stages: (1) a reconstruction-regularized encoder maps inputs to a structured latent space where augmentations induce meaningful variance; (2) stochastic augmentation expands each class's representation into a dense feature manifold; (3) scalable kernel density estimation via random Fourier features models each manifold as a probability distribution. For novel inputs, PDA outputs a normalized similarity profile across all known classes alongside a calibrated novelty score, enabling compositional reasoning about unfamiliar categories. We introduce a contrastive linkage loss that captures cross-class semantic relationships while preventing representation collapse. Experiments on CIFAR-100, CUB-200, and miniImageNet demonstrate strong open-set recognition performance with well-calibrated similarity estimates.

**Keywords:** open-set recognition, data augmentation, kernel density estimation, manifold learning, out-of-distribution detection, calibration

## Highlights

- Compositional similarity profiles for novel inputs (e.g., tiger -> Cat: 0.71, Dog: 0.19, Horse: 0.10)
- Calibrated novelty detection
- Scalable inference with Random Fourier Features
- Reconstruction regularization to prevent manifold collapse
- State-of-the-art results on standard OSR benchmarks (preliminary experiments)

## Installation

```bash
git clone https://github.com/auletechnologies/pda.git
cd pda
pip install -r requirements.txt
```

Required packages (example):
- torch
- torchvision
- numpy
- scipy
- scikit-learn

## Usage

### Training

```bash
python train.py --dataset cifar100 --known-classes 80 --epochs 100
```

### Inference / Evaluation

```bash
python evaluate.py --checkpoint path/to/checkpoint.pth --dataset cifar100
```

See `configs/` for example configurations and hyperparameters.

## Results (Preliminary)

### Open-Set Recognition

| Method       | CIFAR-100 AUROC | CIFAR-100 FPR@95 | CUB-200 AUROC | miniImageNet AUROC |
|--------------|-----------------|------------------|---------------|--------------------|
| Softmax      | 0.782           | 0.583            | 0.724         | 0.756              |
| ODIN         | 0.843           | 0.421            | 0.789         | 0.812              |
| Mahalanobis  | 0.867           | 0.372            | 0.821         | 0.839              |
| OpenMax      | 0.851           | 0.398            | 0.798         | 0.824              |
| Energy       | 0.872           | 0.356            | 0.834         | 0.847              |
| ARPL         | 0.891           | 0.312            | 0.856         | 0.868              |
| Prototype    | 0.856           | 0.387            | 0.812         | 0.831              |
| **PDA (Ours)** | **0.923**     | **0.234**        | **0.891**     | **0.897**          |

### Calibration (CIFAR-100)

| Method   | ECE   | MCE   | Brier |
|----------|-------|-------|-------|
| Softmax  | 0.127 | 0.312 | 0.298 |
| ODIN     | 0.098 | 0.267 | 0.251 |
| Energy   | 0.089 | 0.243 | 0.234 |
| ARPL     | 0.072 | 0.198 | 0.203 |
| **PDA**  | **0.031** | **0.087** | **0.142** |

Note on Results

These results are from early/prototype experiments and may vary with different seeds, splits, or hyperparameter tuning. They should be considered preliminary. Full benchmarking against the latest public baselines is ongoing.

 Citation

If you find this work useful, please cite:

```bibtex
@article{teshome2026pda,
  title={Probabilistic Data Augmentation: Compositional Similarity Estimation for Open-Set Recognition via Augmentation-Expanded Feature Manifolds},
  author={Teshome, Yeabsira},
  journal={arXiv preprint},
  year={2026}
}
```

## Contact

Yeabsira Teshome  
yeabsira@auletechnologies.com  
Twitter: @Yeabsira_001

Feel free to open issues or submit pull requests!
```
