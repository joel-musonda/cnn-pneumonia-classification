# Pneumonia Classification Under Label Deprivation

How many labeled chest X-rays does a convolutional neural network need to detect pneumonia, and how much of the performance lost to scarce labels can data augmentation and self-supervised pretraining recover?

This repository holds the code, figures and experimental protocol for a study by Joel Musonda (Crucible Leadership Academy of Lusaka, October 2026). It accompanies the paper *How Much Labeled Data Does a Pneumonia Classifier Need? Data Augmentation and Self-Supervised Pretraining Under Label Deprivation*.

## Contents

1. [Motivation](#1-motivation)
2. [Research questions](#2-research-questions)
3. [Repository layout](#3-repository-layout)
4. [Data](#4-data)
5. [Methods](#5-methods)
6. [Baseline results](#6-baseline-results)
7. [The label deprivation experiment](#7-the-label-deprivation-experiment)
8. [Model v2: transfer learning and threshold selection](#8-model-v2-transfer-learning-and-threshold-selection)
9. [Discussion](#9-discussion)
10. [Limitations](#10-limitations)
11. [Reproducing the results](#11-reproducing-the-results)
12. [Related work](#12-related-work)
13. [References](#13-references)
14. [Citation](#14-citation)

## 1. Motivation

Rajpurkar et al. [4] showed in 2017 that a 121-layer DenseNet trained on more than 100,000 frontal chest radiographs could detect pneumonia with an F1 score above the average of four practicing radiologists. That result rested on a dataset assembled from one large hospital archive, with labels text-mined from radiology reports [5]. A hospital in Zambia that wanted a model calibrated to its own patients and equipment would begin with a few hundred labeled images.

The cost lies in labeling, not in imaging. Unlabeled radiographs accumulate in every radiology department, while each expert label consumes radiologist time. Two families of methods try to make better use of the labels that exist:

- Data augmentation, which creates new training examples by applying label-preserving transformations to existing ones [6, 11].
- Self-supervised learning, which trains an encoder on unlabeled images using a pretext task that needs no annotation, then fine-tunes it on the small labeled set [7, 8, 9, 10].

This project measures how quickly a pneumonia classifier degrades as labeled data is withheld, and how much of that degradation each method recovers, on hardware a student can access for free (a single Google Colab T4 GPU).

## 2. Research questions

- RQ1. How does test ROC-AUC change as the number of labeled training images falls from about 21,000 to about 200?
- RQ2. Does augmentation of the labeled images reduce that drop, and at which training set sizes does it matter most?
- RQ3. Does SimCLR pretraining on the unlabeled images give a further gain when labels are scarce?

## 3. Repository layout

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_baseline_cnn.ipynb                  # three-block CNN trained from scratch (outputs saved)
│   ├── 02_pneumonia_v2_densenet.ipynb         # ImageNet DenseNet-121, class weighting, tuned thresholds, Grad-CAM
│   └── 03_data_deprivation_experiments.ipynb  # 1% to 100% labels x {baseline, augmentation, SimCLR} x 3 seeds
└── figures/
    ├── sample_xray.png
    ├── class_distribution.png
    ├── baseline_cnn_code.png
    ├── baseline_training_log.png
    ├── baseline_confusion_matrix.png
    └── class_weighting_code.png
```

All notebooks are written for Google Colab. Notebook 01 contains the outputs of the run reported in Section 6. Notebooks 02 and 03 have not yet been run on the full dataset, so their results are reported as pending.

## 4. Data

### 4.1 Source

The images come from the RSNA Pneumonia Detection Challenge [3], hosted on Kaggle. The dataset is a subset of the NIH ChestX-ray14 collection [5] that radiologists re-annotated with bounding boxes around lung opacities suggestive of pneumonia [12]. Images are 1024 x 1024 DICOM files.

### 4.2 Labels

`stage_2_train_labels.csv` has 30,227 rows, because a patient with several opacities has one row per bounding box. `stage_2_detailed_class_info.csv` assigns each row to one of three groups:

| Group | Rows | Label used here |
|---|---:|---:|
| No Lung Opacity / Not Normal | 11,821 | 0 |
| Lung Opacity | 9,555 | 1 |
| Normal | 8,851 | 0 |

Collapsing rows to one label per patient (`groupby('patientId')['Target'].max()`) gives 26,684 images, of which roughly 22% are positive.

The middle group deserves attention. "No Lung Opacity / Not Normal" covers radiographs with other abnormalities, such as nodules, effusions or devices, that radiologists judged not to show pneumonia. Placing them in the negative class makes the task harder and more realistic than the common "pneumonia versus healthy" framing, since the classifier must separate pneumonia from other disease and not only from normal anatomy.

<p align="center"><img src="figures/class_distribution.png" width="520"><br><em>Figure 1. Rows per class in the RSNA detailed class file.</em></p>

<p align="center"><img src="figures/sample_xray.png" width="360"><br><em>Figure 2. A training radiograph as loaded in notebook 01. Burned-in text and markers, visible at the upper right, are a known source of shortcut learning in chest X-ray models.</em></p>

## 5. Methods

### 5.1 Preprocessing

Each DICOM pixel array is rescaled to [0, 255], converted to an 8-bit grayscale image and resized. The baseline uses 128 x 128 inputs normalized with mean 0.485 and standard deviation 0.229. Notebook 02 caches every image once at 256 x 256, applies augmentation on the GPU, resizes to 224 x 224, replicates the channel three times and normalizes with the ImageNet channel statistics expected by the pretrained backbone.

### 5.2 Baseline architecture

`PneumoniaCNN` is a three-block network. Each block applies a 3 x 3 convolution with padding 1, batch normalization, ReLU and 2 x 2 max pooling. The blocks have 32, 64 and 128 filters, so a 128 x 128 input becomes a 128 x 16 x 16 feature map. A 128-unit fully connected layer with dropout 0.5 and a single output logit complete the classifier.

| Layer | Output shape | Parameters |
|---|---|---:|
| Conv 3x3, 32 + BN + pool | 32 x 64 x 64 | 384 |
| Conv 3x3, 64 + BN + pool | 64 x 32 x 32 | 18,624 |
| Conv 3x3, 128 + BN + pool | 128 x 16 x 16 | 74,112 |
| Linear 32,768 to 128 | 128 | 4,194,432 |
| Linear 128 to 1 | 1 | 129 |
| Total | | about 4.29 M |

Almost 98% of the parameters sit in the first fully connected layer. This matters for the deprivation study: with 213 labeled images, a model of this shape has about 20,000 parameters per training example in that layer alone, and will overfit unless regularized.

<p align="center"><img src="figures/baseline_cnn_code.png" width="560"><br><em>Figure 3. The baseline model definition in notebook 01.</em></p>

### 5.3 Loss and class imbalance

The baseline minimizes binary cross-entropy on the logit $z$ with target $y \in \{0, 1\}$:

$$\mathcal{L}(z, y) = -\left[\, w_{+}\, y \log \sigma(z) + (1 - y) \log\left(1 - \sigma(z)\right) \right]$$

In the first run $w_{+} = 1$. With about 78% negatives, the loss is minimized early by pushing every prediction toward the negative class, which is what happened (Section 6). The second run sets $w_{+} = 3.2$, close to the ratio of negatives to positives, so that a missed pneumonia costs as much in total as all the negatives it competes with. Notebooks 02 and 03 compute $w_{+} = N_{-} / N_{+}$ from whichever training subset is in use.

### 5.4 Decision threshold

A classifier outputs a probability $p = \sigma(z)$, and a threshold $t$ turns it into a decision. The default $t = 0.5$ is only optimal when classes are balanced and errors cost the same, and neither holds for pneumonia screening. Notebook 02 chooses $t$ on a validation set that the model never trains on, using two rules:

- Youden's index, $t^{*} = \arg\max_t \left[\mathrm{TPR}(t) - \mathrm{FPR}(t)\right]$, which balances sensitivity against specificity.
- A screening rule, the highest $t$ with $\mathrm{TPR}(t) \ge 0.90$, which fixes sensitivity first and accepts the false positives that follow.

The test set is then scored once at each threshold. Choosing $t$ on the test set itself would leak information and overstate performance.

### 5.5 Evaluation metrics

Accuracy is reported for continuity, but it is not the primary measure. On this dataset a model that calls every image normal scores about 78%. The study reports:

- ROC-AUC, the probability that a randomly chosen positive is ranked above a randomly chosen negative. It does not depend on the threshold.
- Sensitivity (recall) $= TP / (TP + FN)$ and specificity $= TN / (TN + FP)$.
- Balanced accuracy, the mean of sensitivity and specificity, which is 0.5 for any trivial classifier.
- Precision and F1 at the chosen threshold.

### 5.6 Data augmentation

Augmentations are applied per batch on the GPU with an affine sampling grid:

| Transformation | Supervised training | SimCLR pretraining |
|---|---|---|
| Random resized crop (fraction of image kept) | 0.80 to 1.00 | 0.50 to 1.00 |
| Rotation | up to 10 degrees | up to 10 degrees |
| Brightness shift | up to 0.10 | up to 0.20 |
| Contrast scaling | 0.90 to 1.10 | 0.60 to 1.40 |
| Gaussian noise | none | standard deviation 0.03 |
| 3 x 3 blur | none | with probability 0.5 |
| Horizontal flip | never | never |

Horizontal flips are excluded on purpose. Flipping a frontal radiograph moves the heart to the right side of the chest, which in real patients occurs only with dextrocardia. A flipped image is therefore out of distribution, and the network could learn features that do not exist in clinical data. Perez and Wang [6] found simple geometric transformations to be among the most effective augmentations on small datasets, and the choices above restrict those transformations to ones a radiograph can plausibly undergo through patient positioning and exposure differences.

### 5.7 Self-supervised pretraining with SimCLR

SimCLR [7] trains an encoder $f$ and a small projection head $g$ so that two augmented views of the same image map to nearby points, while views of different images are pushed apart. For a batch of $N$ images, each augmented twice, the loss for a positive pair $(i, j)$ is the normalized temperature-scaled cross-entropy (NT-Xent):

$$\ell_{i,j} = -\log \frac{\exp\left(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau\right)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \ne i]} \exp\left(\mathrm{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau\right)}$$

where $\mathbf{z} = g(f(\mathbf{x}))$, $\mathrm{sim}$ is cosine similarity and $\tau$ is a temperature. Notebook 03 uses the baseline CNN's convolutional blocks as $f$, a two-layer projection head (512 hidden units, 128 outputs), batch size 256, $\tau = 0.2$ and 30 epochs over all 21,347 training-pool images with their labels discarded. After pretraining, $g$ is thrown away and $f$ initializes the supervised classifier.

The rationale comes from Sowrirajan et al. [9], who found that contrastive pretraining on chest X-rays helped most when labeled data was limited, and Azizi et al. [10], who showed that a second stage of self-supervised pretraining on unlabeled medical images improved chest X-ray classification over ImageNet pretraining alone. One open question here is whether a 4.3-million-parameter CNN has enough capacity to benefit, since Chen et al. [7] found that contrastive learning gains more from model size than supervised learning does.

## 6. Baseline results

The baseline CNN was trained for five epochs (Adam, learning rate 0.001, batch size 32) on a random 80/20 split: 21,347 training and 5,337 validation patients.

<p align="center"><img src="figures/baseline_training_log.png" width="520"><br><em>Figure 4. Training loss and validation metrics, captured from notebook 01.</em></p>

<p align="center"><img src="figures/baseline_confusion_matrix.png" width="400"><br><em>Figure 5. Validation confusion matrix, threshold 0.5.</em></p>

| Metric | Baseline CNN | Always predict "normal" |
|---|---:|---:|
| Accuracy | 79.60% | 78.13% |
| Sensitivity | 0.083 | 0.000 |
| Specificity | 0.995 | 1.000 |
| Precision | 0.836 | undefined |
| F1 | 0.151 | 0.000 |
| Balanced accuracy | 0.539 | 0.500 |

Of 1,167 validation patients with pneumonia, the model identified 97 and missed 1,070. It flagged 19 of 4,170 negatives. The 79.60% accuracy is 1.47 percentage points above the trivial rule.

Two conclusions follow. The first is that the network did learn something: its precision of 0.836 means the few images it flags are mostly true positives, so its scores carry signal that a lower threshold could exploit. The second is that accuracy cannot be the outcome measure for the deprivation study. A model trained on 1% of the labels that collapses to "normal" would lose only 1.5 points of accuracy against the full-data model, hiding the very effect under study.

A class-weighted rerun ($w_{+} = 3.2$, threshold 0.35) is in notebook 01 (Figure 6), but its output was not saved, so its numbers are not reported.

<p align="center"><img src="figures/class_weighting_code.png" width="560"><br><em>Figure 6. The class-weighted retraining cell.</em></p>

## 7. The label deprivation experiment

### 7.1 Protocol

1. Split the 26,684 patients once, stratified by label, into a training pool (80%, 21,347) and a held-out test set (20%, 5,337). The test set is used only for final scoring.
2. From the pool, draw stratified subsets of 1%, 5%, 10%, 25%, 50% and 100%: 213, 1,067, 2,134, 5,336, 10,673 and 21,347 labeled images.
3. Train three conditions on each subset: baseline, augmentation, and SimCLR pretraining followed by fine-tuning with augmentation.
4. Repeat each cell with three seeds, which change both the subset drawn and the weight initialization.
5. Score ROC-AUC, sensitivity, specificity, precision and F1 on the test set.

Smaller subsets receive more epochs (`epochs = max(15, 15 x 2000 / n)`). At a fixed 15 epochs, a 213-image run would make about 60 gradient updates against about 5,000 for the full set, and the comparison would confound too little data with too little training. The rule raises the 213-image run to about 560 updates, which narrows that gap while still letting the smallest subsets see each image many times.

### 7.2 Results

Test ROC-AUC, mean and standard deviation over three seeds. Pending a full run of notebook 03.

| Labeled images | Baseline | Augmentation | SimCLR and augmentation |
|---:|:---:|:---:|:---:|
| 213 (1%) | pending | pending | pending |
| 1,067 (5%) | pending | pending | pending |
| 2,134 (10%) | pending | pending | pending |
| 5,336 (25%) | pending | pending | pending |
| 10,673 (50%) | pending | pending | pending |
| 21,347 (100%) | pending | pending | pending |

### 7.3 Analysis plan

The main figure plots test AUC against the number of labeled images on a logarithmic axis, one line per condition. Sun et al. [2] found vision performance to grow roughly linearly in the logarithm of dataset size, so a straight line on this plot is the null expectation. Following Cho et al. [1], a learning curve fitted to the baseline points gives the number of labeled images the baseline would need to match each method's AUC at a given subset size. The ratio of the two is a direct estimate of how many labels a method saves.

### 7.4 Hypotheses

- H1. Augmentation raises AUC most at 5% to 25% of the labels and has little effect at 100%, consistent with Perez and Wang [6].
- H2. SimCLR pretraining gives its largest gain at 1% and 5%, following the pattern MoCo-CXR showed [9].
- H3. The SimCLR gain is smaller than in published studies, because the encoder is far smaller than the ResNet-50 and larger backbones those studies used [7, 10].

## 8. Model v2: transfer learning and threshold selection

Notebook 02 targets the failure in Section 6 directly: higher sensitivity without giving up overall performance. It changes four things relative to the baseline.

| Component | Baseline (v1) | v2 |
|---|---|---|
| Backbone | 3-block CNN, random initialization | DenseNet-121, ImageNet weights (CheXNet backbone [4]) |
| Input | 128 x 128, 1 channel | 224 x 224, 3 channels |
| Class imbalance | none, then $w_{+} = 3.2$ | $w_{+} = N_{-}/N_{+}$ from training data |
| Augmentation | none | crop, rotation, intensity (Section 5.6) |
| Optimizer | Adam, fixed rate 0.001, 5 epochs | AdamW, one-cycle schedule up to 3e-4, 8 epochs, mixed precision |
| Model selection | last epoch | epoch with best validation AUC |
| Split | 80 / 20 | 70 / 15 / 15 (train / validation / test) |
| Threshold | 0.5 | Youden and 0.90-sensitivity rules, chosen on validation |
| Interpretability | none | Grad-CAM heat maps |

The notebook reports test metrics at three thresholds beside the v1 numbers, saves ROC and confusion-matrix figures, and writes weights and metrics to Google Drive.

Accuracy may not rise above the baseline's 79.60%. Raising sensitivity on an imbalanced set converts some true negatives into false positives, and each of those costs accuracy. The improvement should be judged by AUC and balanced accuracy, and by sensitivity at a stated specificity.

Results: pending a full run of notebook 02.

## 9. Discussion

The baseline shows that the binding constraint on this dataset was not the quantity of data. With 21,347 labeled images, the unweighted network still predicted the majority class for almost every patient. More data would not have changed that. Any study of data volume has to fix the imbalance first, through the loss, the sampling or the threshold, and has to measure outcomes with metrics that a trivial classifier cannot score well on.

The deprivation design adds two controls that are easy to omit. Matching the number of gradient updates across subset sizes separates "less data" from "less training". Drawing a new subset for each seed means the reported variance includes the effect of which images happened to be labeled, which is a real source of uncertainty for a hospital labeling its first few hundred cases.

## 10. Limitations

- Resolution. Downsampling from 1024 x 1024 to 128 x 128 or 224 x 224 removes fine texture radiologists use to identify early consolidation.
- Label noise. RSNA labels reflect radiologist consensus on opacity, not confirmed clinical diagnosis, and the "Not Normal" negatives contain disease that can look like pneumonia.
- Patient population. The images come from a single United States health system. Performance on Zambian patients, on different X-ray equipment and with different disease prevalence (including tuberculosis and HIV-associated pneumonias) is unknown and would need local validation.
- Shortcut learning. Burned-in text, markers and positioning differences between inpatient and outpatient films can correlate with labels. Grad-CAM in notebook 02 is a first check, not a guarantee.
- Baseline protocol. The baseline used a random split without a fixed seed, five epochs, no learning-rate schedule and the last epoch rather than the best, so its numbers will move slightly between runs.
- Clinical use. None of these models is validated for diagnosis.

## 11. Reproducing the results

1. Create a Kaggle account, accept the rules of the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge), and download `kaggle.json` from Kaggle account settings.
2. Open a notebook from `notebooks/` in Google Colab and set Runtime > Change runtime type > T4 GPU.
3. Run all cells and upload `kaggle.json` when prompted. The download is about 3.7 GB.
4. Notebook 03 expects the dataset folder created by notebook 01 or 02 in the same session.

Approximate time on a free T4: notebook 01 roughly 30 minutes (CPU-bound DICOM loading), notebook 02 about an hour, notebook 03 several hours because of its 54 training runs and the SimCLR stage. Notebook 03 caches images and the pretrained encoder, so it can resume after a disconnection.

Local installation:

```bash
pip install -r requirements.txt
```

## 12. Related work

| Work | Contribution | Relevance here |
|---|---|---|
| Wang et al. 2017 [5] | ChestX-ray8/14: 108,948 radiographs, 32,717 patients, report-mined labels | Parent dataset of RSNA |
| Rajpurkar et al. 2017 [4] | CheXNet, DenseNet-121, F1 above average radiologist | v2 backbone |
| Shih et al. 2019 [12] | Radiologist bounding boxes for possible pneumonia | Source of RSNA labels |
| Cho et al. 2015 [1] | Learning curve to predict required training set size | Method for Section 7.3 |
| Sun et al. 2017 [2] | Performance grows logarithmically with data (JFT-300M) | Null expectation for the deprivation curve |
| Perez and Wang 2017 [6] | Compared augmentation strategies on small ImageNet subsets | Augmentation design and H1 |
| Shorten and Khoshgoftaar 2019 [11] | Survey of image augmentation | Background |
| Chen et al. 2020 [7] | SimCLR; augmentation composition and projection head matter | Pretraining method |
| He et al. 2020 [8] | MoCo; momentum encoder and negative queue | Alternative contrastive method |
| Sowrirajan et al. 2021 [9] | MoCo-CXR helps most with limited labels; transfers to TB | Basis for H2 |
| Azizi et al. 2021 [10] | In-domain self-supervision after ImageNet improves CXR AUC | Basis for H3 and future work |

## 13. References

[1] J. Cho, K. Lee, E. Shin, G. Choy and S. Do. How much data is needed to train a medical image deep learning system to achieve necessary high accuracy? arXiv:1511.06348, 2015.

[2] C. Sun, A. Shrivastava, S. Singh and A. Gupta. Revisiting unreasonable effectiveness of data in deep learning era. ICCV, 2017. arXiv:1707.02968.

[3] Radiological Society of North America. RSNA Pneumonia Detection Challenge. Kaggle, 2018. https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

[4] P. Rajpurkar, J. Irvin, K. Zhu, B. Yang, H. Mehta, T. Duan, D. Ding, A. Bagul, C. Langlotz, K. Shpanskaya, M. P. Lungren and A. Y. Ng. CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. arXiv:1711.05225, 2017.

[5] X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri and R. M. Summers. ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. CVPR, 2017. arXiv:1705.02315.

[6] L. Perez and J. Wang. The effectiveness of data augmentation in image classification using deep learning. arXiv:1712.04621, 2017.

[7] T. Chen, S. Kornblith, M. Norouzi and G. Hinton. A simple framework for contrastive learning of visual representations. ICML, 2020. arXiv:2002.05709.

[8] K. He, H. Fan, Y. Wu, S. Xie and R. Girshick. Momentum contrast for unsupervised visual representation learning. CVPR, 2020. arXiv:1911.05722.

[9] H. Sowrirajan, J. Yang, A. Y. Ng and P. Rajpurkar. MoCo-CXR: MoCo pretraining improves representation and transferability of chest X-ray models. MIDL, 2021. arXiv:2010.05352.

[10] S. Azizi, B. Mustafa, F. Ryan, Z. Beaver, J. Freyberg, J. Deaton, A. Loh, A. Karthikesalingam, S. Kornblith, T. Chen, V. Natarajan and M. Norouzi. Big self-supervised models advance medical image classification. ICCV, 2021. arXiv:2101.05224.

[11] C. Shorten and T. M. Khoshgoftaar. A survey on image data augmentation for deep learning. Journal of Big Data 6, 60, 2019.

[12] G. Shih, C. C. Wu, S. S. Halabi et al. Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia. Radiology: Artificial Intelligence 1(1), e180041, 2019.

## 14. Citation

```bibtex
@misc{musonda2026pneumonia,
  author = {Musonda, Joel},
  title  = {How Much Labeled Data Does a Pneumonia Classifier Need? Data Augmentation and Self-Supervised Pretraining Under Label Deprivation},
  year   = {2026},
  note   = {Crucible Leadership Academy of Lusaka},
  url    = {https://github.com/joel-musonda/cnn-from-scratch-for-pneumonia-classification}
}
```
