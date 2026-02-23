# Person Search via Gallery Filter Network: A Backbone Efficiency Analysis
## How to Run

This project is designed to be easily reproducible and evaluated using Google Colab. The entry point for the code is the `main.ipynb` notebook. In the **Setup** section at the beginning of the notebook one can set boolean flags to decide what should be re-run and what to use as precomputed:
- RUN_TRAINING - whether to run the training process.
- RUN_EVALUATION - whether to run the evaluation loops. If False precomputed metrics are printed.
- RUN_EFFICIENCY_BENCHMARKING - whether to run the efficiency benchmarking (FPS).
- USE_PRETRAINED_MODELS - whether to load the already trained weights.
- USE_PRECOMPUTED_RESULTS - whether to load the results I computed, greatly recomended to be set to true. This allows to visualize some additional performance comparison between models, model errors and inference samples.

The loading of model weights and precomputed results is handled internally in the notebook. However I also provide the download links at the end of this Readme.md.

## Project Overview
This project addresses the **Person Search** task on the PRW dataset. Given a query person and a gallery of scenes, the system must detect people in the scenes and rank them based on their similarity to the query. The ultimate goal is to rank the true query person at the top of the retrieved gallery scenes.

This project utilizes an **end-to-end** architecture.


## Model Architecture
The core model utilized in this project is the [**Gallery Filter Network (GFN)** by Jaffe et al.](https://arxiv.org/abs/2210.12903) 

This architecture improves upon previous approaches by introducing a unique attention mechanism. The GFN trains the model to predict whether a specific person is likely present in a given gallery scene. This query-scene similarity score is then used to re-weight the standard query-detection similarity score, refining the ranking performance.

### Motivation & Experimentation
The original GFN paper relies on a heavy **ConvNeXt-Base** backbone. However, real-world applications of person search heavily depend on real-time processing speeds. 

The primary question of this project is: **Is such a large, computationally expensive backbone truly necessary to maintain high accuracy?** To answer this, the model was trained and evaluated using three different backbones from the ConvNeXt family:
1. **ConvNeXt-Base** (Original)
2. **ConvNeXt-Small**
3. **ConvNeXt-Tiny**

## Results & Metrics

### Performance Metrics

| Backbone | mAP (%) | Rank-1 (%) | Cross-Camera mAP (%) | Cross-Camera Rank-1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **ConvNeXt-Base** | 56.52% | 91.44% | 54.38% | 79.14% |
| **ConvNeXt-Small** | 56.03% | 90.28% | 53.92% | 81.04% |
| **ConvNeXt-Tiny** | 50.99% | 88.92% | 48.55% | 76.96% |

### Efficiency Metrics

| Backbone | Parameters (M) | MACs (G) | FPS |
| :--- | :---: | :---: | :---: |
| **ConvNeXt-Base** | 172.90 | 68.14 | 8.23 |
| **ConvNeXt-Small** | 97.67 |40.10 | 12.62 |
| **ConvNeXt-Tiny** | 76.04 | 38.20 | 16.48 |

## Key Findings & Conclusion

1. **Sweet Spot of 'ConvNeXt-Small':** Empirically, the **ConvNeXt-Small** backbone produces features robust enough to obtain highly competitive results. It suffers only a marginal drop in performance (0.51% drop in mAP and 1.16% in Rank-1) compared to the Base model, while significantly reducing the number of parameters and increasing inference speed from 8.23 to 12.62 FPS. 
2. **The Limit of Downsizing:** While the **ConvNeXt-Tiny** backbone improves efficiency and speed even further, it fails to provide sufficiently discriminative features for this complex task. This results in a severe performance penalty (5.53% drop in mAP and 2.52% in Rank-1).
3. **Cross-Camera Robustness:** In the more challenging cross-camera setting, the relative performance drops between the models remain roughly identical to the standard setting. This suggests that the massive parameter count of the ConvNeXt-Base does *not* necessarily translate to more robust, view-independent features; if it did, the performance gap between Base and Small would have widened in this specific setting.

**Conclusion:** Balancing performance and efficiency, the **ConvNeXt-Small** backbone is the optimal choice for this architecture. It is large enough to closely match the accuracy of the ConvNeXt-Base, while drastically cutting down MACs and parameter counts, keeping its operational efficiency closer to that of the Tiny backbone.

## Trained Weights & Precomputed Results

The model weights and precomputed prediction files for all three backbones are provided below.

### Trained Weights
* **ConvNeXt-Base:** [Download weights (.pth)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQA5UonSX75lSYAxlC4s6daAAcE-RzuZ5epZry77-plG3CI?e=hUNup8)
* **ConvNeXt-Small:** [Download weights (.pth)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQCIFjBwT6ofS6dKyWIWoWFfAT4O7PprdnP7weMsKl6LvmU?e=BKjfD6)
* **ConvNeXt-Tiny:** [Download weights (.pth)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQCNfblGti6CSawh8HUZx2bHAfCgtb8wnWToHNlVfT_eiCI?e=np6p74)

### Precomputed Results
These files contain the raw evaluation outputs and metrics generated during testing.
* **ConvNeXt-Base:** [Download results (.json)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQCNk37s9Rr9S4CTtV3PlssXAfEuDcIo0lMriixFApGIOT0?e=WxPDJQ)
* **ConvNeXt-Small:** [Download results (.json)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQBQlC55jLUMRo_-nJcXQU0AAWY0HKrQNAEhrjG42J2ZXqg?e=akS8ho)
* **ConvNeXt-Tiny:** [Download results (.json)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQDPzvfKL3eiSJYSGZmlZEAnAeQ062GiwXJpdalFNDevcWg?e=wS4bu1)
  
Cross-camera setting:

* **ConvNeXt-Base:** [Download results (.json)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQBMZPJhh7YQS5gI7qF25rLTAbeDHnCLZGVLu3r5Pf7sUFM?e=XnZIVw)
* **ConvNeXt-Small:** [Download results (.json)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQBgXfACLSUbSKXIg3DQ8IuNAUgQr_kiAy2b3FKl1PyOJjw?e=ZLO1Bd)
* **ConvNeXt-Tiny:** [Download results (.json)](https://liveunibo-my.sharepoint.com/:u:/g/personal/tomaz_cotic_studio_unibo_it/IQCAzC0OKxHpSKzBmj5VrI50AfLB2YGv2oSmtb-Xs9tlFrU?e=Jk63vq)

---
*Developed as part of [Machine Learning for Computer Vision] - [Tomaž Cotič/ 0001180192]*
