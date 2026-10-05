# Code for "Lightweight Preprocessing and Feature Extraction for LoRa RF Fingerprint Identification"

This repository contains the code associated with the following publication:

> R. Gerzaguet, M. Gautier, J. Zhang, A. Chillet, A. Marshall, O. Berder, "Lightweight Preprocessing and Feature Extraction for LoRa RF Fingerprint Identification," *IEEE International Conference on Communications (ICC)*, Glasgow, United Kingdom, 2026, pp. 1–6.  
> DOI: 10.1109/ICC59461.2026.11587750
>
> https://ieeexplore.ieee.org/abstract/document/11587750

The code provided in this repository was used as the basis for generating the experimental results presented in the paper. It implements the preprocessing, feature extraction, device enrollment, authentication, and rogue-device detection pipeline described in the proposed approach.

## Repository contents

The main processing pipeline is implemented in Julia and includes:

- IQ signal preprocessing based on instantaneous phase differences;
- feature extraction using the proposed lightweight approach;
- feature selection using ExtraTrees;
- dimensionality reduction using Linear Discriminant Analysis (LDA);
- device enrollment;
- device authentication using GMM or kNN;
- rogue-device detection;
- performance evaluation using ROC curves, AUC, and EER.

The main files are:

- `main.jl`: main execution pipeline and experimental configuration;
- `selected_indexes.bin`: list of selected feature indexes obtained during feature selection;
- `lda_proj.bin`: LDA projection matrix used for dimensionality reduction.

The files `selected_indexes.bin` and `lda_proj.bin` respectively contain the selected feature index list **I** and the LDA projection matrix **M** used by the feature extraction pipeline.

## Reproducibility

The provided implementation is intended to facilitate the reproduction of the results reported in the paper. The configuration parameters, datasets, and pre-trained feature-selection and projection parameters should be set according to the experimental setup described in the paper.

## Status

A detailed description of the implementation, dataset organization, configuration parameters, and instructions for reproducing the reported experiments will be provided soon.
