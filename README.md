# USD-Bench
Official repository for USD-Bench: a benchmark for user-profile based and explainable stance detection with structured user profiles and dimension ranking.

## Overview

USD-Bench is a benchmark for user-profile-based stance detection (USD).  
Unlike traditional stance detection datasets that rely only on tweet content, USD-Bench incorporates structured user profiles to support:

- User-specific stance prediction
- Explainable stance reasoning
- Dimension ranking for interpretability

The benchmark contains:
- Tweets
- Targets
- Structured user profiles
- Stance labels
- Ranked profile dimensions for explainability

We further propose USD-LLM, a strong baseline built on Qwen3-8B with LoRA fine-tuning and user profile contrastive learning.

## Features

- First benchmark for explicit user-profile based stance detection
- Explainable stance prediction with dimension ranking
- Structured target-specific user profiles
- Support for user-specific and ambiguous stance reasoning

## Dataset

USD-Bench contains:
- 2,742 annotated samples
- 5 stance targets
- Structured user profiles with basic and target-specific dimensions
- Top-6 dimension rankings for explainability evaluation

## Model

We provide:
- USD-LLM training code
- Data processing pipeline
- Evaluation scripts
- Prompt templates
- Benchmark splits

## Citation

If you find this repository useful, please cite our paper.

