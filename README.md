# Generative Adversarial Network for Synthetic Image Generation

This repository contains the implementation of Generative Adversarial Networks (GANs) and Conditional GANs (cGANs) to generate synthetic images of MNIST digits. The project demonstrates the power of GANs in generating realistic data and explores conditional inputs for controlled generation.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Project Overview](#project-overview)
5. [Results](#results)

---

## Introduction

Generative Adversarial Networks (GANs) have revolutionized the field of synthetic data generation by using adversarial training between a generator and a discriminator. This project implements both a basic GAN and a Conditional GAN to:

- Generate realistic MNIST digit images.
- Demonstrate control over class-specific image generation using conditional input labels.

---

## Features

- **Basic GAN**:
  - Learns to generate MNIST digits through adversarial training.
  - Outputs progressively improving image quality as training epochs increase.

- **Conditional GAN (cGAN)**:
  - Allows class-specific image generation by conditioning on input labels.
  - Produces diverse samples with clear digit structures.

- **Visualization**:
  - Real-time visualization of generated images during training.
  - Loss curves to monitor the performance of generator and discriminator.

---

## Dataset

- **Source**: MNIST Dataset
  - Handwritten digit images (28x28 pixels) for numbers 0 through 9.
- **Format**:
  - Training set: 60,000 samples.
  - Test set: 10,000 samples.
- **Preprocessing**:
  - Normalized pixel values to [-1, 1] for GAN training.

---

## Project Overview

### Objectives
1. Train a GAN to generate realistic MNIST digits.
2. Extend the model to a Conditional GAN for controlled generation based on digit class.

### Workflow
1. **Model Architecture**:
   - **GAN**:
     - Generator: Fully connected neural network.
     - Discriminator: Binary classifier to distinguish real vs. generated images.
   - **cGAN**:
     - Generator and Discriminator with class conditioning.
2. **Training**:
   - Adversarial training loop to optimize generator and discriminator.
   - Loss functions: Binary Cross-Entropy.
3. **Evaluation**:
   - Visual inspection of generated images.
   - Plotting loss curves for convergence analysis.

### Tools and Technologies
- **Programming Language**: Python
- **Libraries**: TensorFlow, NumPy, Matplotlib

---

## Results

### Basic GAN
- Generated digits become progressively clearer over epochs.
- Achieved a balance between generator and discriminator performance.

### Conditional GAN
- Successfully generated class-specific images based on input labels.
- Achieved high diversity and clarity in the generated samples.

### Visualizations
- Loss curves indicate stable training dynamics.
- Example outputs for digits 0 through 9 included in the repository.

---

## Acknowledgements

- TensorFlow and the developers of open-source libraries used in this project.
- Yann LeCun and collaborators for creating the MNIST dataset.

