# NNSS — Stable-by-Design Neural Network State-Space Models

This repository provides the **MATLAB implementation** of the Neural Network–based State-Space (NN-SS) modeling framework with **stability guarantees**, as proposed in the following work:

> **Sertbaş, Ahmet & Kumbasar, Tufan (2025)**  
> *Stable-by-Design Neural Network-Based LPV State-Space Models for System Identification*


---

## 📌 Repository Scope

This repository **only contains source code**.

- ✅ MATLAB implementation of NN-SS
- ✅ Training, validation, and testing pipelines
- ✅ Benchmark systems (Powerplant, Robot Arm, Two-Tank)
- ✅ SIMBa and subspace identification baselines


## 📂 Repository Structure

```text
NNSS/
├─ Training.m              % Main training pipeline
├─ Testing.m               % Testing & evaluation pipeline
├─ SIMBa_Layer.m           % SIMBa neural state-space layer
├─ Local_Functions.m       % Shared helper functions
├─ Powerplant.m            % Powerplant benchmark
├─ Robot_Arm.m             % Robot arm benchmark
├─ Two_tank.m              % Two-tank benchmark
├─ Bayesian.m              % Bayesian optimization utilities (optional)
├─ README.md
└─ LICENSE
