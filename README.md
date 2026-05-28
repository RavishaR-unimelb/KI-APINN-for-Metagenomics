# KI-APINN: Knowledge-Inclusive Adaptive Physics-Informed Neural Network
 
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

 
## Overview
 
**KI-APINN** (Knowledge-Inclusive Adaptive Physics-Informed Neural Network) is a general framework for integrating diverse knowledge sources with physics-informed neural networks through interpretable adaptation mechanisms. The framework enables:
 
- **Multimodal knowledge integration**: Combines network structure, textual descriptions, numerical data, and physical constraints
- **Physics-informed learning**: Enforces domain-specific equations and constraints
- **Interpretable adaptation**: Mechanism reveals how different knowledge sources contribute to predictions
- **Cross-domain applicability**: Can be instantiated for various scientific domains
### A²G²: Metagenomics Instantiation
 
**A²G²** is a specific instantiation of KI-APINN for microbial community modeling. It infers interaction networks from microbiome abundance data by integrating:
 
- **Network structure** (Graph Neural Networks)
- **Biological text** (Text embeddings)
- **Abundance data** (Experimental data)
- **Physics-informed constraints** (Generalized Lotka–Volterra equations)
Through interpretable adaptive mechanisms, A²G² learns how different knowledge modalities contribute to predicting microbial community dynamics across diverse ecosystems (human gut, plant rhizosphere, in vitro communities).
 

## Installation
 
This repository contains the KI-APINN framework with the A²G² metagenomics instantiation. Follow the steps below to set up the environment and install dependencies.
 
### Requirements
 
- **Python**: 3.9 or higher
- **GPU** (optional but recommended): NVIDIA GPU with CUDA support

### Quick Start
 
#### Step 1: Clone the Repository
 
```bash
git clone https://github.com/RavishaR-unimelb/KI-APINN-for-Metagenomics.git
cd KI-APINN-for-Metagenomics
```
 
#### Step 2: Create a Python Environment
 
```bash
# Create virtual environment
python3.9 -m venv ki_apinn_env
 
# Activate environment
# On macOS/Linux:
source ki_apinn_env/bin/activate
 
# On Windows:
ki_apinn_env\Scripts\activate
```
 
#### Step 3: Install Dependencies
 
```bash
# Install required packages
pip install -r requirements.txt
```
 
#### Step 4: Verify Installation
 
```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

```

---

## Contact & Support
 
**Author**: Ravisha Rupasinghe  
**Affiliation**: University of Melbourne  
**Email**: [ravisha.rupasinghe@student.unimelb.edu.au](mailto:ravisha.rupasinghe@student.unimelb.edu.au)
 
