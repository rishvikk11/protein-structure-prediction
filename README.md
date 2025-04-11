# Protein Structure Prediction Using Deep Learning

## Overview

This project aims to predict the **3D coordinates of protein structures** from amino acid sequences using deep learning framework PyTorch, with interactive 3D visualization capabilities. By combining sequence processing, coordinate normalization, and a custom PyTorch model, it provides a computational biology pipeline to approximate protein folding — a crucial task in structural bioinformatics.

---

## Features

- **Custom Dataset Pipeline**: Preprocessed amino acid sequences and their associated 3D coordinates using `.pdb` files from the Protein Data Bank (PDB), with normalization and padding for model readiness.
- **End-to-End Deep Learning Pipeline**: Built a PyTorch-based sequence-to-structure model capable of learning and predicting protein backbone atom coordinates (CA atoms).
- **Evaluation & Visualization**:
  - Calculated **RMSD (Root Mean Square Deviation)** between true and predicted structures.
  - Visualized predicted vs. true 3D structures using both `py3Dmol`.

---

## Tools and Technologies

- **Programming Language**: Python  
- **Frameworks/Libraries**: PyTorch, NumPy, Pandas, Pickle, Py3Dmol  
- **Data Source**: PDB (Protein Data Bank)  
- **Others**: Biopython for structure parsing

---

## Key Achievements

- Implemented a full training and evaluation pipeline to predict 3D atomic coordinates from raw amino acid sequences.
- Achieved a test-time **average RMSD < 20.0Å** on validation samples after model tuning.
- Visualized predicted structures in **interactive 3D** to compare predictions against ground truth.

---

## How It Works

1. **Data Extraction**  
   - Extracts protein sequences and their 3D coordinates (specifically CA atoms) from `.pdb` files.
   - Applies sequence encoding, coordinate normalization, and padding to standardize input/output lengths.

2. **Model Training**  
   - A sequence model (LSTM) processes amino acid sequences and predicts the spatial coordinates of the protein backbone atoms.

3. **Model Evaluation**  
   - Predictions are evaluated using **RMSD**, a standard metric in structural biology for comparing 3D coordinates.
   - Visualizations highlight how closely the model’s predicted structure matches the real protein.

4. **Visualization**  
   - Uses `py3Dmol` to render interactive 3D models of predicted and true protein structures for qualitative comparison.

---

## Future Enhancements/Considerations

- Replace the current model with a **Transformer-based architecture** to see if it captures long-range dependencies in protein sequences better.
- Extend coordinate prediction to include **side-chain atoms** for full atomic-level accuracy.
- Build a **web interface** to allow users to input a sequence and visualize its predicted 3D structure in real-time.
- Incorporate **AlphaFold-like confidence scoring** to assess prediction reliability.
