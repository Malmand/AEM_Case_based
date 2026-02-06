# 🔬 Modeling of AEM Electrolyser: Wet vs. Dry Cathode Performance

This repository contains a Python-based modeling framework for analyzing the performance of an **Anion Exchange Membrane (AEM) electrolyser**. The project focuses on comparing **wet cathode** and **dry cathode** operating modes using electrochemical, thermal, and impedance-based validation methods.

---

## Overview

The main simulation script (`main.py`) performs the following tasks:

1. **Data Ingestion**  
   Loads experimental polarization and thermal data from Excel files.

2. **Electrochemical Fitting**  
   Fits a dual-reaction activation model to experimental V–I curves to extract:
   - Charge transfer coefficient (α)
   - Exchange current density (i₀)  
   for both anode and cathode.

3. **Thermal Modeling**  
   Simulates stack temperature evolution and compares predictions at the membrane center with experimental reference points.

4. **Efficiency Analysis**  
   Calculates system efficiency using:
   - Standard HHV-based definitions  
   - Thermodynamic definitions from Lamy & Millet

5. **Calibration & Validation**  
   Uses Electrochemical Impedance Spectroscopy (EIS) data to calibrate the relationship between temperature and Ohmic resistance and validate thermal self-heating predictions.

---

## Prerequisites

- **Python 3.8+**
- Required Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `openpyxl`

### Install Dependencies

```bash
pip install pandas numpy matplotlib scipy openpyxl
