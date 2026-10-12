# AEM Water Electrolyser Model: Wet vs Dry Cathode Operation

Python model of an **anion exchange membrane (AEM) water electrolyser** that combines electrochemical fitting, a 1D thermal model and efficiency analysis to compare **wet-cathode** and **dry-cathode** operation against experimental data. Ohmic resistance is calibrated against temperature using electrochemical impedance spectroscopy (EIS), which is then used to check the predicted self-heating of the cell.

Developed in the case-based module of the Erasmus Mundus DENSYS Master's (M1, Université de Lorraine) as a team project of four students; I was responsible for the electrochemical model.

## What the model does

| Step | Content |
|---|---|
| 1. Data | Loads measured polarisation and temperature data for wet- and dry-cathode tests |
| 2. Electrochemistry | Open-circuit voltage, membrane resistance (Arrhenius) and KOH electrolyte resistance; a dual Butler-Volmer activation model is fitted to the measured V-I curves to extract the charge transfer coefficient (α) and exchange current density (i₀) at the anode and cathode |
| 3. Thermal model | 1D through-plane model of cell temperature, compared with measured temperatures at the membrane centre; thermal gradient across the layers |
| 4. Efficiency | HHV-based efficiency and thermodynamic definitions following Lamy and Millet (2020), for both operating modes and across temperatures |
| 5. Sensitivity | Tornado charts for voltage and efficiency, and a parameter sensitivity curve |
| 6. EIS calibration | Ohmic resistance extracted from EIS spectra at 60 °C and 80 °C, used to build a resistance-temperature relation and validate the thermal self-heating prediction |

## How to run

```bash
pip install -r requirements.txt
python aem_model.py
```

Figures (polarisation fits, thermal response, efficiency comparisons, sensitivity and EIS calibration, about 20 figures) are saved to `results/`.

### Input data

The experimental data were provided for the course and are **not included** in this repository. To run the model with your own measurements, place the following in `data/` (or point `AEM_DATA_DIR` to another folder):

```
data/
├── Polarization curve.xlsx                 wet-cathode test: cdensity (A/cm²), voltage, tempCout, tempAout, time
├── Polarization curve dry cathode.xlsx     dry-cathode test, same columns
└── Membrane resistance/
    ├── 3A 60C.txt                          EIS spectrum at 3 A, 60 °C (Real, Imaginary columns)
    └── 3A 80C.txt                          EIS spectrum at 3 A, 80 °C
```

The output folder can be changed with `AEM_OUTPUT_DIR`.

## Repository structure

```
AEM_Case_based/
├── aem_model.py       full model: data, fitting, thermal, efficiency, sensitivity, EIS
├── requirements.txt
└── README.md
```

## Author

Mohamed Al-Mandhari · [github.com/Malmand](https://github.com/Malmand)
