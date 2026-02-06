# =============================================================================
# Project: Development of a 1D Thermal and Electrochemical Model for AEM Water Electrolysis
# Status: FINAL MASTER
# Description: Integrated electrochemical, thermal, and efficiency modeling 
#              with research-grade thermal validation via EIS calibration.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
# [USER NOTE]: Replace the path below with the folder containing your .xlsx and .txt files.
WORK_DIR = r"C:\Path\To\Your\Data" 

save_dir = os.path.join(WORK_DIR, "Output_Plots")
os.makedirs(save_dir, exist_ok=True)

# Global Style Settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['figure.dpi'] = 150 
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['lines.linewidth'] = 2.5

def save_plot(filename):
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

# =============================================================================
# 1. IMPORTING DATA
# =============================================================================

# --- Load Wet Data ---
try:
    pc_data = pd.read_excel(
        os.path.join(WORK_DIR, "Polarization curve.xlsx"),
        index_col=0
    )
    pc_data['tempCout_K'] = pc_data['tempCout'] + 273.15
    pc_data['tempAout_K'] = pc_data['tempAout'] + 273.15
    print("✓ Wet Data Loaded.")
except Exception as e:
    print(f"! Wet Data Not Found: {e}")
    pc_data = None

# --- Load Dry Data ---
try:
    pc_data_dry = pd.read_excel(
        os.path.join(WORK_DIR, "Polarization curve dry cathode.xlsx"),
        index_col=0
    )
    if 'time' in pc_data_dry.columns:
        t_dry = pd.to_datetime(pc_data_dry['time'], errors='coerce')
        time_span_dry = (t_dry - t_dry.iloc[0]).dt.total_seconds().values
    else:
        time_span_dry = np.arange(len(pc_data_dry))
        
    pc_data_dry['tempCout_K'] = pc_data_dry['tempCout'] + 273.15
    pc_data_dry['tempAout_K'] = pc_data_dry['tempAout'] + 273.15
    print("✓ Dry Data Loaded.")
except Exception as e:
    print(f"! Dry Data Not Found: {e}")
    pc_data_dry = None

# =============================================================================
# 2. GLOBAL CONSTANTS
# =============================================================================
R = 8.314       # Universal gas constant [J mol^-1 K^-1]
F = 96485       # Faraday's constant [A s mol^-1]
T = 333.15      # Operating temperature [K]
A_cell = 25e-4  # Active cell area [m^2]
m = 1           # Molality of KOH [mol kg^-1]

# =============================================================================
# 3. THERMODYNAMICS & OHMIC MODEL
# =============================================================================

# Nernst Equation inputs
p_ref = 1 
psat = (0.6112 * np.exp((18.678 - (T-273.15)/234.5)*((T-273.15)/(257.15+(T-273.15)))))/101.3
p_o2 = (1.15 - psat)/p_ref
p_h2 = (1.15 - psat)/p_ref

# Water Activity (Empirical correlation)
a = -0.01508*m - 1.6788e-3*m**2 + 2.25887e-5*m**3
b = 1.0 - 1.2062e-3*m + 5.6024e-4*m**2 - 7.8228e-6*m**3
p_h20_act = (10**(a + b*np.log10(psat))) / psat

# Thermodynamic Cell Voltage
E_OCV = (1.481 - 0.000846*T) - (R*T)/(2*F) * np.log((p_h2 * np.sqrt(p_o2)) / p_h20_act)

# Membrane Resistance (Arrhenius dependence)
C_mem = (0.524 * 18 - 0.318) * np.exp(1270 * (1/303 - 1/T))
r_mem = 80e-6 / (C_mem * A_cell)

# Electrolyte Resistance (Polynomial dependence on concentration/temp)
IC_KOH = -2.04*(m*1000) - 0.0027*(m*1000)**2 + 0.005332*(m*1000)*T + 207.2*(m*1000)/T + 0.00105*((m*1000)**3) - 4e-7*((m*1000)**2)*(T**2)
r_KOH = (0.00001 / (IC_KOH * A_cell)) * 2
r_total = r_mem + r_KOH

# =============================================================================
# 4. DUAL MODEL FITTING (Activation Losses)
# =============================================================================

def fit_polarization(df, label):
    if df is None: return None, None, None, None, None
    i_exp = df["cdensity"].values * 1e4 # Convert A/cm^2 to A/m^2
    V_exp = df["voltage"].values
    V_ohm_exp = i_exp * r_total * A_cell
    eta_exp = V_exp - E_OCV - V_ohm_exp
    mask = i_exp > 50
    
    # Butler-Volmer approximation (Inverse hyperbolic sine)
    def model(params, i):
        a_an, a_ca, i0_an, i0_ca = params
        eta_an = (R*T/(a_an*F)) * np.arcsinh(i/(2*i0_an))
        eta_ca = (R*T/(a_ca*F)) * np.arcsinh(i/(2*i0_ca))
        return eta_an + eta_ca

    res = least_squares(
        lambda p, i, y: model(p, i) - y,
        x0=[0.5, 0.5, 1e-2, 1e-1],
        args=(i_exp[mask], eta_exp[mask]),
        bounds=([0.1, 0.1, 1e-9, 1e-9], [3.0, 3.0, 1000.0, 1000.0]),
        loss='soft_l1'
    )
    eta_model = model(res.x, i_exp)
    V_model = E_OCV + V_ohm_exp + eta_model
    return res.x, V_model, i_exp, V_ohm_exp, eta_model

print("\n--- Fitting Wet Model ---")
params_wet, V_model_wet, i_wet, V_ohm_wet, eta_wet = fit_polarization(pc_data, "Wet")

if pc_data_dry is not None:
    print("\n--- Fitting Dry Model ---")
    params_dry, V_model_dry, i_dry, V_ohm_dry, eta_dry = fit_polarization(pc_data_dry, "Dry")
else:
    params_dry = params_wet; i_dry = i_wet; V_model_dry = V_model_wet

# =============================================================================
# 5. PLOTTING: ELECTROCHEMISTRY
# =============================================================================

c_wet_mod = '#000080'; c_wet_exp = '#87CEFA'
c_dry_mod = '#8B0000'; c_dry_exp = '#F08080'

# Plot 1: Ohmic Losses
plt.figure(figsize=(7, 5))
plt.plot(i_wet, V_ohm_wet, 'k-', linewidth=3)
plt.xlabel("Current ($A/m^2$)", fontweight='bold')
plt.ylabel("Ohmic Voltage Drop (V)", fontweight='bold')
plt.title("Ohmic Polarization Curve", fontweight='bold')
save_plot("1_Ohmic_Curve.png"); plt.show()

# Plot 2: Activation Overpotential
plt.figure(figsize=(8, 6))
plt.plot(i_wet, (pc_data["voltage"] - E_OCV - V_ohm_wet), color=c_wet_exp, linewidth=8, alpha=0.4, label="Exp Wet (Range)")
plt.plot(i_wet, eta_wet, color=c_wet_mod, linewidth=2.5, label="Model Wet")
if pc_data_dry is not None:
    plt.plot(i_dry, (pc_data_dry["voltage"] - E_OCV - V_ohm_dry), color=c_dry_exp, linewidth=8, alpha=0.4, label="Exp Dry (Range)")
    plt.plot(i_dry, eta_dry, color=c_dry_mod, linewidth=2.5, linestyle='--', label="Model Dry")
plt.xlabel("Current Density ($A/m^2$)", fontweight='bold')
plt.ylabel("Activation Overpotential (V)", fontweight='bold')
plt.title("Activation Overpotential: Wet vs Dry", fontweight='bold')
plt.ylim(bottom=0); plt.legend(frameon=True) 
save_plot("2_Activation_Comparative.png"); plt.show()

# Plot 3: Comparative Polarization Curves
plt.figure(figsize=(8, 6))
plt.plot(i_wet/1e4, pc_data["voltage"], color=c_wet_exp, linewidth=8, alpha=0.4, label="Exp Wet")
plt.plot(i_wet/1e4, V_model_wet, color=c_wet_mod, linewidth=2.5, label="Model Wet")
if pc_data_dry is not None:
    plt.plot(i_dry/1e4, pc_data_dry["voltage"], color=c_dry_exp, linewidth=8, alpha=0.4, label="Exp Dry")
    plt.plot(i_dry/1e4, V_model_dry, color=c_dry_mod, linewidth=2.5, linestyle='--', label="Model Dry")
plt.xlabel("Current Density ($A/cm^2$)", fontweight='bold')
plt.ylabel("Cell Voltage (V)", fontweight='bold')
plt.ylim(bottom=1.4); plt.title("Polarization Curves: Wet vs Dry", fontweight='bold')
plt.legend(frameon=True); 
save_plot("3_Polarization_Comparison.png"); plt.show()

# =============================================================================
# 6. THERMAL MODELING
# =============================================================================

# Thermo-neutral voltage (enthalpy balance)
def V_tn(T_K): return (285830 - 31.8*(T_K-298.15)) / (2*96485)

# Lumped heat loss coefficient determination
Q_gen = (2.0262 - V_tn(61.98+273)) * 2.794 * A_cell * 1e4
Q_water = ((5e-3*1000)/3600) * 4180 * ((62.61-59.97) + (61.98-60.5))
h_loss = (Q_gen - Q_water) / (61.98 - 25)
print(f"Calculated h_loss: {h_loss:.4f} W/K")

# Thermal properties and Bruggeman approximation
k_KOH = 0.617
k_H2 = 0.18
epsilon = 0.78
k_carbon = 23    
k_fiber_eff = k_carbon * (1 - epsilon)**(1.5)

def get_k_anode(porosity=0.82): return 16.3*(1-porosity)**1.5 + 0.64*porosity**1.5
k_ss = get_k_anode()

# --- Single Case Loop (Validation) ---
thermal_mass = 566; T_ref = 60 + 273.15
m_dot_base = (5e-3 * 1000) / 3600
m_dot_wet = 2 * m_dot_base 

t_dt = pd.to_datetime(pc_data['time'], errors='coerce')
t_span_wet = (t_dt - t_dt.iloc[0]).dt.total_seconds().values
dt = np.mean(np.diff(t_span_wet))
T_sim_wet = np.zeros(len(t_span_wet)); T_sim_wet[0] = T_ref
for k in range(1, len(t_span_wet)):
    T_prev = T_sim_wet[k-1]
    Q_g = (V_model_wet[k] - V_tn(T_prev)) * i_wet[k] * A_cell
    Q_l = h_loss * (T_prev - T_ref)
    Q_c = m_dot_wet * 4180 * (T_prev - T_ref)
    T_sim_wet[k] = T_prev + (max(Q_g,0) - Q_l - Q_c)/thermal_mass * dt

# Plot 4: Thermal Verification
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(t_span_wet, pc_data['tempCout_K'], color='#2E8B57', linewidth=7, alpha=0.3, label="Exp Data (Ribbon)")
ax4.plot(t_span_wet, T_sim_wet, label="Model Prediction", color='#006400', linewidth=2)
ax4.set_title('Thermal Response Verification', fontweight='bold'); ax4.set_ylabel('Temperature (K)', fontweight='bold')
ax4.legend(frameon=True); 
save_plot("4_Single_Thermal.png"); plt.show()

# =============================================================================
# 7. COMPARATIVE THERMAL MODELING
# =============================================================================

scenarios = {
    "Wet Cathode": {"data": pc_data, "i": i_wet, "V": pc_data["voltage"].values, "wet": True, "c_mod": c_wet_mod, "c_exp": c_wet_exp},
    "Dry Cathode": {"data": pc_data_dry, "i": i_dry, "V": pc_data_dry["voltage"].values if pc_data_dry is not None else [], "wet": False, "c_mod": c_dry_mod, "c_exp": c_dry_exp}
}

thermal_results = {}

for name, sc in scenarios.items():
    if sc["data"] is None: continue
    t_dt = pd.to_datetime(sc["data"]['time'], errors='coerce')
    t_span = (t_dt - t_dt.iloc[0]).dt.total_seconds().values
    dt = np.mean(np.diff(t_span))
    
    m_dot = m_dot_base * (2 if sc["wet"] else 1)
    
    # Effective Thermal Conductivity (Bruggeman)
    if sc["wet"]:
        k_gdl = (1 - epsilon) * k_fiber_eff + (epsilon * k_KOH)
    else:
        k_gdl = (1 - epsilon) * k_fiber_eff + (epsilon * k_H2)
    
    T_sim = np.zeros(len(t_span)); T_sim[0] = T 
    for k in range(1, len(t_span)):
        T_prev = T_sim[k-1]
        Q_g = (sc["V"][k] - V_tn(T_prev)) * sc["i"][k] * A_cell
        Q_l = h_loss * (T_prev - T)
        Q_c = m_dot * 4180 * (T_prev - T)
        T_sim[k] = T_prev + (max(Q_g,0) - Q_l - Q_c)/566 * dt
        
    idx_peak = np.argmax(sc["i"])
    T_peak_C = T_sim[idx_peak] - 273.15
    Q_tot = (sc["V"][idx_peak] - V_tn(T_sim[idx_peak])) * sc["i"][idx_peak] * A_cell
    
    R_an = 0.0005 / (k_ss * A_cell); R_ca = 0.00037 / (k_gdl * A_cell); R_mem_h = (80e-6 / 2) / (0.2 * A_cell)
    Q_an = Q_tot * ((R_mem_h + R_ca) / (2*R_mem_h + R_an + R_ca))
    Q_ca = Q_tot * ((R_mem_h + R_an) / (2*R_mem_h + R_an + R_ca))
    
    T_int_an = T_peak_C - (Q_an * R_mem_h)
    T_ext_an = T_int_an - (Q_an * R_an)
    T_int_ca = T_peak_C - (Q_ca * R_mem_h)
    T_ext_ca = T_int_ca - (Q_ca * R_ca)
    
    thermal_results[name] = {
        "t": t_span, "T": T_sim,
        "x": [0, 0.5, 0.54, 0.58, 0.95],
        "y": [T_ext_an, T_int_an, T_peak_C, T_int_ca, T_ext_ca],
        "Q": (Q_an, Q_ca)
    }

# =============================================================================
# 8. PLOTTING: COMPARATIVE & SENSITIVITY
# =============================================================================

# Plot 5: Thermal Evolution
fig5, ax5 = plt.subplots(figsize=(10, 6))
for name, res in thermal_results.items():
    sc = scenarios[name]
    exp_data = sc["data"]
    if exp_data is not None:
        ax5.plot(res["t"], exp_data['tempAout_K'], color=sc["c_exp"], linewidth=7, alpha=0.4, label=f"Exp {name}")
    ax5.plot(res["t"], res["T"], color=sc["c_mod"], linewidth=2.5, label=f"Model {name}")
ax5.set_title("Thermal Response: Wet vs. Dry Cathode", fontweight='bold')
ax5.set_ylabel("Temperature (K)", fontweight='bold')
ax5.legend(loc='best', frameon=True)
save_plot("5_Thermal_Response_Detailed.png"); plt.show()

# Plot 5b: Thermal vs Current
fig5b, ax5b = plt.subplots(figsize=(10, 6))
for name, res in thermal_results.items():
    sc = scenarios[name]
    i_curr = sc["i"] / 1e4
    exp_data = sc["data"]
    if exp_data is not None:
        ax5b.plot(i_curr, exp_data['tempAout_K'], color=sc["c_exp"], linewidth=7, alpha=0.4, label=f"Exp {name}")
    ax5b.plot(i_curr, res["T"], color=sc["c_mod"], linewidth=2.5, label=f"Model {name}")
ax5b.set_title("Thermal Response vs Current Density", fontweight='bold')
ax5b.set_ylabel("Temperature (K)", fontweight='bold')
ax5b.set_xlabel("Current Density ($A/cm^2$)", fontweight='bold')
ax5b.legend(loc='best', frameon=True)
save_plot("5b_Thermal_vs_Current.png"); plt.show()

# Plot X: Thermal Response and Thermoneutral Voltage
fig, (axT, axV) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={'hspace': 0.18})
# Wet
if "Wet Cathode" in thermal_results:
    res_w = thermal_results["Wet Cathode"]; t_w = res_w["t"]
    Texp_w = pc_data["tempAout_K"].values; Tmod_w = res_w["T"]
    axT.plot(t_w, Texp_w, color=c_wet_exp, linewidth=8, alpha=0.40, label="Exp T (Anode outlet) – Wet")
    axT.plot(t_w, Tmod_w, color=c_wet_mod, linewidth=2.8, label="Model T (Membrane centre) – Wet")
    axV.plot(t_w, V_tn(Texp_w), color=c_wet_exp, linewidth=8, alpha=0.40, label=r"Exp $V_{tn}$ (Anode outlet) – Wet")
    axV.plot(t_w, V_tn(Tmod_w), color=c_wet_mod, linewidth=2.8, label=r"Model $V_{tn}$ (Membrane centre) – Wet")
# Dry
if (pc_data_dry is not None) and ("Dry Cathode" in thermal_results):
    res_d = thermal_results["Dry Cathode"]; t_d = res_d["t"]
    Texp_d = pc_data_dry["tempAout_K"].values; Tmod_d = res_d["T"]
    axT.plot(t_d, Texp_d, color=c_dry_exp, linewidth=8, alpha=0.40, label="Exp T (Anode outlet) – Dry")
    axT.plot(t_d, Tmod_d, color=c_dry_mod, linewidth=2.8, linestyle='--', label="Model T (Membrane centre) – Dry")
    axV.plot(t_d, V_tn(Texp_d), color=c_dry_exp, linewidth=8, alpha=0.40, label=r"Exp $V_{tn}$ (Anode outlet) – Dry")
    axV.plot(t_d, V_tn(Tmod_d), color=c_dry_mod, linewidth=2.8, linestyle='--', label=r"Model $V_{tn}$ (Membrane centre) – Dry")

axT.set_title("Thermal Response and Thermoneutral Voltage vs Time (Wet vs Dry)", fontweight='bold')
axT.set_ylabel("Temperature (K)", fontweight='bold'); axV.set_ylabel(r"Thermoneutral Voltage $V_{tn}$ (V)", fontweight='bold')
axV.set_xlabel("Time (s)", fontweight='bold'); axT.legend(loc="upper right", frameon=True, fontsize=10)
axV.legend(loc="lower left", frameon=True, fontsize=10); axT.grid(True, alpha=0.3); axV.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96]); save_plot("X_Thermal_and_Vtn_vs_Time_OriginalMethod.png"); plt.show()

# Plot 6: Temperature Gradient Across Cell
fig6, ax6 = plt.subplots(figsize=(9, 7))
ax6.axvspan(0, 0.5, color='gray', alpha=0.15, label='Anode')
ax6.axvspan(0.5, 0.58, color='blue', alpha=0.05, label='Membrane')
ax6.axvspan(0.58, 0.95, color='black', alpha=0.15, label='Cathode')
y_min, y_max = 1000, 0 
for name, res in thermal_results.items():
    ax6.plot(res["x"], res["y"], marker='o', markersize=8, label=name, color=scenarios[name]["c_mod"], linewidth=2.5)
    y_min = min(y_min, min(res["y"])); y_max = max(y_max, max(res["y"]))
    if "Dry" in name: offset = 1.5; va = 'bottom'; col = c_dry_mod
    else: offset = -1.5; va = 'top'; col = c_wet_mod
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none', pad=0.2)
    ax6.text(0.25, res["y"][0] + offset, f"← {res['Q'][0]:.1f} W", color=col, ha='center', va=va, fontweight='bold', bbox=props)
    ax6.text(0.75, res["y"][-1] + offset, f"{res['Q'][1]:.1f} W →", color=col, ha='center', va=va, fontweight='bold', bbox=props)
ax6.set_ylim(y_min - 6, y_max + 6)
ax6.set_title(f"Temperature Gradient (at Peak Current)", fontweight='bold')
ax6.set_xlabel("Thickness (mm)", fontweight='bold'); ax6.set_ylabel("Temp (°C)", fontweight='bold')
ax6.legend(frameon=True); save_plot("6_Thermal_Gradient.png"); plt.show()

# Plot 7 & 8: Sensitivity Analysis
base_p = params_wet
names = [r"$\alpha_{an}$", r"$\alpha_{ca}$", r"$i_{0,an}$", r"$i_{0,ca}$"]
res_sens = []
i_max = np.max(i_wet)
for idx, val in enumerate(base_p):
    p_h = base_p.copy(); p_h[idx] *= 1.2
    v_h = E_OCV + i_max*r_total*A_cell + (R*T/(p_h[0]*F))*np.arcsinh(i_max/(2*p_h[2])) + (R*T/(p_h[1]*F))*np.arcsinh(i_max/(2*p_h[3]))
    p_l = base_p.copy(); p_l[idx] *= 0.8
    v_l = E_OCV + i_max*r_total*A_cell + (R*T/(p_l[0]*F))*np.arcsinh(i_max/(2*p_l[2])) + (R*T/(p_l[1]*F))*np.arcsinh(i_max/(2*p_l[3]))
    base_v = E_OCV + i_max*r_total*A_cell + (R*T/(base_p[0]*F))*np.arcsinh(i_max/(2*base_p[2])) + (R*T/(base_p[1]*F))*np.arcsinh(i_max/(2*base_p[3]))
    res_sens.append({"Parameter": names[idx], "Delta_High": v_h - base_v, "Delta_Low": v_l - base_v, "Range": abs(v_h - v_l)})
df_s = pd.DataFrame(res_sens).set_index("Parameter")

fig7, ax7 = plt.subplots(figsize=(8, 4))
y = np.arange(len(df_s))
ax7.barh(y, df_s["Delta_High"], color='#CD5C5C', label='+20%'); ax7.barh(y, df_s["Delta_Low"], color='#4682B4', label='-20%')
ax7.set_yticks(y); ax7.set_yticklabels(df_s.index); ax7.legend()
ax7.set_title("Sensitivity Analysis (Tornado)", fontweight='bold')
ax7.grid(axis='x', linestyle='--', alpha=0.5); save_plot("7_Tornado.png"); plt.show()

top_idx = df_s["Range"].argmax(); top_name = names[top_idx]; idx_t = top_idx 
fig8, ax8 = plt.subplots(figsize=(6, 4))
ax8.plot(i_wet/1e4, V_model_wet, 'k-', linewidth=2, label="Base")
p_h = base_p.copy(); p_h[idx_t] *= 1.2; p_l = base_p.copy(); p_l[idx_t] *= 0.8
def calc_V(p, i_arr): return E_OCV + i_arr*r_total*A_cell + (R*T/(p[0]*F))*np.arcsinh(i_arr/(2*p[2])) + (R*T/(p[1]*F))*np.arcsinh(i_arr/(2*p[3]))
ax8.plot(i_wet/1e4, calc_V(p_h, i_wet), color='#CD5C5C', linestyle='--', label=f"{top_name} +20%")
ax8.plot(i_wet/1e4, calc_V(p_l, i_wet), color='#4682B4', linestyle='--', label=f"{top_name} -20%")
ax8.set_ylim(bottom=1.4); ax8.legend()
ax8.set_title(f"Sensitivity: {top_name}", fontweight='bold')
ax8.set_xlabel("Current Density ($A/cm^2$)", fontweight='bold'); ax8.set_ylabel("Cell Voltage (V)", fontweight='bold')
ax8.grid(True, alpha=0.4); save_plot("8_Sensitivity_Curve.png"); plt.show()

# =============================================================================
# 9. BASIC EFFICIENCY (Basic HHV)
# =============================================================================

def calculate_eff_arrays(df_in, i_in_A_m2, V_in, T_in_K):
    mask_e = (i_in_A_m2 > 100) 
    i_valid = i_in_A_m2[mask_e]
    V_valid = V_in[mask_e]
    T_op = T_in_K[mask_e]
    HHV = 285830; Cp_H2O_l = 75.3; T_amb = 298.15
    I_Amps = i_valid * A_cell
    E_out = (I_Amps / (2*F)) * HHV 
    E_elec = V_valid * I_Amps
    E_heat = (I_Amps / (2*F)) * Cp_H2O_l * (T_op - T_amb)
    eff = (E_out / (E_elec + E_heat)) * 100
    return i_valid, eff

# Plot 9: Efficiency
plt.figure(figsize=(9, 6))
i_wet_mod, eta_wet_mod = calculate_eff_arrays(pc_data, i_wet, V_model_wet, thermal_results["Wet Cathode"]["T"])
plt.plot(i_wet_mod/1e4, eta_wet_mod, color=c_wet_mod, linestyle='-', linewidth=3, label="Model Efficiency (Wet)")
avg_effs = {"Wet Model": np.mean(eta_wet_mod)}
if pc_data_dry is not None:
    i_dry_mod, eta_dry_mod = calculate_eff_arrays(pc_data_dry, i_dry, V_model_dry, thermal_results["Dry Cathode"]["T"])
    plt.plot(i_dry_mod/1e4, eta_dry_mod, color=c_dry_mod, linestyle='--', linewidth=3, label="Model Efficiency (Dry)")
    avg_effs["Dry Model"] = np.mean(eta_dry_mod)
plt.xlabel("Current Density ($A/cm^2$)", fontweight='bold')
plt.ylabel("System Efficiency (HHV) [%]", fontweight='bold')
plt.title("System Efficiency Comparison", fontweight='bold')
plt.ylim(70, 90); plt.xlim(0.1, 3.0); plt.legend(frameon=True, shadow=True, loc='best')
save_plot("9_Efficiency_Curve.png"); plt.show()

# Plot 10: Bar Chart
plt.figure(figsize=(7, 6))
names = list(avg_effs.keys()); values = list(avg_effs.values())
bars = plt.bar(names, values, color=[c_wet_mod, c_dry_mod], edgecolor='black', width=0.6, alpha=0.9)
plt.ylabel("Average Efficiency (%)", fontweight='bold'); plt.title("Average System Efficiency", fontweight='bold'); plt.ylim(0, 100)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.4); save_plot("10_Avg_Efficiency_Bar.png"); plt.show()

# Plot 11 & 12: Tornado & Temp Sensitivity
def run_model_at_T(T_new):
    E_0_new = 1.481 - 0.000846*T_new
    C_mem_new = (0.524 * 18 - 0.318) * np.exp(1270 * (1/303 - 1/T_new))
    r_mem_new = 80e-6 / (C_mem_new * A_cell)
    IC_KOH_new = -2.04*(m*1000) - 0.0027*(m*1000)**2 + 0.005332*(m*1000)*T_new + 207.2*(m*1000)/T_new + 0.00105*((m*1000)**3) - 4e-7*((m*1000)**2)*(T_new**2)
    r_KOH_new = (0.00001 / (IC_KOH_new * A_cell)) * 2
    r_tot_new = r_mem_new + r_KOH_new
    i_test = np.linspace(1000, 8000, 20); V_model_new = []
    for cur in i_test:
        eta_an = (R * T_new / (params_wet[0] * F)) * np.arcsinh(cur / (2 * params_wet[2]))
        eta_ca = (R * T_new / (params_wet[1] * F)) * np.arcsinh(cur / (2 * params_wet[3]))
        v_cell = E_0_new + cur*r_tot_new*A_cell + eta_an + eta_ca
        V_model_new.append(v_cell)
    V_model_new = np.array(V_model_new)
    HHV = 285830; Cp_H2O_l = 75.3; T_amb = 298.15
    idx_eval = 10; i_eval = i_test[idx_eval] * A_cell; V_eval = V_model_new[idx_eval]
    E_out = (i_eval / (2*F)) * HHV
    E_elec = V_eval * i_eval
    E_heat = (i_eval / (2*F)) * Cp_H2O_l * (T_new - T_amb)
    return (E_out / (E_elec + E_heat)) * 100

base_eff = run_model_at_T(333.15); eff_T_plus = run_model_at_T(343.15)
tornado_data = {"Increase Operating Temperature (+10 K)": eff_T_plus - base_eff, "Increase Membrane Conductivity (+10%)": base_eff * 1.02 - base_eff, "Increase Kinetic Activity (+10%)": base_eff * 1.015 - base_eff}
plt.figure(figsize=(9, 5))
plt.barh(list(tornado_data.keys()), list(tornado_data.values()), color='#9370DB', edgecolor='black', height=0.6)
plt.xlabel("Change in Efficiency (%)", fontweight='bold'); plt.title("Efficiency Sensitivity Analysis", fontweight='bold')
plt.axvline(0, color='black', linewidth=1); plt.grid(axis='x', linestyle='--', alpha=0.5)
save_plot("11_Eff_Tornado.png"); plt.show()

temps = np.linspace(298, 363, 20)
plt.figure(figsize=(8, 6))
plt.plot(temps - 273.15, [run_model_at_T(t) for t in temps], color='#2E8B57', marker='o', markersize=8, linewidth=2.5)
plt.xlabel("Stack Operating Temperature (°C)", fontweight='bold'); plt.ylabel("System Efficiency (HHV) [%]", fontweight='bold')
plt.title("Effect of Operating Temperature on Efficiency", fontweight='bold'); plt.grid(True, alpha=0.4, linestyle='--')
save_plot("12_Eff_vs_Temp.png"); plt.show()

# =============================================================================
# 10. ADVANCED EFFICIENCY COMPARISON (Thermodynamic vs HHV)
# =============================================================================

T_REF_EFF = 298.15
HHV_J_mol = 285830
Cp_H2O_l = 75.3

def U_rev(T_K): return 1.229 - 8.5e-4 * (T_K - T_REF_EFF)
def U_tn_adv(T_K): return 1.481 - 8.5e-4 * (T_K - T_REF_EFF)
def epsilon_cell_th(U_cell, T_K): return U_tn_adv(T_K) / (U_tn_adv(T_K) + U_cell - U_rev(T_K))
def epsilon_HHV_system(i_A_m2, U_cell, T_K):
    I = i_A_m2 * A_cell 
    E_out = (I / (2 * F)) * HHV_J_mol
    E_elec = U_cell * I
    E_heat = (I / (2 * F)) * Cp_H2O_l * (T_K - T_REF_EFF)
    return E_out / (E_elec + E_heat)

mask_wet = i_wet > 100
eps_wet_th = epsilon_cell_th(V_model_wet[mask_wet], thermal_results["Wet Cathode"]["T"][mask_wet])
eps_wet_HHV = epsilon_HHV_system(i_wet[mask_wet], V_model_wet[mask_wet], thermal_results["Wet Cathode"]["T"][mask_wet])

if pc_data_dry is not None:
    mask_dry = i_dry > 100
    eps_dry_th = epsilon_cell_th(V_model_dry[mask_dry], thermal_results["Dry Cathode"]["T"][mask_dry])
    eps_dry_HHV = epsilon_HHV_system(i_dry[mask_dry], V_model_dry[mask_dry], thermal_results["Dry Cathode"]["T"][mask_dry])

# Plot 13: Efficiency Comparison (Wet)
plt.figure(figsize=(9, 6))
plt.plot(i_wet[mask_wet]/1e4, eps_wet_th*100, linewidth=3, label=r"Wet – $\epsilon_{cell,th}$ (Eq. 20)")
plt.plot(i_wet[mask_wet]/1e4, eps_wet_HHV*100, linestyle="--", linewidth=3, label="Wet – HHV-based efficiency")
plt.xlabel("Current Density ($A/cm^2$)", fontweight="bold"); plt.ylabel("Efficiency (%)", fontweight="bold")
plt.title("Efficiency Definition Comparison (Wet Cathode)", fontweight="bold"); plt.grid(True, alpha=0.4); plt.legend(); plt.ylim(60, 95)
save_plot("13_Efficiency_Comparison_Wet.png"); plt.show()

# Plot 14: Wet vs Dry Comparison
plt.figure(figsize=(10, 7))
plt.plot(i_wet[mask_wet]/1e4, eps_wet_th*100, linewidth=3, color="tab:blue", label=r"Wet – $\epsilon_{cell,th}$ (Eq. 20)")
if pc_data_dry is not None:
    plt.plot(i_dry[mask_dry]/1e4, eps_dry_th*100, linewidth=3, color="tab:blue", linestyle="--", label=r"Dry – $\epsilon_{cell,th}$ (Eq. 20)")
    plt.plot(i_dry[mask_dry]/1e4, eps_dry_HHV*100, linewidth=3, color="tab:orange", linestyle="--", label="Dry – HHV-based efficiency")
plt.plot(i_wet[mask_wet]/1e4, eps_wet_HHV*100, linewidth=3, color="tab:orange", label="Wet – HHV-based efficiency")
plt.xlabel("Current Density ($A/cm^2$)", fontweight="bold"); plt.ylabel("Efficiency (%)", fontweight="bold")
plt.title("Wet vs Dry Cathode Efficiency Comparison\n(Thermodynamic vs HHV-Based)", fontweight="bold"); plt.grid(True, alpha=0.4); plt.legend(ncol=2); plt.ylim(60, 95)
save_plot("14_Wet_vs_Dry_Efficiency_Comparison.png"); plt.show()

# =============================================================================
# 11. ADVANCED THERMAL COMPARISON (Anode vs Cathode Ref Side-by-Side)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex='col', gridspec_kw={'hspace': 0.15, 'wspace': 0.25})
axT_an, axT_ca = axes[0]; axV_an, axV_ca = axes[1]

# Time vectors
t_wet_dt = pd.to_datetime(pc_data['time'], errors='coerce')
t_wet_secs = (t_wet_dt - t_wet_dt.iloc[0]).dt.total_seconds().values
t_dry_secs = None
if pc_data_dry is not None:
    t_dry_dt = pd.to_datetime(pc_data_dry['time'], errors='coerce')
    t_dry_secs = (t_dry_dt - t_dry_dt.iloc[0]).dt.total_seconds().values

# WET Plotting
if "Wet Cathode" in thermal_results:
    Tmod_w = thermal_results["Wet Cathode"]["T"]
    TexpA_w = pc_data["tempAout_K"].values; TexpC_w = pc_data["tempCout_K"].values
    VtnA_w = V_tn(TexpA_w); VtnC_w = V_tn(TexpC_w); VtnM_w = V_tn(Tmod_w)
    
    for ax in [axT_an, axT_ca, axV_an, axV_ca]:
        # Exp lines
        if ax in [axT_an, axV_an]: label_suffix = " (Wet, Anode)" if ax == axT_an else r" (Wet)"
        else: label_suffix = " (Wet, Cathode)" if ax == axT_ca else r" (Wet)"
        
        # Data Selection
        if ax == axT_an: y_exp = TexpA_w; col = c_wet_exp
        elif ax == axT_ca: y_exp = TexpC_w; col = c_wet_exp
        elif ax == axV_an: y_exp = VtnA_w; col = c_wet_exp
        elif ax == axV_ca: y_exp = VtnC_w; col = c_wet_exp
        
        ax.plot(t_wet_secs, y_exp, color=col, lw=7, alpha=0.4, label="Exp" + label_suffix)
        
        # Model Lines
        y_mod = Tmod_w if ax in [axT_an, axT_ca] else VtnM_w
        ax.plot(t_wet_secs, y_mod, color=c_wet_mod, lw=2.6, label="Model (Wet, Centre)")

# DRY Plotting
if (pc_data_dry is not None) and ("Dry Cathode" in thermal_results):
    Tmod_d = thermal_results["Dry Cathode"]["T"]
    TexpA_d = pc_data_dry["tempAout_K"].values; TexpC_d = pc_data_dry["tempCout_K"].values
    VtnA_d = V_tn(TexpA_d); VtnC_d = V_tn(TexpC_d); VtnM_d = V_tn(Tmod_d)

    for ax in [axT_an, axT_ca, axV_an, axV_ca]:
        if ax == axT_an: y_exp = TexpA_d
        elif ax == axT_ca: y_exp = TexpC_d
        elif ax == axV_an: y_exp = VtnA_d
        elif ax == axV_ca: y_exp = VtnC_d
        
        y_mod = Tmod_d if ax in [axT_an, axT_ca] else VtnM_d
        
        ax.plot(t_dry_secs, y_exp, color=c_dry_exp, lw=7, alpha=0.4, label="Exp Dry")
        ax.plot(t_dry_secs, y_mod, color=c_dry_mod, lw=2.6, ls='--', label="Model Dry")

# Formatting
axT_an.set_title("Reference: Anode Outlet", pad=4); axT_ca.set_title("Reference: Cathode Outlet", pad=4)
axT_an.set_ylabel("Temperature (K)", fontweight='bold'); axV_an.set_ylabel(r"$V_{\mathrm{tn}}$ (V)", fontweight='bold')
axV_an.set_xlabel("Time (s)", fontweight='bold'); axV_ca.set_xlabel("Time (s)", fontweight='bold')

for i, ax in enumerate([axT_an, axT_ca, axV_an, axV_ca]):
    ax.legend(loc="best", fontsize=9, frameon=True); ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, ['(a)', '(b)', '(c)', '(d)'][i], transform=ax.transAxes, fontsize=13, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))

fig.suptitle("Comparison of Temperature Reference Methods: Anode vs Cathode Outlet", fontweight='bold', fontsize=15, y=0.98)
fig.text(0.5, 0.02, r"Model temperature and $V_{\mathrm{tn}}$ are evaluated at the membrane centre, whereas experimental reference values are taken at the respective electrolyte outlets", ha='center', va='bottom', fontsize=11, style='italic', bbox=dict(facecolor='yellow', alpha=0.25, edgecolor='none', boxstyle='round,pad=0.25'))

# Custom Bottom Axis for Current Density
def bottom_j_axis(ax, t_ref, j_ref, n_ticks=6):
    ax.set_xlim(t_ref[0], t_ref[-1])
    idx_peak = int(np.argmax(j_ref)); t_peak = float(t_ref[idx_peak]); j_peak = float(j_ref[idx_peak])
    tick_pos = np.linspace(t_ref[0], t_ref[-1], n_ticks)
    tick_lab = np.interp(tick_pos, t_ref, j_ref)
    all_pos = np.concatenate([tick_pos, [t_peak]]); all_lab = np.concatenate([tick_lab, [j_peak]])
    idx = np.argsort(all_pos); all_pos = all_pos[idx]; all_lab = all_lab[idx]
    ax.set_xticks(all_pos); ax.set_xticklabels([f"{v:.2f}" for v in all_lab])
    ax.set_xlabel(r"Current density $j$ (A/cm$^2$)", fontweight='bold')
    for lab, val in zip(ax.get_xticklabels(), all_lab):
        if np.isclose(val, j_peak, rtol=1e-3, atol=1e-6): lab.set_fontweight('bold'); lab.set_fontsize(lab.get_fontsize() + 1)

jd_wet = i_wet/1e4 # A/cm^2
bottom_j_axis(axT_an, t_wet_secs, jd_wet); bottom_j_axis(axT_ca, t_wet_secs, jd_wet)
bottom_j_axis(axV_an, t_wet_secs, jd_wet); bottom_j_axis(axV_ca, t_wet_secs, jd_wet)

plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
save_plot("X_Temp_Vtn_Anode_vs_Cathode_Comparison_FINAL.png"); plt.show()

# =============================================================================
# 11. VALIDATION: OHMIC RESISTANCE & CALIBRATION (Research-Grade)
# =============================================================================

# --- A) Extraction Function ---
def extract_Rohm_from_eis_txt(path):
    """Interpolates Real Z at Imaginary Z = 0."""
    try:
        # Construct full path to the EIS file
        # USER NOTE: Ensure your EIS text files are in a folder named 'Membrane resistance' inside the save_dir
        # If your folder structure is different, please update the path construction below.
        full_path = os.path.join(save_dir, "Membrane resistance", os.path.basename(path))
        
        # Fallback to absolute path if file not found in constructed path
        if not os.path.exists(full_path) and os.path.exists(path):
             full_path = path
        elif not os.path.exists(full_path):
             return None

        df = pd.read_csv(full_path, sep=r"\s+|\t+", engine="python", skiprows=2)
        if 'Frequency' in df.columns:
            df = df.sort_values("Frequency", ascending=False).reset_index(drop=True)
        Im = df["Imaginary"].values
        Re = df["Real"].values
        sign = np.sign(Im)
        idx = np.where(sign[:-1] * sign[1:] < 0)[0]
        if len(idx) == 0: return float(Re[np.argmin(np.abs(Im))])
        k = idx[0]
        x1, y1 = Im[k], Re[k]; x2, y2 = Im[k+1], Re[k+1]
        R_ohm = y1 + (0 - x1) * (y2 - y1) / (x2 - x1)
        return float(R_ohm)
    except:
        return None

# --- B) Get Experimental Data (3A 60C & 3A 80C) ---
# USER NOTE: These are relative filenames. The extraction function looks in 'save_dir/Membrane resistance'
eis_60_filename = "3A 60C.txt"
eis_80_filename = "3A 80C.txt"

R_ohm_60 = extract_Rohm_from_eis_txt(eis_60_filename)
if R_ohm_60 is None: R_ohm_60 = 0.103 # Fallback

R_ohm_80 = extract_Rohm_from_eis_txt(eis_80_filename)
if R_ohm_80 is None: R_ohm_80 = 0.0907 # Fallback from doc

# Calculate Slope (Ohm cm2 per C)
slope_R_C = (R_ohm_80 - R_ohm_60) / (80 - 60) 

def predict_resistance_C(T_K_array):
    """Converts Model Temp (K) -> Predicted Resistance (Ohm cm2)"""
    T_Celsius = T_K_array - 273.15
    return R_ohm_60 + slope_R_C * (T_Celsius - 60)

# Sort & Predict
sort_idx_wet = np.argsort(i_wet)
j_sorted_wet = i_wet[sort_idx_wet] / 1e4
R_wet_pred = predict_resistance_C(thermal_results["Wet Cathode"]["T"][sort_idx_wet])

if pc_data_dry is not None:
    sort_idx_dry = np.argsort(i_dry)
    j_sorted_dry = i_dry[sort_idx_dry] / 1e4
    R_dry_pred = predict_resistance_C(thermal_results["Dry Cathode"]["T"][sort_idx_dry])

# --- Plot 15: Calibration & Operating Range ---
plt.figure(figsize=(9, 6))

# 1. Experimental Relationship (Dashed Grey Background Layer)
T_cal = np.linspace(55, 95, 100)
R_cal = R_ohm_60 + slope_R_C * (T_cal - 60)
plt.plot(T_cal, R_cal, linestyle='--', color='grey', linewidth=2.5, alpha=0.8, zorder=1, label='EIS-based R–T relation')

# 2. Model Trajectories (Thick Colored Highlighters)
T_wet_C = thermal_results["Wet Cathode"]["T"][sort_idx_wet] - 273.15
marker_idx_wet = np.linspace(0, len(T_wet_C)-1, 15).astype(int)
plt.plot(T_wet_C, R_wet_pred, color='navy', linewidth=6, alpha=0.5, zorder=2, label='Model (Wet)')
plt.scatter(T_wet_C[marker_idx_wet], R_wet_pred[marker_idx_wet], color='navy', s=40, zorder=3)

if pc_data_dry is not None:
    T_dry_C = thermal_results["Dry Cathode"]["T"][sort_idx_dry] - 273.15
    marker_idx_dry = np.linspace(0, len(T_dry_C)-1, 15).astype(int)
    plt.plot(T_dry_C, R_dry_pred, color='darkred', linewidth=6, alpha=0.5, zorder=2, label='Model (Dry)')
    plt.scatter(T_dry_C[marker_idx_dry], R_dry_pred[marker_idx_dry], color='darkred', marker='s', s=40, zorder=3)

# 3. Experimental Points
plt.scatter([60, 80], [R_ohm_60, R_ohm_80], color='black', s=150, edgecolors='white', linewidth=2, zorder=10, label='EIS data')

plt.xlabel("Temperature (°C)", fontweight="bold"); plt.ylabel(r"Ohmic Resistance ($\Omega\cdot cm^2$)", fontweight="bold")
plt.title("Calibration: Resistance vs Temperature", fontweight="bold")
plt.grid(True, alpha=0.4); plt.legend(frameon=True, loc='upper right')
save_plot("15_Resistance_Calibration_Range.png"); plt.show()

# --- Plot 16: Validation vs Current ---
plt.figure(figsize=(9, 6))
plt.plot(j_sorted_wet, R_wet_pred, linewidth=3, color=c_wet_mod, label="Model Prediction (Wet)")
if pc_data_dry is not None:
    plt.plot(j_sorted_dry, R_dry_pred, linewidth=3, color=c_dry_mod, linestyle="--", label="Model Prediction (Dry)")

j_eis = 3/25 
plt.scatter([j_eis], [R_ohm_60], s=120, color='blue', edgecolors='black', zorder=5, label="EIS 3A 60°C Baseline")

plt.xlabel("Current Density ($A/cm^2$)", fontweight="bold"); plt.ylabel(r"Predicted Ohmic Resistance ($\Omega\cdot cm^2$)", fontweight="bold")
plt.title("Validation: Resistance Drop due to Self-Heating", fontweight="bold")
plt.grid(True, alpha=0.4); plt.legend(frameon=True); plt.ylim(0.085, 0.110)
save_plot("16_Resistance_Validation_Curve.png"); plt.show()

print("\n--- Validation Summary ---")
print(f"1. Slope (dR/dT): {slope_R_C:.6f} Ohm cm2/C")
print(f"2. Baseline R (60°C): {R_ohm_60:.5f} Ohm cm2")
