# Generated from: Modeling of AEM Electrolyser
# Status: FINAL MASTER (Consistent Bold Styling + LaTeX Symbols + Highlighter Plots)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
save_dir = r"D:\UL Study Materials\Case Based Module\Experimental Data\Good Simulation Files\All Pictures"
os.makedirs(save_dir, exist_ok=True)

# Global Style Settings for Professional Publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
# Note: Specific bolding is applied manually to axes for maximum control

def save_plot(filename):
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

# =============================================================================
# 1. IMPORTING DATA
# =============================================================================

# --- Load Wet Data ---
pc_data = pd.read_excel(
    r"c:\Users\ASUS\Downloads\Polarization curve.xlsx",
    index_col=0
)
pc_data['tempCout_K'] = pc_data['tempCout'] + 273.15
pc_data['tempAout_K'] = pc_data['tempAout'] + 273.15

# --- Load Dry Data ---
try:
    pc_data_dry = pd.read_excel(
        r"C:\Users\ASUS\Downloads\Polarization curve dry cathode.xlsx",
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
except:
    print("! Dry Data Not Found.")
    pc_data_dry = None

# Constants
R = 8.314; F = 96485; T = 333.15


# =============================================================================
# 2. THERMODYNAMICS & OHMIC (Exact Original Equations)
# =============================================================================

p_ref = 1 
psat = (0.6112 * np.exp((18.678 - (T-273.15)/234.5)*((T-273.15)/(257.15+(T-273.15)))))/101.3
p_o2 = (1.15 - psat)/p_ref; p_h2 = (1.15 - psat)/p_ref
m = 1
a = -0.01508*m - 1.6788e-3*m**2 + 2.25887e-5*m**3
b = 1.0 - 1.2062e-3*m + 5.6024e-4*m**2 - 7.8228e-6*m**3
p_h20_act = (10**(a + b*np.log10(psat))) / psat
E_OCV = (1.481 - 0.000846*T) - (R*T)/(2*F) * np.log((p_h2 * np.sqrt(p_o2)) / p_h20_act)

C_mem = (0.524 * 18 - 0.318) * np.exp(1270 * (1/303 - 1/T))
r_mem = 80e-6 / (C_mem * 25e-4)
IC_KOH = -2.04*(m*1000) - 0.0027*(m*1000)**2 + 0.005332*(m*1000)*T + 207.2*(m*1000)/T + 0.00105*((m*1000)**3) - 4e-7*((m*1000)**2)*(T**2)
r_KOH = (0.00001 / (IC_KOH * 25e-4)) * 2
r_total = r_mem + r_KOH

# =============================================================================
# 3. DUAL MODEL FITTING
# =============================================================================

def fit_polarization(df, label):
    i_exp = df["cdensity"].values * 1e4
    V_exp = df["voltage"].values
    V_ohm_exp = i_exp * r_total * 25e-4
    eta_exp = V_exp - E_OCV - V_ohm_exp
    mask = i_exp > 50
    
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
# 4. PLOTTING: ELECTROCHEMISTRY ("Highlighter" Style)
# =============================================================================

# Colors
c_wet_mod = '#000080' # Navy
c_wet_exp = '#87CEFA' # Light Sky Blue
c_dry_mod = '#8B0000' # Dark Red
c_dry_exp = '#F08080' # Light Coral

# Plot 1: Ohmic
plt.figure(figsize=(7, 5))
plt.plot(i_wet, V_ohm_wet, 'k-', linewidth=3)
plt.xlabel("Current ($A/m^2$)", fontweight='bold')
plt.ylabel("Ohmic Voltage Drop (V)", fontweight='bold')
plt.title("Ohmic Polarization Curve", fontweight='bold')
save_plot("1_Ohmic_Curve.png"); plt.show()

# Plot 2: Activation
plt.figure(figsize=(8, 6))
# Wet
plt.plot(i_wet, (pc_data["voltage"] - E_OCV - V_ohm_wet), color=c_wet_exp, linewidth=8, alpha=0.4, label="Exp Wet (Range)")
plt.plot(i_wet, eta_wet, color=c_wet_mod, linewidth=2.5, label="Model Wet")
# Dry
if pc_data_dry is not None:
    plt.plot(i_dry, (pc_data_dry["voltage"] - E_OCV - V_ohm_dry), color=c_dry_exp, linewidth=8, alpha=0.4, label="Exp Dry (Range)")
    plt.plot(i_dry, eta_dry, color=c_dry_mod, linewidth=2.5, linestyle='--', label="Model Dry")
plt.xlabel("Current Density ($A/m^2$)", fontweight='bold')
plt.ylabel("Activation Overpotential (V)", fontweight='bold')
plt.title("Activation Overpotential: Wet vs Dry", fontweight='bold')
plt.ylim(bottom=0); plt.legend(frameon=True) 
save_plot("2_Activation_Comparative.png"); plt.show()

# Plot 3: Comparative Polarization
plt.figure(figsize=(8, 6))
# Wet
plt.plot(i_wet/1e4, pc_data["voltage"], color=c_wet_exp, linewidth=8, alpha=0.4, label="Exp Wet")
plt.plot(i_wet/1e4, V_model_wet, color=c_wet_mod, linewidth=2.5, label="Model Wet")
# Dry
if pc_data_dry is not None:
    plt.plot(i_dry/1e4, pc_data_dry["voltage"], color=c_dry_exp, linewidth=8, alpha=0.4, label="Exp Dry")
    plt.plot(i_dry/1e4, V_model_dry, color=c_dry_mod, linewidth=2.5, linestyle='--', label="Model Dry")
plt.xlabel("Current Density ($A/cm^2$)", fontweight='bold')
plt.ylabel("Cell Voltage (V)", fontweight='bold')
plt.ylim(bottom=1.4); plt.title("Polarization Curves: Wet vs Dry", fontweight='bold')
plt.legend(frameon=True); 
save_plot("3_Polarization_Comparison.png"); plt.show()

# =============================================================================
# 5. THERMAL MODELING
# =============================================================================

def V_tn(T_K): return (285830 - 31.8*(T_K-298.15)) / (2*96485)

Q_gen = (2.0262 - V_tn(61.98+273)) * 2.794 * 25e-4 * 1e4
Q_water = ((5e-3*1000)/3600) * 4180 * ((62.61-59.97) + (61.98-60.5))
h_loss = (Q_gen - Q_water) / (61.98 - 25)
print(f"Calculated h_loss: {h_loss:.4f} W/K")


k_KOH = 0.617
k_H2 = 0.18
epsilon = 0.78
k_carbon = 23    
k_fiber_eff = k_carbon * (1 - epsilon)**(1.5)

# This is the ANODE side function
def get_k_anode(porosity=0.82): return 16.3*(1-porosity)**1.5 + 0.64*porosity**1.5
k_ss = get_k_anode()

# --- Single Case Loop ---
thermal_mass = 566; T_ref = 60 + 273.15
m_dot_base = (5e-3 * 1000) / 3600
m_dot_wet = 2 * m_dot_base 

t_dt = pd.to_datetime(pc_data['time'], errors='coerce')
t_span_wet = (t_dt - t_dt.iloc[0]).dt.total_seconds().values
dt = np.mean(np.diff(t_span_wet))
T_sim_wet = np.zeros(len(t_span_wet)); T_sim_wet[0] = T_ref
for k in range(1, len(t_span_wet)):
    T_prev = T_sim_wet[k-1]
    Q_g = (V_model_wet[k] - V_tn(T_prev)) * i_wet[k] * 25e-4
    Q_l = h_loss * (T_prev - T_ref)
    Q_c = m_dot_wet * 4180 * (T_prev - T_ref)
    T_sim_wet[k] = T_prev + (max(Q_g,0) - Q_l - Q_c)/thermal_mass * dt

# Plot 4
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(t_span_wet, pc_data['tempCout_K'], color='#2E8B57', linewidth=7, alpha=0.3, label="Exp Data (Ribbon)")
ax4.plot(t_span_wet, T_sim_wet, label="Model Prediction", color='#006400', linewidth=2)
ax4.set_title('Thermal Response Verification', fontweight='bold'); ax4.set_ylabel('Temperature (K)', fontweight='bold')
ax4.legend(frameon=True); 
save_plot("4_Single_Thermal.png"); plt.show()

# =============================================================================
# 6. COMPARATIVE THERMAL MODELING (Anode Constant, Cathode Varies)
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
    
    # --- CATHODE-ONLY LOGIC ---
    if sc["wet"]:
        k_gdl = (1 - epsilon) * k_fiber_eff + (epsilon * k_KOH)
    else:
        k_gdl = (1 - epsilon) * k_fiber_eff + (epsilon * k_H2)
    # --------------------------
    
    T_sim = np.zeros(len(t_span)); T_sim[0] = T 
    for k in range(1, len(t_span)):
        T_prev = T_sim[k-1]
        Q_g = (sc["V"][k] - V_tn(T_prev)) * sc["i"][k] * 25e-4
        Q_l = h_loss * (T_prev - T)
        Q_c = m_dot * 4180 * (T_prev - T)
        T_sim[k] = T_prev + (max(Q_g,0) - Q_l - Q_c)/566 * dt
        
    idx_peak = np.argmax(sc["i"])
    T_peak_C = T_sim[idx_peak] - 273.15
    Q_tot = (sc["V"][idx_peak] - V_tn(T_sim[idx_peak])) * sc["i"][idx_peak] * 25e-4
    
    R_an = 0.0005 / (k_ss * 25e-4); R_ca = 0.00037 / (k_gdl * 25e-4); R_mem_h = (80e-6 / 2) / (0.2 * 25e-4)
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
# 7. PLOTTING: COMPARATIVE & SENSITIVITY
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

# Plot 6: Thermal Gradient (PERFECTED LABELS - NO OVERLAP)
fig6, ax6 = plt.subplots(figsize=(9, 7)) # Increased height further
ax6.axvspan(0, 0.5, color='gray', alpha=0.15, label='Anode')
ax6.axvspan(0.5, 0.58, color='blue', alpha=0.05, label='Membrane')
ax6.axvspan(0.58, 0.95, color='black', alpha=0.15, label='Cathode')

y_min, y_max = 1000, 0 

for name, res in thermal_results.items():
    ax6.plot(res["x"], res["y"], marker='o', markersize=8, label=name, color=scenarios[name]["c_mod"], linewidth=2.5)
    y_min = min(y_min, min(res["y"])); y_max = max(y_max, max(res["y"]))
    
    # --- Smart Label Placement ---
    if "Dry" in name:
        offset = 1.5 
        va = 'bottom'
        col = c_dry_mod
    else: 
        offset = -1.5
        va = 'top'
        col = c_wet_mod
        
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none', pad=0.2)
    ax6.text(0.25, res["y"][0] + offset, f"← {res['Q'][0]:.1f} W", color=col, ha='center', va=va, fontweight='bold', bbox=props)
    ax6.text(0.75, res["y"][-1] + offset, f"{res['Q'][1]:.1f} W →", color=col, ha='center', va=va, fontweight='bold', bbox=props)

ax6.set_ylim(y_min - 6, y_max + 6)
ax6.set_title(f"Temperature Gradient (at Peak Current)", fontweight='bold')
ax6.set_xlabel("Thickness (mm)", fontweight='bold'); ax6.set_ylabel("Temp (°C)", fontweight='bold')
ax6.legend(frameon=True); save_plot("6_Thermal_Gradient.png"); plt.show()

# Plot 7 & 8: Sensitivity
base_p = params_wet
names = [r"$\alpha_{an}$", r"$\alpha_{ca}$", r"$i_{0,an}$", r"$i_{0,ca}$"]
res_sens = []
i_max = np.max(i_wet)

for idx, val in enumerate(base_p):
    p_h = base_p.copy(); p_h[idx] *= 1.2
    v_h = E_OCV + i_max*r_total*25e-4 + (R*T/(p_h[0]*F))*np.arcsinh(i_max/(2*p_h[2])) + (R*T/(p_h[1]*F))*np.arcsinh(i_max/(2*p_h[3]))
    p_l = base_p.copy(); p_l[idx] *= 0.8
    v_l = E_OCV + i_max*r_total*25e-4 + (R*T/(p_l[0]*F))*np.arcsinh(i_max/(2*p_l[2])) + (R*T/(p_l[1]*F))*np.arcsinh(i_max/(2*p_l[3]))
    base_v = E_OCV + i_max*r_total*25e-4 + (R*T/(base_p[0]*F))*np.arcsinh(i_max/(2*base_p[2])) + (R*T/(base_p[1]*F))*np.arcsinh(i_max/(2*base_p[3]))
    res_sens.append({"Parameter": names[idx], "Delta_High": v_h - base_v, "Delta_Low": v_l - base_v, "Range": abs(v_h - v_l)})

df_s = pd.DataFrame(res_sens).set_index("Parameter")

fig7, ax7 = plt.subplots(figsize=(8, 4))
y = np.arange(len(df_s))
ax7.barh(y, df_s["Delta_High"], color='#CD5C5C', label='+20%'); ax7.barh(y, df_s["Delta_Low"], color='#4682B4', label='-20%')
ax7.set_yticks(y); ax7.set_yticklabels(df_s.index); ax7.legend()
ax7.set_title("Sensitivity Analysis (Tornado)", fontweight='bold')
ax7.grid(axis='x', linestyle='--', alpha=0.5); save_plot("7_Tornado.png"); plt.show()

top_idx = df_s["Range"].argmax()
top_name = names[top_idx]
idx_t = top_idx 

fig8, ax8 = plt.subplots(figsize=(6, 4))
ax8.plot(i_wet/1e4, V_model_wet, 'k-', linewidth=2, label="Base")
p_h = base_p.copy(); p_h[idx_t] *= 1.2; p_l = base_p.copy(); p_l[idx_t] *= 0.8
def calc_V(p, i_arr): return E_OCV + i_arr*r_total*25e-4 + (R*T/(p[0]*F))*np.arcsinh(i_arr/(2*p[2])) + (R*T/(p[1]*F))*np.arcsinh(i_arr/(2*p[3]))
ax8.plot(i_wet/1e4, calc_V(p_h, i_wet), color='#CD5C5C', linestyle='--', label=f"{top_name} +20%")
ax8.plot(i_wet/1e4, calc_V(p_l, i_wet), color='#4682B4', linestyle='--', label=f"{top_name} -20%")
ax8.set_ylim(bottom=1.4); ax8.legend()
ax8.set_title(f"Sensitivity: {top_name}", fontweight='bold')
ax8.set_xlabel("Current Density ($A/cm^2$)", fontweight='bold'); ax8.set_ylabel("Cell Voltage (V)", fontweight='bold')
ax8.grid(True, alpha=0.4); save_plot("8_Sensitivity_Curve.png"); plt.show()

# =============================================================================
# 8. COMPARISON OF EFFICIENCY DEFINITIONS
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
F = 96485                     # Faraday constant [C/mol]
HHV = 285830                  # J/mol
Cp_H2O_l = 75.3               # J/mol/K
T_REF = 298.15                # K (ambient)

# -----------------------------------------------------------------------------
# THERMODYNAMIC VOLTAGES
# -----------------------------------------------------------------------------
def U_rev(T_K):
    return 1.229 - 8.5e-4 * (T_K - T_REF)

def U_tn(T_K):
    return 1.481 - 8.5e-4 * (T_K - T_REF)

# -----------------------------------------------------------------------------
# METHOD 1: BEST ESTIMATION (LAMY & MILLET – Eq. 20)
# -----------------------------------------------------------------------------
def epsilon_cell_th(U_cell, T_K):
    return U_tn(T_K) / (U_tn(T_K) + U_cell - U_rev(T_K))

# -----------------------------------------------------------------------------
# METHOD 2: HHV-BASED SYSTEM EFFICIENCY (SIMPLIFIED)
# -----------------------------------------------------------------------------
def epsilon_HHV_system(i_A_m2, U_cell, T_K):
    I = i_A_m2 * 25e-4                     # A
    E_out = (I / (2 * F)) * HHV
    E_elec = U_cell * I
    E_heat = (I / (2 * F)) * Cp_H2O_l * (T_K - T_REF)
    return E_out / (E_elec + E_heat)

# -----------------------------------------------------------------------------
# MASKS (SEPARATE — IMPORTANT)
# -----------------------------------------------------------------------------
mask_wet = i_wet > 100
mask_dry = i_dry > 100

# -----------------------------------------------------------------------------
# COMPUTE EFFICIENCIES — WET
# -----------------------------------------------------------------------------
eps_wet_th = epsilon_cell_th(
    V_model_wet[mask_wet],
    thermal_results["Wet Cathode"]["T"][mask_wet]
)

eps_wet_HHV = epsilon_HHV_system(
    i_wet[mask_wet],
    V_model_wet[mask_wet],
    thermal_results["Wet Cathode"]["T"][mask_wet]
)

# -----------------------------------------------------------------------------
# COMPUTE EFFICIENCIES — DRY
# -----------------------------------------------------------------------------
eps_dry_th = epsilon_cell_th(
    V_model_dry[mask_dry],
    thermal_results["Dry Cathode"]["T"][mask_dry]
)

eps_dry_HHV = epsilon_HHV_system(
    i_dry[mask_dry],
    V_model_dry[mask_dry],
    thermal_results["Dry Cathode"]["T"][mask_dry]
)

# =============================================================================
# PLOT 1: EFFICIENCY COMPARISON (WET)
# =============================================================================
plt.figure(figsize=(9, 6))

plt.plot(
    i_wet[mask_wet] / 1e4,
    eps_wet_th * 100,
    linewidth=3,
    label=r"Wet – $\epsilon_{cell,th}$ (Eq. 20)"
)

plt.plot(
    i_wet[mask_wet] / 1e4,
    eps_wet_HHV * 100,
    linestyle="--",
    linewidth=3,
    label="Wet – HHV-based efficiency"
)

plt.xlabel("Current Density ($A/cm^2$)", fontweight="bold")
plt.ylabel("Efficiency (%)", fontweight="bold")
plt.title("Efficiency Definition Comparison (Wet Cathode)", fontweight="bold")
plt.grid(True, alpha=0.4)
plt.legend()
plt.ylim(60, 95)

save_plot("11_Efficiency_Comparison_Wet.png")
plt.show()

# =============================================================================
# PLOT 2: EFFICIENCY COMPARISON (DRY)
# =============================================================================
plt.figure(figsize=(9, 6))

plt.plot(
    i_dry[mask_dry] / 1e4,
    eps_dry_th * 100,
    linewidth=3,
    label=r"Dry – $\epsilon_{cell,th}$ (Eq. 20)"
)

plt.plot(
    i_dry[mask_dry] / 1e4,
    eps_dry_HHV * 100,
    linestyle="--",
    linewidth=3,
    label="Dry – HHV-based efficiency"
)

plt.xlabel("Current Density ($A/cm^2$)", fontweight="bold")
plt.ylabel("Efficiency (%)", fontweight="bold")
plt.title("Efficiency Definition Comparison (Dry Cathode)", fontweight="bold")
plt.grid(True, alpha=0.4)
plt.legend()
plt.ylim(60, 95)

save_plot("12_Efficiency_Comparison_Dry.png")
plt.show()

# =============================================================================
# PLOT 3: AVERAGE EFFICIENCY COMPARISON (BAR CHART)
# =============================================================================
avg_eff = {
    "Wet – Eq. 20": np.mean(eps_wet_th) * 100,
    "Wet – HHV": np.mean(eps_wet_HHV) * 100,
    "Dry – Eq. 20": np.mean(eps_dry_th) * 100,
    "Dry – HHV": np.mean(eps_dry_HHV) * 100,
}

plt.figure(figsize=(9, 6))
bars = plt.bar(
    avg_eff.keys(),
    avg_eff.values(),
    edgecolor="black",
    alpha=0.9
)

plt.ylabel("Average Efficiency (%)", fontweight="bold")
plt.title("Average Efficiency: Thermodynamic vs HHV-Based", fontweight="bold")
plt.ylim(0, 100)

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        h + 1,
        f"{h:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.grid(axis="y", linestyle="--", alpha=0.4)
save_plot("13_Avg_Efficiency_Comparison.png")
plt.show()



# =============================================================================
# PLOT 4: WET vs DRY COMPARISON (BOTH EFFICIENCY DEFINITIONS)
# =============================================================================

plt.figure(figsize=(10, 7))

# --- Thermodynamic efficiency (Eq. 20)
plt.plot(
    i_wet[mask_wet] / 1e4,
    eps_wet_th * 100,
    linewidth=3,
    color="tab:blue",
    label=r"Wet – $\epsilon_{cell,th}$ (Eq. 20)"
)

plt.plot(
    i_dry[mask_dry] / 1e4,
    eps_dry_th * 100,
    linewidth=3,
    color="tab:blue",
    linestyle="--",
    label=r"Dry – $\epsilon_{cell,th}$ (Eq. 20)"
)

# --- HHV-based efficiency
plt.plot(
    i_wet[mask_wet] / 1e4,
    eps_wet_HHV * 100,
    linewidth=3,
    color="tab:orange",
    label="Wet – HHV-based efficiency"
)

plt.plot(
    i_dry[mask_dry] / 1e4,
    eps_dry_HHV * 100,
    linewidth=3,
    color="tab:orange",
    linestyle="--",
    label="Dry – HHV-based efficiency"
)

plt.xlabel("Current Density ($A/cm^2$)", fontweight="bold")
plt.ylabel("Efficiency (%)", fontweight="bold")
plt.title(
    "Wet vs Dry Cathode Efficiency Comparison\n(Thermodynamic vs HHV-Based)",
    fontweight="bold"
)

plt.grid(True, alpha=0.4)
plt.legend(ncol=2)
plt.ylim(60, 95)

save_plot("14_Wet_vs_Dry_Efficiency_Comparison.png")
plt.show()
