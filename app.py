import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load models
scaler_eff = joblib.load("scaler_efficiency.pkl")
model_efficiency = joblib.load("model_efficiency.pkl")
scaler_res = joblib.load("scaler_residuo.pkl")
model_residuo = joblib.load("model_residuo.pkl")

st.title("🔬 Predicción de Lixiviación - Optimizado")

# Sidebar Inputs
st.sidebar.header("📌 Parámetros de Entrada")

# 1️⃣ Total Mass of Material
MP_gr = st.sidebar.number_input("Masa de Sólido (g)", value=100, format="%.2f")

# 2️⃣ Acid Concentration (User Input)
Acid_Concentration = st.sidebar.number_input("Concentración de Ácido (M)", value=1.0, format="%.2f")

# 3️⃣ Process Conditions
Temperature = st.sidebar.number_input("Temperatura (°C)", value=70)
Time = st.sidebar.number_input("Tiempo (min)", value=90)

# 4️⃣ Select Metal
metal = st.sidebar.selectbox("Metal", ["Mn", "Fe"], index=0)

# Cte1: Fixed Composition (Stored in Code)
Cte1_Composition = {"Mn": 24.67, "Fe": 24.67}  # Fixed values
selected_composition = Cte1_Composition[metal]

# ✅ pH Calculation
if Acid_Concentration > 1:
    pH = 0
else:
    pH = -np.log10(Acid_Concentration) if Acid_Concentration > 0 else 0
st.sidebar.write(f"🔹 **pH Calculado:** {pH:.2f}")

# ✅ Acid Volume Calculation (DP = 200 g/L)
DP = 200
Volume_of_Acid = MP_gr / DP if MP_gr > 0 else 0
st.sidebar.write(f"🔹 **Volumen de Ácido Calculado (L):** {Volume_of_Acid:.5f}")

# 📝 Data Preparation for Model Prediction
input_data = pd.DataFrame([{
    "Masa de Sólido (g)": MP_gr,
    "Concentración de Ácido": Acid_Concentration,
    "pH": pH,
    "Temperatura": Temperature,
    "Tiempo": Time,
    "Volumen de Ácido (L)": Volume_of_Acid,
    "Metal_Mn": 1 if metal == "Mn" else 0,
    "Metal_Fe": 1 if metal == "Fe" else 0
}])

# ✅ Model Prediction
if st.button("🔮 Predecir"):
    eff_prediction = model_efficiency.predict(input_data)[0]
    res_prediction = model_residuo.predict(input_data)[0]

    # ✅ Calculate Cte2 and Cte3 based on predictions
    Cte2 = eff_prediction * MP_gr / 100  
    Cte3 = res_prediction  # Solid Residue

    # Display Predictions
    st.write(f"### ✅ Eficiencia de Lixiviación Predicha: {eff_prediction:.2f}%")
    st.write(f"### 🏗️ Residuo Predicho (g): {res_prediction:.2f}g")
    st.write(f"📌 **Cte2 (Neutralized Output):** {Cte2:.2f}g")
    st.write(f"📌 **Cte3 (Solid Residue):** {Cte3:.2f}g")

# 🔎 Feature Importance
st.markdown("### 🔍 Importancia de Variables en la Predicción")

eff_importances = model_efficiency.feature_importances_
res_importances = model_residuo.feature_importances_
feature_names = input_data.columns

# Handling Feature Importance Display
if len(feature_names) == len(eff_importances):
    eff_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': eff_importances}).sort_values(by='Importance', ascending=False)
    res_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': res_importances}).sort_values(by='Importance', ascending=False)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.barh(eff_importance_df['Feature'], eff_importance_df['Importance'], color='skyblue')
    ax1.invert_yaxis()
    ax1.set_title("🔬 Importancia de Características - Eficiencia de Lixiviación")

    st.pyplot(fig1)
else:
    st.error("❌ Error: Mismatch in feature count. Check model training features.")
