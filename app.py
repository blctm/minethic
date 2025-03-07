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

st.title("Predicción de la Eficiencia Sólida en Lixiviación")

# Sidebar inputs
st.sidebar.header("Parámetros de Entrada")
MP_gr = st.sidebar.number_input("MP(gr)", value=0.0, format="%.2f")
Cantidad_Total = st.sidebar.number_input("Cantidad Total(gr)", value=50)
Tiempo = st.sidebar.number_input("Tiempo (horas)", value=1)
Temperatura = st.sidebar.number_input("Temperatura (°C)", value=40)
Disolvente = st.sidebar.number_input("Disolvente", value=0.0, format="%.2f")
LicorLavado = st.sidebar.number_input("Licor de Lavado", value=0.0, format="%.2f")
Acid_Concentration = st.sidebar.number_input("Concentración de ácido", value=0.0, format="%.2f")
RSS_porcentajes = st.sidebar.number_input("Residuo Seco (%)", value=0.0, format="%.2f")
cte2_por_MP = st.sidebar.number_input("cte2 (g/L)", value=0.0, format="%.2f")

# Calculate pH based on Acid Concentration
if Acid_Concentration > 1:
    pH = 0
else:
    pH = -np.log10(Acid_Concentration) if Acid_Concentration > 0 else 0
st.sidebar.write(f"🔹 **pH Calculado:** {pH:.2f}")

# Calculate Volume of Acid using pulp density (DP)
DP = 200  # Fixed pulp density in g/L
Volume_of_Acid = MP_gr / DP if MP_gr > 0 else 0
st.sidebar.write(f"🔹 **Volumen de Ácido Calculado (L):** {Volume_of_Acid:.5f}")

# Dropdown for metals
metal = st.sidebar.selectbox("Metal", ["Fe", "Mg", "Mn", "Zn"], index=0)
metal_dict = {"Fe": [1, 0, 0, 0], "Mg": [0, 1, 0, 0], "Mn": [0, 0, 1, 0], "Zn": [0, 0, 0, 1]}
metal_features = metal_dict[metal]

# DataFrame for input
input_data = pd.DataFrame([{
    "MP(gr)": MP_gr,
    "Cantidad Total(gr)": Cantidad_Total,
    "Tiempo": Tiempo,
    "Temperatura": Temperatura,
    "Disolvente": Disolvente,
    "LicorLavado": LicorLavado,
    "Concentración de ácido": Acid_Concentration,
    "Volúmen de ácido (L)": Volume_of_Acid,
    "Residuo Seco (%)": RSS_porcentajes,
    "cte2 (g/L)": cte2_por_MP,
    "Metal_Fe": metal_features[0],
    "Metal_Mg": metal_features[1],
    "Metal_Mn": metal_features[2],
    "Metal_Zn": metal_features[3]
}])

# Prediction
if st.button("Predecir"):
    eff_prediction = round(((MP_gr - (RSS_porcentajes * MP_gr / 100)) / MP_gr) * 100, 2)
    res_prediction = RSS_porcentajes * MP_gr / 100

    st.write(f"### ✅ Eficiencia Predicha (BS): {eff_prediction:.2f}%")
    st.write(f"### 🏗️ Residuo Predicho (gr): {res_prediction:.2f}g")

# Feature Importance
st.markdown("### 🔎 Importancia de las Características")
eff_importances = model_efficiency.feature_importances_
res_importances = model_residuo.feature_importances_

feature_names = list(input_data.columns)
eff_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': eff_importances}).sort_values(by='Importance', ascending=False)
res_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': res_importances}).sort_values(by='Importance', ascending=False)

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.barh(eff_importance_df['Feature'][:10], eff_importance_df['Importance'][:10])
ax1.invert_yaxis()
ax1.set_title("Top 10 Características - Modelo de Eficiencia")

st.pyplot(fig1)
