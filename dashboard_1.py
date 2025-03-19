import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Cargar modelos y escaladores
scaler_eff = joblib.load("Bscaler_efficiencyb.pkl")
model_efficiency = joblib.load("Bmodel_efficiencyb.pkl")
scaler_res = joblib.load("Bscaler_residuob.pkl")
model_residuo = joblib.load("Bmodel_residuob.pkl")

# Título de la app
st.title("Predicción de la Eficiencia Sólida")

# Explicación detallada
st.markdown("""
### Acerca del Proceso de Lixiviación
El proceso de lixiviación es una técnica utilizada para extraer metales valiosos de materiales sólidos. Esta aplicación predice la **Eficiencia de Base Sólida (BS)** en este proceso. A continuación, se explica cómo se han calculado algunos parámetros clave:

- **Concentración de Ácido**: Se calcula a partir de datos experimentales relacionados con el pH.
- **Volumen de Ácido (L)**: Corresponde al volumen de solución utilizada durante la reacción química.
- **Disolvente**: Representan la cantidad de solvente utilizado

### Proceso Inicial
Para utilizar este modelo:
1. Ingresa los valores experimentales en el panel izquierdo.
2. Selecciona el metal principal a extraer.
3. Haz clic en "Predecir" para obtener los resultados esperados de **eficiencia** y **residuos**.

El modelo utiliza técnicas avanzadas de machine learning para predecir estos valores con base en los datos ingresados.
""")

# Inputs del usuario
st.sidebar.header("Parámetros de Entrada")
MP_gr = st.sidebar.number_input("MP(gr)", value=0.0, format="%.2f")
Cantidad_Total = st.sidebar.number_input("Cantidad Total(gr)", value=0)
Tiempo = st.sidebar.number_input("Tiempo (hours)", value=0)
Temperatura = st.sidebar.number_input("Temperatura (°C)", value=0)
Disolvente = st.sidebar.number_input("Disolvente", value=0.0, format="%.2f")
Acid_Concentration = st.sidebar.number_input("Concentración de ácido", value=0.0, format="%.2f")
Volume_of_Acid = st.sidebar.number_input("Volúmen de ácido (L)", value=0.0, format="%.2f")

# LicorLavado = st.sidebar.number_input("LicorLavado", value=0.0, format="%.2f")
# Dropdown para metales
metal = st.sidebar.selectbox(
    "Metales",
    options=["Fe", "Mg", "Mn", "Zn"],
    index=0
)

# Codificación One-Hot para el metal
metal_dict = {
    "Fe": [1, 0, 0, 0],
    "Mg": [0, 1, 0, 0],
    "Mn": [0, 0, 1, 0],
    "Zn": [0, 0, 0, 1]
}
metal_features = metal_dict[metal]

# Crear DataFrame de entrada
input_data = pd.DataFrame([{
    "MP(gr)": MP_gr,
    "Cantidad Total(gr)": Cantidad_Total,
    "Tiempo": Tiempo,
    "Temperatura": Temperatura,
    "Disolvente": Disolvente,
    "Concentración de ácido": Acid_Concentration,
    "Volúmen de ácido (L)": Volume_of_Acid,
    "Metal_Fe": metal_features[0],
    "Metal_Mg": metal_features[1],
    "Metal_Mn": metal_features[2],
    "Metal_Zn": metal_features[3]
}])

# Escalar datos
scaled_data_eff = scaler_eff.transform(input_data)
scaled_data_res = scaler_res.transform(input_data)

# Predicción
if st.button("Predecir"):
    eff_prediction = model_efficiency.predict(scaled_data_eff)
    res_prediction = model_residuo.predict(scaled_data_res)
    
    st.write(f"### Eficiencia Predicha (BS): {eff_prediction[0]:.2f}")
    st.write(f"### Residuo Predicho (gr): {res_prediction[0]:.2f}")

# Importancias de características
st.markdown("### Importancia de las Características")

# Obtener importancias de los modelos
eff_importances = model_efficiency.feature_importances_
res_importances = model_residuo.feature_importances_

# Nombres de las características
feature_names = [
    "MP(gr)", "Cantidad Total(gr)", "Tiempo", "Temperatura", "Disolvente", 
    "LicorLavado", "Concentración de ácido", "Volúmen de ácido (L)", 
    "Metal_Fe", "Metal_Mg", "Metal_Mn", "Metal_Zn"
]

# Crear DataFrames ordenados por importancia
eff_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': eff_importances}).sort_values(by='Importance', ascending=False)
res_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': res_importances}).sort_values(by='Importance', ascending=False)

# Gráficos de barras
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.barh(eff_importance_df['Feature'][:10], eff_importance_df['Importance'][:10], color='skyblue')
ax1.invert_yaxis()
ax1.set_title("Top 10 Características - Modelo de Eficiencia")
ax1.set_xlabel("Importancia")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.barh(res_importance_df['Feature'][:10], res_importance_df['Importance'][:10], color='salmon')
ax2.invert_yaxis()
ax2.set_title("Top 10 Características - Modelo de Residuo")
ax2.set_xlabel("Importancia")

# Mostrar gráficos en Streamlit
st.pyplot(fig1)
st.pyplot(fig2)
