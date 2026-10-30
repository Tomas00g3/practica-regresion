import streamlit as st
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Práctica 4.2 - Regresión Lineal", layout="wide")
st.title("Predicción del Consumo Per Cápita de Agua Potable (CPC)")

# --- PASO 0: SIMULACIÓN DE DATOS RELACIONALES EN DUCKDB ---
duckdb.sql("SELECT setseed(0.42);")
duckdb.sql("""
CREATE TABLE IF NOT EXISTS agua AS
SELECT
    1 AS residencia_id,
    20 + random() * (35 - 20) AS temperatura_max,
    50 + random() * (90 - 50) AS humedad_max,
    2 + CAST(floor(random() * 5) AS INT) AS personas_familia,
    1.0 + random() * (2.0 - 1.0) AS consumo_bidon_pc,
    80 + (2.5 * (20 + random() * (35 - 20))) 
       - (0.8 * (50 + random() * (90 - 50))) 
       - (5.0 * (2 + CAST(floor(random() * 5) AS INT))) 
       + (15.0 * (1.0 + random() * (2.0 - 1.0))) 
       + (random() * 4 - 2) AS cpc_consumo
FROM range(150);
""")

df_agua = duckdb.sql("SELECT * FROM agua").df()

# --- PASO 1: PRIMEROS 5 RESULTADOS OBTENIDOS DE DUCKDB ---
st.subheader("PASO 1: PRIMEROS 5 RESULTADOS OBTENIDOS DE DUCKDB")
st.dataframe(df_agua.head(5), use_container_width=True)

# --- PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r) ---
st.subheader("PASO 2: MATRIZ DE CORRELACIÓN DE PEARSON (r)")
matriz_corr = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc', 'cpc_consumo']].corr()
st.dataframe(matriz_corr, use_container_width=True)

st.write("**Correlación lineal de cada variable con la respuesta 'cpc_consumo':**")
corr_cpc = matriz_corr['cpc_consumo'].drop('cpc_consumo')
st.dataframe(corr_cpc.to_frame(), use_container_width=True)

# --- AJUSTE DEL MODELO OLS ---
X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

y_pred = modelo_ols.predict(X_const)
residuos = modelo_ols.resid

# --- PASO 3: RESUMEN DEL MODELO OLS (statsmodels) ---
st.subheader("PASO 3: RESUMEN DEL MODELO OLS (statsmodels)")
st.text("OLS Regression Results")
st.code(str(modelo_ols.summary()), language="text")

# --- PASO 4: DIAGNÓSTICO DE RESIDUOS Y SUPUESTOS ---
st.subheader("PASO 4: DIAGNÓSTICO DE RESIDUOS Y SUPUESTOS")

# 1. Prueba de Normalidad (Shapiro-Wilk)
stat_shapiro, p_val_shapiro = stats.shapiro(residuos)
st.write(f"**1. Prueba de Normalidad (Shapiro-Wilk):** Estadístico = {stat_shapiro:.4f}, valor p = {p_val_shapiro:.4f}")
if p_val_shapiro > 0.05:
    st.success("--> CONCLUSIÓN: Los residuos cumplen con el supuesto de NORMALIDAD (p > 0.05).")
else:
    st.warning("--> CONCLUSIÓN: Los residuos NO cumplen con el supuesto de normalidad (p <= 0.05).")

# 2. Estadístico Durbin-Watson
dw_stat = durbin_watson(residuos)
st.write(f"**2. Estadístico Durbin-Watson (Independencia):** {dw_stat:.4f} (Valores entre 1.5 y 2.5 indican ausencia de autocorrelación).")

# --- PASO 5: MÉTRICAS DE RENDIMIENTO DEL MODELO (scikit-learn) ---
st.subheader("PASO 5: MÉTRICAS DE RENDIMIENTO DEL MODELO (scikit-learn)")
r2 = r2_score(Y, y_pred)
rmse = np.sqrt(mean_squared_error(Y, y_pred))

col1, col2 = st.columns(2)
col1.metric("Coeficiente de Determinación (R²)", f"{r2*100:.2f}%")
col2.metric("Error Cuadrático Medio Raíz (RMSE)", f"{rmse:.4f} lt/hab-día")

coef = modelo_ols.params
st.markdown(f"""
**Ecuación del Modelo Ajustado:**  
`CPC = {coef['const']:.2f} + ({coef['temperatura_max']:.2f} * Temp) + ({coef['humedad_max']:.2f} * Humedad) + ({coef['personas_familia']:.2f} * Personas) + ({coef['consumo_bidon_pc']:.2f} * Bidon)`
""")

# --- PANEL VISUAL DE DIAGNÓSTICO ---
st.subheader("Panel de Diagnóstico Visual")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Panel de Diagnóstico Visual: Regresión Lineal Múltiple (Subtema 4.2)', fontsize=16, fontweight='bold')

# 1. Mapa de Calor de Correlación
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

# 2. Scatter Plot: Humedad vs Consumo
sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1], line_kws={'color': 'red', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Humedad Máx vs Consumo')
axes[0, 1].set_xlabel('Humedad Máxima (%)')
axes[0, 1].set_ylabel('CPC Consumo (lt/hab-día)')

# 3. Valores Reales vs Predichos
axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2*100:.1f}%)')
axes[0, 2].set_xlabel('CPC Predicho (ŷ)')
axes[0, 2].set_ylabel('CPC Real (y)')

# 4. Residuos vs Valores Ajustados (Homocedasticidad)
axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')
axes[1, 0].set_xlabel('Valores Predichos (ŷ)')
axes[1, 0].set_ylabel('Residuos (e = y - ŷ)')

# 5. Gráfico Q-Q Normal (Normalidad de Errores)
sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

# Ocultar la celda sobrante
axes[1, 2].axis('off')

plt.tight_layout()
st.pyplot(fig)
