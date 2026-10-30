import streamlit as st
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Práctica 4.2 - Regresión Lineal", layout="wide")
st.title("Predicción del Consumo Per Cápita de Agua Potable (CPC)")

# Generación de datos determinista en DuckDB
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

# Ajuste del modelo
X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

y_pred = modelo_ols.predict(X_const)
residuos = modelo_ols.resid
r2 = r2_score(Y, y_pred)
rmse = np.sqrt(mean_squared_error(Y, y_pred))

# Visualizaciones
st.subheader("Métricas del Modelo")
col1, col2 = st.columns(2)
col1.metric("Coeficiente R²", f"{r2*100:.2f}%")
col2.metric("RMSE", f"{rmse:.4f} lt/hab-día")

st.subheader("Panel Visual de Diagnóstico")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

matriz_corr = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc', 'cpc_consumo']].corr()
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación')

sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1], line_kws={'color':'red'})
axes[0, 1].set_title('2. Humedad Máx vs Consumo')

axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
axes[0, 2].set_title('3. Val. Reales vs Predichos')

axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_title('4. Residuos vs Ajustados')

sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal')

axes[1, 2].axis('off')
plt.tight_layout()

st.pyplot(fig)
