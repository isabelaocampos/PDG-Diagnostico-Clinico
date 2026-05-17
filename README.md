# PDG — Diagnóstico Clínico de Hallazgos Radiológicos Críticos

Sistema de aprendizaje automático para la clasificación automática de hallazgos radiológicos críticos y su evaluación diagnóstica, desarrollado como Proyecto de Grado en la Universidad ICESI.

---

## ¿Qué hace este proyecto?

El sistema recibe el texto libre de un informe radiológico y determina si el hallazgo es **Crítico** o **No Crítico**. Cuando se detecta un hallazgo crítico, el caso avanza a la etapa de evaluación de diagnóstico diferencial para identificar la patología específica.

```
Informe radiológico (texto libre)
        │
        ▼
┌─────────────────────┐
│   ¿Es CRÍTICO?      │──── No Crítico ──► Flujo estándar
│   (Clasificación)   │
└─────────────────────┘
        │ Crítico
        ▼
┌─────────────────────┐
│   ¿Qué diagnóstico? │──► Evaluación de diagnóstico diferencial
│   (Identificación)  │
└─────────────────────┘
```

---

## Pipeline de notebooks

Ejecutar en orden. Cada notebook es independiente y carga su propio estado desde los archivos de datos procesados.

| # | Notebook | Descripción |
|---|---|---|
| 0.0 | `0.0_Analisis_y_Transformacion.ipynb` | Exploración, limpieza y normalización del dataset de informes clínicos |
| 0.1 | `0.1_Inferencia_RAD_ALERT.ipynb` | Etiquetado inicial del dataset mediante inferencia externa *(preparación de datos, no parte del modelo final)* |
| 1 | `1_Sobremuestreo.ipynb` | Balanceo del conjunto de entrenamiento con SMOTE |
| 1b | `1_Sobremuestreo_ROS.ipynb` | Balanceo alternativo con Random Over-Sampling |
| 2 | `2_Modelos_Tradicionales_editado.ipynb` | Entrenamiento y evaluación de SVM, Regresión Logística y Naive Bayes |
| 3 | `3_Modelo_LSTM.ipynb` | Red neuronal LSTM con embeddings de texto clínico |
| 4 | `4_Submuestreo.ipynb` | Experimento de balanceo con submuestreo aleatorio |
| 5 | `5_Modelo_XGBoost.ipynb` | Clasificador XGBoost con datos aumentados por LLM |
| 6 | `6_Comparacion_SVM_LSTM_XGBoost.ipynb` | Comparación final de modelos y selección del mejor |
| 7 | `7_Modelo_RoBERTa.ipynb` | Fine-tuning de DistilBERT y RoBERTa-BNE *(requiere GPU)* |

---

## Resultado principal

El modelo seleccionado es **SVM** (Kernel RBF, validación cruzada 5-fold), por obtener el mejor equilibrio entre precisión, recall y reproducibilidad. Fue entrenado sobre el conjunto balanceado con SMOTE.

---

## Instalación y ejecución

### Requisitos previos
- Python 3.12
- `pip` actualizado: `python -m pip install --upgrade pip`

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/isabelaocampos/PDG-Diagnostico-Clinico
cd PDG-Diagnostico-Clinico

# 2. Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Abrir los notebooks
jupyter notebook
```

> ⚠️ TensorFlow ≥ 2.16 es obligatorio para Python 3.12. El notebook `7_Modelo_RoBERTa` requiere GPU para ejecutarse completo.

---

## Estructura del repositorio

```
PDG-Diagnostico-Clinico/
├── requirements.txt
├── README.md
└── notebooks/
    ├── 0.0_Analisis_y_Transformacion.ipynb
    ├── 0.1_Inferencia_RAD_ALERT.ipynb
    ├── 1_Sobremuestreo.ipynb
    ├── 1_Sobremuestreo_ROS.ipynb
    ├── 2_Modelos_Tradicionales_editado.ipynb
    ├── 3_Modelo_LSTM.ipynb
    ├── 4_Submuestreo.ipynb
    ├── 5_Modelo_XGBoost.ipynb
    ├── 6_Comparacion_SVM_LSTM_XGBoost.ipynb
    └── 7_Modelo_RoBERTa.ipynb
```

---

## Licencia

Uso académico — Proyecto de Grado, Universidad ICESI.