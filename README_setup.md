# Configuración del entorno — Proyecto RAD-ALERT

## Requisitos previos
- **Python 3.12** (versión usada en todos los notebooks)
- `pip` actualizado: `python -m pip install --upgrade pip`

---

## 1. Clonar / descargar el repositorio

```bash
git clone <url-del-repo>
cd <carpeta-del-proyecto>
```

---

## 2. Crear entorno virtual

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow ≥ 2.16 es obligatorio para Python 3.12.  
> Si ves errores con `pandas-profiling`, usa `ydata-profiling` (ya incluido en el requirements).

---

## 4. Configurar variables de entorno

```bash
# Copiar la plantilla
cp .env.example .env   # en Windows: copy .env.example .env

# Editar .env con tus rutas reales (DATA_DIR, RAW_DATA_PATH, etc.)
```

---

## 5. Cargar el .env en cada notebook

Agrega esta celda al inicio de cada cuaderno (ya está preparada para funcionar):

```python
from dotenv import load_dotenv
import os

load_dotenv()  # carga automáticamente el .env de la raíz

DATA_DIR           = os.getenv("DATA_DIR", "./data")
RAW_DATA_PATH      = os.getenv("RAW_DATA_PATH", "./data/raw/dataset.xlsx")
PROCESSED_DATA_PATH= os.getenv("PROCESSED_DATA_PATH", "./data/processed/dataset_procesado.pkl")
MODELS_DIR         = os.getenv("MODELS_DIR", "./models")
RANDOM_SEED        = int(os.getenv("RANDOM_SEED", 42))
```

---

## 6. Registrar el kernel en Jupyter

```bash
python -m ipykernel install --user --name rad-alert --display-name "Python 3.12 (RAD-ALERT)"
```

Luego, al abrir cualquier notebook, selecciona el kernel **Python 3.12 (RAD-ALERT)**.

---

## Estructura sugerida de carpetas

```
proyecto/
├── .env                  ← tu config local (NO subir a git)
├── .env.example          ← plantilla (SÍ subir a git)
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
└── notebooks/
    ├── 0_Analisis_y_Transformacion_v2.ipynb
    ├── 0_1_Inferencia_RAD_ALERT.ipynb
    ├── 1_Sobremuestreo.ipynb
    ├── 2_Modelos_Tradicionales.ipynb
    ├── 3_Modelo_LSTM.ipynb
    ├── 4_Submuestreo.ipynb
    ├── 5_Modelo_XGBoost.ipynb
    └── 6_Comparacion_SVM_LSTM_XGBoost.ipynb
```

---

## .gitignore recomendado

```
.env
.venv/
__pycache__/
*.pkl
*.h5
*.pt
data/raw/
models/
```
