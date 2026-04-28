# File: src/evaluate.py
"""Métricas y visualizaciones para evaluación multicategoría en TC-DIAG."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluar_modelo(y_true: List[int], y_pred: List[int], y_proba: np.ndarray, nombres_clases: List[str]) -> Dict[str, Any]:
    """Evalúa un modelo multicategoría y resume sus métricas principales.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones discretas.
        y_proba: Probabilidades por clase con forma (n, c).
        nombres_clases: Nombres de las clases en el mismo orden de y_proba.

    Returns:
        Diccionario con métricas globales y por clase.
    """
    reporte = classification_report(y_true, y_pred, target_names=nombres_clases, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    y_true_bin = label_binarize(y_true, classes=list(range(len(nombres_clases))))
    auc_roc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")

    reporte_por_clase = classification_report(y_true, y_pred, target_names=nombres_clases, output_dict=True, zero_division=0)
    recall_por_clase = {clase: float(reporte_por_clase[clase]["recall"]) for clase in nombres_clases}

    return {
        "full_report": reporte,
        "confusion_matrix": cm,
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "accuracy": float(accuracy),
        "auc_roc_macro": float(auc_roc_macro),
        "recall_por_clase": recall_por_clase,
    }


def comparar_modelos(resultados: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Compara múltiples modelos a partir de sus resultados de evaluación.

    Args:
        resultados: Diccionario con métricas por modelo.

    Returns:
        DataFrame ordenado por Macro_Recall descendente.
    """
    filas = []
    for nombre_modelo, metricas in resultados.items():
        filas.append(
            {
                "Modelo": nombre_modelo,
                "Accuracy": metricas.get("accuracy", 0.0),
                "Macro_Precision": metricas.get("macro_precision", 0.0),
                "Macro_Recall": metricas.get("macro_recall", 0.0),
                "Macro_F1": metricas.get("macro_f1", 0.0),
                "AUC_ROC": metricas.get("auc_roc_macro", 0.0),
            }
        )
    dataframe = pd.DataFrame(filas)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by="Macro_Recall", ascending=False).reset_index(drop=True)
    return dataframe


def plot_confusion_matrix(cm: np.ndarray, nombres_clases: List[str], model_name: str, output_path: str) -> None:
    """Grafica y guarda una matriz de confusión.

    Args:
        cm: Matriz de confusión.
        nombres_clases: Nombres de las clases.
        model_name: Nombre del modelo para el título.
        output_path: Ruta de salida de la imagen.

    Returns:
        None.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_multiclass(y_true: List[int], y_proba: np.ndarray, nombres_clases: List[str], output_path: str) -> None:
    """Grafica curvas ROC multicategoría bajo estrategia One-vs-Rest.

    Args:
        y_true: Etiquetas reales.
        y_proba: Probabilidades por clase.
        nombres_clases: Nombres de clases.
        output_path: Ruta de salida de la figura.

    Returns:
        None.
    """
    from sklearn.metrics import roc_curve

    y_true_bin = label_binarize(y_true, classes=list(range(len(nombres_clases))))
    plt.figure(figsize=(10, 8))

    for indice, clase in enumerate(nombres_clases):
        fpr, tpr, _ = roc_curve(y_true_bin[:, indice], y_proba[:, indice])
        auc_clase = roc_auc_score(y_true_bin[:, indice], y_proba[:, indice])
        plt.plot(fpr, tpr, linewidth=2, label=f"{clase} (AUC = {auc_clase:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC Multicategoría")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
