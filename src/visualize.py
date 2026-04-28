# File: src/visualize.py
"""Visualizaciones de distribución, longitud de texto y nubes de palabras para TC-DIAG."""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_distribucion_clases(df: pd.DataFrame, columna: str = "patologia", output_path: Optional[str] = None) -> None:
    """Grafica la distribución de clases en el dataset.

    Args:
        df: DataFrame con la columna de clase.
        columna: Nombre de la columna de clases.
        output_path: Ruta de salida opcional para guardar la figura.

    Returns:
        None.
    """
    if df.empty:
        logger.warning("El DataFrame está vacío; no se puede graficar la distribución.")
        return

    plt.figure(figsize=(10, 6))
    orden = df[columna].value_counts().index.tolist()
    palette = sns.color_palette("Set2", n_colors=len(orden))
    ax = sns.countplot(data=df, x=columna, order=orden, palette=palette)
    ax.set_title("Distribución de Patologías en el Dataset")
    ax.set_xlabel("Patología")
    ax.set_ylabel("Cantidad")

    for patch in ax.patches:
        altura = int(patch.get_height())
        ax.annotate(
            f"{altura}",
            (patch.get_x() + patch.get_width() / 2.0, altura),
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_wordclouds_por_clase(
    df: pd.DataFrame,
    texto_col: str = "texto_combinado",
    clase_col: str = "patologia",
    output_path: Optional[str] = None,
) -> None:
    """Genera nubes de palabras por clase diagnóstica.

    Args:
        df: DataFrame con textos clínicos y clases.
        texto_col: Columna que contiene el texto a visualizar.
        clase_col: Columna de la clase diagnóstica.
        output_path: Ruta de salida opcional para guardar la figura.

    Returns:
        None.
    """
    if df.empty:
        logger.warning("El DataFrame está vacío; no se pueden generar wordclouds.")
        return

    clases = list(df[clase_col].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_planas = axes.flatten()

    for indice, clase in enumerate(clases[:4]):
        textos = " ".join(df.loc[df[clase_col] == clase, texto_col].dropna().astype(str).tolist())
        wordcloud = WordCloud(width=800, height=500, background_color="white", colormap="viridis").generate(textos or "sin datos")
        axes_planas[indice].imshow(wordcloud, interpolation="bilinear")
        axes_planas[indice].set_title(str(clase))
        axes_planas[indice].axis("off")

    for indice in range(len(clases), 4):
        axes_planas[indice].axis("off")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_longitud_texto(
    df: pd.DataFrame,
    texto_col: str = "texto_combinado",
    clase_col: str = "patologia",
    output_path: Optional[str] = None,
) -> None:
    """Grafica la distribución de longitud de texto por patología.

    Args:
        df: DataFrame con textos y clases.
        texto_col: Columna de texto clínico.
        clase_col: Columna de la clase diagnóstica.
        output_path: Ruta de salida opcional para guardar la figura.

    Returns:
        None.
    """
    if df.empty:
        logger.warning("El DataFrame está vacío; no se puede graficar la longitud de texto.")
        return

    dataframe = df.copy()
    dataframe["n_words"] = dataframe[texto_col].fillna(0).astype(str).str.split().str.len()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe, x=clase_col, y="n_words", palette="Set3")
    plt.title("Distribución de Longitud de Texto por Patología")
    plt.xlabel("Patología")
    plt.ylabel("Número de palabras")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
