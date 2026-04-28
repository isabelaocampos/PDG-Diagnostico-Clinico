# File: src/augmentation.py
"""Aumento de datos clínicos mediante parafraseo controlado con OpenAI."""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

PROMPT_TEMPLATE: str = (
    "Eres un radiólogo experto en neuroimagen.\n"
    "Parafrasea el siguiente informe de tomografía de cráneo simple \n"
    "manteniendo exactamente el mismo diagnóstico y hallazgos clínicos. \n"
    "Usa sinónimos médicos apropiados y cambia la estructura de las frases. \n"
    "No agregues, omitas ni inventes información. \n"
    "La longitud debe ser similar y debe sonar natural para un radiólogo. \n"
    "El diagnóstico del informe es: {patologia}. \n"
    "Responde solo con el informe parafraseado, sin explicaciones.\n"
    "Informe original: {texto}"
)


def aumentar_texto(texto: str, patologia: str, client: OpenAI, model: str = "gpt-3.5-turbo") -> str:
    """Genera una paráfrasis clínica conservando el mismo diagnóstico.

    Args:
        texto: Informe radiológico original.
        patologia: Etiqueta clínica asociada al informe.
        client: Cliente autenticado de OpenAI.
        model: Nombre del modelo de chat a utilizar.

    Returns:
        Texto parafraseado o el original si ocurre un error.
    """
    prompt = PROMPT_TEMPLATE.format(patologia=patologia, texto=texto)
    try:
        respuesta = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en redacción radiológica clínica."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        time.sleep(1)
        contenido = respuesta.choices[0].message.content if respuesta.choices else None
        return contenido.strip() if contenido else texto
    except Exception as exc:  # noqa: BLE001
        logger.error("Error al aumentar texto para patología %s: %s", patologia, exc)
        time.sleep(1)
        return texto


def balancear_clases(df: pd.DataFrame, client: OpenAI, target_per_class: Optional[int] = None) -> pd.DataFrame:
    """Balancea clases generando ejemplos aumentados para las minoritarias.

    Args:
        df: DataFrame con al menos la columna patologia y texto clínico.
        client: Cliente autenticado de OpenAI.
        target_per_class: Número objetivo de ejemplos por clase.

    Returns:
        DataFrame balanceado con ejemplos originales y aumentados.
    """
    if "patologia" not in df.columns:
        raise ValueError("El DataFrame debe incluir la columna 'patologia'.")

    dataframe = df.copy()
    if "texto_combinado" not in dataframe.columns:
        columnas_texto = [col for col in ["datos_clinicos", "hallazgos", "opinion"] if col in dataframe.columns]
        if not columnas_texto:
            raise ValueError("Se requiere una columna de texto como 'texto_combinado', 'hallazgos' o equivalentes.")
        dataframe["texto_combinado"] = dataframe.apply(
            lambda fila: " ".join(str(fila.get(col, "") or "") for col in columnas_texto).strip(), axis=1
        )

    conteos = dataframe["patologia"].value_counts()
    if conteos.empty:
        return dataframe

    objetivo = int(target_per_class or conteos.max())
    filas_nuevas = []
    for patologia, conteo in conteos.items():
        if conteo >= objetivo:
            continue
        necesarios = objetivo - conteo
        subset = dataframe[dataframe["patologia"] == patologia]
        muestras = subset.sample(n=necesarios, replace=True, random_state=42)
        for _, fila in muestras.iterrows():
            texto_original = str(fila.get("texto_combinado", "") or "")
            texto_aumentado = aumentar_texto(texto_original, str(patologia), client)
            nueva_fila = fila.to_dict()
            nueva_fila["texto_combinado"] = texto_aumentado
            if "hallazgos" in nueva_fila:
                nueva_fila["hallazgos"] = texto_aumentado
            filas_nuevas.append(nueva_fila)

    dataframe_balanceado = pd.concat([dataframe, pd.DataFrame(filas_nuevas)], ignore_index=True)
    dataframe_balanceado = dataframe_balanceado.sample(frac=1.0, random_state=42).reset_index(drop=True)
    logger.info("Balanceo completado. Registros finales: %d", len(dataframe_balanceado))
    return dataframe_balanceado


def guardar_aumentado(df: pd.DataFrame, output_path: str) -> None:
    """Guarda el conjunto balanceado en CSV y reporta la distribución por clase.

    Args:
        df: DataFrame a guardar.
        output_path: Ruta de salida del archivo CSV.

    Returns:
        None.
    """
    conteo_antes = df["patologia"].value_counts().to_dict() if "patologia" in df.columns else {}
    df.to_csv(output_path, index=False, encoding="utf-8")
    conteo_despues = df["patologia"].value_counts().to_dict() if "patologia" in df.columns else {}
    logger.info("Archivo guardado en %s", output_path)
    logger.info("Distribución antes/después: %s -> %s", conteo_antes, conteo_despues)
