# File: src/train.py
"""Rutinas de entrenamiento para los modelos de clasificación TC-DIAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from transformers import Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_roberta(
    model: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: str,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> Dict[str, float]:
    """Entrena un modelo RoBERTa con la API de HuggingFace Trainer.

    Args:
        model: Modelo de clasificación a entrenar.
        train_dataset: Dataset de entrenamiento tokenizado.
        val_dataset: Dataset de validación tokenizado.
        output_dir: Directorio para checkpoints y artefactos.
        num_epochs: Número de épocas de entrenamiento.
        batch_size: Tamaño de lote para entrenamiento.
        learning_rate: Tasa de aprendizaje inicial.

    Returns:
        Diccionario con las métricas finales del mejor modelo.
    """
    logger.info("Iniciando entrenamiento RoBERTa en %s", output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "accuracy": accuracy_score(labels, predictions),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_recall",
        greater_is_better=True,
        logging_dir="logs/",
        logging_steps=10,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Entrenamiento completado con métricas: %s", metrics)
    return {
        "macro_recall": float(metrics.get("eval_macro_recall", 0.0)),
        "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
        "accuracy": float(metrics.get("eval_accuracy", 0.0)),
    }


def train_baseline(pipeline: Any, X_train: List[str], y_train: List[int], X_val: List[str], y_val: List[int]) -> Dict[str, Any]:
    """Entrena un baseline clásico y reporta métricas multicategoría.

    Args:
        pipeline: Pipeline de scikit-learn a entrenar.
        X_train: Textos de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        X_val: Textos de validación.
        y_val: Etiquetas de validación.

    Returns:
        Diccionario con accuracy, macro_recall, macro_f1 y el reporte completo.
    """
    logger.info("Entrenando baseline clásico.")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_recall": recall_score(y_val, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
        "classification_report": classification_report(y_val, y_pred, zero_division=0),
    }


def guardar_modelo(model: Any, tokenizer: Any, output_dir: str) -> None:
    """Guarda el modelo y el tokenizer en disco.

    Args:
        model: Modelo entrenado con interfaz save_pretrained o equivalente.
        tokenizer: Tokenizer asociado al modelo.
        output_dir: Directorio de destino.

    Returns:
        None.
    """
    ruta_salida = Path(output_dir)
    ruta_salida.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    elif hasattr(model, "state_dict"):
        import torch

        torch.save(model.state_dict(), ruta_salida / "pytorch_model.bin")
    else:
        raise TypeError("El modelo no expone una forma compatible de guardado.")

    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    logger.info("Modelo y tokenizer guardados en %s", output_dir)
