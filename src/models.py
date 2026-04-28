# File: src/models.py
"""Arquitecturas de modelos para clasificación multicategoría de informes radiológicos."""

from __future__ import annotations

import logging
from typing import Literal

import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from transformers import AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoBERTaMulticlass(nn.Module):
    """Cabeza de clasificación multicategoría sobre un encoder RoBERTa."""

    def __init__(self, model_name: str, num_labels: int = 4, dropout_rate: float = 0.1) -> None:
        """Inicializa el encoder base y la capa de clasificación.

        Args:
            model_name: Identificador del modelo HuggingFace base.
            num_labels: Número de clases de salida.
            dropout_rate: Probabilidad de dropout antes del clasificador.

        Returns:
            None.
        """
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propaga el lote hacia adelante y devuelve logits crudos.

        Args:
            input_ids: Tensores de identificadores de token.
            attention_mask: Máscara de atención.
            token_type_ids: Segment IDs opcionales.

        Returns:
            Tensor con logits sin softmax.
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled_output))
        return logits


def build_bilstm_multiclass(
    vocab_size: int,
    embed_dim: int,
    max_len: int,
    num_classes: int = 4,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
) -> tf.keras.Model:
    """Construye un BiLSTM multicategoría para texto clínico.

    Args:
        vocab_size: Tamaño del vocabulario.
        embed_dim: Dimensión de los embeddings.
        max_len: Longitud máxima de secuencia.
        num_classes: Número de clases de salida.
        lstm_units_1: Unidades de la primera capa LSTM bidireccional.
        lstm_units_2: Unidades de la segunda capa LSTM bidireccional.

    Returns:
        Modelo Keras compilado.
    """
    model = Sequential(
        [
            Embedding(vocab_size, embed_dim, input_length=max_len),
            Bidirectional(LSTM(lstm_units_1, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(lstm_units_2)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_baseline_pipeline(classifier_type: str = "svm") -> Pipeline:
    """Construye un pipeline clásico para clasificación de texto clínico.

    Args:
        classifier_type: Tipo de clasificador base. Opciones: svm, logistic, naive_bayes.

    Returns:
        Pipeline de scikit-learn listo para entrenamiento.
    """
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )

    if classifier_type == "svm":
        classifier = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
    elif classifier_type == "logistic":
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial",
        )
    elif classifier_type == "naive_bayes":
        classifier = MultinomialNB(alpha=0.1)
    else:
        raise ValueError("classifier_type debe ser 'svm', 'logistic' o 'naive_bayes'.")

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])
    return pipeline
