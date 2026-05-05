"""
train_v5.py — VERSION DEFINITIVA
=================================
Mejoras vs V4:
  • Modelo base: Narrativaai/fake-news-detection-spanish
    (ya pre-entrenado en español con F1=0.77 — partimos desde ahí)
  • Dataset maestro unificado (~5,500+ noticias vs 4,247 en V4)
  • FAKE_WEIGHT_MULTIPLIER reducido a 1.2 (v4 era 1.5 — demasiado agresivo)
  • lr_scheduler_type = "cosine" (mejor convergencia que lineal)
  • warmup_ratio = 0.15 (más calentamiento para un modelo ya fine-tuneado)
  • early_stopping_patience = 3 (v4 era 2 — demasiado impaciente)
  • num_train_epochs = 8 (más tiempo para convergencia)
  • Un solo archivo CSV maestro desde GCP (más limpio que 3 separados)
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset
from google.cloud import storage
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# =============================================================================
# CONFIGURACIÓN V5
# =============================================================================
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

# Modelo base: ya fine-tuneado en fake news en español (F1=0.77 de base)
# Ventaja enorme vs xlm-roberta-base que parte desde cero en esta tarea.
MODEL_NAME = "Narrativaai/fake-news-detection-spanish"

MAX_LENGTH             = 256
SEED                   = 42
OUTPUT_DIR             = "./models/fake_radar_v5"
GCP_MODEL_DEST         = "models/fake_radar_v5"
FAKE_WEIGHT_MULTIPLIER = 1.2   # Reducido de 1.5 — precision_fake era muy baja en V4


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_v5.log"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# GCP
# =============================================================================
def download_from_gcp(bucket_name: str, source_blob: str, dest: str) -> None:
    log.info(f"Descargando gs://{bucket_name}/{source_blob} → {dest}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    bucket.blob(source_blob).download_to_filename(dest)
    log.info("Descarga completa.")


def upload_folder_to_gcp(bucket_name: str, local_folder: str, gcp_prefix: str) -> None:
    log.info(f"Subiendo {local_folder} → gs://{bucket_name}/{gcp_prefix}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for local_file in Path(local_folder).rglob("*"):
        if not local_file.is_file():
            continue
        blob_path = f"{gcp_prefix}/{local_file.relative_to(local_folder)}"
        bucket.blob(blob_path).upload_from_filename(str(local_file))
        log.info(f"  ✓ {blob_path}")
    log.info(f"Modelo disponible en gs://{bucket_name}/{gcp_prefix}")


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================
def validate_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"text", "label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Faltan columnas: {missing}")

    original = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().str.len() > 10]
    df["label"] = df["label"].astype(int)

    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"[{name}] Etiquetas inválidas: {unique_labels}")

    removed = original - len(df)
    if removed > 0:
        log.warning(f"[{name}] {removed} filas eliminadas por datos inválidos.")

    log.info(
        f"[{name}] {len(df)} filas | "
        f"Reales: {(df['label']==0).sum()} | Falsas: {(df['label']==1).sum()}"
    )
    return df.reset_index(drop=True)


def prepare_full_data() -> pd.DataFrame:
    """
    Descarga el dataset maestro unificado desde GCP.
    Generado por fetch_all_datasets.py — contiene todas las fuentes combinadas.
    """
    log.info("=" * 60)
    log.info("FASE 1: CARGANDO DATASET MAESTRO V5")
    log.info("=" * 60)

    local_path = "data/dataset_master.csv"
    download_from_gcp(GCP_BUCKET_NAME, "datasets/dataset_master.csv", local_path)

    df = pd.read_csv(local_path)
    df = validate_dataframe(df, "dataset_master")
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    log.info(f"Dataset V5: {len(df)} noticias totales")
    log.info(f"  → Reales (0): {(df['label']==0).sum()}")
    log.info(f"  → Falsas (1): {(df['label']==1).sum()}")
    return df


# =============================================================================
# MÉTRICAS
# =============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":       accuracy_score(labels, preds),
        "f1_fake":        f1_score(labels, preds, pos_label=1, zero_division=0),
        "f1_macro":       f1_score(labels, preds, average="macro", zero_division=0),
        "precision_fake": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall_fake":    recall_score(labels, preds, pos_label=1, zero_division=0),
    }


# =============================================================================
# TRAINER CON PÉRDIDA PONDERADA
# =============================================================================
def build_weighted_trainer(weights_tensor: torch.Tensor):
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.get("labels")
            outputs = model(**inputs)
            logits  = outputs.get("logits")
            device  = next(model.parameters()).device
            loss    = nn.CrossEntropyLoss(weight=weights_tensor.to(device))(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


# =============================================================================
# TOKENIZACIÓN
# =============================================================================
def build_tokenize_fn(tokenizer, max_length: int):
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    return tokenize_fn


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    set_seed(SEED)

    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        log.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                 f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        log.warning("Sin GPU — entrenando en CPU (lento).")

    # ------------------------------------------------------------------
    # FASE 1: Datos
    # ------------------------------------------------------------------
    df = prepare_full_data()
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    log.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # ------------------------------------------------------------------
    # FASE 2: Class Weights
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 2: CALCULANDO PESOS POR CLASE")
    log.info("=" * 60)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    # 1.2 en vez de 1.5 — en V4 la precision_fake era solo 0.51
    # porque el modelo era demasiado agresivo clasificando todo como fake
    class_weights[1] *= FAKE_WEIGHT_MULTIPLIER

    log.info(f"Peso Real (0): {class_weights[0]:.4f}")
    log.info(f"Peso Fake (1): {class_weights[1]:.4f}  (×{FAKE_WEIGHT_MULTIPLIER})")

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # ------------------------------------------------------------------
    # FASE 3: Modelo y tokenizer
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info(f"FASE 3: CARGANDO {MODEL_NAME}")
    log.info("=" * 60)
    log.info("Este modelo ya fue entrenado en fake news en español (F1 base ~0.77).")
    log.info("Partimos desde un punto mucho mejor que xlm-roberta-base.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True,   # Por si el head del modelo base difiere
    )

    # ------------------------------------------------------------------
    # FASE 4: Tokenización
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 4: TOKENIZANDO (256 tokens)")
    log.info("=" * 60)

    tokenize_fn     = build_tokenize_fn(tokenizer, MAX_LENGTH)
    tokenized_train = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    tokenized_val   = Dataset.from_pandas(val_df).map(tokenize_fn,   batched=True)

    # ------------------------------------------------------------------
    # FASE 5: Configuración de entrenamiento
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 5: CONFIGURANDO ENTRENAMIENTO V5")
    log.info("=" * 60)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Evaluación y checkpoints
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_fake",
        greater_is_better=True,

        # Hiperparámetros
        learning_rate=2e-5,
        num_train_epochs=8,            # Más tiempo (V4 era 5, pero paró en 2-3)
        weight_decay=0.01,

        # Scheduler coseno: converge más suavemente que lineal
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,             # 15% de pasos para calentar el modelo

        # Batch efectivo: 8 × 2 = 16 (mismo que V2 pero sin el RAM issue)
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        # Velocidad
        fp16=use_fp16,
        dataloader_num_workers=2,

        # Storage
        save_total_limit=2,

        # Reproducibilidad
        seed=SEED,

        # Logging
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",
    )

    # ------------------------------------------------------------------
    # FASE 6: Entrenamiento
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 6: INICIANDO ENTRENAMIENTO V5")
    log.info("=" * 60)

    WeightedLossTrainer = build_weighted_trainer(weights_tensor)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # ------------------------------------------------------------------
    # FASE 7: Guardado local
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 7: GUARDANDO MODELO V5")
    log.info("=" * 60)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log.info(f"Modelo guardado en {OUTPUT_DIR}")

    # Reporte de métricas finales
    final_metrics = trainer.evaluate()
    log.info("=" * 60)
    log.info("MÉTRICAS FINALES V5:")
    for k, v in final_metrics.items():
        log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # FASE 8: Subida a GCP
    # ------------------------------------------------------------------
    log.info("FASE 8: SUBIENDO MODELO V5 A GCP")
    upload_folder_to_gcp(GCP_BUCKET_NAME, OUTPUT_DIR, GCP_MODEL_DEST)

    log.info("★ Entrenamiento V5 completado exitosamente.")
    log.info("  Revisa train_v5.log para el historial completo de métricas.")
