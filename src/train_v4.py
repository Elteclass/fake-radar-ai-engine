import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset
from google.cloud import storage
from dotenv import load_dotenv

# =============================================================================
# CONFIGURACIÓN GLOBAL V4
# =============================================================================
load_dotenv()

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
MODEL_NAME      = "xlm-roberta-base"
MAX_LENGTH      = 256
SEED            = 42
OUTPUT_DIR      = "./models/fake_radar_v4"
GCP_MODEL_DEST  = "models/fake_radar_v4"

# Multiplicador de penalización para la clase Fake (Clase 1).
# 1.5 = moderado. Sube a 2.0 si recall de fake sigue bajo tras entrenar.
FAKE_WEIGHT_MULTIPLIER = 1.5

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_v4.log"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# GCP: DESCARGA Y SUBIDA
# =============================================================================
def download_from_gcp(bucket_name: str, source_blob: str, destination: str) -> None:
    """Descarga un archivo de GCP Storage al disco local."""
    log.info(f"Descargando gs://{bucket_name}/{source_blob} → {destination}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(source_blob)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    blob.download_to_filename(destination)
    log.info("Descarga completa.")


def upload_folder_to_gcp(bucket_name: str, local_folder: str, gcp_prefix: str) -> None:
    """Sube recursivamente una carpeta local a GCP Storage."""
    log.info(f"Subiendo {local_folder} → gs://{bucket_name}/{gcp_prefix}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for local_file in Path(local_folder).rglob("*"):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_folder)
        blob_path = f"{gcp_prefix}/{relative}"
        bucket.blob(blob_path).upload_from_filename(str(local_file))
        log.info(f"  ✓ {blob_path}")

    log.info(f"Modelo disponible en gs://{bucket_name}/{gcp_prefix}")


# =============================================================================
# PREPARACIÓN DE DATOS
# =============================================================================
def validate_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Valida columnas requeridas, elimina nulos y textos vacíos."""
    required = {"text", "label"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Faltan columnas: {missing}. Columnas encontradas: {list(df.columns)}")

    original_len = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip().str.len() > 10]
    df["label"] = df["label"].astype(int)

    # Verificar que solo haya etiquetas 0 y 1
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"[{name}] Etiquetas inválidas: {unique_labels}. Solo se permiten 0 y 1.")

    removed = original_len - len(df)
    if removed > 0:
        log.warning(f"[{name}] Se eliminaron {removed} filas inválidas.")

    log.info(f"[{name}] {len(df)} filas válidas | Reales: {(df['label']==0).sum()} | Falsas: {(df['label']==1).sum()}")
    return df.reset_index(drop=True)


def prepare_full_data() -> pd.DataFrame:
    """Descarga, valida y combina los datasets desde GCP."""
    log.info("=" * 60)
    log.info("FASE 1: PREPARACIÓN DE DATOS V4")
    log.info("=" * 60)

    path_esp   = "data/temp_spanish.csv"
    path_latam = "data/temp_latam.csv"

    download_from_gcp(GCP_BUCKET_NAME, "datasets/open_source_spanish.csv", path_esp)
    download_from_gcp(GCP_BUCKET_NAME, "datasets/dataset_latam_master.csv",  path_latam)

    df_esp   = pd.read_csv(path_esp)
    df_latam = pd.read_csv(path_latam)

    df_esp   = validate_dataframe(df_esp,   "open_source_spanish")
    df_latam = validate_dataframe(df_latam, "dataset_latam_master")

    df_master = pd.concat([df_esp, df_latam], ignore_index=True)
    df_master = df_master.sample(frac=1, random_state=SEED).reset_index(drop=True)

    log.info(f"Dataset V4 listo: {len(df_master)} noticias totales.")
    log.info(f"  → Reales (0): {(df_master['label']==0).sum()}")
    log.info(f"  → Falsas (1): {(df_master['label']==1).sum()}")
    return df_master


# =============================================================================
# MÉTRICAS DE EVALUACIÓN
# =============================================================================
def compute_metrics(eval_pred):
    """
    Calcula accuracy, F1, precision y recall para la clase Fake (label=1).
    El F1-fake es la métrica principal de decisión.
    """
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)

    return {
        "accuracy":       accuracy_score(labels, predictions),
        "f1_fake":        f1_score(labels, predictions, pos_label=1, zero_division=0),
        "f1_macro":       f1_score(labels, predictions, average="macro", zero_division=0),
        "precision_fake": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "recall_fake":    recall_score(labels, predictions, pos_label=1, zero_division=0),
    }


# =============================================================================
# TRAINER PERSONALIZADO CON PÉRDIDA PONDERADA
# =============================================================================
def build_weighted_trainer_class(weights_tensor: torch.Tensor):
    """
    Factory que devuelve una subclase de Trainer con CrossEntropyLoss ponderado.
    Se usa una factory para evitar variables globales.
    """
    class WeightedLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.get("labels")
            outputs = model(**inputs)
            logits  = outputs.get("logits")

            device   = next(model.parameters()).device
            weights  = weights_tensor.to(device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss     = loss_fct(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


# =============================================================================
# TOKENIZACIÓN
# =============================================================================
def build_tokenize_fn(tokenizer, max_length: int):
    """Devuelve una función de tokenización con el tokenizer y max_length fijados."""
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

    # Reproducibilidad total
    set_seed(SEED)

    # Info de hardware
    if torch.cuda.is_available():
        log.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        log.info(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        use_fp16 = True
    else:
        log.warning("No se detectó GPU. Entrenando en CPU (lento).")
        use_fp16 = False

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
    log.info("FASE 2: CÁLCULO DE PESOS POR CLASE")
    log.info("=" * 60)

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    class_weights_np[1] = class_weights_np[1] * FAKE_WEIGHT_MULTIPLIER

    log.info(f"Peso clase Real (0): {class_weights_np[0]:.4f}")
    log.info(f"Peso clase Fake (1): {class_weights_np[1]:.4f}  (×{FAKE_WEIGHT_MULTIPLIER} aplicado)")

    weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

    # ------------------------------------------------------------------
    # FASE 3: Modelo y tokenizer
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info(f"FASE 3: CARGANDO {MODEL_NAME}")
    log.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # ------------------------------------------------------------------
    # FASE 4: Tokenización de datasets
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 4: TOKENIZANDO (256 tokens)")
    log.info("=" * 60)

    tokenize_fn = build_tokenize_fn(tokenizer, MAX_LENGTH)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset   = Dataset.from_pandas(val_df)

    tokenized_train = train_dataset.map(tokenize_fn, batched=True)
    tokenized_val   = val_dataset.map(tokenize_fn,   batched=True)

    # ------------------------------------------------------------------
    # FASE 5: Argumentos de entrenamiento
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 5: CONFIGURANDO ENTRENAMIENTO")
    log.info("=" * 60)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Evaluación y checkpoints
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,        # Necesario para EarlyStopping
        metric_for_best_model="f1_fake",    # Optimizamos por F1 de noticias falsas
        greater_is_better=True,

        # Hiperparámetros
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,                   # 10% de pasos de calentamiento

        # Batch y gradientes
        # batch 8 × gradient_accumulation 2 = batch efectivo de 16
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        # Velocidad
        fp16=use_fp16,                      # Mixed precision en GPU NVIDIA
        dataloader_num_workers=2,

        # Almacenamiento
        save_total_limit=2,                 # Mantiene solo los 2 mejores checkpoints

        # Semilla
        seed=SEED,

        # Logging
        logging_dir="./logs",
        logging_steps=50,
        report_to="none",                   # Cambia a "wandb" si usas Weights & Biases
    )

    # ------------------------------------------------------------------
    # FASE 6: Trainer y entrenamiento
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 6: INICIANDO ENTRENAMIENTO V4")
    log.info("=" * 60)

    WeightedLossTrainer = build_weighted_trainer_class(weights_tensor)

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ],
    )

    trainer.train()

    # ------------------------------------------------------------------
    # FASE 7: Guardado local
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("FASE 7: GUARDANDO MODELO V4")
    log.info("=" * 60)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log.info(f"Modelo guardado localmente en {OUTPUT_DIR}")

    # Reporte final de métricas
    final_metrics = trainer.evaluate()
    log.info("=" * 60)
    log.info("MÉTRICAS FINALES:")
    for k, v in final_metrics.items():
        log.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # FASE 8: Subida a GCP
    # ------------------------------------------------------------------
    log.info("FASE 8: SUBIENDO MODELO A GCP")
    upload_folder_to_gcp(GCP_BUCKET_NAME, OUTPUT_DIR, GCP_MODEL_DEST)

    log.info("¡Entrenamiento V4 completado exitosamente!")
