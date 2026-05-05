"""
fetch_all_datasets.py — VERSION DEFINITIVA
==========================================
Combina TODAS las fuentes de datos en un único CSV maestro y lo sube a GCP.

Fuentes incluidas:
  [HuggingFace]
    1. mariagrandury/fake_news_corpus_spanish
    2. sayalaruano/FakeNewsSpanish_Kaggle2
    3. IsaacRodgz/Fake-news-latam-omdena

  [Archivos locales — sube a Colab antes de ejecutar]
    4. train.xlsx                      (jpposadas — FakeNewsCorpusSpanish train)
    5. test.xlsx                       (jpposadas — FakeNewsCorpusSpanish test)
    6. Dataset_FakeNewsEspañol2024_.xlsx
    7. spanishFakeNews.csv             (mismo origen que sayalaruano — los
                                        duplicados se eliminan automáticamente)

USO EN COLAB:
  1. Sube los 4 archivos locales al entorno de Colab.
  2. !python src/fetch_all_datasets.py
  3. El CSV resultante queda en GCP: datasets/dataset_master.csv
"""

import os
import pandas as pd
from datasets import load_dataset
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")

# =============================================================================
# RUTAS DE ARCHIVOS LOCALES
# Ajusta aquí si los subiste a una subcarpeta distinta en Colab.
# =============================================================================
LOCAL_FILES = {
    "jpposadas_train": "train.xlsx",
    "jpposadas_test":  "test.xlsx",
    "fakenews_2024":   "Dataset_FakeNewsEspañol2024_.xlsx",
    "spanish_csv":     "spanishFakeNews.csv",
}

GCP_DESTINATION = "datasets/dataset_master.csv"
LOCAL_OUTPUT    = "data/dataset_master.csv"


# =============================================================================
# HELPERS
# =============================================================================

def clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def to_standard(df, text_col, label_col, mapping: dict) -> pd.DataFrame:
    """Normaliza al formato {text, label} con mapeo explícito de etiquetas."""
    df = df.copy()
    df["text"]  = clean_text(df[text_col])
    df["label"] = df[label_col].astype(str).str.strip().map(mapping)
    return df[["text", "label"]].dropna()


# =============================================================================
# FUENTES — HUGGING FACE
# =============================================================================

def load_mariagrandury() -> pd.DataFrame:
    print("  Descargando mariagrandury/fake_news_corpus_spanish...")
    ds = load_dataset("mariagrandury/fake_news_corpus_spanish", split="test").to_pandas()
    ds["text"]  = clean_text(ds["TEXT"])
    ds["label"] = ds["CATEGORY"].astype(int)
    return ds[["text", "label"]]


def load_sayalaruano() -> pd.DataFrame:
    print("  Descargando sayalaruano/FakeNewsSpanish_Kaggle2...")
    ds = load_dataset("sayalaruano/FakeNewsSpanish_Kaggle2", split="train").to_pandas()
    return to_standard(ds, "texto", "clase", {"fake": 1, "real": 0})


def load_latam_omdena() -> pd.DataFrame:
    print("  Descargando IsaacRodgz/Fake-news-latam-omdena...")
    train = load_dataset("IsaacRodgz/Fake-news-latam-omdena", split="train").to_pandas()
    test  = load_dataset("IsaacRodgz/Fake-news-latam-omdena", split="test").to_pandas()
    df    = pd.concat([train, test], ignore_index=True)
    return to_standard(df, "Content", "Corrected_label", {"Fake": 1, "True": 0})


# =============================================================================
# FUENTES — ARCHIVOS LOCALES
# =============================================================================

def load_jpposadas_train() -> pd.DataFrame:
    """
    train.xlsx — jpposadas/FakeNewsCorpusSpanish
    Labels: 'Fake' → 1 | 'True' → 0
    """
    df = pd.read_excel(LOCAL_FILES["jpposadas_train"])
    return to_standard(df, "Text", "Category", {"Fake": 1, "True": 0})


def load_jpposadas_test() -> pd.DataFrame:
    """
    test.xlsx — jpposadas/FakeNewsCorpusSpanish (split test)
    ATENCIÓN: este split usa 'False' en vez de 'Fake' — se normaliza aquí.
    Labels: 'False' → 1 | 'True' → 0
    """
    df = pd.read_excel(LOCAL_FILES["jpposadas_test"])
    return to_standard(df, "TEXT", "CATEGORY", {"False": 1, "True": 0})


def load_fakenews_2024() -> pd.DataFrame:
    """
    Dataset_FakeNewsEspañol2024_.xlsx
    Labels: 'FALSO' → 1 | 'VERDADERO' → 0
    NOTA: algunas filas tienen 'VERDADERO ' con espacio extra — str.strip() lo resuelve.
    """
    df = pd.read_excel(LOCAL_FILES["fakenews_2024"])
    return to_standard(df, "TEXT", "CATEGORY", {"FALSO": 1, "VERDADERO": 0})


def load_spanish_csv() -> pd.DataFrame:
    """
    spanishFakeNews.csv — mismo origen que sayalaruano (HuggingFace).
    Los duplicados exactos se eliminarán en la deduplicación global.
    Labels: 'fake' → 1 | 'real' → 0
    """
    df = pd.read_csv(LOCAL_FILES["spanish_csv"])
    return to_standard(df, "texto", "clase", {"fake": 1, "real": 0})


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def build_master_dataset(output_path: str) -> None:
    print("=" * 65)
    print("CONSTRUCCIÓN DEL DATASET MAESTRO — VERSION DEFINITIVA")
    print("=" * 65)

    sources = [
        # Nombre para el reporte           Función cargadora       Obligatoria
        ("mariagrandury       [HF]",        load_mariagrandury,     True),
        ("sayalaruano         [HF]",        load_sayalaruano,       True),
        ("latam_omdena        [HF]",        load_latam_omdena,      True),
        ("jpposadas_train  [local]",        load_jpposadas_train,   False),
        ("jpposadas_test   [local]",        load_jpposadas_test,    False),
        ("fakenews_2024    [local]",        load_fakenews_2024,     False),
        ("spanish_csv      [local]",        load_spanish_csv,       False),
    ]

    all_dfs = []
    summary = []

    for name, fn, required in sources:
        print(f"\n[{name}]")
        try:
            df = fn()
            df = df.dropna(subset=["text", "label"])
            df["label"] = df["label"].astype(int)
            df = df[df["text"].str.len() > 10]
            df = df.drop_duplicates(subset=["text"])

            r = (df["label"] == 0).sum()
            f = (df["label"] == 1).sum()
            all_dfs.append(df)
            summary.append((name, len(df), r, f))
            print(f"  ✓ {len(df)} filas | Real: {r} | Fake: {f}")

        except FileNotFoundError:
            msg = "REQUERIDO — falla crítica" if required else "opcional — se omite"
            print(f"  {'✗' if required else '⚠'} Archivo no encontrado ({msg})")
            if required:
                raise
            summary.append((name, 0, 0, 0))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            if required:
                raise
            summary.append((name, 0, 0, 0))

    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FUSIONANDO Y DEDUPLICANDO...")
    df_master = pd.concat(all_dfs, ignore_index=True)
    before    = len(df_master)
    df_master = df_master.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df_master = df_master.sample(frac=1, random_state=42).reset_index(drop=True)
    after     = len(df_master)

    # -------------------------------------------------------------------------
    total_real = (df_master["label"] == 0).sum()
    total_fake = (df_master["label"] == 1).sum()
    ratio      = total_fake / total_real if total_real > 0 else 0

    print("\n" + "=" * 65)
    print(f"{'FUENTE':<38} {'TOTAL':>5} {'REAL':>5} {'FAKE':>5}")
    print("-" * 65)
    for name, total, r, f in summary:
        icon = "✓" if total > 0 else "✗"
        print(f"  {icon} {name:<36} {total:>5} {r:>5} {f:>5}")
    print("-" * 65)
    print(f"  {'Antes de deduplicación':<36} {before:>5}")
    print(f"  {'Duplicados eliminados':<36} {before - after:>5}")
    print(f"  {'★ TOTAL FINAL':<36} {after:>5} {total_real:>5} {total_fake:>5}")
    print(f"\n  Ratio fake/real : {ratio:.2f}")
    print(f"  % Reales        : {total_real / after * 100:.1f}%")
    print(f"  % Falsas        : {total_fake / after * 100:.1f}%")

    if ratio < 0.4:
        print("\n  ⚠ DESBALANCE: pocos fakes. El class_weight compensará.")
    else:
        print("\n  ✓ Balance de clases adecuado.")
    print("=" * 65)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_master.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✓ Dataset guardado en: {output_path}")


def upload_to_gcp(bucket_name: str, local_file: str, gcp_path: str) -> None:
    print(f"\nSubiendo a gs://{bucket_name}/{gcp_path} ...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(gcp_path).upload_from_filename(local_file)
    print(f"✓ Dataset maestro asegurado en GCP.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if not GCP_BUCKET_NAME:
        raise ValueError("GCP_BUCKET_NAME no encontrado en .env")

    build_master_dataset(LOCAL_OUTPUT)
    upload_to_gcp(GCP_BUCKET_NAME, LOCAL_OUTPUT, GCP_DESTINATION)

    print("\n★ LISTO: ejecuta train_v5.py para entrenar con el nuevo dataset.")
