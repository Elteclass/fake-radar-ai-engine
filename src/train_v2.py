import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from google.cloud import storage
from dotenv import load_dotenv

# 1. Cargar configuración
load_dotenv()
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
MODEL_NAME = "xlm-roberta-base" 
MAX_LENGTH = 128 

def download_from_gcp(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)

def prepare_balanced_data():
    """Descarga los datos y aplica balanceo estricto 50/50."""
    print("--- FASE 1: PREPARACIÓN Y BALANCEO DE DATOS (V2) ---")
    
    path_esp = "data/temp_spanish.csv"
    path_latam = "data/temp_latam.csv"
    
    download_from_gcp(GCP_BUCKET_NAME, "datasets/open_source_spanish.csv", path_esp)
    download_from_gcp(GCP_BUCKET_NAME, "datasets/dataset_latam_master.csv", path_latam)
    
    df_esp = pd.read_csv(path_esp)
    df_latam = pd.read_csv(path_latam)
    df_master = pd.concat([df_esp, df_latam], ignore_index=True)
    
    # --- LA MAGIA DE V2: BALANCEO 50/50 ---
    print("Aplicando Undersampling para curar el desbalance de clases...")
    df_fake = df_master[df_master['label'] == 1]
    df_real = df_master[df_master['label'] == 0]
    
    # Encontramos la clase minoritaria
    min_count = min(len(df_fake), len(df_real))
    
    # Recortamos ambas clases para que sean exactamente iguales
    df_fake_balanced = df_fake.sample(n=min_count, random_state=42)
    df_real_balanced = df_real.sample(n=min_count, random_state=42)
    
    # Unimos y mezclamos
    df_balanced = pd.concat([df_fake_balanced, df_real_balanced], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset V2 listo: {len(df_balanced)} noticias ({min_count} Reales y {min_count} Falsas).")
    return df_balanced

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"¡Excelente! Entrenando con GPU: {torch.cuda.get_device_name(0)}")
    
    df = prepare_balanced_data()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print(f"--- FASE 2: DESCARGANDO {MODEL_NAME} ---")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    print("--- FASE 3: TOKENIZANDO DATOS ---")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    print("--- FASE 4: INICIANDO ENTRENAMIENTO V2 ---")
    training_args = TrainingArguments(
        output_dir="./models/fake_radar_model_v2", # Directorio nuevo
        eval_strategy="epoch", 
        learning_rate=2e-5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        num_train_epochs=3, 
        weight_decay=0.01,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
    )

    trainer.train()

    print("--- FASE 5: GUARDANDO MODELO V2 ---")
    final_model_path = "./models/fake_radar_v2" # Guardamos en carpeta separada
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path) 
    print(f"¡Entrenamiento V2 exitoso! Modelo guardado en {final_model_path}")
