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
    """Descarga un archivo desde Google Cloud Storage a la máquina local (Colab)."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)
    print(f"Descargado: {source_blob_name} -> {destination_file_name}")

def prepare_data():
    """Descarga los ingredientes de GCP y los pasa por la licuadora."""
    print("--- FASE 1: PREPARACIÓN DE DATOS ---")
    
    path_esp = "data/temp_spanish.csv"
    path_latam = "data/temp_latam.csv"
    
    download_from_gcp(GCP_BUCKET_NAME, "datasets/open_source_spanish.csv", path_esp)
    download_from_gcp(GCP_BUCKET_NAME, "datasets/dataset_latam_master.csv", path_latam)
    
    df_esp = pd.read_csv(path_esp)
    df_latam = pd.read_csv(path_latam)
    
    df_master = pd.concat([df_esp, df_latam], ignore_index=True)
    df_master = df_master.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Súper-Dataset listo: {len(df_master)} noticias en total.")
    return df_master

def tokenize_function(examples):
    """Convierte el texto humano a números (tokens)."""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"¡Excelente! Entrenando con GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("ADVERTENCIA: No se detectó GPU. El entrenamiento será extremadamente lento.")

    df = prepare_data()
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

    print("--- FASE 4: INICIANDO ENTRENAMIENTO ---")
    training_args = TrainingArguments(
        output_dir="./models/fake_radar_model",
        eval_strategy="epoch", 
        learning_rate=2e-5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        num_train_epochs=3, 
        weight_decay=0.01,
        save_strategy="epoch",
        # Eliminamos logging_dir para limpiar el warning deprecado
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer, # <-- EL FIX MAESTRO ESTÁ AQUÍ
    )

    trainer.train()

    print("--- FASE 5: GUARDANDO MODELO ---")
    final_model_path = "./models/fake_radar_final"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path) 
    print(f"¡Entrenamiento exitoso! Modelo guardado en {final_model_path}")
