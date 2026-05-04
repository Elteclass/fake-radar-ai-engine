import pandas as pd
from datasets import load_dataset
from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()

def download_and_clean_dataset(output_path: str):
    print("Iniciando extracción múltiple desde Hugging Face Hub...")
    
    datasets_list = []
    
    # --- DATASET 1: mariagrandury ---
    try:
        print("Descargando Dataset 1 (mariagrandury)...")
        ds1 = load_dataset("mariagrandury/fake_news_corpus_spanish", split="test").to_pandas()
        ds1 = ds1.rename(columns={'TEXT': 'text'})
        ds1['label'] = ds1['CATEGORY'].astype(int) # True/False a 1/0
        datasets_list.append(ds1[['text', 'label']])
        print(f"Dataset 1 listo: {len(ds1)} registros.")
    except Exception as e:
        print(f"Error en Dataset 1: {e}")

    # --- DATASET 2: sayalaruano ---
    try:
        print("Descargando Dataset 2 (sayalaruano)...")
        ds2 = load_dataset("sayalaruano/FakeNewsSpanish_Kaggle2", split="train").to_pandas()
        ds2 = ds2.rename(columns={'texto': 'text'})
        # Convertir 'fake' a 1 y 'real' a 0
        ds2['label'] = ds2['clase'].map({'fake': 1, 'real': 0}).fillna(0).astype(int) 
        datasets_list.append(ds2[['text', 'label']])
        print(f"Dataset 2 listo: {len(ds2)} registros.")
    except Exception as e:
        print(f"Error en Dataset 2: {e}")

    # --- FUSIÓN (MERGE) ---
    print("Fusionando fuentes de datos...")
    if datasets_list:
        # Concatenar todos los dataframes en uno solo
        df_master = pd.concat(datasets_list, ignore_index=True)
        
        # Limpieza general
        df_master = df_master.dropna(subset=['text'])
        df_master['text'] = df_master['text'].astype(str).str.strip()
        df_master['text'] = df_master['text'].str.replace(r'\s+', ' ', regex=True)
        
        # Eliminar posibles duplicados que existan entre ambos datasets
        df_master = df_master.drop_duplicates(subset=['text'])
        
        # Guardar el master dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_master.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Éxito rotundo: Dataset Maestro procesado con {len(df_master)} noticias únicas. Guardado en {output_path}")
    else:
        print("Error crítico: No se pudo descargar ningún dataset.")

def upload_to_gcp(bucket_name: str, source_file_name: str, destination_blob_name: str):
    print(f"Enviando base de conocimiento a gs://{bucket_name}/{destination_blob_name} ...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print("¡Súper Base de conocimiento asegurada en la nube!")

if __name__ == "__main__":
    LOCAL_CSV_PATH = "data/open_source_spanish.csv"
    GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    
    if not GCP_BUCKET_NAME:
        raise ValueError("Error: GCP_BUCKET_NAME no encontrado en el archivo .env")
        
    GCP_DESTINATION_PATH = "datasets/open_source_spanish.csv"
    
    download_and_clean_dataset(LOCAL_CSV_PATH)
    upload_to_gcp(GCP_BUCKET_NAME, LOCAL_CSV_PATH, GCP_DESTINATION_PATH)
