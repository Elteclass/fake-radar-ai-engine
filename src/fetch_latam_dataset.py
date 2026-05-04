import pandas as pd
from datasets import load_dataset
from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()

def download_latam_dataset(output_path: str):
    print("Iniciando extracción del Dataset LATAM (IsaacRodgz)...")
    
    try:
        # Descargamos el set de entrenamiento (2.78k rows)
        print("Descargando el set de entrenamiento...")
        ds_train = load_dataset("IsaacRodgz/Fake-news-latam-omdena", split="train").to_pandas()
        
        # Descargamos el set de prueba (310 rows) para tener más volumen
        print("Descargando el set de prueba...")
        ds_test = load_dataset("IsaacRodgz/Fake-news-latam-omdena", split="test").to_pandas()
        
        # Unimos ambos sets para tener todo en un solo dataframe
        df_master = pd.concat([ds_train, ds_test], ignore_index=True)
        print(f"Total de registros descargados: {len(df_master)}")
        
        print("Estandarizando formato MLOps...")
        
        # 1. Renombrar la columna Content a text
        df_master = df_master.rename(columns={'Content': 'text'})
        
        # 2. Convertir Corrected_label a label (1 para Fake, 0 para True)
        df_master['label'] = df_master['Corrected_label'].map({'Fake': 1, 'True': 0})
        
        # 3. Descartar basura (Nos quedamos solo con las columnas necesarias)
        df_master = df_master[['text', 'label']]
        
        # Limpieza rigurosa (MLOps standard)
        print("Realizando limpieza de texto...")
        df_master = df_master.dropna(subset=['text', 'label']) # Eliminar filas vacías
        df_master['label'] = df_master['label'].astype(int) # Asegurar que sean enteros
        df_master['text'] = df_master['text'].astype(str).str.strip()
        df_master['text'] = df_master['text'].str.replace(r'\s+', ' ', regex=True)
        
        # Eliminar posibles duplicados
        df_master = df_master.drop_duplicates(subset=['text'])
        
        # Guardar localmente
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_master.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Éxito: Súper Dataset LATAM procesado con {len(df_master)} noticias únicas. Guardado en {output_path}")
        
    except Exception as e:
        print(f"Error crítico en la extracción: {e}")

def upload_to_gcp(bucket_name: str, source_file_name: str, destination_blob_name: str):
    print(f"Enviando dataset maestro a gs://{bucket_name}/{destination_blob_name} ...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print("¡Súper Base de conocimiento LATAM asegurada en la nube!")

if __name__ == "__main__":
    LOCAL_CSV_PATH = "data/dataset_latam_master.csv"
    GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    
    if not GCP_BUCKET_NAME:
        raise ValueError("Error: GCP_BUCKET_NAME no encontrado en el archivo .env")
        
    GCP_DESTINATION_PATH = "datasets/dataset_latam_master.csv"
    
    download_latam_dataset(LOCAL_CSV_PATH)
    upload_to_gcp(GCP_BUCKET_NAME, LOCAL_CSV_PATH, GCP_DESTINATION_PATH)
