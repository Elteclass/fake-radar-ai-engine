import pandas as pd
from google.cloud import storage
import os
from dotenv import load_dotenv

# Cargar las variables ocultas del archivo .env
load_dotenv()

def create_base_dataset(output_path: str):
    print("Iniciando la construcción del dataset de Fake Radar...")
    data = [
        {"text": "El Instituto Nacional Electoral (INE) anunció las nuevas fechas para la actualización de la credencial para votar con fotografía en todos los módulos del país.", "label": 0},
        {"text": "¡ESCÁNDALO! Descubren fraude millonario y urnas falsas escondidas bajo la tierra, el gobierno intenta silenciar a los testigos y cerrar las redes sociales.", "label": 1},
        {"text": "La Secretaría de Salud reportó una disminución del 15% en los casos de influenza estacional durante el primer trimestre del año, según el boletín epidemiológico.", "label": 0},
        {"text": "¡ALERTA MÁXIMA! El nuevo virus creado en laboratorio escapó y aseguran que convierte a las personas en zombies, comparte antes de que lo borren.", "label": 1}
    ]
    df = pd.DataFrame(data)
    df['text'] = df['text'].str.strip()
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Dataset local guardado exitosamente en: {output_path}")

def upload_to_gcp(bucket_name: str, source_file_name: str, destination_blob_name: str):
    print(f"Subiendo a gs://{bucket_name}/{destination_blob_name} ...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print("¡Subida completada con éxito a GCP!")

if __name__ == "__main__":
    LOCAL_CSV_PATH = "data/dataset_fake_radar_v1.csv"

    GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    
    if not GCP_BUCKET_NAME:
        raise ValueError("¡CRÍTICO! La variable GCP_BUCKET_NAME no se encontró en el archivo .env")
        
    GCP_DESTINATION_PATH = "datasets/dataset_fake_radar_v1.csv"
    
    create_base_dataset(LOCAL_CSV_PATH)
    upload_to_gcp(GCP_BUCKET_NAME, LOCAL_CSV_PATH, GCP_DESTINATION_PATH)
