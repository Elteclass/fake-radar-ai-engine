import os
from google.cloud import storage
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def upload_folder_to_gcp(bucket_name: str, source_folder: str, destination_folder: str):
    """Sube un directorio completo recursivamente a Google Cloud Storage."""
    print(f"Iniciando el respaldo del modelo en gs://{bucket_name}/{destination_folder} ...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Recorrer todos los archivos dentro de la carpeta del modelo
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            local_file = os.path.join(root, file)
            
            # Crear la ruta relativa para mantener la estructura de carpetas en la nube
            rel_path = os.path.relpath(local_file, source_folder)
            blob_path = os.path.join(destination_folder, rel_path).replace("\\", "/") 
            
            blob = bucket.blob(blob_path)
            print(f"Subiendo archivo pesado: {rel_path} ...")
            
            # Subir el archivo físico
            blob.upload_from_filename(local_file)
            
    print("✅ ¡Misión Cumplida! Modelo XLM-RoBERTa respaldado de forma segura en Google Cloud Storage.")

if __name__ == "__main__":
    GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    
    if not GCP_BUCKET_NAME:
        raise ValueError("Error CRÍTICO: GCP_BUCKET_NAME no encontrado en el archivo .env")
        
    LOCAL_MODEL_DIR = "./models/fake_radar_final"
    # Lo guardamos en una carpeta especial en tu Bucket
    GCP_MODEL_DIR = "models/fake_radar_v1" 
    
    upload_folder_to_gcp(GCP_BUCKET_NAME, LOCAL_MODEL_DIR, GCP_MODEL_DIR)
