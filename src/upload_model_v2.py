import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

def upload_folder_to_gcp(bucket_name: str, source_folder: str, destination_folder: str):
    print(f"Iniciando el respaldo del modelo V2 en gs://{bucket_name}/{destination_folder} ...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            local_file = os.path.join(root, file)
            rel_path = os.path.relpath(local_file, source_folder)
            blob_path = os.path.join(destination_folder, rel_path).replace("\\", "/") 
            
            blob = bucket.blob(blob_path)
            print(f"Subiendo archivo: {rel_path} ...")
            blob.upload_from_filename(local_file)
            
    print("✅ ¡Misión Cumplida! Modelo V2 respaldado de forma segura en Google Cloud Storage.")

if __name__ == "__main__":
    GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    
    # Apuntamos a las carpetas V2
    LOCAL_MODEL_DIR = "./models/fake_radar_v2"
    GCP_MODEL_DIR = "models/fake_radar_v2" 
    
    upload_folder_to_gcp(GCP_BUCKET_NAME, LOCAL_MODEL_DIR, GCP_MODEL_DIR)
