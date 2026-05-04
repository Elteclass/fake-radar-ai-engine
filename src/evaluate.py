import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

def evaluate_model():
    print("Cargando el modelo entrenado (Fake Radar V2)...")
    
    # --- AQUÍ ESTÁ EL CAMBIO ---
    model_path = "./models/fake_radar_v2" 
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        print("Error CRÍTICO: No se encontró el modelo en local.")
        return

    print("Descargando lote de noticias de prueba...")
    dataset = load_dataset("IsaacRodgz/Fake-news-latam-omdena", split="test")
    df_test = dataset.to_pandas()
    
    df_test = df_test.rename(columns={'Content': 'text'})
    df_test['label'] = df_test['Corrected_label'].map({'Fake': 1, 'True': 0})
    df_test = df_test.dropna(subset=['text', 'label'])
    df_test['label'] = df_test['label'].astype(int)

    print(f"Evaluando {len(df_test)} noticias. Esto tomará unos segundos...")
    
    model.eval() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    true_labels = df_test['label'].tolist()
    texts = df_test['text'].tolist()

    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)

    accuracy = accuracy_score(true_labels, predictions)
    
    print("\n" + "=".center(50, "="))
    print(f"🎯 EXACTITUD (ACCURACY) GLOBAL V2: {accuracy * 100:.2f}%")
    print("=".center(50, "="))
    
    print("\n📊 REPORTE DETALLADO DE MÉTRICAS V2:")
    print(classification_report(true_labels, predictions, target_names=['Verdadera (0)', 'Fake News (1)']))

if __name__ == "__main__":
    evaluate_model()
