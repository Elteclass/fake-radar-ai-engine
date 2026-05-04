import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def test_fake_radar(text: str):
    print(f"\nAnalizando noticia: '{text}'")
    
    # 1. Cargar el modelo entrenado desde la carpeta local
    model_path = "./models/fake_radar_final"
    
    # Evitar que la librería lance warnings innecesarios
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    
    # 2. Tokenizar el texto de entrada
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # 3. Hacer la predicción sin gastar memoria extra
    with torch.no_grad():
        outputs = model(**inputs)
        # Convertir la salida cruda a porcentajes de probabilidad
        probabilities = F.softmax(outputs.logits, dim=-1)
        
    # Extraer porcentajes (Recuerda nuestro mapeo: 0 = True/Real, 1 = Fake)
    prob_real = probabilities[0][0].item() * 100
    prob_fake = probabilities[0][1].item() * 100
    
    print("-" * 50)
    print(f"Probabilidad de ser VERDADERA: {prob_real:.2f}%")
    print(f"Probabilidad de ser FAKE NEWS: {prob_fake:.2f}%")
    print("-" * 50)
    
    if prob_fake > 50:
        print("🚨 VEREDICTO DE FAKE RADAR: ¡ALERTA DE DESINFORMACIÓN!")
    else:
        print("✅ VEREDICTO DE FAKE RADAR: Noticia Confiable.")

if __name__ == "__main__":
    # Prueba 1: Tono institucional (Debería ser clasificada como Real)
    test_fake_radar("El Instituto Nacional Electoral aprobó el nuevo presupuesto para la organización de las elecciones federales del próximo año.")
    
    # Prueba 2: Sensacionalismo y Fake clásica (Debería ser clasificada como Fake)
    test_fake_radar("¡URGENTE! Se filtra audio donde el presidente pacta vender la mitad del territorio nacional por vacunas falsas. ¡Comparte antes de que nos censuren!")
