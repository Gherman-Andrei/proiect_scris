import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Calea către modelul salvat (folderul trocr_finetuned_model)
model_path = "./trocr_finetuned_model"#

# Încărcăm procesorul și modelul
processor = TrOCRProcessor.from_pretrained(model_path)
model = VisionEncoderDecoderModel.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Model încărcat pe {device}")

# Funcție pentru recunoaștere text
def ocr_handwritten(image_path: str) -> str:
    # Deschidem imaginea
    image = Image.open(image_path).convert("RGB")
    
    # Preprocesare
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generare text
    generated_ids = model.generate(
        pixel_values,
        max_length=64,
        num_beams=5,
        early_stopping=True
    )
    
    # Decodare
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

if __name__ == "__main__":
    
    image_path = "image.jpg" 
    try:
        rezultat = ocr_handwritten(image_path)
        print("\n" + "="*50)
        print("TEXT RECUNOSCUT:")
        print(f"'{rezultat}'")
        print("="*50)
    except Exception as e:
        print(f"Eroare: {e}")
        print("Verifică dacă imaginea există și calea este corectă.")