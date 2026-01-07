from datasets import load_dataset
from transformers import Trainer
from transformers import TrOCRProcessor
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List
import torch

HF_TOKEN = 
WANDB_API_KEY = 

from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataclasses import dataclass
from typing import Any, Dict, List
import torch

def main():
    # Verificare GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimizări GPU
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    ds = load_dataset("toghrultahirov/handwritten_text_ocr", split="train[:100]")

    # Încărcăm procesorul TrOCR pentru scris de mână - PREPROCESARE
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    # Funcția de preprocesare a datelor
    def preprocess_data(example):
        # Deschidem imaginea și o convertim în RGB
        image = example['image'].convert("RGB")

        # Procesăm imaginea CU return_tensors="pt" pentru a obține tensor PyTorch direct
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0) 

        # Procesăm textul (deja cu return_tensors="pt")
        labels = processor.tokenizer(
            example["text"],
            padding="max_length",
            max_length=14,  # textul tokenizat, cu padding și trunchiere pentru a avea o lungime fixă de 14 tokeni
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return {
            "pixel_values": pixel_values,  # Acum e tensor [3, H, W]
            "labels": labels
        }

    # Aplicăm funcția de preprocesare asupra dataset-ului
    # map este o metodă care permite aplicarea unei funcții pe fiecare element al setului de date, transformându-l în formatul necesar pentru antrenarea modelului
    dataset = ds.map(preprocess_data, remove_columns=ds.column_names, num_proc=1)  # num_proc=1 evită probleme paralele

    # 3. Definim colatorul de date (cu conversie robustă la tensor)
    @dataclass
    class DataCollatorForOCR:
        processor: Any
        padding: bool = True
        max_length: int = 14  # aceeași valoare ca în preprocess_data

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            # Conversie explicită la tensor pentru siguranță (rezolvă NumPy/list mismatches)
            pixel_values = [torch.as_tensor(feature["pixel_values"]) for feature in features]
            labels = [torch.as_tensor(feature["labels"]) for feature in features]
            
            # Pregătim batch-ul
            batch = {
                "pixel_values": torch.stack(pixel_values),
                "labels": torch.stack(labels)
            }
            return batch

    # 4. Inițializăm colatorul
    data_collator = DataCollatorForOCR(processor=processor)

    # Definirea modelului și a procesului de fine-tuning
    # Încărcăm modelul TrOCR pre-antrenat
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.to(device)  # Mută pe GPU/CPU

    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_trocr_finetuned",
        run_name="experiment_1_fixed",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        predict_with_generate=True,
        eval_strategy="epoch",  # Fix pentru warning
        logging_dir="./logs",
        save_total_limit=1,
        num_train_epochs=3,
        fp16=True if device == "cuda" else False,  # Mixed precision pentru VRAM mic
        dataloader_pin_memory=False,  # Economisește memorie
        remove_unused_columns=False,
    )

    # Definim trainerul
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,  # Adăugat data_collator pentru a fi folosit în trainer
    )

    # Antrenarea modelului
    decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.decoder_start_token_id = decoder_start_token_id

    pad_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = pad_token_id
    trainer.train()

    # SALVARE MODELUL ANTRENAT LOCAL
    # După antrenare, salvăm modelul final complet (inclusiv configurația) într-un director local.
    # De asemenea, salvăm procesorul (processor) pentru a putea fi încărcat ulterior împreună cu modelul.
    # Directorul va conține toate fișierele necesare pentru a încărca modelul cu from_pretrained().
    model_save_path = "./trocr_finetuned_model"  # Specifică calea unde vrei să salvezi (poate fi schimbată)
    model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)
    
    print(f"Modelul antrenat a fost salvat local în: {model_save_path}")
    print("Poți încărca modelul ulterior cu: model = VisionEncoderDecoderModel.from_pretrained('{}')".format(model_save_path))
    print("Și procesorul cu: processor = TrOCRProcessor.from_pretrained('{}')".format(model_save_path))

if __name__ == '__main__':
    main()