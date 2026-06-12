"""
Python-Code um das Modell manuell runter zu laden, damit es dann im Offline-Modus läuft.
Das Modell wird im Ordner main/python/offline_models abgelegt. Das Dockerfile greift darauf zu.
"""


from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pathlib import Path

# Modellname
model_name = "seara/rubert-tiny2-russian-emotion-detection-cedr"

# Tokenizer und Modell herunterladen (online)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
local_dir = Path("../../main/python/offline_models/rubert_tiny2_russian")

model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)

print(f"Modell wurde heruntergeladen und in {local_dir} gespeichert.")