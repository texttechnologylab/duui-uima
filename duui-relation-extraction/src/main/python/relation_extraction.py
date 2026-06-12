"""
Pipeline für Huggingface Model Loader
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

class RebelRelationExtraction:

    def __init__(self, model_path: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    def extract_relations(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=4
        )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        relations = [self.parse_triplets(text) for text in decoded]
        return relations

    @staticmethod
    def parse_triplets(text):
        triplets = []
        # Splitte Triplets durch Doppel-Leerzeichen oder Tabs
        raw_triplets = re.split(r"\t+|\s{2,}", text.strip())

        # Prüfen, ob genug Teile da sind
        if len(raw_triplets) % 3 != 0:
            # Kann passieren bei mehreren Triplets, Fehlermeldung ignorieren
            print("[WARN] Triplet-Parsing unvollständig:", text)
            return triplets

        # Jedes 3er-Paket = Subject, Object, Predicate
        for i in range(0, len(raw_triplets), 3):
            subject = raw_triplets[i].strip()
            object_ = raw_triplets[i + 1].strip()
            predicate = raw_triplets[i + 2].strip()
            triplets.append({
                "subject": subject,
                "predicate": predicate,
                "object": object_,
                "confidence": 1.0
            })

        return triplets


class KnowGLRelationExtraction:

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    def extract_relations(self, texts):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=4
        )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        relations = [self.parse_triplets(text) for text in decoded]
        return relations

    @staticmethod
    def parse_triplets(text):
        """
        Parse KnowGL output:
        [(subj_mention # subj_label # subj_type) | relation | (obj_mention # obj_label # obj_type)]$
        """
        triplets = []

        raw_triplets = text.split("$")

        for triple in raw_triplets:
            triple = triple.strip()

            match = re.match(
                r"\[\((.*?)#(.*?)#(.*?)\)\s*\|\s*(.*?)\s*\|\s*\((.*?)#(.*?)#(.*?)\)\]",
                triple
            )

            if not match:
                continue

            subj_mention, subj_label, subj_type, relation, obj_mention, obj_label, obj_type = match.groups()

            triplets.append({
                "subject": subj_label.strip(),
                "predicate": relation.strip(),
                "object": obj_label.strip(),
                "subject_mention": subj_mention.strip(),
                "object_mention": obj_mention.strip(),
                "subject_type": subj_type.strip(),
                "object_type": obj_type.strip(),
                "confidence": 1.0
            })

        return triplets
