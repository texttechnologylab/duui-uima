"""
Das Modell stammt von Huggingface https://huggingface.co/blanchefort/rubert-base-cased-sentiment/tree/main

Kann nicht getestet werden, da das Dockerimage sich aufgrund vom RUN des ersten Modells nicht bauen lässt,
wenn man das Dockerfile im Ordner duui-transformers-sentiment hat. Wenn Dockerfile im Unterordner docker ist, dann
die Pfade der Copy-Aufrufe nicht gefunden.

Mittels preload(RUBERT) wird das Modell aber auf jeden Fall geladen.
"""



def deeppavlov_preprocess(text: str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return text.strip()

SUPPORTED_MODEL = {
    "blanchefort/rubert-base-cased-sentiment": {
        "version": "main",
        "max_length": 512,
        "mapping": {
            "POSITIVE": 1,
            "NEUTRAL": 0,
            "NEGATIVE": -1
        },
        "3sentiment": {
            "pos": ["POSITIVE"],
            "neu": ["NEUTRAL"],
            "neg": ["NEGATIVE"]
        },
        "preprocess": deeppavlov_preprocess,
        "languages": ["ru"]
    }
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)