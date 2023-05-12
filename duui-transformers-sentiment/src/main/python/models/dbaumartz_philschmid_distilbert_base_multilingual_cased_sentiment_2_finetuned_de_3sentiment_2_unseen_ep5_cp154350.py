SUPPORTED_MODEL = {
    "dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-2-unseen-ep5-cp154350": {
        "version": "v1.0_ep5_cp154350",
        "type": "local",
        "path": "/models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-unseen/checkpoint-154350",
        "max_length": 512,
        "mapping": {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        },
        "3sentiment": {
            "pos": ["positive"],
            "neu": ["neutral"],
            "neg": ["negative"]
        },
        "preprocess": lambda text: text,
        "languages": ["de"]
    },
}
