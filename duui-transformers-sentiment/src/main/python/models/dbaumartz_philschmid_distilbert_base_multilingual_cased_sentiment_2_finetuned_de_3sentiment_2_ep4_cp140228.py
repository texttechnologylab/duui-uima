SUPPORTED_MODEL = {
    "dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-2-ep4-cp140228": {
        "version": "v1.0_ep4_cp140228",
        "type": "local",
        "path": "/models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-140228",
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
