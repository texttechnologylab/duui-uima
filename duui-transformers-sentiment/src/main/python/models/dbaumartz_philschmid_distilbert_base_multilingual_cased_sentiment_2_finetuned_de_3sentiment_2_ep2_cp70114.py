SUPPORTED_MODEL = {
    "dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-2-ep2-cp70114": {
        "version": "v1.0_ep2_cp70114",
        "type": "local",
        "path": "/models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment/checkpoint-70114",
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
