SUPPORTED_MODEL = {
    "dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-2-exact-ep13-cp402727": {
        "version": "v1.0_ep13_cp402727",
        "type": "local",
        "path": "/models/dbaumartz/philschmid_distilbert-base-multilingual-cased-sentiment-2-finetuned-de-3sentiment-exact/checkpoint-402727",
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
