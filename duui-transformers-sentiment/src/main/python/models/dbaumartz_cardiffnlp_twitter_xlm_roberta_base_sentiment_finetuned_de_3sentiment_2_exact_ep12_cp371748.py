if __name__ != "__main__":
    from .cardiffnlp_twitter_roberta_base_sentiment import cardiffnlp_preprocess

SUPPORTED_MODEL = {
    "dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-2-exact-ep12-cp371748": {
    "version": "v1.0_ep12_cp371748",
    "type": "local",
    "path": "/models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-371748",
    "max_length": 512,
    "mapping": {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    },
    "3sentiment": {
        "pos": ["Positive"],
        "neu": ["Neutral"],
        "neg": ["Negative"]
    },
    "preprocess": lambda text: cardiffnlp_preprocess(text),
    "languages": ["de"]
    },
}

