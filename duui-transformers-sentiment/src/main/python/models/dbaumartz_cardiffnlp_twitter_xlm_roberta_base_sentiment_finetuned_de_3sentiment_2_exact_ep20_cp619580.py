if __name__ != "__main__":
    from .cardiffnlp_twitter_roberta_base_sentiment import cardiffnlp_preprocess

SUPPORTED_MODEL = {
    "dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-2-exact-ep20-cp619580": {
    "version": "v1.0_ep20_cp619580",
    "type": "local",
    "path": "/models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-619580",
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

