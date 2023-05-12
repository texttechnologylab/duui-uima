if __name__ != "__main__":
    from .cardiffnlp_twitter_roberta_base_sentiment import cardiffnlp_preprocess

SUPPORTED_MODEL = {
    "dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-2-exact-ep14-cp433706": {
    "version": "v1.0_ep14_cp433706",
    "type": "local",
    "path": "/models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-433706",
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

