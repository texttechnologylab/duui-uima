if __name__ != "__main__":
    from .cardiffnlp_twitter_roberta_base_sentiment import cardiffnlp_preprocess

SUPPORTED_MODEL = {
    "dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-2-exact-ep2-cp61958": {
        "version": "v1.0_ep2_cp61958",
        "type": "local",
        "path": "/models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-exact/checkpoint-61958",
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
