if __name__ != "__main__":
    from .cardiffnlp_twitter_roberta_base_sentiment import cardiffnlp_preprocess

SUPPORTED_MODEL = {
    "dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-2-unseen-ep5-cp175050": {
        "version": "v1.0_ep5_cp175050",
        "type": "local",
        "path": "/models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen/checkpoint-175050",
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
