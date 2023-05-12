if __name__ != "__main__":
    from .cardiffnlp_twitter_roberta_base_sentiment import cardiffnlp_preprocess

SUPPORTED_MODEL = {
    "dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-2-ep3-cp105171": {
        "version": "v1.0_ep3_cp105171",
        "type": "local",
        "path": "/models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment/checkpoint-105171",
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
