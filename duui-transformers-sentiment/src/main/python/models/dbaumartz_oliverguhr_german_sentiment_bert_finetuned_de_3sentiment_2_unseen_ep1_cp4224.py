if __name__ != "__main__":
    from .oliverguhr_german_sentiment_bert import oliverguhr_clean_text

SUPPORTED_MODEL = {
    "dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-2-unseen-ep1-cp4224": {
        "version": "v1.0_ep1_cp4224",
        "type": "local",
        "path": "/models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen/checkpoint-4224",
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
        "preprocess": lambda text: oliverguhr_clean_text(text),
        "languages": ["de"]
    },
}
