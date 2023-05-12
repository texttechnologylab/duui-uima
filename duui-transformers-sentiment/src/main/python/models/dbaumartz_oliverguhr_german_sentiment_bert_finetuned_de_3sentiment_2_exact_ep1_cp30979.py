if __name__ != "__main__":
    from .oliverguhr_german_sentiment_bert import oliverguhr_clean_text

SUPPORTED_MODEL = {
    "dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-2-exact-ep1-cp30979": {
        "version": "v1.0_ep1_cp30979",
        "type": "local",
        "path": "/models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-30979",
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
