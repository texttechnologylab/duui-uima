if __name__ != "__main__":
    from .oliverguhr_german_sentiment_bert import oliverguhr_clean_text

SUPPORTED_MODEL = {
    "dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-2-exact-ep2-cp61958": {
        "version": "v1.0_ep2_cp61958",
        "type": "local",
        "path": "/models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-61958",
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
