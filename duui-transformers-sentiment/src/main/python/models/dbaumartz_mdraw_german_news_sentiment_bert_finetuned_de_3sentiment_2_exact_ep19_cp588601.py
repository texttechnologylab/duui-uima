if __name__ != "__main__":
    from .oliverguhr_german_sentiment_bert import oliverguhr_clean_text

SUPPORTED_MODEL = {
    "dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-2-exact-ep19-cp588601": {
    "version": "v1.0_ep19_cp588601",
    "type": "local",
    "path": "/models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-588601",
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
