if __name__ != "__main__":
    from .oliverguhr_german_sentiment_bert import oliverguhr_clean_text

SUPPORTED_MODEL = {
    "dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-2-exact-ep8-cp247832": {
    "version": "v1.0_ep8_cp247832",
    "type": "local",
    "path": "/models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-exact/checkpoint-247832",
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
