if __name__ != "__main__":
    from .oliverguhr_german_sentiment_bert import oliverguhr_clean_text

SUPPORTED_MODEL = {
    "mdraw/german-news-sentiment-bert": {
        "version": "7b4abebe1c3fcfbc62dc0435e480807a80c18210",
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
    }
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)
