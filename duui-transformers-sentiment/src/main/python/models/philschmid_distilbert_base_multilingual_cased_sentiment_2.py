SUPPORTED_MODEL = {
    "philschmid/distilbert-base-multilingual-cased-sentiment-2": {
        "version": "83ff874f93aacbba79642abfe2a274a3c874232b",
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
        "preprocess": lambda text: text,
        "languages": ["en", "de", "fr", "es", "zh", "ja"]
    },
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)
