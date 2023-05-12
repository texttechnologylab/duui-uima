SUPPORTED_MODEL = {
    "j-hartmann/sentiment-roberta-large-english-3-classes": {
        "version": "f995433eb6d79d26702ab9335bfde472a9933ee4",
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
        "languages": ["en"]
    }
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)
