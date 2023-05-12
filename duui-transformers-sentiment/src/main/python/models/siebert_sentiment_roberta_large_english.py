SUPPORTED_MODEL = {
    "siebert/sentiment-roberta-large-english": {
        "version": "6eac71655a474ee4d6d0eee7fa532300c537856d",
        "max_length": 512,
        "mapping": {
            "POSITIVE": 1,
            "NEGATIVE": -1
        },
        "3sentiment": {
            "pos": ["POSITIVE"],
            "neu": [],
            "neg": ["NEGATIVE"]
        },
        "preprocess": lambda text: text,
        "languages": ["en"]
    },
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)
