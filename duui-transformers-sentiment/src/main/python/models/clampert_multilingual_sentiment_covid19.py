SUPPORTED_MODEL = {
    "clampert/multilingual-sentiment-covid19": {
        "version": "eea3f8e26d2828dbf9f0f1d939dd868396ec863c",
        "max_length": 512,
        "mapping": {
            "positive": 1,
            "negative": -1
        },
        "3sentiment": {
            "pos": ["positive"],
            "neu": [],
            "neg": ["negative"]
        },
        "preprocess": lambda text: text,
        # TODO
        "languages": ["en", "fr", "de"]
    },
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)
