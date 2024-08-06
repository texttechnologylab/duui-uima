# See "TweetEval" paper sec. 2.2 and https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
def cardiffnlp_preprocess(text: str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


SUPPORTED_MODEL = {
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": {
        "version": "f3e34b6c30bf27b6649f72eca85d0bbe79df1e55",
        "max_length": 512,
        "mapping": {
            "Positive": 1,
            "Neutral": 0,
            "Negative": -1
        },
        "3sentiment": {
            "pos": ["Positive"],
            "neu": ["Neutral"],
            "neg": ["Negative"]
        },
        "preprocess": lambda text: cardiffnlp_preprocess(text),
        "languages": ["ar", "en", "fr", "de", "hi", "it", "sp", "pt"]
    },
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)
