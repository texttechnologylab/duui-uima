if __name__ != "__main__":
    from .oliverguhr_german_sentiment_bert import oliverguhr_clean_text

SUPPORTED_MODEL = {
    "dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-2-unseen-adapter-pfeiffer-ep10-cp125780": {
        "version": "v1.0_ep10_cp125780",
        "type": "adapter",
        "model_type": "huggingface",
        "model_name": "mdraw/german-news-sentiment-bert",
        "model_version": "7b4abebe1c3fcfbc62dc0435e480807a80c18210",
        "adapter_type": "local",
        "adapter_path": "/models/dbaumartz/mdraw_german-news-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-125780/3sentiment-unseen",
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
