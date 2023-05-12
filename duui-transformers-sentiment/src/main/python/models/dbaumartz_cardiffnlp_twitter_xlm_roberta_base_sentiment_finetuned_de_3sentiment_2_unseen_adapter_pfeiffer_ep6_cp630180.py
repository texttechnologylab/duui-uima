if __name__ != "__main__":
    from .cardiffnlp_twitter_roberta_base_sentiment import cardiffnlp_preprocess

SUPPORTED_MODEL = {
    "dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-2-unseen-adapter-pfeiffer-ep6-cp630180": {
        "version": "v1.0_ep6_cp630180",
        "type": "adapter",
        "model_type": "huggingface",
        "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "model_version": "f3e34b6c30bf27b6649f72eca85d0bbe79df1e55",
        "adapter_type": "local",
        "adapter_path": "/models/dbaumartz/cardiffnlp_twitter-xlm-roberta-base-sentiment-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-630180/3sentiment-unseen",
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
        "languages": ["de"]
    },
}
