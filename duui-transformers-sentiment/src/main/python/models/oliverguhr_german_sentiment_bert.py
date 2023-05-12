import re


# See "Guhr et al. 2020" paper sec 3.2 and https://github.com/oliverguhr/german-sentiment-lib/blob/master/germansentiment/sentimentmodel.py#L40
def oliverguhr_clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = oliverguhr_clean_http_urls.sub('', text)
    text = oliverguhr_clean_at_mentions.sub('', text)
    text = oliverguhr_replace_numbers(text)
    text = oliverguhr_clean_chars.sub('', text)  # use only text chars
    text = ' '.join(text.split())  # substitute multiple whitespace with single whitespace
    text = text.strip().lower()
    return text


def oliverguhr_replace_numbers(text: str) -> str:
    return text.replace("0", " null").replace("1", " eins").replace("2", " zwei") \
        .replace("3", " drei").replace("4", " vier").replace("5", " fünf") \
        .replace("6", " sechs").replace("7", " sieben").replace("8", " acht") \
        .replace("9", " neun")


SUPPORTED_MODEL = {
    "oliverguhr/german-sentiment-bert": {
        "version": "c5c8dd0c5b966460dce1b7c5851bd90af1d2c6b6",
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


oliverguhr_clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)
oliverguhr_clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
oliverguhr_clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)
