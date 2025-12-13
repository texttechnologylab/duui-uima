import torch
from transformers import AutoTokenizer, AutoModel
from skdim.id import MLE
from tqdm import tqdm
from InstrinsicDim import PHD


class PDHScorer:
    def __init__(self, model_name, device, alpha, metric, n_points=9, n_reruns=3):
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.MIN_SUBSAMPLE = 40
        self.INTERMEDIATE_POINTS = 7
        self.phd = PHD(alpha=alpha, metric=metric, n_reruns=n_reruns, n_points=n_points, n_points_min=self.MIN_SUBSAMPLE)
        self.MLE_solver = MLE()

    def _preprocess_text(self, text):
        inputs = self.tokenizer(text.replace('\n', ' ').replace('  ', ' '), truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outp = self.model(**inputs)
        return outp, inputs

    def _get_mle_single(self, outp):
        return self.MLE_solver.fit_transform(outp[0][0].numpy()[1:-1])

    def _get_phd_single(self, outp, inputs):
        # We omit the first and last tokens (<CLS> and <SEP> because they do not directly correspond to any part of the)
        mx_points = inputs['input_ids'].shape[1] - 2

        mn_points = self.MIN_SUBSAMPLE
        step = (mx_points - mn_points) // self.INTERMEDIATE_POINTS

        return self.phd.fit_transform(outp[0][0].numpy()[1:-1], min_points=mn_points, max_points=mx_points - step, point_jump=step)

    def process_texts(self, texts):
        """
        Calculate the PDH score for a list of texts.
        :param texts: List of strings to be scored.
        :return: List of PDH scores.
        """
        scores = []
        for text in tqdm(texts, desc="Calculating PDH scores"):
            try:
                outp, inputs = self._preprocess_text(text)
                phd_score = self._get_phd_single(outp, inputs)
                mle_score = self._get_mle_single(outp)
                score_i = {"PHD": phd_score, "MLE": mle_score}
                scores.append(score_i)
            except Exception as e:
                print(f"Error processing text: {text[:50]}... Error: {e}")
                scores.append({"PHD": None, "MLE": None})
        return scores


if __name__ == '__main__':
    device_i = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model = AutoModel.from_pretrained("FacebookAI/xlm-roberta-base").to(device_i)
    model_name = "FacebookAI/xlm-roberta-base"
    scorer = PDHScorer(model_name=model_name, device=device_i, alpha=1.0, metric='euclidean', n_points=9, n_reruns=3)
    texts = [
        "Speaking of festivities, there is one day in China that stands unrivaled - the first day of the Lunar New Year, commonly referred to as the Spring Festival. Even if you're generally uninterested in celebratory events, it's hard to resist the allure of the family reunion dinner, a quintessential aspect of the Spring Festival. Throughout the meal, family members raise their glasses to toast one another, expressing wishes for happiness, peace, health, and prosperity in the upcoming year.",
        "The Spring Festival, also known as the Lunar New Year, is a time of great significance in Chinese culture.",
    ]
    scores = scorer.process_texts(texts)
    for i, text in enumerate(texts):
        print(f"Text: {text}\nPHD Score: {scores[i]['PHD']}\nMLE Score: {scores[i]['MLE']}\n")

    # sample_text = "Speaking of festivities, there is one day in China that stands unrivaled - the first day of the Lunar New Year, commonly referred to as the Spring Festival. Even if you're generally uninterested in celebratory events, it's hard to resist the allure of the family reunion dinner, a quintessential aspect of the Spring Festival. Throughout the meal, family members raise their glasses to toast one another, expressing wishes for happiness, peace, health, and prosperity in the upcoming year."
    # print("MLE estimation of the Intrinsic dimension of sample text is ", get_mle_single(sample_text))

