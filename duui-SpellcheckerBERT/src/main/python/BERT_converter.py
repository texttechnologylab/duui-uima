from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch


class BertConverter:
    def __init__(self, model_name='bert-base-uncased', device_number=0, run_gpu=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device_number = device_number
        self.run_gpu = run_gpu
        if torch.cuda.is_available() and run_gpu:
            self.model = AutoModel.from_pretrained(model_name).to(device_number)
        else:
            self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, sentences, ret_input=False):
        if type(sentences) == str:
            sentences = [sentences]
        if torch.cuda.is_available() and self.run_gpu:
            encoded_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device_number)
        else:
            encoded_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)

        output = self.model(**encoded_input)
        if ret_input:
            return output, encoded_input
        else:
            return output

    def encode_to_vec(self, sentences):
        bert_results = self.encode(sentences)
        if torch.cuda.is_available() and self.run_gpu:
            vec_list = bert_results.last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()
        else:
            vec_list = bert_results.last_hidden_state[:, 0, :].detach().numpy().tolist()
        return vec_list


class BertSentenceConverter:
    def __init__(self, model_name='all-MiniLM-L6-v2', device_number=0, run_gpu=False):
        self.run_gpu = run_gpu
        if torch.cuda.is_available() and run_gpu:
            self.model = SentenceTransformer(model_name).to(device_number)
        else:
            self.model = SentenceTransformer(model_name)
        self.model.eval()

    def encode_to_vec(self, sentences, token=None, nlp=False):
        if type(sentences) == str:
            sentences = [sentences]

        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        if torch.cuda.is_available() and self.run_gpu:
            vec_list = embeddings.detach().cpu().numpy().tolist()
        else:
            vec_list = embeddings.detach().cpu().numpy().tolist()
        return vec_list


if __name__ == '__main__':
    bert_converter = BertConverter("bert-base-multilingual-cased", 0)
    print(len(bert_converter.encode_to_vec(["Satz1", "sentence1", "Satz1"])))
