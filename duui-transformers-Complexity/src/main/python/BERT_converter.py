from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


class BertConverter:
    def __init__(self, model_name, device_number):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device_number = device_number
        self.model = AutoModel.from_pretrained(model_name).to(device_number)
        self.model.eval()

    def encode(self, sentences, ret_input=False):
        if type(sentences) == str:
            sentences = [sentences]

        # for sentence in sentences:
        #     if len(sentence) > 0 and sentence[-1] != ".":
        #         sentence += "."

        encoded_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True,
                                       max_length=512).to(self.device_number)

        output = self.model(**encoded_input)
        if ret_input:
            return output, encoded_input
        else:
            return output

    def encode_to_vec(self, sentences):
        bert_results = self.encode(sentences)
        return bert_results.last_hidden_state[:, 0, :].detach().cpu().numpy().tolist()


class BertSentenceConverter:
    def __init__(self, model_name, device_number):
        self.model = SentenceTransformer(model_name).to(device_number)
        self.model.eval()

    def encode_to_vec(self, sentences, token=None, nlp=False):
        if type(sentences) == str:
            sentences = [sentences]

        for sentence in sentences:
            if len(sentence) > 0 and sentence[-1] != ".":
                sentence += "."

        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        return embeddings.detach().cpu().numpy().tolist()


class BertSentence:

    def __init__(self, model_name, device_number):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device_number = device_number
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name).to(device_number)
        self.model.eval()

    def encode_to_vec(self, sentences, ret_input=False):
        if type(sentences) == str:
            sentences = [sentences]

        # for sentence in sentences:
        #     if len(sentence) > 0 and sentence[-1] != ".":
        #         sentence += "."

        encoded_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True,
                                       max_length=512).to(self.device_number)

        output = self.model(**encoded_input)
        if self.model_name == "intfloat/multilingual-e5-base":
            last_hidden = output.last_hidden_state.masked_fill(~encoded_input['attention_mask'][..., None].bool(), 0.0)
            output = last_hidden.sum(dim=1) / encoded_input['attention_mask'].sum(dim=1)[..., None]
        else:
            output = output.pooler_output
        if ret_input:
            return output.detach().cpu().numpy().tolist(), encoded_input
        else:
            return output.detach().cpu().numpy().tolist()
