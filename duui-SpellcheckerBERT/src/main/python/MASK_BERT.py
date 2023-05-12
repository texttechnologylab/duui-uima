import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoModel, AutoTokenizer
import logging
logging.basicConfig(level=logging.INFO)


class Bert_Predicter:
    def __init__(self, model_name="bert-base-uncased", device_number=0):
        self.device_number = device_number
        self.model = BertForMaskedLM.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        if torch.cuda.is_available():
            self.model.to(device_number)
        self.model.eval()

    def mask_prediction(self, mask_text, topk=20):
        MaskPredict = {}
        text = f"[CLS] {mask_text} [SEP]"
        if torch.cuda.is_available():
            tokenized = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=False).to("cuda")
        else:
            tokenized = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=False)
        # tokenizer.encode_plus(sentence_masked, return_tensors='pt', add_special_tokens=False)
        input_ids = tokenized['input_ids']
        input_id_list = input_ids[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_id_list)
        masked_index = tokens.index("[MASK]")
        # indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized)

        with torch.no_grad():
            out = self.model(**tokenized, output_hidden_states=True)

        out_p = torch.nn.functional.softmax(out[0][0], dim=-1)
        out_p_argsort = torch.argsort(out_p, dim=-1, descending=True)
        input_max, max_indices = torch.max(out_p, -1)

        for i, idx in enumerate(out_p_argsort[masked_index, :topk]):
            MaskPredict[self.tokenizer.convert_ids_to_tokens(idx.item())] = out_p[masked_index, idx].item()
            # print(i, idx.item(), self.tokenizer.convert_ids_to_tokens(idx.item()), f'{out_p[masked_index, idx].item():4f}')
        # topk_weight, topk_index = torch.topk(probs, topk, sorted=True)
        # for c, pred_index in enumerate(topk_index):
        #     predict_word = self.tokenizer.convert_ids_to_tokens([pred_index])[0]
        #     prob = topk_weight[c]
        #     MaskPredict[predict_word] = prob

        return MaskPredict


if __name__ == '__main__':
    text_in = "I am the best [MASK] in the world!"
    predicter = Bert_Predicter()
    predicter.mask_prediction(mask_text=text_in)
