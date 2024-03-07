import copy
import itertools
from symspellpy import SymSpell, Verbosity
from BERT_converter import BertSentenceConverter
from MASK_BERT import Bert_Predicter
from cos_sim import list_cos_sim
from typing import List, Dict, Union
from spellchecker import spellchecker


class SentenceBestPrediction:
    def __init__(self, sentence: str, model_bert: str, sentence_model: str, device_number: int):
        self.sen_org = sentence
        self.model_bert = model_bert
        self.sentence_model = sentence_model
        self.device_number = device_number
        self.mask_predicter = Bert_Predicter(model_bert, device_number)
        self.sentence_embedding_model = BertSentenceConverter(sentence_model, device_number)

    def set_sen_org(self, sen_org):
        self.sen_org=sen_org

    def mask_sentence(self, sentence: List[Dict[str, Union[str, int]]]):
        sen_words = []
        sen_to_predict = {}
        mask_token_index = []
        for token in sentence:
            token_in = token["token"]
            if token["spellout"] == "unknown":
                mask_token_index.append(token["index"])
            elif token["spellout"] == "wrong":
                token_in = token["suggestion"]
            sen_words.append(token_in)
        for c, index in enumerate(mask_token_index):
            sen_i = copy.deepcopy(sen_words)
            sen_i[index] = "[MASK]"
            sen_to_predict[f"sen{c}"] = {
                "sentence": " ".join(sen_i),
                "index": index,
                "begin": sentence[index]["begin"],
                "end": sentence[index]["end"],
            }
        return sen_to_predict, sen_words

    def get_Mask_prediction(self, sentence: Dict[str, Dict[str, Union[str, int]]]) -> dict:
        for sen_i in sentence:
            sentence[sen_i]["Prediction"] = self.mask_predicter.mask_prediction(sentence[sen_i]["sentence"], 10)
        return sentence

    def get_sentence_sim(self, sentence: dict):
        vec_sen_org = self.sentence_embedding_model.encode_to_vec(self.sen_org)[0]
        all_sen = {}
        for sen_i in sentence:
            all_sen[sen_i] = {}
            pred_sen = []
            for pred_i in sentence[sen_i]["Prediction"]:
                sen_changed = sentence[sen_i]["sentence"].replace("[MASK]", pred_i).replace(" ##", "")
                all_sen[sen_i][sen_changed] = pred_i
                pred_sen.append(sen_changed)
            vec_list_org = list(itertools.repeat(vec_sen_org, len(pred_sen)))
            vec_list_pred = self.sentence_embedding_model.encode_to_vec(pred_sen)
            cos_sim = list_cos_sim(vec_list_org, vec_list_pred)
            sentence[sen_i]["cos_sim"] = {}
            for count, sim_i in enumerate(cos_sim):
                sentence[sen_i]["cos_sim"][all_sen[sen_i][pred_sen[count]]] = sim_i
        return sentence

    def get_best_word(self, sentence:dict, sen_sym_spell: list):
        best_sim_ad = 0.0
        best_sen = ""
        best_word = ""
        best_cos = 0
        best_pred = 0
        predictions = {
            "Bert": {
                "word": "",
                "probability": 0.0
            },
            "SenBert": {
                "word": "",
                "probability": 0.0
            },
            "Sen-Cos-Bert": {
                "word": "",
                "probability": 0.0
            }
        }
        for sen_i in sentence:
            for pred_i in sentence[sen_i]["Prediction"]:
                cos_i = sentence[sen_i]["cos_sim"][pred_i]
                bert_pred = float(sentence[sen_i]["Prediction"][pred_i])
                best_i = float(cos_i)*bert_pred

                if predictions["Bert"]["probability"] < bert_pred:
                    predictions["Bert"]["probability"] = bert_pred
                    predictions["Bert"]["word"] = pred_i
                if predictions["SenBert"]["probability"] < cos_i:
                    predictions["SenBert"]["probability"] = cos_i
                    predictions["SenBert"]["word"] = pred_i
                if predictions["Sen-Cos-Bert"]["probability"] < best_i:
                    predictions["Sen-Cos-Bert"]["probability"] = best_i
                    predictions["Sen-Cos-Bert"]["word"] = pred_i
                    best_sen = sentence[sen_i]["sentence"].replace("[MASK]", pred_i).replace(" ##", "")
            index = sentence[sen_i]["index"]
            sen_sym_spell[index]["toolPred"] = copy.deepcopy(predictions)
            print(f"Best Prediction for {self.sen_org}: {best_word} with {best_sim_ad}; cos_sim{best_cos} prediction_possibility {best_pred} => {best_sen}")
        return sen_sym_spell


if __name__ == '__main__':
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = "de-100k.txt"
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    except Exception as ex:
        print(f"de-100k.txt, Error: {ex}")
        exit()
    sentence = ["Ich", "habe", "im", "Landtag3", "L34t3", "angesprochen", ",", "ich", "Int3ll3g3nt", "halte", "!"]
    begin = [0, 4, 9, 12, 21, 27, 40, 42, 46, 50, 54, 66, 72]
    end = [3, 8, 11, 20, 26, 39, 41, 45, 49, 53, 65, 71, 73]
    sen_org = " ".join(sentence)
    spell_out = spellchecker(sentence, begin, end, sym_spell, lower_case=True)
    # sen_org = "I am the best gIrl3 in the world!"
    # sen_test = {
    #     "sen1": {
    #         "sentence": "I am the best [MASK] in the world!",
    #         "index": 5
    #     },
    #     # "sen2": {
    #     #     "sentence": "I am the best gI [MASK] in the world!",
    #     #     "index": 6
    #     # },
    #     # "sen3":{
    #     #     "sentence": "I am the best [MASK] rl3 in the world!",
    #     #     "index": 5,
    #     # }
    # }
    sen_pred = SentenceBestPrediction(sen_org, "bert-base-uncased", "all-mpnet-base-v2", 0)
    sen_test, sen_org = sen_pred.mask_sentence(spell_out)
    sen_pred.set_sen_org(sen_org)
    pred_sentences = sen_pred.get_Mask_prediction(sen_test)
    cos_sim_sentences = sen_pred.get_sentence_sim(pred_sentences)
    sen_pred.get_best_word(cos_sim_sentences, spell_out)

