import os.path

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from iso639 import languages
from gtts import gTTS
import whisper
import json


LANGUAGES_FLORES200 = [
"ace_Arab",  "bam_Latn",  "dzo_Tibt",  "hin_Deva",	"khm_Khmr",  "mag_Deva",  "pap_Latn",  "sot_Latn",	"tur_Latn",
"ace_Latn",  "ban_Latn",  "ell_Grek",  "hne_Deva",	"kik_Latn",  "mai_Deva",  "pbt_Arab",  "spa_Latn",	"twi_Latn",
"acm_Arab",  "bel_Cyrl",  "eng_Latn",  "hrv_Latn",	"kin_Latn",  "mal_Mlym",  "pes_Arab",  "srd_Latn",	"tzm_Tfng",
"acq_Arab",  "bem_Latn",  "epo_Latn",  "hun_Latn",	"kir_Cyrl",  "mar_Deva",  "plt_Latn",  "srp_Cyrl",	"uig_Arab",
"aeb_Arab",  "ben_Beng",  "est_Latn",  "hye_Armn",	"kmb_Latn",  "min_Arab",  "pol_Latn",  "ssw_Latn",	"ukr_Cyrl",
"afr_Latn",  "bho_Deva",  "eus_Latn",  "ibo_Latn",	"kmr_Latn",  "min_Latn",  "por_Latn",  "sun_Latn",	"umb_Latn",
"ajp_Arab",  "bjn_Arab",  "ewe_Latn",  "ilo_Latn",	"knc_Arab",  "mkd_Cyrl",  "prs_Arab",  "swe_Latn",	"urd_Arab",
"aka_Latn",  "bjn_Latn",  "fao_Latn",  "ind_Latn",	"knc_Latn",  "mlt_Latn",  "quy_Latn",  "swh_Latn",	"uzn_Latn",
"als_Latn",  "bod_Tibt",  "fij_Latn",  "isl_Latn",	"kon_Latn",  "mni_Beng",  "ron_Latn",  "szl_Latn",	"vec_Latn",
"amh_Ethi",  "bos_Latn",  "fin_Latn",  "ita_Latn",	"kor_Hang",  "mos_Latn",  "run_Latn",  "tam_Taml",	"vie_Latn",
"apc_Arab",  "bug_Latn",  "fon_Latn",  "jav_Latn",	"lao_Laoo",  "mri_Latn",  "rus_Cyrl",  "taq_Latn",	"war_Latn",
"arb_Arab",  "bul_Cyrl",  "fra_Latn",  "jpn_Jpan",	"lij_Latn",  "mya_Mymr",  "sag_Latn",  "taq_Tfng",	"wol_Latn",
"arb_Latn",  "cat_Latn",  "fur_Latn",  "kab_Latn",	"lim_Latn",  "nld_Latn",  "san_Deva",  "tat_Cyrl",	"xho_Latn",
"ars_Arab",  "ceb_Latn",  "fuv_Latn",  "kac_Latn",	"lin_Latn",  "nno_Latn",  "sat_Olck",  "tel_Telu",	"ydd_Hebr",
"ary_Arab",  "ces_Latn",  "gaz_Latn",  "kam_Latn",	"lit_Latn",  "nob_Latn",  "scn_Latn",  "tgk_Cyrl",	"yor_Latn",
"arz_Arab",  "cjk_Latn",  "gla_Latn",  "kan_Knda",	"lmo_Latn",  "npi_Deva",  "shn_Mymr",  "tgl_Latn",	"yue_Hant",
"asm_Beng",  "ckb_Arab",  "gle_Latn",  "kas_Arab",	"ltg_Latn",  "nso_Latn",  "sin_Sinh",  "tha_Thai",	"zho_Hans",
"ast_Latn",  "crh_Latn",  "glg_Latn",  "kas_Deva",	"ltz_Latn",  "nus_Latn",  "slk_Latn",  "tir_Ethi",	"zho_Hant",
"awa_Deva",  "cym_Latn",  "grn_Latn",  "kat_Geor",	"lua_Latn",  "nya_Latn",  "slv_Latn",  "tpi_Latn",	"zsm_Latn",
"ayr_Latn",  "dan_Latn",  "guj_Gujr",  "kaz_Cyrl",	"lug_Latn",  "oci_Latn",  "smo_Latn",  "tsn_Latn",	"zul_Latn",
"azb_Arab",  "deu_Latn",  "hat_Latn",  "kbp_Latn",	"luo_Latn",  "ory_Orya",  "sna_Latn",  "tso_Latn",
"azj_Latn",  "dik_Latn",  "hau_Latn",  "kea_Latn",	"lus_Latn",  "pag_Latn",  "snd_Arab",  "tuk_Latn",
"bak_Cyrl",  "dyu_Latn",  "heb_Hebr",  "khk_Cyrl",	"lvs_Latn",  "pan_Guru",  "som_Latn",  "tum_Latn"
]

MBArt_FACEBOOK = ["ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN", "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO", "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL", "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF", "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA", "gl_ES", "sl_SI"]

class TranslationTransformer:
    def __init__(self, model_name: str, device='cuda:0'):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    def translate(self, texts: str, langin: str, langout: str):
        with torch.no_grad():
            prefix = f"translate {langin} to {langout}: "
            inputs = self.tokenizer(f"{prefix}{texts}", return_tensors="pt", padding=True, truncation=True,
                                    max_length=512).to(self.device)
            outputs = self.model.generate(**inputs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class LanguageM2M:
    def __init__(self, model_name: str, device='cuda:0'):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    def translate(self, texts: str, langin: str, langout: str):
        self.tokenizer.src_lang = langin
        encoded_in = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        generated_tokens = self.model.generate(**encoded_in, forced_bos_token_id=self.tokenizer.lang_code_to_id[langout])
        output = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return output[0]

class LanguageNLLB:
    def __init__(self, model_name: str, device='cuda:0'):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    def translate(self, text: str, langin: str, langout: str):
        translator = pipeline("translation", model =self.model, tokenizer =self.tokenizer, src_lang = langin, tgt_lang = langout, max_length = 400)
        output = translator(text)
        translated_text = output[0]["translation_text"]
        return translated_text

def language_match(lang: str):
    lang_name = ""
    match lang:
        case lang if lang in languages.part1:
            lang_name = languages.part1[lang].name
        case lang if lang in languages.part2b:
            lang_name = languages.part2b[lang].name
        case lang if lang in languages.part2t:
            lang_name = languages.part2t[lang].name
        case lang if lang in languages.part3:
            lang_name = languages.part3[lang].name
        case lang if lang in languages.part5:
            lang_name = languages.part5[lang].name
        case lang if lang in languages.name:
            lang_name = languages.name
        case "_":
            lang_name = "Unknown"
    if lang_name == "":
        lang_name = "Unknown"
    return lang_name


class WhisperTranslation:
    def __init__(self, device_i: str, model_name: str):
        self.device = device_i
        self.model = whisper.load_model(model_name, device=device_i)


    def translate(self, text: str, langin: str, langout: str):
        decoded_options = {
            "Language": langin,
            "task": "translation",
        }
        self.text2speech(text, langin)
        audio = whisper.load_audio("speech.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions(task='translation', language=langout)
        result = whisper.decode(self.model, mel, options)
        if os.path.exists("speech.mp3"):
            os.remove("speech.mp3")
        print(result.text)

    def text2speech(self, text: str, lang: str):
        tts = gTTS(text, lang=lang)
        tts.save("speech.mp3")

def language_flores200():
    lang_dict = {}
    for lang in LANGUAGES_FLORES200:
        lang_part1 = lang.split("_")[0]
        lang_part2 = lang.split("_")[1]
        if lang_part1 not in lang_dict:
            lang_dict[lang_part1] = []
        lang_dict[lang_part1].append(lang_part2)
    with open("languages_flores200.json", "w", encoding="UTF-8") as f:
        json.dump(lang_dict, f, indent=2)

def language_MBArt():
    lang_dict = {}
    for lang in MBArt_FACEBOOK:
        lang_part1 = lang.split("_")[0]
        lang_part2 = lang.split("_")[1]
        if lang_part1 not in lang_dict:
            lang_dict[lang_part1] = []
        lang_dict[lang_part1].append(lang_part2)
    with open("languages_MBArt.json", "w", encoding="UTF-8") as f:
        json.dump(lang_dict, f, indent=2)


def language_to_flores200(lang: str, flores200: dict):
    match lang:
        case lang if lang in languages.part1:
            lang_name = languages.part1[lang].part3
        case lang if lang in languages.part2b:
            lang_name = languages.part2b[lang].part3
        case lang if lang in languages.part2t:
            lang_name = languages.part2t[lang].part3
        case lang if lang in languages.part3:
            lang_name = languages.part3[lang].part3
        case lang if lang in languages.part5:
            lang_name = languages.part5[lang].part3
        case lang if lang in languages.name:
            lang_name = languages.name[lang].part3
        case "_":
            lang_name = lang
    try:
        if lang_name in flores200:
            part2 = flores200[lang_name][0]
            lang_out = f"{lang_name}_{part2}"
        return lang_out, True
    except:
        return f"Language {lang} not found in Flores200.", False

def language_to_MBART(lang, mbart_languages: dict):
    match lang:
        case lang if lang in languages.part1:
            lang_name = languages.part1[lang].part1
        case lang if lang in languages.part2b:
            lang_name = languages.part2b[lang].part1
        case lang if lang in languages.part2t:
            lang_name = languages.part2t[lang].part1
        case lang if lang in languages.part3:
            lang_name = languages.part3[lang].part1
        case lang if lang in languages.part5:
            lang_name = languages.part5[lang].part1
        case lang if lang in languages.name:
            lang_name = languages.name[lang].part1
        case "_":
            lang_name = lang
    try:
        if lang_name in mbart_languages:
            part2 = mbart_languages[lang_name][0]
            lang_out = f"{lang_name}_{part2}"
        return lang_out, True
    except:
        return f"Language {lang} not found in MBart.", False




if __name__ == '__main__':
    language_flores200()
    language_MBArt()
    with open("languages_flores200.json", "r", encoding="UTF-8") as f:
        flores200 = json.load(f)
    with open("languages_MBArt.json", "r", encoding="UTF-8") as f:
        mbart_languages = json.load(f)
    test1 = language_to_flores200("en", flores200)
    test2 = language_to_MBART("en", mbart_languages)
    device_i = "cuda:0" if torch.cuda.is_available() else "cpu"
    # translater = TranslationTransformer("google/flan-t5-base", device_i)
    # texts = "How old are you?"
    # langin = "en"
    # langout = "de"
    # translation = translater.translate(texts, language_match(langin), language_match(langout))
    # print(translation)

    # translater = LanguageM2M("facebook/nllb-200-distilled-600M", device_i)
    # translater = LanguageM2M("facebook/mbart-large-50-many-to-many-mmt", device_i)
    # texts = "How old are you?"
    # langin = "en_XX"
    # langout = "tr_TR"
    # translation = translater.translate(texts, langin, langout)
    # print(translation)

    # translater = LanguageNLLB("facebook/nllb-200-distilled-600M", device_i)
    # texts = "How old are you?"
    # langin = "eng_Latn"
    # langout = "tur_Latn"
    # translation = translater.translate(texts, langin, langout)
    # print(translation)

    # translater = WhisperTranslation(device_i, "small")
    # texts = "Wie alt bist du?"
    # langin = "de"
    # langout = "en"
    # translation = translater.translate(texts, langin, langout)