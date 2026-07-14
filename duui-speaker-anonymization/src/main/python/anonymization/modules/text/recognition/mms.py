from tqdm import tqdm
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline

from ..text import Text


LANGS = {
    'en': 'eng',
    'de': 'deu',
    'it': 'ita',
    'fr': 'fra',
    'pt': 'por',
    'nl': 'nld',
    'es': 'spa',
    'ru': 'rus',
    'pl': 'pol'
}


class MMSASR:

    def __init__(self, model_path, device, utt_start_token='', utt_end_token='', lang=None, batch_size=16, **kwargs):
        self.device = device
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_path = model_path  # TODO
        self.utt_start_token = utt_start_token
        self.utt_end_token = utt_end_token
        self.lang = LANGS[lang]

        model_id = 'facebook/mms-1b-all'
        model = Wav2Vec2ForCTC.from_pretrained(model_id, cache_dir=model_path, torch_dtype=torch_dtype)
        model.to(self.device)
        processor = Wav2Vec2Processor.from_pretrained(model_id, cache_dir=model_path)
        self.speech2text = pipeline('automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
                                    feature_extractor=processor.feature_extractor, batch_size=batch_size,
                                    torch_dtype=torch_dtype, device=self.device, framework='pt',
                                    model_kwargs={"target_lang": self.lang, "ignore_mismatched_sizes": True})

        self.output = 'text'

    def recognize_speech_of_audio(self, audio_file):
        text = self.speech2text(audio_file)['text']
        text = self.utt_start_token + text.strip() + self.utt_end_token
        return text

    def recognize_speech_of_dataset(self, audio_dataset, out_dir, save_intermediate=True, job_id=None):
        texts = Text(is_phones=(self.output == 'phones'))

        outputs = self.speech2text(audio_dataset)

        if job_id is None:  # single processing
            add_suffix = None
            tqdm_params = {}
        else: # process amongst multiple processes
            add_suffix = f'_{job_id}'
            tqdm_params = {'desc': f'Job {job_id}', 'leave': True}

        i = 0
        for output in tqdm(outputs, **tqdm_params):
            utt = output['utt'][0]
            spk = output['spk'][0]
            sentence = self.utt_start_token + output['text'].strip() + self.utt_end_token
            texts.add_instance(sentence=sentence, utterance=utt, speaker=spk)

            i += 1
            if i % 100 == 0 and save_intermediate:
                texts.save_text(out_dir=out_dir, add_suffix=add_suffix)

        if save_intermediate:
            texts.save_text(out_dir=out_dir, add_suffix=add_suffix)
        return texts
