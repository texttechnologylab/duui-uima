from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ..text import Text


LANGS = {
    'en': 'english',
    'de': 'german',
    'it': 'italian',
    'fr': 'french',
    'nl': 'dutch',
    'es': 'spanish',
    'pt': 'portuguese',
    'pl': 'polish',
    'ru': 'russian',
    'tr': 'turkish'
}


class WhisperASR:

    def __init__(self, model_path, device, utt_start_token='', utt_end_token='', lang=None, batch_size=16, **kwargs):
        self.device = device
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_path = model_path  # TODO
        self.use_flash_attention_2 = False
        self.utt_start_token = utt_start_token
        self.utt_end_token = utt_end_token
        self.lang = lang

        model_id = 'openai/whisper-large-v3'
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                                                          use_safetensors=True,
                                                          use_flash_attention_2=self.use_flash_attention_2,
                                                          cache_dir=model_path)
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=model_path)
        self.speech2text = pipeline('automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
                                    feature_extractor=processor.feature_extractor, batch_size=batch_size,
                                    return_timestamps=False, torch_dtype=torch_dtype, device=self.device,
                                    max_new_tokens=128,)

        self.output = 'text'

    def recognize_speech_of_audio(self, audio_file):
        if self.lang is None:
            text = self.speech2text(audio_file)['text']
        else:
            text = self.speech2text(audio_file, generate_kwargs={'language': LANGS[self.lang]})['text']
        text = self.utt_start_token + text.strip() + self.utt_end_token
        return text

    def recognize_speech_of_dataset(self, audio_dataset, out_dir, save_intermediate=True, job_id=None):
        texts = Text(is_phones=(self.output == 'phones'))

        if self.lang is None:
            outputs = self.speech2text(audio_dataset)
        else:
            outputs = self.speech2text(audio_dataset, generate_kwargs={'language': LANGS[self.lang]})

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
