from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path('anonymization/modules/tts/IMSToucan').absolute()))

from .IMSToucan.InferenceInterfaces.InferenceArchitectures.InferenceToucanTTS import ToucanTTS
from .IMSToucan.InferenceInterfaces.InferenceArchitectures.InferenceAvocodo import HiFiGANGenerator
from .IMSToucan.Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from .IMSToucan.Preprocessing.TextFrontend import get_language_id
from utils import setup_logger

logger = setup_logger(__name__)

class AnonFastSpeech2(torch.nn.Module):

    def __init__(self, vocoder_model_path, tts_model_path, device="cpu", language="en"):
        super().__init__()
        self.device = device

        ################################
        #   build text to phone        #
        ################################
        self.text2phone = ArticulatoryCombinedTextFrontend(language=language, add_silence_to_end=True, silent=False) #TODO

        ################################
        #   load weights               #
        ################################
        checkpoint = torch.load(tts_model_path, map_location='cpu')

        ################################
        #   load phone to mel model    #
        ################################
        self.use_lang_id = True
        try:
            self.phone2mel = ToucanTTS(weights=checkpoint["model"])  # multi speaker multi language
            logger.info('Load multi speaker multi languages TTS model')
        except RuntimeError:
            try:
                self.use_lang_id = False
                self.phone2mel = ToucanTTS(weights=checkpoint["model"], lang_embs=None)  # multi speaker single language
                logger.info('Load multi speaker single language TTS model')
            except RuntimeError:
                self.phone2mel = ToucanTTS(weights=checkpoint["model"], lang_embs=None,
                                           utt_embed_dim=None)  # single speaker
                logger.info('Load single speaker TTS model')
        with torch.no_grad():
            self.phone2mel.store_inverse_all()  # this also removes weight norm
        self.phone2mel = self.phone2mel.to(torch.device(device))

        ################################
        #  load mel to wave model      #
        ################################
        self.mel2wav = HiFiGANGenerator(path_to_weights=vocoder_model_path).to(torch.device(device))
        self.mel2wav.remove_weight_norm()
        self.mel2wav = torch.jit.trace(self.mel2wav, torch.randn([80, 5]).to(torch.device(device)))

        ################################
        #  set defaults                #
        ################################
        self.default_utterance_embedding = checkpoint["default_emb"].to(self.device)
        logger.info(f'AnonFastSpeech2 Language: {language}')
        self.phone2mel.eval()
        self.mel2wav.eval()
        if self.use_lang_id:
            self.lang_id = get_language_id(language)
        else:
            self.lang_id = None
        self.to(torch.device(device))
        self.eval()

    def forward(self,
                text,
                view=False,
                duration_scaling_factor=1.0,
                pitch_variance_scale=1.0,
                energy_variance_scale=1.0,
                pause_duration_scaling_factor=1.0,
                durations=None,
                pitch=None,
                energy=None,
                text_is_phonemes=False,
                return_plot_as_filepath=False):
        """
        duration_scaling_factor: reasonable values are 0.5 < scale < 1.5.
                                     1.0 means no scaling happens, higher values increase durations for the whole
                                     utterance, lower values decrease durations for the whole utterance.
        pitch_variance_scale: reasonable values are 0.0 < scale < 2.0.
                                  1.0 means no scaling happens, higher values increase variance of the pitch curve,
                                  lower values decrease variance of the pitch curve.
        energy_variance_scale: reasonable values are 0.0 < scale < 2.0.
                                   1.0 means no scaling happens, higher values increase variance of the energy curve,
                                   lower values decrease variance of the energy curve.
        """
        with torch.inference_mode():
            phones = self.text2phone.string_to_tensor(text, input_phonemes=text_is_phonemes).to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(phones,
                                                           return_duration_pitch_energy=True,
                                                           utterance_embedding=self.default_utterance_embedding,
                                                           durations=durations,
                                                           pitch=pitch,
                                                           energy=energy,
                                                           lang_id=self.lang_id,
                                                           duration_scaling_factor=duration_scaling_factor,
                                                           pitch_variance_scale=pitch_variance_scale,
                                                           energy_variance_scale=energy_variance_scale,
                                                           pause_duration_scaling_factor=pause_duration_scaling_factor)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel)

        return wave
