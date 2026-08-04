from pathlib import Path

import numpy as np

from utils import read_kaldi_format, save_kaldi_format


class Text:
    """Collection of utterance text used by the upstream ASR helpers."""

    def __init__(self, is_phones=False):
        self.sentences = []
        self.utterances = []
        self.speakers = []
        self.utt2idx = {}
        self.new = True
        self.is_phones = is_phones

    def __len__(self):
        return len(self.utterances)

    def __iter__(self):
        for i in range(len(self)):
            yield self.sentences[i], self.utterances[i], self.speakers[i]

    def __getitem__(self, utterance):
        return self.get_instance(utterance)[0]

    def __contains__(self, item):
        return item in self.utterances

    def add_instance(self, sentence, utterance, speaker):
        self.utt2idx[utterance] = len(self)
        self.sentences.append(self.remove_special_characters(sentence))
        self.utterances.append(utterance)
        self.speakers.append(speaker)

    def add_instances(self, sentences, utterances, speakers):
        duplicate_indices = [i for i, utt in enumerate(utterances) if utt in self.utterances]
        for idx in reversed(duplicate_indices):
            del sentences[idx]
            del utterances[idx]
            del speakers[idx]

        self.utt2idx.update({utt: len(self) + i for i, utt in enumerate(utterances)})
        self.sentences.extend(self.remove_special_characters(sentence) for sentence in sentences)
        self.utterances.extend(utterances)
        self.speakers.extend(speakers)

    def get_instance(self, utterance):
        idx = self.utt2idx[utterance]
        return self.sentences[idx], self.speakers[idx]

    @staticmethod
    def remove_special_characters(sentence):
        return sentence.replace("«", "").replace("»", "").replace("^", "")

    def get_iterators(self, count):
        def instances(indices):
            for i in indices:
                yield self.sentences[i], self.utterances[i], self.speakers[i]

        iterator_length = len(self) // count
        slices = [
            [iterator_length * i, iterator_length * (i + 1)]
            if i < count - 1
            else [iterator_length * i, len(self)]
            for i in range(count)
        ]
        return [instances(range(*iterator_slice)) for iterator_slice in slices]

    def update_instance(self, utterance, sentence):
        self.sentences[self.utt2idx[utterance]] = self.remove_special_characters(sentence)

    def remove_instances(self, utterance_list):
        indices = sorted(
            (self.utt2idx[utt] for utt in utterance_list if utt in self.utt2idx),
            reverse=True,
        )
        for idx in indices:
            del self.sentences[idx]
            del self.utterances[idx]
            del self.speakers[idx]
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}

    def get_instances_of_speaker(self, speaker):
        indices = [i for i, current in enumerate(self.speakers) if current == speaker]
        return (
            [self.sentences[i] for i in indices],
            [self.utterances[i] for i in indices],
        )

    def shuffle(self):
        order = np.random.permutation(len(self))
        self.sentences = [self.sentences[i] for i in order]
        self.utterances = [self.utterances[i] for i in order]
        self.speakers = [self.speakers[i] for i in order]
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}

    def save_text(self, out_dir: Path, add_suffix=None):
        out_dir.mkdir(exist_ok=True, parents=True)
        suffix = add_suffix or ""
        save_kaldi_format(dict(zip(self.utterances, self.sentences)), out_dir / f"text{suffix}")
        save_kaldi_format(dict(zip(self.utterances, self.speakers)), out_dir / f"utt2spk{suffix}")

    def load_text(self, in_dir, add_suffix=None):
        self.new = False
        suffix = add_suffix or ""
        utt_1, sentences = read_kaldi_format(
            filename=in_dir / f"text{suffix}", return_as_dict=False, values_as_string=True
        )
        sentences = [self.remove_special_characters(sentence) for sentence in sentences]
        utt_2, speakers = read_kaldi_format(
            filename=in_dir / f"utt2spk{suffix}", return_as_dict=False
        )

        if utt_1 == utt_2:
            self.utterances = utt_1
            self.sentences = sentences
            self.speakers = speakers
        elif sorted(utt_1) == sorted(utt_2):
            self.utterances, self.sentences = zip(
                *sorted(zip(utt_1, sentences), key=lambda value: value[0])
            )
            _, self.speakers = zip(*sorted(zip(utt_2, speakers), key=lambda value: value[0]))
        else:
            raise ValueError(
                f"{in_dir / f'text{suffix}'} and {in_dir / f'utt2spk{suffix}'} "
                "have mismatching utterance keys"
            )
        self.utt2idx = {utt: i for i, utt in enumerate(self.utterances)}
