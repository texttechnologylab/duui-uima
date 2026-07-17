import torch


class SpeakerEmbeddings:

    def __init__(self, vec_type='xvector', emb_level='spk', device=torch.device('cpu')):
        self.vec_type = vec_type
        self.emb_level = emb_level
        self.device = device

        self.identifiers2idx = {}
        self.idx2identifiers = {}
        self.vectors = None
        self.original_speakers = []
        self.genders = []

        self.new = True

    def __iter__(self):
        assert self.identifiers2idx and self.vectors is not None, \
            'Speaker vectors need to be extracted or loaded before they can be iterated!'

        for identifier, idx in sorted(self.identifiers2idx.items(), key=lambda x: x[1]):
            yield identifier, self.vectors[idx]

    def __len__(self):
        return len(self.identifiers2idx)

    def __getitem__(self, item):
        assert (self.identifiers2idx is not None) and (self.vectors is not None), \
            'Speaker vectors need to be extracted or loaded before they can be accessed!'
        assert item <= len(self), 'Index needs to be smaller or equal the number of speakers!'
        return self.idx2identifiers[item], self.vectors[item]

    def set_vectors(self, identifiers, vectors, speakers, genders):
        if not isinstance(identifiers, dict):
            self.identifiers2idx = {identifier: idx for idx, identifier in enumerate(identifiers)}
        else:
            self.identifiers2idx = identifiers
        self.vectors = torch.tensor(vectors) if not isinstance(vectors, torch.Tensor) else vectors
        self.genders = genders
        self.original_speakers = speakers
        self.idx2identifiers = {idx: identifier for identifier, idx in self.identifiers2idx.items()}

