import logging
import os
import sys
import torch

from datetime import datetime

sys.path.append('crfsrl')

##################################################################
from supar.utils.logging import progress_bar
from torch.distributions.utils import lazy_property
from supar.utils import Config, Embedding
from supar.utils.common import BOS, MIN, PAD, UNK, INF, CACHE
from supar.utils.data import DataLoader, Sampler, PrefetchGenerator, collate_fn
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.fn import binarize, debinarize, kmeans
from supar.utils.fn import pad, download, get_rng_state, set_rng_state
from supar.utils.logging import get_logger, init_logger, progress_bar, logger
from supar.utils.metric import Metric
from supar.utils.optim import InverseSquareRootLR, LinearLR
from supar.utils.parallel import DistributedDataParallel as DDP
from supar.utils.parallel import gather, is_master, reduce, get_device_count, get_free_port
from supar.utils.tokenizer import TransformerTokenizer
from crfsrl.metric import SpanSRLMetric
from crfsrl.model import CRF2oSemanticRoleLabelingModel, CRFSemanticRoleLabelingModel
from crfsrl import CRFSemanticRoleLabelingParser, CRF2oSemanticRoleLabelingParser
#
from supar.utils.transform import Sentence, Batch, Transform
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

class Dataset(torch.utils.data.Dataset):
    r"""
    Dataset that is compatible with :class:`torch.utils.data.Dataset`, serving as a wrapper for manipulating all data fields
    with the operating behaviours defined in :class:`~supar.utils.transform.Transform`.
    The data fields of all the instantiated sentences can be accessed as an attribute of the dataset.

    Args:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform` or its derivations.
            The instance holds a series of loading and processing behaviours with regard to the specific data format.
        data (Union[str, Iterable]):
            A filename or a list of instances that will be passed into :meth:`transform.load`.
        cache (bool):
            If ``True``, tries to use the previously cached binarized data for fast loading.
            In this way, sentences are loaded on-the-fly according to the meta data.
            If ``False``, all sentences will be directly loaded into the memory.
            Default: ``False``.
        binarize (bool):
            If ``True``, binarizes the dataset once building it. Only works if ``cache=True``. Default: ``False``.
        bin (str):
            Path for saving binarized files, required if ``cache=True``. Default: ``None``.
        max_len (int):
            Sentences exceeding the length will be discarded. Default: ``None``.
        kwargs (Dict):
            Together with `data`, kwargs will be passed into :meth:`transform.load` to control the loading behaviour.

    Attributes:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform`.
        sentences (List[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in ``transform``.
            If ``cache=True``, each is a pointer to the sentence stored in the cache file.
    """

    def __init__(
        self,
        transform: Transform,
        data: Union[str, Iterable],
        cache: bool = False,
        binarize: bool = False,
        bin: str = None,
        max_len: int = None,
        **kwargs
    ):# -> Dataset:
        super(Dataset, self).__init__()

        self.transform = transform
        self.data = data
        self.cache = False
        self.binarize = binarize
        self.bin = bin
        self.max_len = max_len or INF
        self.kwargs = kwargs

        if isinstance(data, str) or isinstance(data, Path):
            print('loading', data, kwargs)
            self.sentences = list(transform.load(data, **kwargs))
#                 print('loaded', len(self.sentences))
        elif isinstance(data, list):
            self.sentences = list(transform.load(data))

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"
        if hasattr(self, 'loader'):
            s += f", n_batches={len(self.loader)}"
        if hasattr(self, 'buckets'):
            s += f", n_buckets={len(self.buckets)}"
        if self.cache:
            s += f", cache={self.cache}"
        if self.binarize:
            s += f", binarize={self.binarize}"
        if self.max_len < INF:
            s += f", max_len={self.max_len}"
        s += ")"
        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return debinarize(self.fbin, self.sentences[index]) if self.cache else self.sentences[index]

    def __getattr__(self, name):
        if name not in {f.name for f in self.transform.flattened_fields}:
            raise AttributeError
        if self.cache:
            if os.path.exists(self.fbin) and not self.binarize:
                sentences = self
            else:
                sentences = self.transform.load(self.data, **self.kwargs)
            return (getattr(sentence, name) for sentence in sentences)
        return [getattr(sentence, name) for sentence in self.sentences]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @lazy_property
    def sizes(self):
        if not self.cache:
            return [s.size for s in self.sentences]
        return debinarize(self.fbin, 'sizes')

    def build(
        self,
        batch_size: int,
        n_buckets: int = 1,
        shuffle: bool = False,
        distributed: bool = False,
        n_workers: int = 0,
        pin_memory: bool = True,
        chunk_size: int = 1000,
    ):# -> Dataset:
        self.cache = False
        self.sentences = [i for i in self.transform(self.sentences) if len(i) < self.max_len]
        self.buckets = dict(zip(*kmeans(self.sizes, n_buckets)))
        self.loader = DataLoader(transform=self.transform,
                                 dataset=self,
                                 batch_sampler=Sampler(self.buckets, batch_size, shuffle, distributed),
                                 num_workers=n_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=pin_memory)
        return self
#
## -*- coding: utf-8 -*-
#import os
#
from supar.utils.logging import progress_bar
from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import CoNLLSentence, Transform


class CoNLL(Transform):
    r"""
    The CoNLL object holds ten fields required for CoNLL-X data format :cite:`buchholz-marsi-2006-conll`.
    Each field can be bound to one or more :class:`~supar.utils.field.Field` objects. For example,
    ``FORM`` can contain both :class:`~supar.utils.field.Field` and :class:`~supar.utils.field.SubwordField`
    to produce tensors for words and subwords.

    Attributes:
        ID:
            Token counter, starting at 1.
        FORM:
            Words in the sentence.
        LEMMA:
            Lemmas or stems (depending on the particular treebank) of words, or underscores if not available.
        CPOS:
            Coarse-grained part-of-speech tags, where the tagset depends on the treebank.
        POS:
            Fine-grained part-of-speech tags, where the tagset depends on the treebank.
        FEATS:
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or underscores if not available.
        HEAD:
            Heads of the tokens, which are either values of ID or zeros.
        DEPREL:
            Dependency relations to the HEAD.
        PHEAD:
            Projective heads of tokens, which are either values of ID or zeros, or underscores if not available.
        PDEPREL:
            Dependency relations to the PHEAD, or underscores if not available.
    """

    fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'MORPH', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
    #fields = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'MORPH', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']

    def __init__(self,
                 ID=None, FORM=None, LEMMA=None, CPOS=None, POS=None, MORPH=None,
                 FEATS=None, HEAD=None, DEPREL=None, PHEAD=None, PDEPREL=None):
        super().__init__()

        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.CPOS = CPOS
        self.POS = POS
        #self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.PHEAD = PHEAD
        self.PDEPREL = PDEPREL
        self.MORPH = MORPH

    @property
    def src(self):
        #return self.FORM, self.LEMMA, self.CPOS, self.POS, self.FEATS
        return self.FORM, self.LEMMA, self.CPOS, self.POS, self.MORPH#, self.FEATS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL, self.PHEAD, self.PDEPREL

    @classmethod
    def get_arcs(cls, sequence, placeholder='_'):
        return [-1 if i == placeholder else int(i) for i in sequence]

    @classmethod
    def get_sibs(cls, sequence, placeholder='_'):
        sibs = [[0] * (len(sequence) + 1) for _ in range(len(sequence) + 1)]
        heads = [0] + [-1 if i == placeholder else int(i) for i in sequence]

        for i, hi in enumerate(heads[1:], 1):
            for j, hj in enumerate(heads[i+1:], i + 1):
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i][hi] = j
                    else:
                        sibs[j][hj] = i
                    break
        return sibs[1:]

    @classmethod
    def get_edges(cls, sequence):
        edges = [[0]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edges[i][int(pair.split(':')[0])] = 1
        return edges

    @classmethod
    def get_labels(cls, sequence):
        labels = [[None]*(len(sequence)+1) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence, 1):
            if s != '_':
                for pair in s.split('|'):
                    edge, label = pair.split(':')
                    labels[i][int(edge)] = label
        return labels

    @classmethod
    def get_srl_edges(cls, sequence):
        edges = [[[False]*(len(sequence)+1) for _ in range(len(sequence)+1)] for _ in range(len(sequence)+1)]
        spans = [['O']*len(sequence) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence):
            if s != '_':
                for pair in s.split('|'):
                    head, label = pair.split(':')
                    if label != '[prd]':
                        spans[int(head)][i] = label
                    else:
                        spans[i + 1][i] = label
        for i, label in enumerate(sequence):
            edges[i+1][i+1][0] = '[prd]' in label

        def factorize(tags):
            spans = []
            for i, tag in enumerate(tags, 1):
                if tag.startswith('B'):
                    spans.append([i, i+1, tag[2:]])
                elif tag.startswith('O') and (len(spans) == 0 or spans[-1][-1] != 'O'):
                    spans.append([i, i+1, 'O'])
                elif tag.startswith('['):
                    spans.append([i, i+1, tag])
                else:
                    spans[-1][1] += 1
            return spans
        for prd, arg_labels in enumerate(spans[1:], 1):
            for *span, label in factorize(arg_labels):
                if span[0] != prd:
                    if label != 'O':
                        edges[prd][span[0] if span[0] < prd else span[1]-1][prd] = True
                    else:
                        for i in range(*span):
                            edges[prd][i][prd] = True
                for i in range(*span):
                    if i != prd:
                        for j in range(*span):
                            if i != j:
                                edges[prd][i][j] = True
        return edges

    @classmethod
    def get_srl_roles(cls, sequence):
        labels = [['O']*(len(sequence)+1) for _ in range(len(sequence)+1)]
        spans = [['O']*len(sequence) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence):
            if s != '_':
                for pair in s.split('|'):
                    head, label = pair.split(':')
                    if label != '[prd]':
                        spans[int(head)][i] = label
                    else:
                        spans[i + 1][i] = label

        def factorize(tags):
            spans = []
            for i, tag in enumerate(tags, 1):
                if tag.startswith('B'):
                    spans.append([i, i+1, tag[2:]])
                elif tag.startswith('O') and (len(spans) == 0 or spans[-1][-1] != 'O'):
                    spans.append([i, i+1, 'O'])
                elif tag.startswith('['):
                    spans.append([i, i+1, tag])
                else:
                    spans[-1][1] += 1
            return spans
        for prd, arg_labels in enumerate(spans[1:], 1):
            if '[prd]' not in sequence[prd-1]:
                continue
            labels[prd][0] = '[prd]'
            for *span, label in factorize(arg_labels):
                if not label.startswith('['):
                    for i in range(*span):
                        labels[prd][i] = label
            labels[prd][prd] = '[prd]'
        return labels

    @classmethod
    def get_srl_spans(cls, sequence):
        labels = []
        spans = [['O']*len(sequence) for _ in range(len(sequence)+1)]
        for i, s in enumerate(sequence):
            if s != '_':
                for pair in s.split('|'):
                    head, label = pair.split(':')
                    if label != '[prd]':
                        spans[int(head)][i] = label
                    else:
                        spans[i + 1][i] = label

        def factorize(tags):
            spans = []
            for i, tag in enumerate(tags, 1):
                if tag.startswith('B'):
                    spans.append([i, i+1, tag[2:]])
                elif tag.startswith('O') and (len(spans) == 0 or spans[-1][-1] != 'O'):
                    spans.append([i, i+1, 'O'])
                elif tag.startswith('['):
                    spans.append([i, i+1, tag])
                else:
                    spans[-1][1] += 1
            return spans
        for prd, arg_labels in enumerate(spans[1:], 1):
            if '[prd]' not in sequence[prd-1]:
                continue
            for i, j, label in factorize(arg_labels):
                if i != prd and not label.startswith('['):
                    labels.append((prd, i, j-1, label))
        return labels

    @classmethod
    def build_relations(cls, chart):
        sequence = ['_'] * len(chart)
        for i, row in enumerate(chart):
            pairs = [(j, label) for j, label in enumerate(row) if label is not None]
            if len(pairs) > 0:
                sequence[i] = '|'.join(f"{head}:{label}" for head, label in pairs)
        return sequence

    @classmethod
    def build_srl_roles(cls, spans, length):
        labels = [''] * length
        for span in spans:
            prd, head, start, end, label = span
            if label == 'O':
                continue
            if '[prd]' not in labels[prd-1]:
                labels[prd-1] = '|'.join((labels[prd-1], '0:[prd]'))
            labels[start-1] = '|'.join((labels[start-1], f'{prd}:B-{label}'))
            for i in range(start, end):
                labels[i] = '|'.join((labels[i], f'{prd}:I-{label}'))
        labels = [('_' if not label else label).lstrip('|') for label in labels]
        return labels

    @classmethod
    def toconll(cls, tokens):
        r"""
        Converts a list of tokens to a string in CoNLL-X format.
        Missing fields are filled with underscores.

        Args:
            tokens (list[str] or list[tuple]):
                This can be either a list of words, word/pos pairs or word/lemma/pos triples.

        Returns:
            A string in CoNLL-X format.

        Examples:
            >>> print(CoNLL.toconll(['She', 'enjoys', 'playing', 'tennis', '.']))
            1       She     _       _       _       _       _       _       _       _
            2       enjoys  _       _       _       _       _       _       _       _
            3       playing _       _       _       _       _       _       _       _
            4       tennis  _       _       _       _       _       _       _       _
            5       .       _       _       _       _       _       _       _       _

            >>> print(CoNLL.toconll([('She',     'she',    'PRP'),
                                     ('enjoys',  'enjoy',  'VBZ'),
                                     ('playing', 'play',   'VBG'),
                                     ('tennis',  'tennis', 'NN'),
                                     ('.',       '_',      '.')]))
            1       She     she     PRP     _       _       _       _       _       _
            2       enjoys  enjoy   VBZ     _       _       _       _       _       _
            3       playing play    VBG     _       _       _       _       _       _
            4       tennis  tennis  NN      _       _       _       _       _       _
            5       .       _       .       _       _       _       _       _       _

        """

        if isinstance(tokens[0], str):
            s = '\n'.join([f"{i}\t{word}\t" + '\t'.join(['_']*8)
                           for i, word in enumerate(tokens, 1)])
        elif len(tokens[0]) == 2:
            s = '\n'.join([f"{i}\t{word}\t_\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, tag) in enumerate(tokens, 1)])
        elif len(tokens[0]) == 3:
            s = '\n'.join([f"{i}\t{word}\t{lemma}\t{tag}\t" + '\t'.join(['_']*6)
                           for i, (word, lemma, tag) in enumerate(tokens, 1)])
        else:
            raise RuntimeError(f"Invalid sequence {tokens}. Only list of str or list of word/pos/lemma tuples are support.")
        return s + '\n'

    @classmethod
    def isprojective(cls, sequence):
        r"""
        Checks if a dependency tree is projective.
        This also works for partial annotation.

        Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
        which are hard to detect in the scenario of partial annotation.

        Args:
            sequence (list[int]):
                A list of head indices.

        Returns:
            ``True`` if the tree is projective, ``False`` otherwise.

        Examples:
            >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
            False
            >>> CoNLL.isprojective([3, -1, 2])
            False
        """

        pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
        for i, (hi, di) in enumerate(pairs):
            for hj, dj in pairs[i+1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri or li < rj < ri) and (li - lj)*(ri - rj) > 0:
                    return False
        return True

    @classmethod
    def istree(cls, sequence, proj=False, multiroot=False):
        r"""
        Checks if the arcs form an valid dependency tree.

        Args:
            sequence (list[int]):
                A list of head indices.
            proj (bool):
                If ``True``, requires the tree to be projective. Default: ``False``.
            multiroot (bool):
                If ``False``, requires the tree to contain only a single root. Default: ``True``.

        Returns:
            ``True`` if the arcs form an valid tree, ``False`` otherwise.

        Examples:
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

        from supar.utils.alg import tarjan
        if proj and not cls.isprojective(sequence):
            return False
        n_roots = sum(head == 0 for head in sequence)
        if n_roots == 0:
            return False
        if not multiroot and n_roots > 1:
            return False
        if any(i == head for i, head in enumerate(sequence, 1)):
            return False
        return next(tarjan(sequence), None) is None

    @classmethod
    def factorize(cls, tags):
        spans = []
        for i, tag in enumerate(tags, 1):
            if tag.startswith('B'):
                spans.append([i, i+1, tag[2:]])
            elif tag.startswith('I'):
                if len(spans) > 0 and spans[-1][-1] == tag[2:]:
                    spans[-1][1] += 1
                else:
                    spans.append([i, i+1, tag[2:]])
            elif len(spans) == 0 or spans[-1][-1] != tag:
                spans.append([i, i+1, tag])
            else:
                spans[-1][1] += 1
        return spans

    def load(self, data, conllus=False, lang=None, proj=False, max_len=None, **kwargs):
        r"""
        Loads the data in CoNLL-X format.
        Also supports for loading data from CoNLL-U file with comments and non-integer IDs.

        Args:
            data (list[list] or str):
                A list of instances or a filename.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            proj (bool):
                If ``True``, discards all non-projective sentences. Default: ``False``.
            max_len (int):
                Sentences exceeding the length will be discarded. Default: ``None``.

        Returns:
            A list of :class:`CoNLLSentence` instances.
        """

        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        elif isinstance(data, list):
            lines = []
            for conllu in data:
                lines.extend('\n'.join(conllu).split('\n'))
                lines.extend([''])
        else:
            if lang is not None:
                tokenizer = Tokenizer(lang)
                data = [tokenizer(i) for i in ([data] if isinstance(data, str) else data)]
            else:
                data = [data] if isinstance(data[0], str) else data
            lines = '\n'.join([self.toconll(i) for i in data]).split('\n')

        i, start, sentences = 0, 0, []
        for line in lines:
            if not line:
                sentences.append(CoNLLSentence(self, lines[start:i]))
                start = i + 1
            i += 1
        if proj:
            sentences = [i for i in sentences if self.isprojective(list(map(int, i.arcs)))]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences

class CoNLLSentence(Sentence):
    r"""
    Sencence in CoNLL-X format.

    Args:
        transform (CoNLL):
            A :class:`~supar.utils.transform.CoNLL` object.
        lines (List[str]):
            A list of strings composing a sentence in CoNLL-X format.
            Comments and non-integer IDs are permitted.
        index (Optional[int]):
            Index of the sentence in the corpus. Default: ``None``.

    Examples:
        >>> lines = ['# text = But I found the location wonderful and the neighbors very kind.',
                     '1\tBut\t_\t_\t_\t_\t_\t_\t_\t_',
                     '2\tI\t_\t_\t_\t_\t_\t_\t_\t_',
                     '3\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '4\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_',
                     '6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7\tand\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '8\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_',
                     '10\tvery\t_\t_\t_\t_\t_\t_\t_\t_',
                     '11\tkind\t_\t_\t_\t_\t_\t_\t_\t_',
                     '12\t.\t_\t_\t_\t_\t_\t_\t_\t_']
        >>> sentence = CoNLLSentence(transform, lines)  # fields in transform are built from ptb.
        >>> sentence.arcs = [3, 3, 0, 5, 6, 3, 6, 9, 11, 11, 6, 3]
        >>> sentence.rels = ['cc', 'nsubj', 'root', 'det', 'nsubj', 'xcomp',
                             'cc', 'det', 'dep', 'advmod', 'conj', 'punct']
        >>> sentence
        # text = But I found the location wonderful and the neighbors very kind.
        1       But     _       _       _       _       3       cc      _       _
        2       I       _       _       _       _       3       nsubj   _       _
        3       found   _       _       _       _       0       root    _       _
        4       the     _       _       _       _       5       det     _       _
        5       location        _       _       _       _       6       nsubj   _       _
        6       wonderful       _       _       _       _       3       xcomp   _       _
        7       and     _       _       _       _       6       cc      _       _
        7.1     found   _       _       _       _       _       _       _       _
        8       the     _       _       _       _       9       det     _       _
        9       neighbors       _       _       _       _       11      dep     _       _
        10      very    _       _       _       _       11      advmod  _       _
        11      kind    _       _       _       _       6       conj    _       _
        12      .       _       _       _       _       3       punct   _       _
    """

    def __init__(self, transform: CoNLL, lines: List[str], index: Optional[int] = None):
        super().__init__(transform, index)
        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i - 1] = line
            else:
                self.annotations[len(self.values)] = line
#                 morphs = value[5].split('|')
#                 for morph_value in morphs:
#                     if 'Case' in morph_value:
#                         g_split = morph_value.split('=')
#                         value[5] = g_split[-1]
#                         #break to prevent overwriting
#                         break
#                     else:
#                         value[5] = 'X'
#                 gns = value[9].split('|')

                self.values.append(value)

        self.values = list(zip(*self.values))


    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values))}}
        return '\n'.join(merged.values()) + '\n'

class CRF2oSemanticRoleLabelingParser(CRFSemanticRoleLabelingParser):
    r"""
    The implementation of Semantic Role Labeling Parser using second-order span-constrained CRF.
    """

    NAME = 'crf2o-semantic-role-labeling'
    MODEL = CRF2oSemanticRoleLabelingModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.EDGE, self.ROLE, self.SPAN = self.transform.PHEAD

    def train(
            self,
            train: Union[str, Iterable],
            dev: Union[str, Iterable],
            test: Union[str, Iterable],
            epochs: int,
            patience: int,
            batch_size: int = 5000,
            update_steps: int = 1,
            buckets: int = 32,
            workers: int = 0,
            clip: float = 5.0,
            amp: bool = False,
            cache: bool = False,
            verbose: bool = True,
            train_data = None,
            dev_data = None,
            epoch_start: int = 0,
            **kwargs
    ) -> None:
        r"""
        Args:
            train/dev/test (Union[str, Iterable]):
                Filenames of the train/dev/test datasets.
            epochs (int):
                The number of training iterations.
            patience (int):
                The number of consecutive iterations after which the training process would be early stopped if no improvement.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            clip (float):
                Clips gradient of an iterable of parameters at specified value. Default: 5.0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
        """
        logger.info('Starting training')
        self.args.update(locals())
        args = self.args
        init_logger(logger, verbose=verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info("Loading the data")
        #         if args.cache:
        #             args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        if train_data is None:
            train = Dataset(self.transform, train, args).build(batch_size, buckets,
                                                               True, dist.is_initialized(), workers)
            self.train_data = train
        else:
            self.train_data = train_data
            train = train_data

        if dev_data is None:
            dev = Dataset(self.transform, dev, args).build(batch_size, buckets,
                                                           False, dist.is_initialized(), workers)
            self.dev_data = dev
        else:
            self.dev_data = dev_data
            dev = dev_data

        #         dev = Dataset(self.transform, dev, args).build(batch_size, buckets, False, dist.is_initialized(), workers)
        self.dev_data = dev
        logger.info(f"{'train:':6} {train}")
        if not args.test:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = Dataset(self.transform, args.test, args).build(batch_size, buckets, False,
                                                                  dist.is_initialized(), workers)
            self.test_data = test
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, args.decay**(1/args.decay_steps))
        elif args.encoder == 'transformer':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = InverseSquareRootLR(self.optimizer, args.warmup_steps)
        else:
            # we found that Huggingface's AdamW is more robust and empirically better than the native implementation
            from transformers import AdamW
            steps = len(train.loader) * epochs // args.update_steps
            self.optimizer = AdamW(
                [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                 for n, p in self.model.named_parameters()],
                args.lr,
                (args.mu, args.nu),
                args.eps,
                args.weight_decay
            )
            #             self.scheduler = LinearLR(self.optimizer, int(steps*args.warmup), steps)
            self.scheduler = ConstantLR(self.optimizer, factor=0.5, total_iters=1)
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get('find_unused_parameters', True))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(dist.group.WORLD, fp16_compress_hook)
        #TODO continue learning
        self.step, self.epoch, self.best_e, self.patience, self.n_batches = 1, 1, 1, patience, len(train.loader)
        #         self.step, self.epoch, self.best_e, self.patience, self.n_batches = 1, 1, 1, patience, len(train.loader)

        self.best_metric, self.elapsed = Metric(), timedelta()
        if epoch_start > 1:
            self.epoch = epoch_start
            args.epochs += epoch_start
        if self.args.checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                train.loader.batch_sampler.epoch = self.epoch
                print('EPOCH', self.epoch)
            except AttributeError:
                logger.warning("No checkpoint found. Try re-launching the traing procedure instead")

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(train.loader), Metric()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.model.train()
            with self.join():
                # we should zero `step` as the number of batches in different processes is not necessarily equal
                self.step = 0
                for batch in bar:
                    with self.sync():
                        with torch.autocast(self.device, enabled=self.args.amp):
                            loss = self.train_step(batch)
                        self.backward(loss)
                    if self.sync_grad:
                        self.clip_grad_norm_(self.model.parameters(), self.args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        ###
                        self.last_lr = self.scheduler.get_last_lr()[0]
                        ###
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)


                    bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
                    self.step += 1
                    if args.predict_intra_epoch > 0 and self.step != len(bar):
                        #                         bar.set_postfix_str(f'predicting step {self.step} of {len(bar)}')
                        if epoch <= 1 and args.predict_first_epoch:
                            self.model.eval()
                            #                 #################
                            for batch in dev.loader:
                                batch = self.pred_step(batch)

                            if is_master():
                                with open(f'data/progress/{self.args.path.split("/")[-2]}_{epoch}_{self.step}.conllu',
                                          'w') as f:
                                    for s in progress_bar(dev):
                                        f.write(str(s) + '\n')
                            #                 ######################
                            self.model.train()

                        elif isinstance(args.predict_intra_epoch, float) and self.step % ceil(args.predict_intra_epoch*len(bar)) == 0:
                            self.model.eval()
                            #                 #################
                            for batch in dev.loader:
                                batch = self.pred_step(batch)

                            if is_master():
                                with open(f'data/progress/{self.args.path.split("/")[-2]}_{epoch}_{self.step}.conllu',
                                          'w') as f:
                                    for s in progress_bar(dev):
                                        f.write(str(s) + '\n')
                            #                 ######################
                            self.model.train()
            #                         if isinstance(args.predict_intra_epoch, int) and self.step % args.predict_intra_epoch == 0:
            #                             self.model.eval()
            #             #                 #################
            #                             for batch in dev.loader:
            #                                 batch = self.pred_step(batch)

            #                             if is_master():
            #                                 with open(f'data/progress/{self.args.path.split("/")[-2]}_{epoch}_{self.step}.conllu',
            #                                           'w') as f:
            #                                     for s in progress_bar(dev):
            #                                         f.write(str(s) + '\n')
            #                 ######################
            #                             self.model.train()

            self.model.eval()
            with self.join(), torch.autocast(self.device, enabled=self.args.amp):
                metric = self.reduce(sum([self.eval_step(i) for i in progress_bar(dev.loader)], Metric()))
                logger.info(f"{'dev:':5} {metric}")
                #                 #################
                if args.predict_every_epoch:
                    for batch in dev.loader:
                        batch = self.pred_step(batch)

                    if is_master():
                        with open(f'data/progress/{self.args.path.split("/")[-2]}_{epoch}_{len(bar)+1}.conllu', 'w') as f:
                            for s in progress_bar(dev):
                                f.write(str(s) + '\n')
                #                 ######################

                if args.test:
                    test_metric = sum([self.eval_step(i) for i in progress_bar(test.loader)], Metric())
                    logger.info(f"{'test:':5} {self.reduce(test_metric)}")

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, metric
                if is_master() and args.save:
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved) to {args.path}\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        if dist.is_initialized():
            dist.barrier()
        #         print(args.save)
        if args.save:
            best = self.load(**args)
        else:
            best = self

        # only allow the master device to save models
        #         if is_master():
        #             best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        if args.test:
            best.model.eval()
            with best.join():
                test_metric = sum([best.eval_step(i) for i in progress_bar(test.loader)], Metric())
                logger.info(f"{'test:':5} {best.reduce(test_metric)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")


    def evaluate(
            self,
            data: Union[str, Iterable],
            batch_size: int = 5000,
            buckets: int = 8,
            workers: int = 0,
            amp: bool = False,
            cache: bool = False,
            verbose: bool = True,
            **kwargs
    ):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for evaluation. Both a filename and a list of instances are allowed.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.

        Returns:
            The evaluation results.
        """
        self.args.update(locals())
        args = self.args
        init_logger(logger, verbose=verbose)

        self.transform.train()
        logger.info("Loading the data")
        #         if args.cache:
        #             args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        data = Dataset(self.transform, data, args)
        #         data = Dataset(self.transform, **args)
        data.build(batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"\n{data}")

        logger.info("Evaluating the data")
        start = datetime.now()
        self.model.eval()
        with self.join():
            metric = self.reduce(sum([self.eval_step(i) for i in progress_bar(data.loader)], Metric()))
        elapsed = datetime.now() - start
        logger.info(f"{metric}")
        logger.info(f"{elapsed}s elapsed, {len(data)/elapsed.total_seconds():.2f} Sents/s")

        return metric

    def predict(
            self,
            data: Union[str, Iterable],
            pred: str = None,
            lang: str = None,
            prob: bool = False,
            batch_size: int = 5000,
            buckets: int = 8,
            workers: int = 0,
            cache: bool = False,
            **kwargs
    ):
        self.args.update(locals())
        args = self.args
        init_logger(logger, verbose=verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        data = Dataset(self.transform, **args)
        data.build(batch_size, buckets, False, dist.is_initialized(), workers)
        logger.info(f"\n{data}")

        logger.info("Making predictions on the data")
        start = datetime.now()
        self.model.eval()
        with tempfile.TemporaryDirectory() as t:
            # we have clustered the sentences by length here to speed up prediction,
            # so the order of the yielded sentences can't be guaranteed
            for batch in progress_bar(data.loader):
                batch = self.pred_step(batch)
                if args.cache:
                    for s in batch:
                        with open(os.path.join(t, f"{s.index}"), 'w') as f:
                            f.write(str(s) + '\n')
            elapsed = datetime.now() - start

            if dist.is_initialized():
                dist.barrier()
            if args.cache:
                tdirs = gather(t) if dist.is_initialized() else (t,)
            if pred is not None and is_master():
                logger.info(f"Saving predicted results to {pred}")
                with open(pred, 'w') as f:
                    # merge all predictions into one single file
                    if args.cache:
                        sentences = (os.path.join(i, s) for i in tdirs for s in os.listdir(i))
                        for i in progress_bar(sorted(sentences, key=lambda x: int(os.path.basename(x)))):
                            with open(i) as s:
                                shutil.copyfileobj(s, f)
                    else:
                        for s in progress_bar(data):
                            f.write(str(s) + '\n')
            # exit util all files have been merged
            if dist.is_initialized():
                dist.barrier()
        logger.info(f"{elapsed}s elapsed, {len(data) / elapsed.total_seconds():.2f} Sents/s")

        #         if not cache:
        return data

    ##################################################
    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, edges, roles, spans = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        s_edge, s_sib, s_role = self.model(words, feats)
        loss = self.model.loss(s_edge, s_sib, s_role, edges, roles, mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> SpanSRLMetric:
        words, *feats, edges, roles, spans = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        s_edge, s_sib, s_role = self.model(words, feats)
        loss = self.model.loss(s_edge, s_sib, s_role, edges, roles, mask)
        role_preds = self.model.decode(s_edge, s_sib, s_role, mask)
        return SpanSRLMetric(
            loss,
            [[(i[0], *i[2:-1], self.ROLE.vocab[i[-1]]) for i in s if i[-1] != self.ROLE.unk_index] for s in role_preds],
            [[i for i in s if i[-1] != 'O'] for s in spans]
        )

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, *feats = batch.compose(self.transform)
        word_mask = words.ne(self.args.pad_index)
        mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
        mask[:, 0] = 0
        lens = mask.sum(-1)
        s_edge, s_sib, s_role = self.model(words, feats)
        if self.args.prd:
            prd_mask = pad([word_mask.new_tensor([0]+['[prd]' in i for i in s.values[8]]) for s in batch.sentences])
            s_edge[:, 0].masked_fill_(~prd_mask, MIN)
            s_role[..., 0, 0].masked_fill_(prd_mask, MIN)
            s_role[..., 0, self.args.prd_index].masked_fill_(~prd_mask, MIN)
        role_preds = [[(*i[:-1], self.ROLE.vocab[i[-1]]) for i in s]
                      for s in self.model.decode(s_edge, s_sib, s_role, mask)]
        batch.roles = [CoNLL.build_srl_roles(pred, length) for pred, length in zip(role_preds, lens.tolist())]
        if self.args.prob:
            scores = zip(*(s.cpu().unbind() for s in (s_edge, s_sib, s_role)))
            batch.probs = [(s[0][:i+1, :i+1], s[1][:i+1, :i+1, :i+1], s[2][:i+1, :i+1])
                           for i, s in zip(lens.tolist(), scores)]
        return batch

    @classmethod
    def build(cls, path, args, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        #         args = Config(**locals())
        lower = args.lower
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=lower)
        TAG, CHAR, LEMMA, ELMO, BERT = None, None, None, None, None
        DEPREL, MORPH = None, None
        if args.encoder == 'bert':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)

            WORD = SubwordField('words',
                                pad=t.pad_token,
                                unk=t.unk_token,
                                bos=t.bos_token or t.cls_token,
                                fix_len=fix_len,
                                tokenize=t.tokenize,
                                fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=lower)
            if 'morph' in args.feat:
                MORPH = Field('morphs', bos=BOS)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS)
            if 'dep' in args.feat:
                DEPREL = Field('deps', bos=BOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=PAD, unk=UNK, bos=BOS, lower=lower)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField('bert',
                                    pad=t.pad_token,
                                    unk=t.unk_token,
                                    bos=t.bos_token or t.cls_token,
                                    fix_len=args.fix_len,
                                    tokenize=t.tokenize,
                                    fn=None if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast)) else lambda x: ' '+x)
                BERT.vocab = t.get_vocab()
        EDGE = ChartField('edges', use_vocab=False, fn=CoNLL.get_srl_edges)
        ROLE = ChartField('roles', unk='O', fn=CoNLL.get_srl_roles)
        SPAN = RawField('spans', fn=CoNLL.get_srl_spans)
        transform = CoNLL(FORM=(WORD, CHAR, ELMO, BERT), LEMMA=LEMMA, POS=TAG, DEPREL=DEPREL,
                          MORPH=MORPH, PHEAD=(EDGE, ROLE, SPAN))

        train = Dataset(transform, args.train)
        if args.encoder != 'bert':
            WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None),
                       lambda x: x / torch.std(x))
            if MORPH is not None:
                MORPH.build(train)
            if DEPREL is not None:
                DEPREL.build(train)
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        ROLE.build(train)
        args.update({
            'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_roles': len(ROLE.vocab),
            'n_tags': len(TAG.vocab) if TAG is not None else None,
            'n_morphs': len(MORPH.vocab) if MORPH is not None else None,
            'n_deps': len(DEPREL.vocab) if DEPREL is not None else None,
            'n_chars': len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index': CHAR.pad_index if CHAR is not None else None,
            'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'prd_index': ROLE.vocab['[prd]'],
            'nul_index': ROLE.vocab['O']
        })
        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)

        parser.model.to(parser.device)
        ###########
        try:
            parser.tokenizer = t
        except UnboundLocalError:
            parser.tokenizer = None
        ###########
        return parser

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_args():
    args = dotdict()
    args.prd = False
    # args.encoder = 'lstm'
    args.encoder = 'bert'
    args.feat = ['morph', 'lemma']
    args.embed = '../../../data/vectors_mini.txt'
    args.decay = 0.75
    args.decay_steps = 5000
    args.n_char_embed = 50
    args.n_embed = 300
    args.n_feat_embed = 100
    args.min_freq = 1
    args.n_edge_mlp = 500
    # args.embed = 'fasttext-de'
    args.bert = 'dbmdz/bert-base-german-cased'
    args.train = '../data/ttlab.conllu'
    args.test = '../data/ttlab.conllu'
    args.dev = '../data/ttlab.conllu'
    args.device = 'cuda'
    ##training
    args.fix_len = 20
    # args.epochs = 1
    args.lr = 1e-5
    args.lr_rate = 20
    args.mu = 0.9
    args.nu = 0.9
    args.eps = 1e-12
    args.weight_decay = 0
    args.clip = 5.0
    args.warmup = 0.1
    args.update_steps = 1
    args.verbose = True
    args.cache = False
    args.path = 'exp/test/model'
    args.build = True
    args.lower = False
    args.save = True
    args.predict_intra_epoch = 0.0
    args.predict_every_epoch = True
    args.predict_first_epoch = True
    args.prob = False
    args.batch_size = 1024
    #the order in model for pop() or pop(0)
    return args

dt = datetime.now()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

def load_parser(device, model_path=None, model_type=None):
    # parser = load_cache_diaparser_model(diaparser_model_name)

    dt = datetime.now()

#############################################################
    args = get_args()
    args.device = device
    if model_path is None:
        args.path = 'crfsrl/exp/xlm_roberta_base_de_final/model'
        args.bert = 'xlm-roberta-base'
    else:
        args.path = 'crfsrl/' + model_path
        args.bert = model_type
    args.checkpoint = True
    Parser = CRF2oSemanticRoleLabelingParser
    parser = Parser.load(**args)

    return parser, args

#def predict_roles(tokens__, parser, model_path=None, model_type=None):
def predict_roles(tokens__, parser, args):
    # print(dt, 'Start processing', flush=True)

    tset = []
    for sentence in tokens__:
        tokens = []
        for i, token in enumerate(sentence):
            if isinstance(token, str):
                split_ = [token] + ['_'] * 8
                #split_ = [str(i+1), token] + ['_'] * 8
                tokens.append('\t'.join(split_))
            else:
                split_ = [token['text']] + ['_'] * 8
                tokens.append('\t'.join(split_))
        if len(tokens) > 0:
            tset.append(tokens)
        else:
            tset.append(['X'])
    buckets = 32
    workers = 0
    args.data = tset
    parser.transform.eval()
    #
    data = Dataset(parser.transform, **args)
    # print(args.batch_size)
    data.build(args.batch_size, buckets, False, False, workers)

    start = datetime.now()
    parser.model.eval()

    for batch in data.loader:
       batch = parser.pred_step(batch)

    s_i = 0
    for s in data:
        t_i = 0
        for idx, token, arg in zip(s.values[0], s.values[1], s.values[8]):
            #print(f'{idx:<3} {token:<20} {arg:20}')
            try:
                token_ = tokens__[s_i][t_i]['text']
            except IndexError:
                continue
            #assert tokens__[s_i][t_i]['text'] == token
            tokens__[s_i][t_i]['arg'] = arg#.replace('B-', '').replace('I-', '')
            t_i += 1
        s_i += 1


    return tokens__

if __name__ == '__main__':
    tokens = [[
        {'text': 'Die'},
        {'text': 'Rhizocephala'},
        {'text': 'tanzen'}, 
        {'text': '.'}
        ], 
        [
         {'text': 'Der'},
         {'text': 'Buntspecht'},
         {'text': 'frisst'},
         {'text': 'die'},
         {'text': 'Rhizocephala'},
         {'text': '.'}
        ]]

    parser, args = load_parser('cpu')
    predict_roles(tokens, parser, args)
