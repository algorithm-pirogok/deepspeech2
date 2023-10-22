from collections import defaultdict
from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        results = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            if not ind:
                continue
            if last_char != self.ind2char[ind]:
                results.append(self.ind2char[ind])
            last_char = self.ind2char[ind]
        return ''.join(results)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        def _extend_and_merge(frame_distr, state_dict):
            new_state_dict = defaultdict(float)
            for next_char_index, next_char_proba in enumerate(frame_distr):
                for (pref, last_char), pref_proba in state_dict.items():
                    next_char = self.ind2char[next_char_index]
                    if next_char == last_char:
                        new_pref = pref
                    else:
                        if next_char != self.EMPTY_TOK:
                            new_pref = pref + next_char
                        else:
                            new_pref = pref
                        last_char = next_char
                    new_state_dict[(new_pref, last_char)] += pref_proba * next_char_proba
            return new_state_dict

        def _truncate(state_dict):
            state_list = list(state_dict.items())
            return dict(state_list.sort(key=lambda x: -x[1])[:beam_size])

        state = {('', self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = _extend_and_merge(frame, state)
            state = _truncate(state)

        hypos: List[Hypothesis] = [Hypothesis(seq, prob) for (seq, _), prob in state.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
