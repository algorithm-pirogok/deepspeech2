from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class WERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        lengths = log_probs_length.detach().numpy()
        if self.mode == 'argmax':
            predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
            for log_prob_vec, length, target_text in zip(predictions, lengths, text):
                target_text = BaseTextEncoder.normalize_text(target_text)
                if hasattr(self.text_encoder, "ctc_decode"):
                    pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
                else:
                    pred_text = self.text_encoder.decode(log_prob_vec[:length])
                wers.append(calc_wer(target_text, pred_text))
        elif self.mode == 'beam-search':
            if not hasattr(self.text_encoder, "ctc_beam_search"):
                raise Exception('Where is beam search?')
            beam_search = getattr(self.text_encoder, "ctc_beam_search")
            for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
                pred_text = beam_search(log_prob_vec, length, **kwargs)[0].text
                target_text = BaseTextEncoder.normalize_text(target_text)
                wers.append(calc_wer(target_text, pred_text))
        elif self.mode == 'lm':
            if not hasattr(self.text_encoder, "ctc_beam_search"):
                raise Exception('Where is lm?')
            lm = getattr(self.text_encoder, "lm_beam_search")
            for log_prob_vec, length, target_text in zip(log_probs, lengths, text):
                pred_text = lm(log_prob_vec, length, **kwargs)
                target_text = BaseTextEncoder.normalize_text(target_text)
                wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
