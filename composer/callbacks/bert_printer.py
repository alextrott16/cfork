# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor learning rate during training."""
from composer.core import Callback, State
from composer.loggers import Logger
from composer.models import BERTModel

__all__ = ["BERTPrinter"]


class BERTPrinter(Callback):
    """Periodically prints the input text and output predictions of a BERT model.

    Args:
        tokenizer_name (str): Name of the tokenizer used in the dataloader.
        print_interval_in_batches (int): Number of batches between printouts.
    """

    def __init__(self, tokenizer_name: str, print_interval_in_batches: int) -> None:
        super().__init__()

        try:
            import transformers
        except ImportError:
            raise ImportError('huggingface transformers and datasets are not installed. '
                              'Please install with `pip install mosaicml-composer[nlp]`')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)  #type: ignore (thirdparty)

        assert print_interval_in_batches > 0
        self.print_interval_in_batches = print_interval_in_batches
        self._last_printout = -1

    # def fit_start(self, state: State, logger: Logger) -> None:
    #     if not isinstance(state.model, BERTModel):
    #         raise TypeError(f"Model class is required to be type :class:`~.BERTModel`, not type {type(state.model)}")

    def after_forward(self, state: State, logger: Logger) -> None:
        nb = int(state.timestamp.batch)
        if (nb % self.print_interval_in_batches) == 0:
            if nb == self._last_printout:
                # If grad_accum > 1 this prevents multiple printouts
                return
            seq_length = state.batch['attention_mask'][0].sum()
            tokens_in = state.batch['input_ids'][0, :seq_length]
            tokens_out = state.outputs['logits'][0, :seq_length].argmax(-1)

            print(
                f'\n\nNumber of batches: {nb}'
                f'\n\nINPUT:  {self.tokenizer.decode(tokens_in)}'
                f'\n\nOUTPUT: {self.tokenizer.decode(tokens_out)}\n'
            )

            self._last_printout = nb
