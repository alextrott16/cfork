# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Adds a --datadir flag to conveniently set a common
data directory for both train and validation datasets.

Example that trains MNIST with label smoothing::

    >>> python examples/run_composer_trainer.py
    -f composer/yamls/models/classify_mnist_cpu.yaml
    --algorithms label_smoothing --alpha 0.1
    --datadir ~/datasets
"""
import sys
import tempfile
import warnings
from typing import Type

from composer.loggers.logger import LogLevel
from composer.loggers.logger_hparams import WandBLoggerHparams
from composer.trainer import TrainerHparams
from composer.utils import dist


def warning_on_one_line(message: str, category: Type[Warning], filename: str, lineno: int, file=None, line=None):
    # From https://stackoverflow.com/questions/26430861/make-pythons-warnings-warn-not-mention-itself
    return f'{category.__name__}: {message} (source: {filename}:{lineno})\n'


def main() -> None:
    warnings.formatwarning = warning_on_one_line

    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv

    # if using wandb, store the config inside the wandb run
    for logger_hparams in hparams.loggers:
        if isinstance(logger_hparams, WandBLoggerHparams):
            logger_hparams.config = hparams.to_dict()

    trainer = hparams.initialize_object()

    # # Only log the config once, since it should be the same on all ranks.
    # if dist.get_global_rank() == 0:
    #     with tempfile.NamedTemporaryFile(mode="x+") as f:
    #         f.write(hparams.to_yaml())
    #         trainer.logger.file_artifact(LogLevel.FIT,
    #                                      artifact_name=f"{trainer.logger.run_name}/hparams.yaml",
    #                                      file_path=f.name,
    #                                      overwrite=True)

    # # Print the config to the terminal and log to artifact store if on each local rank 0
    # if dist.get_local_rank() == 0:
    #     print("*" * 30)
    #     print("Config:")
    #     print(hparams.to_yaml())
    #     print("*" * 30)

    # trainer.fit()

    import transformers
    from composer.algorithms.seq_length_warmup import set_batch_sequence_length
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    for batch_idx, batch in enumerate(trainer.state.dataloader):
        batch = trainer._device.batch_to_device(batch)
        batch = set_batch_sequence_length(batch, 96, truncate=True, preserve_eos=True)
        batch = {k: v[:1] for k, v in batch.items()}
        output = trainer.state.model.forward(batch)

        n_tokens = batch['attention_mask'][0].sum().item()
        input_tokens = batch['input_ids'][0, :n_tokens]
        top_output_tokens = output['logits'][0, :n_tokens].argmax(-1)
        print('Input:')
        print(tokenizer.decode(input_tokens))
        print('Output:')
        print(tokenizer.decode(top_output_tokens))
        print('')

        if batch_idx >= 20:
            break




if __name__ == "__main__":
    main()
