# debugging2
- Unsure of this. Wasn't taking good notes!
- Ran old code on top of the older code.
- Everything runs and metric logging works well.

# debugging3
- This is the original `run_glue_trainer_old` on top of the latest code
- It works. Everything runs.
- Metric logging fails, as expected, but the fine tuning doesn't hang!

# debugging4
- Introduced most recent `log_metrics` and `ingest_finetuning_result`
- Hangs after round 1.
- Metric logging "works" but I forgot to set it to "eval_metrics"

# debugging--removed-log_metrics
- Commented out `log_metrics` call inside `ingest_finetuning_result`
- It works. Everything runs!
- But no metric logging happens.

# debugging--moved-metrics-compute
- Put most recent `log_metrics` and `ingest_finetuning_result` but now all `metric.compute().item()` calls happen inside their `train_finetune` process
- Still deadlocking after round 1, even though only a dictionary of floats gets sent to `log_metrics`... Must be `GlueState` object...

# debugging--removed-glue-state
- Basically took `GlueState` object out of `log_metrics`.
- It works. Everything runs....
- BUT NO METRIC LOGGING

# debugging--removed-glue-state--new-code
- Starting from the new code, made `log_metrics` ignore the `GlueState` object.
- It works! But metric logging is absent.

# debugging--check-old-trainer
- This is just to confirm whether the "run_glue_trainer_old" is in a workable state
- Nope. It hangs in the second round... No stack traces, though...
- Re-running with non-torch multiprocessing in hopes of better stack traces...
- The process finishes, but the second round of runs seem to... crash?

# debugging--all-one-round
- Trying with everything in one round (RTE, MRPC, STSB moved to the first round)
- Everything works (metric logging still removed)
- Next, trying with no MNLI checkpoint

# debugging--two-rounds-same-checkpoints
- This time, I passed the same parent checkpoints to both rounds.
- It worked! This says the problem must be with the checkpointing. Noticing some saving/load name mismatch...

# debugging--fixed-mnli-load-name
- Fixed suffix-name error in second round (`.pt` was missing from MNLI checkpoints)!
- Fixed! `run_glue_trainer_old.py` now works!
- Saved this configuration as `run_glue_trainer_no_printout.py`

# debugging--hanlin-mp--take-1
- Trying a MP version based on Hanlin's approach
- Seems to work just as well!
- Next I'll try to get some of the logging functionality back.

# debugging--hanlin-mp--take-2
- Used glue logging code, but now it all gets compiled at the very end.
- EVERYTHING WORKS -- INCLUDING LOGGING!
- Will now try with multiple checkpoints and seeds.

# debugging--final-testing-1
- A bit of cleanup and testing `run_glue_trainer_old` with multiple seeds and checkpoints
- IT WORKS!

# debugging--final-testing-2
- Moving changes to `run_glue_trainer`:
  - New MP logic in `spawn_finetune` (including new `_setup_gpu_queue` function)
  - New stuff in `train_finetune`
  - Various changes to `run_finetuner`
  - New finetuning logic in `_main`
  - Up-to-date `log_metrics`
  - Copied imports
- Again with multiple seeds and checkpoints
- Success! Will clean up and export <3

# debugging--final-testing-3
- Just testing the cleaned up version. 1 ckpt and 1 seed each.