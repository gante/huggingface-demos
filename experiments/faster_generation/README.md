# Assisted Generation Benchmarks

## Instructions

Each `.py` script is a benchmark using a different dataset, corresponding to a different task. The current benchmarks
work with instruction-tuned models, but no chat template is applied.

All scripts have the same flags, e.g. `python benchmark_code_python.py -h` yields:
```
Run the benchmark, comparing the original to the new generation.

positional arguments:
  model

options:
  -h, --help            show this help message and exit
  --aux-model AUX_MODEL
  --dtype DTYPE
  --temperature TEMPERATURE
  --num-samples NUM_SAMPLES
  --max-gpu-memory [MAX_GPU_MEMORY ...]
```

Example command:
```
python benchmark_decoder_open.py facebook/opt-6.7b --aux-model facebook/opt-125m --dtype fp16
```

## Decoder-only LLMs

If you're looking to benchmark decoder-only LLMs, the following benchmarks are available:
- `benchmark_code_python.py`: prompts the model to continue a code snippet in python
- `benchmark_decoder_open.py`: prompts the model to continue a random entry from the C4 dataset
- `benchmark_decoder_summ.py`: prompts the model to summarize a given piece of news


## Supports

If you're using the latest `transformers` version, the following features are active by default:
- ✅ Assisted generation with static [DISCO](https://arxiv.org/abs/2405.04304) threshold
- ✅ Speculative Decoding, if the assistant model has the same tokenizer and `--temperature` is set
- ✅ [UAG](https://huggingface.co/blog/universal_assisted_generation) if the assistant model has a different tokenizer
