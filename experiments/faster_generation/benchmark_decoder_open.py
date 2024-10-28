from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
from tqdm import tqdm

from utils import get_mismatches, get_parsed_args, run_model, run_model_with_assistant

TORCH_DEVICE = 0
INPUT_LEN = 128  # in characters
GEN_LEN = 128


def run_prediction_loop(model, tokenizer, num_samples, temperature=None, assistant_model=None, assistant_early_exit=None):
    outputs = []
    gen_time = []
    num_tokens = []
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ds_iterator = iter(ds.take(num_samples))

    desc = "ORIGINAL model" if assistant_model is None and assistant_early_exit is None else f"ASSISTED model"
    pbar = tqdm(range(num_samples), desc)
    for i in pbar:
        next_data = next(ds_iterator)["text"]
        inputs = tokenizer([next_data[:INPUT_LEN]], return_tensors="pt")
        inputs = inputs.to(TORCH_DEVICE)

        if temperature is not None:
            do_sample = True
        else:
            do_sample = False

        start = time.time()
        gen_out = model.generate(
            **inputs, do_sample=do_sample, max_new_tokens=GEN_LEN, assistant_model=assistant_model, early_exit=assistant_early_exit, temperature=temperature
        )
        end = time.time()

        outputs.append(tokenizer.decode(gen_out[0]))
        if i >= 2:  # discard first two iterations, warmup
            gen_time.append(end - start)
            num_tokens.append(gen_out.shape[1] - inputs.input_ids.shape[1])

    print(f"Average time per input (ms): {(sum(gen_time) / len(gen_time))*1000:.2f}")
    print(f"Average time per token (ms): {(sum(gen_time) / sum(num_tokens))*1000:.2f}")
    return outputs


if __name__ == "__main__":
    args = get_parsed_args()

    new_outputs = run_model_with_assistant(args, AutoTokenizer, AutoModelForCausalLM, run_prediction_loop)
    og_outputs = run_model(args, AutoTokenizer, AutoModelForCausalLM, run_prediction_loop)

    if args.temperature is None:
        get_mismatches(og_outputs, new_outputs, args.dtype)
