from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
from tqdm import tqdm
from multiprocessing import Process, Queue

from utils import get_mismatches, get_parsed_args, run_og_model, run_new_model

TORCH_DEVICE = 0
GEN_LEN = 128
DBG = False


def run_prediction_loop(model, tokenizer, num_samples, temperature=None, assistant_model=None):
    outputs = []
    gen_time = []
    num_tokens = []
    ds = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=True)
    ds_iterator = iter(ds.take(num_samples))

    desc = "ORIGINAL model" if assistant_model is None else f"ASSISTED model"
    pbar = tqdm(range(num_samples), desc)
    for _ in pbar:
        next_data = "Article: " + next(ds_iterator)["article"] + " Summary:"
        inputs = tokenizer([next_data], return_tensors="pt")
        inputs = inputs.to(TORCH_DEVICE)

        if temperature is not None:
            do_sample = True
        else:
            do_sample = False

        start = time.time()
        gen_out = model.generate(
            **inputs, do_sample=do_sample, max_new_tokens=GEN_LEN, assistant_model=assistant_model, temperature=temperature
        )
        end = time.time()

        outputs.append(tokenizer.decode(gen_out[0]))
        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1] - inputs.input_ids.shape[1])

    print(f"Average time per input (ms): {(sum(gen_time) / len(gen_time))*1000:.2f}")
    print(f"Average time per token (ms): {(sum(gen_time) / sum(num_tokens))*1000:.2f}")
    return outputs


if __name__ == "__main__":
    args = get_parsed_args()

    queue = Queue()

    if DBG:
        run_new_model(args, AutoTokenizer, AutoModelForCausalLM, run_prediction_loop, queue)
        exit()

    if args.temperature is None:
        p = Process(
            target=run_og_model,
            args=(args, AutoTokenizer, AutoModelForCausalLM, run_prediction_loop, queue,)
        )
        p.start()
        p.join()  # this blocks until the process terminates
        og_outputs = queue.get()

    p = Process(
        target=run_new_model,
        args=(args, AutoTokenizer, AutoModelForCausalLM, run_prediction_loop, queue,)
    )
    p.start()
    p.join()  # this blocks until the process terminates
    new_outputs = queue.get()

    if args.temperature is None:
        get_mismatches(og_outputs, new_outputs, args.dtype)
