from transformers import AutoProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import time
from tqdm import tqdm
from multiprocessing import Process, Queue

from utils import get_mismatches, get_parsed_args, run_og_model, run_new_model

TORCH_DEVICE = 0
DBG = False


def run_prediction_loop(model, processor, num_samples, temperature=None, assistant_model=None):
    outputs = []
    gen_time = []
    num_tokens = []

    ds = load_dataset("librispeech_asr", "clean", split="validation")
    speech_samples = ds.select(range(num_samples))[:num_samples]["audio"]

    desc = "ORIGINAL model" if assistant_model is None else f"ASSISTED model"
    pbar = tqdm(range(num_samples), desc)
    for i in pbar:
        inputs = processor.feature_extractor(
            raw_speech=[speech_samples[i]["array"]],
            return_tensors="pt",
            sampling_rate=16000
        )
        inputs = inputs.to(TORCH_DEVICE)

        if temperature is not None:
            do_sample = True
        else:
            do_sample = False

        start = time.time()
        gen_out = model.generate(**inputs, do_sample=do_sample, assistant_model=assistant_model, temperature=temperature)
        end = time.time()

        outputs.append(processor.decode(gen_out[0]))
        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1])

    print(f"Average time per input (ms): {(sum(gen_time) / len(gen_time))*1000:.2f}")
    print(f"Average time per token (ms): {(sum(gen_time) / sum(num_tokens))*1000:.2f}")
    return outputs


if __name__ == "__main__":
    args = get_parsed_args()

    queue = Queue()

    if DBG:
        run_new_model(args, AutoProcessor, WhisperForConditionalGeneration, run_prediction_loop, queue)
        exit()

    if args.temperature is None:
        p = Process(
            target=run_og_model,
            args=(args, AutoProcessor, WhisperForConditionalGeneration, run_prediction_loop, queue,)
        )
        p.start()
        p.join()  # this blocks until the process terminates
        og_outputs = queue.get()

    p = Process(
        target=run_new_model,
        args=(args, AutoProcessor, WhisperForConditionalGeneration, run_prediction_loop, queue,)
    )
    p.start()
    p.join()  # this blocks until the process terminates
    new_outputs = queue.get()

    if args.temperature is None:
        get_mismatches(og_outputs, new_outputs, args.dtype)
