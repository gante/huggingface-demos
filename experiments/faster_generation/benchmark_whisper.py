from transformers import AutoProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import time
from tqdm import tqdm
from multiprocessing import Process, Queue

from utils import get_mismatches, get_parsed_args, run_og_model, run_new_model

TORCH_DEVICE = 0
DBG = False


def run_prediction_loop(model, processor, num_samples):
    outputs = []
    gen_time = []
    num_tokens = []

    ds = load_dataset("librispeech_asr", "clean", split="validation")
    speech_samples = ds.select(range(num_samples))[:num_samples]["audio"]

    desc = "OG model" if not hasattr(model, "fwd_tokens") else f"NEW model ({model.fwd_tokens} tokens forwarded)"
    pbar = tqdm(range(num_samples), desc)
    for i in pbar:
        inputs = processor.feature_extractor(
            raw_speech=[speech_samples[i]["array"]],
            return_tensors="pt",
            sampling_rate=16000
        )
        inputs = inputs.to(TORCH_DEVICE)

        start = time.time()
        gen_out = model.generate(**inputs, do_sample=False)
        end = time.time()

        outputs.append(processor.decode(gen_out[0]))
        gen_time.append(end - start)
        num_tokens.append(gen_out.shape[1])

        if hasattr(model, "fwd_tokens"):
            pbar.set_description(f"NEW model ({model.fwd_tokens} tokens forwarded)")

    print(f"OG Average time per input (ms): {(sum(gen_time) / len(gen_time))*1000:.2f}")
    print(f"OG Average time per token (ms): {(sum(gen_time) / sum(num_tokens))*1000:.2f}")
    return outputs


if __name__ == "__main__":
    args = get_parsed_args()

    queue = Queue()

    if DBG:
        run_new_model(args, AutoProcessor, WhisperForConditionalGeneration, run_prediction_loop, queue)
        exit()

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

    get_mismatches(og_outputs, new_outputs, args.dtype)