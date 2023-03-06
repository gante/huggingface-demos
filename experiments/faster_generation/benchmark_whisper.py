from transformers import AutoProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch
import time
from tqdm import tqdm
from multiprocessing import Process, Queue

from custom_gen_class import modify_generation

TORCH_DEVICE = 0
TORCH_DTYPE = None
MODEL_NAME = "openai/whisper-large-v2"
AUX_MODEL = "openai/whisper-base"
NUM_SAMPLES = 100
DBG = False


def run_prediction_loop(model, processor):
    outputs = []
    gen_time = []
    num_tokens = []

    ds = load_dataset("librispeech_asr", "clean", split="validation")
    speech_samples = ds.select(range(NUM_SAMPLES))[:NUM_SAMPLES]["audio"]

    desc = "OG model" if not hasattr(model, "fwd_tokens") else f"NEW model ({model.fwd_tokens} tokens forwarded)"
    pbar = tqdm(range(NUM_SAMPLES), desc)
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


def run_og_model(queue):
    tokenizer = AutoProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="auto", max_memory={0: "20GiB", "cpu": "50GiB"}, torch_dtype=TORCH_DTYPE
    )
    og_outputs = run_prediction_loop(model, tokenizer)
    queue.put(og_outputs)


def run_new_model(queue):
    tokenizer = AutoProcessor.from_pretrained(MODEL_NAME)

    aux_model = WhisperForConditionalGeneration.from_pretrained(AUX_MODEL)
    aux_model = aux_model.to(TORCH_DEVICE)

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME, device_map="auto", max_memory={0: "20GiB", "cpu": "50GiB"}, torch_dtype=TORCH_DTYPE
    )
    model = modify_generation(model, aux_model)
    new_outputs = run_prediction_loop(model, tokenizer)
    queue.put(new_outputs)


def get_mismatches(og_outputs, new_outputs):
    mismatches = 0
    for i in range(NUM_SAMPLES):
        if og_outputs[i] != new_outputs[i]:
            mismatches += 1
            if TORCH_DTYPE is None:  # float 16 is a bit unstable, float 32 gets the same results
                print("\nOG :", og_outputs[i])
                print("NEW:", new_outputs[i])
    print(f"Mismatches: {mismatches}")


if __name__ == "__main__":
    queue = Queue()

    if DBG:
        run_new_model(queue)
        exit()

    p = Process(target=run_og_model, args=(queue,))
    p.start()
    p.join()  # this blocks until the process terminates
    og_outputs = queue.get()

    p = Process(target=run_new_model, args=(queue,))
    p.start()
    p.join()  # this blocks until the process terminates
    new_outputs = queue.get()

    get_mismatches(og_outputs, new_outputs)
