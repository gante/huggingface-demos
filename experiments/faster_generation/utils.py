import argparse
import torch

TORCH_DEVICE = 0


def get_mismatches(og_outputs, new_outputs, dtype=None):
    mismatches = 0
    num_samples = len(og_outputs)
    for i in range(num_samples):
        if og_outputs[i] != new_outputs[i]:
            mismatches += 1
            if dtype is None:  # float 16 is a bit unstable, float 32 gets the same results
                print("\nOG :", og_outputs[i])
                print("NEW:", new_outputs[i])
    print(f"Mismatches: {mismatches}")
    if dtype is not None:
        print("Note: dtype is NOT float32, so mismatches can happen due to numerical instability")


def get_parsed_args():
    parser = argparse.ArgumentParser(description='Run the benchmark, comparing the original to the new generation.')
    parser.add_argument('model', type=str)
    parser.add_argument('--aux-model', type=str)
    parser.add_argument('--dtype', type=str)
    parser.add_argument('--temperature', type=float)  # non None triggers sampling
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--max-gpu-memory', type=int, nargs="*")

    args = parser.parse_args()

    args.load_in_8bit = False
    if args.dtype is not None:
        if args.dtype == "fp16" or args.dtype == "float16":
            args.dtype = torch.float16
        elif args.dtype == "fp32" or args.dtype == "float32":
            args.dtype = torch.float32
        elif args.dtype == "bf16" or args.dtype == "bfloat16":
            args.dtype = torch.bfloat16
        elif args.dtype == "int8":
            args.dtype = torch.float16
            args.load_in_8bit = True

    return args


def run_og_model(args, processor_cls, model_cls, run_prediction_loop, queue):
    tokenizer = processor_cls.from_pretrained(args.model)

    if args.max_gpu_memory is None:  # fails if it doesn't fit in a GPU
        max_memory = {0: "100GiB", "cpu": "0GiB"}
    else:
        max_memory = {}
        for i in range(len(args.max_gpu_memory)):
            max_memory[i] = str(args.max_gpu_memory[i])+"GiB"
        max_memory["cpu"] = "50GiB"
    print(f"Max memory allocation: {max_memory}")
    model_kwargs = {
        "pretrained_model_name_or_path": args.model,
        "device_map": "auto",
        "max_memory": max_memory,
        "torch_dtype": args.dtype,
        "load_in_8bit": args.load_in_8bit,
    }
    model = model_cls.from_pretrained(**model_kwargs)
    og_outputs = run_prediction_loop(model, tokenizer, args.num_samples, args.temperature)
    queue.put(og_outputs)


def run_new_model(args, processor_cls, model_cls, run_prediction_loop, queue):
    tokenizer = processor_cls.from_pretrained(args.model)

    aux_model = model_cls.from_pretrained(args.aux_model)
    aux_model = aux_model.to(TORCH_DEVICE)

    if args.max_gpu_memory is None:  # fails if it doesn't fit in a GPU
        max_memory = {0: "100GiB", "cpu": "0GiB"}
    else:
        max_memory = {}
        for i in range(len(args.max_gpu_memory)):
            max_memory[i] = str(args.max_gpu_memory[i])+"GiB"
        max_memory["cpu"] = "50GiB"
    print(f"Max memory allocation: {max_memory}")
    model_kwargs = {
        "pretrained_model_name_or_path": args.model,
        "device_map": "auto",
        "max_memory": max_memory,
        "torch_dtype": args.dtype,
        "load_in_8bit": args.load_in_8bit,
    }
    model = model_cls.from_pretrained(**model_kwargs)

    new_outputs = run_prediction_loop(model, tokenizer, args.num_samples, args.temperature, aux_model)
    queue.put(new_outputs)
