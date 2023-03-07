import argparse
from typing import Optional, Union, List
import warnings
import types

from transformers import LogitsProcessorList, StoppingCriteriaList
import torch
import torch.distributed as dist


TORCH_DEVICE = 0


def modify_generation(model, aux_model):

    model.aux_model = aux_model
    model.fwd_tokens = 5
    model.aux_encoder_outputs = None
    original_generate = model.generate

    def generate(self, inputs=None, **kwargs):
        # Encoder-decoder model -> get the encoder outputs for the aux model
        if self.config.is_encoder_decoder:
            inputs_tensor, model_input_name, model_kwargs = self.aux_model._prepare_model_inputs(
                inputs, self.generation_config.bos_token_id, kwargs
            )
            model_kwargs = self.aux_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )
            self.aux_encoder_outputs = model_kwargs["encoder_outputs"]

        return original_generate(inputs, **kwargs)

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ):
        """ Modified greedy search with aux model candidate token generation """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            # stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # =========================================================================================================
            # Major changes start here

            # Use aux model to fast forward
            cur_len = input_ids.shape[-1]
            max_len = stopping_criteria[0].max_length

            # ASSUMPTION -- batch size = 1
            # ASSUMPTION -- no logits processor (would have to be applied in aux model too)
            # forecast next N tokens 👉 this whole loop could have been a simple generate call, if generate
            # returned the past_key_values. Forwarding past_key_values for the the auxiliary model makes a big
            # difference in speed unless the larger model has severe HW bottlenecks.
            candidate_input_ids = input_ids
            for new_token_idx in range(int(self.fwd_tokens)):
                if "aux_past_key_values" in model_kwargs:
                    prev_seq_len = model_kwargs["aux_past_key_values"][0][0].shape[2]
                    # can be 1 or 2 (next token + last token picked by the larger model)
                    new_token_len = candidate_input_ids.shape[1] - prev_seq_len
                    tmp_inputs = candidate_input_ids[:, -new_token_len:]
                    tmp_attn = torch.ones_like(candidate_input_ids)
                    if self.config.is_encoder_decoder:
                        aux_model_outputs = self.aux_model(
                            decoder_input_ids=tmp_inputs,
                            decoder_attention_mask=tmp_attn,
                            past_key_values=model_kwargs["aux_past_key_values"],
                            encoder_outputs=self.aux_encoder_outputs
                        )
                    else:
                        aux_model_outputs = self.aux_model(
                            tmp_inputs, attention_mask=tmp_attn, past_key_values=model_kwargs["aux_past_key_values"]
                        )
                else:
                    if self.config.is_encoder_decoder:
                        aux_model_outputs = self.aux_model(
                            decoder_input_ids=candidate_input_ids, encoder_outputs=self.aux_encoder_outputs
                        )
                    else:
                        aux_model_outputs = self.aux_model(candidate_input_ids)

                model_kwargs["aux_past_key_values"] = aux_model_outputs.past_key_values
                if len(logits_processor) > 0:
                    aux_model_outputs.logits[:, -1, :] = logits_processor(
                        candidate_input_ids, aux_model_outputs.logits[:, -1, :]
                    )
                new_token = aux_model_outputs.logits[:, -1, :].argmax(dim=-1)
                candidate_input_ids = torch.cat((candidate_input_ids, new_token[:, None]), dim=-1)

                # stop on EOS
                if eos_token_id_tensor is not None:
                    last_aux_token_is_eos = new_token.tile(eos_token_id_tensor.shape[0], 1)
                    last_aux_token_is_eos = ~last_aux_token_is_eos.ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0).bool()
                    if last_aux_token_is_eos:
                        break

            new_tokens = new_token_idx + 1

            # original model forward pass using the new candidates. We predict "new_tokens + 1" logits, the +1 is the
            # token after the last candidate given by the smaller model
            if "past_key_values" in model_kwargs:
                og_model_attn = torch.ones_like(candidate_input_ids)
                og_model_input_ids = candidate_input_ids[:, -new_tokens - 1:]
                if self.config.is_encoder_decoder:
                    outputs = self(
                        decoder_input_ids=og_model_input_ids,
                        decoder_attention_mask=og_model_attn,
                        past_key_values=model_kwargs["past_key_values"],
                        encoder_outputs=model_kwargs["encoder_outputs"]
                    )
                else:
                    outputs = self(
                        og_model_input_ids,
                        attention_mask=og_model_attn,
                        past_key_values=model_kwargs["past_key_values"]
                    )
            else:
                if self.config.is_encoder_decoder:
                    outputs = self(
                        decoder_input_ids=candidate_input_ids, encoder_outputs=model_kwargs["encoder_outputs"]
                    )
                else:
                    outputs = self(candidate_input_ids)

            # apply logits processor to the logits (if there is one)
            if len(logits_processor) > 0:
                for i in range(new_tokens):
                    outputs.logits[:, i, :] = logits_processor(
                        candidate_input_ids[:, :cur_len + i], outputs.logits[:, i, :]
                    )

            # confirm whether they are the max logits in the fwd pass
            max_logits = outputs.logits.argmax(dim=-1)[:, -new_tokens - 1:-1]
            candidate_new_tokens = candidate_input_ids[:, -new_tokens:]
            n_matches = ((~(candidate_new_tokens == max_logits)).cumsum(dim=-1) < 1).sum()

            if n_matches == int(self.fwd_tokens):
                self.fwd_tokens += 2.0  # aggressively increase this value
            else:
                self.fwd_tokens -= 1.0  # set to 1.0 if there is a HW bottleneck, 2.0 if not
                if self.fwd_tokens < 1.0:
                    self.fwd_tokens = 1.0

            n_matches = min(n_matches, max_len - cur_len)
            # print(f"{n_matches} / {new_tokens}")
            input_ids = candidate_input_ids[:, 0:cur_len + n_matches]

            # ASSUMPTION -- batch size = 1
            # check stopping criteria here
            if (last_aux_token_is_eos and n_matches == new_tokens) or stopping_criteria(input_ids, None):
                break

            # update other vars
            new_cur_len = input_ids.shape[-1]

            if self.config.is_encoder_decoder:
                new_past = []
                for idx in range(len(outputs.past_key_values)):
                    new_past.append((
                        outputs.past_key_values[idx][0][:, :, :new_cur_len, :],
                        outputs.past_key_values[idx][1][:, :, :new_cur_len, :],
                        outputs.past_key_values[idx][2],
                        outputs.past_key_values[idx][3]
                    ))
                outputs.past_key_values = tuple(new_past)

                new_past = []
                for idx in range(len(model_kwargs["aux_past_key_values"])):
                    new_past.append((
                        model_kwargs["aux_past_key_values"][idx][0][:, :, :new_cur_len, :],
                        model_kwargs["aux_past_key_values"][idx][1][:, :, :new_cur_len, :],
                        model_kwargs["aux_past_key_values"][idx][2],
                        model_kwargs["aux_past_key_values"][idx][3]
                    ))
                model_kwargs["aux_past_key_values"] = new_past

            else:
                new_past = []
                for idx in range(len(outputs.past_key_values)):
                    new_past.append((
                        outputs.past_key_values[idx][0][:, :, :new_cur_len, :],
                        outputs.past_key_values[idx][1][:, :, :new_cur_len, :]
                    ))
                outputs.past_key_values = tuple(new_past)

                new_past = []
                for idx in range(len(model_kwargs["aux_past_key_values"])):
                    new_past.append((
                        model_kwargs["aux_past_key_values"][idx][0][:, :, :new_cur_len, :],
                        model_kwargs["aux_past_key_values"][idx][1][:, :, :new_cur_len, :]
                    ))
                model_kwargs["aux_past_key_values"] = new_past

            if outputs.logits.shape[1] > new_tokens + 1:
                logits_idx = new_cur_len - 1
            else:
                logits_idx = n_matches
            next_token_logits = outputs.logits[:, logits_idx, :]

            # - ignore for now: other aux outputs

            # =========================================================================================================
            # The rest is very similar to the original code

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, None):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        return input_ids

    model.greedy_search = types.MethodType(greedy_search, model)
    model.generate = types.MethodType(generate, model)
    return model


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
    og_outputs = run_prediction_loop(model, tokenizer, args.num_samples)
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
    model = modify_generation(model, aux_model)
    new_outputs = run_prediction_loop(model, tokenizer, args.num_samples)
    queue.put(new_outputs)
