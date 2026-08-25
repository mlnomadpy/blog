#!/usr/bin/env python3
"""Ablate every MLP residual branch in Gemma 4 12B and benchmark the result.

The script uses the MLX 4-bit conversion on Apple Silicon.  It evaluates:

* zero-shot multiple-choice accuracy on a deterministic slice of MMLU,
* next-token negative log-likelihood on WikiText-2,
* qualitative generations from a small fixed prompt set, and
* prefill/decode throughput at several prompt lengths.

The ablation is deliberately surgical: ``layer.mlp(x)`` is replaced with zeros.
The pre/post MLP norms and the residual addition remain, so each transformer
layer becomes an attention-only residual block without changing its depth.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.ipc as arrow_ipc
from mlx_lm import load
from mlx_lm.generate import generate_step, stream_generate
from mlx_lm.models import gemma4
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import MODEL_REMAPPING


DEFAULT_MODEL = "tmp/gemma-4-12B-it-4bit"
DEFAULT_OUTPUT = "scripts/results/gemma4_no_mlp.json"
TOTAL_CHECKPOINT_PARAMETERS = 11_959_730_224


# mlx-lm 0.31.3 has the right Gemma 4 implementation under `gemma4`, while
# Google's unified 12B config uses the more specific `gemma4_unified` label.
MODEL_REMAPPING["gemma4_unified"] = "gemma4"


# This unified checkpoint revision uses `vision_embedder.*` for its lightweight
# patch projection.  The text-only MLX class has no vision module and mlx-lm
# 0.31.3's sanitizer predates that tensor prefix, so discard only those unused
# multimodal weights before the usual strict language-weight load.
_gemma4_sanitize = gemma4.Model.sanitize


def _sanitize_unified_text_weights(self: nn.Module, weights: dict[str, mx.array]):
    weights = {k: v for k, v in weights.items() if not k.startswith("vision_embedder.")}
    return _gemma4_sanitize(self, weights)


gemma4.Model.sanitize = _sanitize_unified_text_weights


@dataclass
class SpeedRun:
    prompt_tokens: int
    prefill_tps: float
    decode_tps: float
    generated_tokens: int


class ZeroMLP(nn.Module):
    """Drop an MLP branch without altering the surrounding residual block."""

    def __call__(self, x: mx.array) -> mx.array:
        return mx.zeros_like(x)


def language_layers(model: nn.Module) -> list[nn.Module]:
    layers = list(model.layers)
    if not layers or not all(hasattr(layer, "mlp") for layer in layers):
        raise RuntimeError("Could not find the Gemma decoder MLP modules")
    return layers


def ablate_all_mlps(model: nn.Module) -> int:
    layers = language_layers(model)
    for layer in layers:
        layer.mlp = ZeroMLP()
    gc.collect()
    mx.clear_cache()
    return len(layers)


def chat_prompt(tokenizer: Any, text: str) -> list[int]:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def cached_arrow(pattern: str) -> list[dict[str, Any]]:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(cache_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No cached dataset matches {pattern!r} under {cache_root}. "
            "Download the dataset once with Hugging Face datasets first."
        )
    with arrow_ipc.open_stream(matches[-1]) as reader:
        return reader.read_all().to_pylist()


def evaluate_mmlu(
    model: nn.Module,
    tokenizer: Any,
    samples: int,
) -> dict[str, Any]:
    """Score A/B/C/D from the next-token distribution, with no sampling."""

    dataset = cached_arrow("cais___mmlu/all/0.0.0/*/mmlu-test.arrow")
    # Spreading indices across the full concatenated dataset avoids accidentally
    # measuring only one alphabetically early subject.
    indices = np.linspace(0, len(dataset) - 1, samples, dtype=int).tolist()
    letters = ["A", "B", "C", "D"]
    candidate_ids = []
    for letter in letters:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"Expected {letter!r} to be one token, got {ids}")
        candidate_ids.append(ids[0])

    correct = 0
    rows = []
    for index in indices:
        row = dataset[index]
        choices = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(letters, row["choices"])
        )
        prompt = chat_prompt(
            tokenizer,
            "Answer this multiple-choice question. Reply with only the letter "
            f"A, B, C, or D.\n\n{row['question']}\n{choices}\n\nAnswer:",
        )
        # With thinking disabled, Gemma 4's template closes an empty thought
        # channel and the model normally emits this final-channel marker next.
        # Supplying it makes A/B/C/D the actual next-token decision we score.
        prompt += tokenizer.encode("<|channel>final\n", add_special_tokens=False)
        logits = model(mx.array(prompt)[None])
        scores = logits[0, -1, candidate_ids]
        mx.eval(scores)
        predicted = int(mx.argmax(scores).item())
        answer = int(row["answer"])
        correct += predicted == answer
        rows.append(
            {
                "index": index,
                "subject": row["subject"],
                "answer": letters[answer],
                "predicted": letters[predicted],
            }
        )
        mx.clear_cache()

    return {
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples,
        "rows": rows,
    }


def evaluate_wikitext(
    model: nn.Module,
    tokenizer: Any,
    token_count: int,
) -> dict[str, Any]:
    """Compute chat-formatted continuation NLL on fixed WikiText-2 text."""

    dataset = cached_arrow(
        "wikitext/wikitext-2-raw-v1/0.0.0/*/wikitext-validation.arrow"
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    context_ids = text_ids[:128]
    target_ids = text_ids[128 : 128 + token_count]
    context = tokenizer.decode(context_ids)
    prefix = chat_prompt(
        tokenizer,
        "Continue the passage exactly, without commentary.\n\n" + context,
    )
    prefix += tokenizer.encode("<|channel>final\n", add_special_tokens=False)
    ids = prefix + target_ids
    inputs = mx.array(ids[:-1])[None]
    targets = mx.array(ids[1:])
    logits = model(inputs)[0]
    start = len(prefix) - 1
    target_logits = logits[start : start + len(target_ids)]
    target_values = targets[start : start + len(target_ids)]
    token_logits = mx.take_along_axis(
        target_logits, target_values[:, None], axis=-1
    ).squeeze(-1)
    nll = mx.logsumexp(target_logits, axis=-1) - token_logits
    mean_nll = float(mx.mean(nll).item())
    mx.clear_cache()
    return {
        "context_tokens": len(context_ids),
        "scored_tokens": len(target_ids),
        "mean_nll_nats": mean_nll,
        "perplexity": math.exp(min(mean_nll, 80.0)),
    }


QUALITATIVE_PROMPTS = [
    "Explain why the sky is blue to a curious twelve-year-old in three sentences.",
    "A farmer has 17 sheep. All but 9 run away. How many remain? Give the answer and one short reason.",
    "Write a Python function that returns True when a string is a palindrome, ignoring spaces and capitalization.",
    "Continue this micro-story in no more than 70 words: The elevator opened onto a floor that was not on the directory.",
]


def qualitative_samples(model: nn.Module, tokenizer: Any, max_tokens: int) -> list[dict[str, str]]:
    sampler = make_sampler(temp=0.0)
    rows = []
    for prompt in QUALITATIVE_PROMPTS:
        pieces = []
        for response in stream_generate(
            model,
            tokenizer,
            mx.array(chat_prompt(tokenizer, prompt)),
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            pieces.append(response.text)
        rows.append({"prompt": prompt, "response": "".join(pieces).strip()})
        mx.clear_cache()
    return rows


def fixed_prompt_tokens(tokenizer: Any, length: int) -> mx.array:
    seed = tokenizer.encode(
        "Local inference is useful because it makes latency, privacy, and model "
        "behavior directly measurable. A controlled benchmark repeats the same "
        "workload so architecture changes can be compared fairly. ",
        add_special_tokens=True,
    )
    repeated = (seed * (length // len(seed) + 1))[:length]
    return mx.array(repeated)


def one_speed_run(
    model: nn.Module,
    prompt: mx.array,
    decode_tokens: int,
) -> SpeedRun:
    sampler = make_sampler(temp=0.0)
    iterator = generate_step(prompt, model, max_tokens=decode_tokens, sampler=sampler)

    start = time.perf_counter()
    token, _ = next(iterator)
    prefill_seconds = time.perf_counter() - start

    generated = 1
    start = time.perf_counter()
    for token, _ in iterator:
        generated += 1
    decode_seconds = time.perf_counter() - start
    return SpeedRun(
        prompt_tokens=int(prompt.size),
        prefill_tps=float(prompt.size / prefill_seconds),
        decode_tps=float(max(generated - 1, 0) / decode_seconds),
        generated_tokens=generated,
    )


def benchmark_speed(
    model: nn.Module,
    tokenizer: Any,
    lengths: Iterable[int],
    repeats: int,
    decode_tokens: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    # Compile/warm the common one-token decode path before timing.
    warm = one_speed_run(model, fixed_prompt_tokens(tokenizer, 32), 4)
    del warm
    mx.clear_cache()

    for length in lengths:
        prompt = fixed_prompt_tokens(tokenizer, length)
        runs = [one_speed_run(model, prompt, decode_tokens) for _ in range(repeats)]
        output[str(length)] = {
            "runs": [asdict(run) for run in runs],
            "prefill_tps_median": statistics.median(run.prefill_tps for run in runs),
            "decode_tps_median": statistics.median(run.decode_tps for run in runs),
        }
        mx.clear_cache()
    return output


def run_condition(
    name: str,
    model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(f"\n[{name}] MMLU ({args.mmlu_samples} samples)", flush=True)
    mmlu = evaluate_mmlu(model, tokenizer, args.mmlu_samples)
    print(f"[{name}] MMLU accuracy: {mmlu['accuracy']:.1%}", flush=True)

    print(f"[{name}] WikiText ({args.ppl_tokens} tokens)", flush=True)
    wikitext = evaluate_wikitext(model, tokenizer, args.ppl_tokens)
    print(f"[{name}] WikiText PPL: {wikitext['perplexity']:.2f}", flush=True)

    print(f"[{name}] qualitative generations", flush=True)
    samples = qualitative_samples(model, tokenizer, args.sample_tokens)

    print(f"[{name}] speed", flush=True)
    speed = benchmark_speed(
        model,
        tokenizer,
        args.prompt_lengths,
        args.speed_repeats,
        args.decode_tokens,
    )
    return {
        "mmlu": mmlu,
        "wikitext": wikitext,
        "samples": samples,
        "speed": speed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--mmlu-samples", type=int, default=40)
    parser.add_argument("--ppl-tokens", type=int, default=512)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--speed-repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    print(f"Loading {model_path}", flush=True)
    model, tokenizer = load(model_path, lazy=True)
    layers = language_layers(model)
    config = model.args.text_config
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    mlp_parameters = len(layers) * 3 * hidden * intermediate

    result: dict[str, Any] = {
        "experiment": {
            "model": str(model_path),
            "quantization": "MLX affine 4-bit, group size 64",
            "hardware": platform.platform(),
            "layers": len(layers),
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "removed_mlp_parameters": mlp_parameters,
            "total_checkpoint_parameters": TOTAL_CHECKPOINT_PARAMETERS,
            "removed_parameter_fraction": mlp_parameters / TOTAL_CHECKPOINT_PARAMETERS,
            "definition": "Replace each layer.mlp output with zeros; retain norms, residuals, attention, embeddings, and LM head.",
            "arguments": vars(args),
        }
    }

    result["intact"] = run_condition("intact", model, tokenizer, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    count = ablate_all_mlps(model)
    print(f"\nAblated {count} MLP modules", flush=True)
    result["no_mlp"] = run_condition("no_mlp", model, tokenizer, args)

    for length in args.prompt_lengths:
        key = str(length)
        intact = result["intact"]["speed"][key]
        ablated = result["no_mlp"]["speed"][key]
        ablated["prefill_speedup_vs_intact"] = (
            ablated["prefill_tps_median"] / intact["prefill_tps_median"]
        )
        ablated["decode_speedup_vs_intact"] = (
            ablated["decode_tps_median"] / intact["decode_tps_median"]
        )

    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nWrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
