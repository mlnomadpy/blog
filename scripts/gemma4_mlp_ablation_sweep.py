#!/usr/bin/env python3
"""Layerwise and partial-MLP ablations for Gemma 4 12B.

This extends the all-MLP intervention with:

* every single MLP removed in isolation,
* first/last/alternating half ablations,
* a uniformly distributed 0/6/12/24/36/48-layer sweep,
* residual-stream RMS and cosine drift through depth, and
* an all-MLP control that matches the intact residual RMS after every layer.

The RMS-matched condition deliberately restores only one scalar statistic.  It
tests whether scale drift explains failure; it cannot restore lost directions.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gemma4_no_mlp import (
    TOTAL_CHECKPOINT_PARAMETERS,
    benchmark_speed,
    cached_arrow,
    chat_prompt,
    evaluate_wikitext,
    fixed_prompt_tokens,
    load,
    qualitative_samples,
)


DEFAULT_MODEL = "tmp/gemma-4-12B-it-4bit"
DEFAULT_OUTPUT = "scripts/results/gemma4_mlp_ablation_sweep.json"
N_LAYERS = 48


DOMAIN_SUBJECTS = {
    "math": [
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
        "high_school_statistics",
    ],
    "physics_engineering": [
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "electrical_engineering",
        "high_school_physics",
    ],
    "computing": [
        "college_computer_science",
        "computer_security",
        "high_school_computer_science",
        "machine_learning",
    ],
    "life_sciences": [
        "anatomy",
        "college_biology",
        "high_school_biology",
        "medical_genetics",
        "virology",
    ],
    "medicine": [
        "clinical_knowledge",
        "college_medicine",
        "human_aging",
        "nutrition",
        "professional_medicine",
    ],
    "law_humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "professional_law",
        "world_religions",
    ],
    "social_business": [
        "business_ethics",
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "management",
        "marketing",
        "professional_accounting",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}


class AblatableMLP(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.enabled = True

    def __call__(self, x: mx.array) -> mx.array:
        if self.enabled:
            return self.inner(x)
        return mx.zeros_like(x)


class TraceLayer(nn.Module):
    """Transparent decoder wrapper with optional tracing and RMS matching."""

    def __init__(self, inner: nn.Module, index: int):
        super().__init__()
        self.inner = inner
        self.index = index
        self.layer_type = inner.layer_type
        self.capture = False
        self.captured: np.ndarray | None = None
        self.target_rms: float | None = None

    def __call__(self, *args, **kwargs):
        h, kvs, offset = self.inner(*args, **kwargs)
        if self.target_rms is not None:
            current_rms = mx.sqrt(mx.mean(h.astype(mx.float32).square()))
            h = h * (self.target_rms / mx.maximum(current_rms, 1e-8))
        if self.capture:
            h32 = h.astype(mx.float32)
            mx.eval(h32)
            self.captured = np.asarray(h32)
        return h, kvs, offset


def install_controls(model: nn.Module) -> tuple[list[AblatableMLP], list[TraceLayer]]:
    # `model.layers` is the live list owned by Gemma4TextModel.  Do not copy it:
    # the trace wrappers must participate in the actual forward pass.
    raw_layers = model.layers
    if len(raw_layers) != N_LAYERS:
        raise RuntimeError(f"Expected {N_LAYERS} decoder layers, got {len(raw_layers)}")
    mlps: list[AblatableMLP] = []
    traces: list[TraceLayer] = []
    for index, layer in enumerate(raw_layers):
        wrapped_mlp = AblatableMLP(layer.mlp)
        layer.mlp = wrapped_mlp
        wrapped_layer = TraceLayer(layer, index)
        raw_layers[index] = wrapped_layer
        mlps.append(wrapped_mlp)
        traces.append(wrapped_layer)
    return mlps, traces


def set_ablation(mlps: list[AblatableMLP], removed: Iterable[int]) -> None:
    removed = set(removed)
    for index, mlp in enumerate(mlps):
        mlp.enabled = index not in removed
    gc.collect()
    mx.clear_cache()


def set_rms_targets(traces: list[TraceLayer], targets: list[float] | None) -> None:
    for index, layer in enumerate(traces):
        layer.target_rms = None if targets is None else targets[index]


def capture_residuals(
    model: nn.Module,
    traces: list[TraceLayer],
    prompt: mx.array,
) -> list[np.ndarray]:
    for layer in traces:
        layer.capture = True
        layer.captured = None
    logits = model(prompt[None])
    mx.eval(logits[0, -1, :1])
    arrays = []
    for layer in traces:
        if layer.captured is None:
            raise RuntimeError(f"Layer {layer.index} did not record an activation")
        arrays.append(layer.captured)
        layer.capture = False
        layer.captured = None
    mx.clear_cache()
    return arrays


def activation_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def compare_trace(current: list[np.ndarray], baseline: list[np.ndarray]) -> list[dict[str, float]]:
    rows = []
    for index, (x, ref) in enumerate(zip(current, baseline)):
        x_flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        r_flat = ref.reshape(-1, ref.shape[-1]).astype(np.float64)
        numerator = np.sum(x_flat * r_flat, axis=-1)
        denominator = np.linalg.norm(x_flat, axis=-1) * np.linalg.norm(r_flat, axis=-1)
        cosine = numerator / np.maximum(denominator, 1e-12)
        x_rms = activation_rms(x)
        ref_rms = activation_rms(ref)
        rows.append(
            {
                "layer": index,
                "rms": x_rms,
                "intact_rms": ref_rms,
                "rms_ratio": x_rms / ref_rms,
                "cosine_to_intact_mean": float(np.mean(cosine)),
                "cosine_to_intact_min": float(np.min(cosine)),
            }
        )
    return rows


def fixed_domain_questions(
    tokenizer: Any, samples_per_domain: int
) -> dict[str, list[dict[str, Any]]]:
    dataset = cached_arrow("cais___mmlu/all/0.0.0/*/mmlu-test.arrow")
    output = {}
    for domain, subjects in DOMAIN_SUBJECTS.items():
        pool = [row for row in dataset if row["subject"] in subjects]
        indices = np.linspace(0, len(pool) - 1, samples_per_domain, dtype=int)
        selected = []
        for i in indices:
            row = pool[int(i)]
            letters = ["A", "B", "C", "D"]
            choices = "\n".join(
                f"{letter}. {choice}"
                for letter, choice in zip(letters, row["choices"])
            )
            prompt = chat_prompt(
                tokenizer,
                "Answer this multiple-choice question. Reply with only the letter "
                f"A, B, C, or D.\n\n{row['question']}\n{choices}\n\nAnswer:",
            )
            prompt += tokenizer.encode("<|channel>final\n", add_special_tokens=False)
            selected.append(
                {
                    "prompt_ids": prompt,
                    "answer": int(row["answer"]),
                    "subject": row["subject"],
                }
            )
        output[domain] = selected
    return output


def evaluate_domains(
    model: nn.Module,
    tokenizer: Any,
    questions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    letters = ["A", "B", "C", "D"]
    candidate_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in letters]
    domains = {}
    total_correct = 0
    total_questions = 0
    all_log_probs = []
    all_margins = []
    for domain, rows in questions.items():
        correct = 0
        correct_log_probs = []
        margins = []
        subject_correct: dict[str, int] = {}
        subject_count: dict[str, int] = {}
        for row in rows:
            logits = model(mx.array(row["prompt_ids"])[None])
            scores = logits[0, -1, candidate_ids].astype(mx.float32)
            mx.eval(scores)
            scores_np = np.asarray(scores, dtype=np.float64)
            answer = int(row["answer"])
            predicted = int(np.argmax(scores_np))
            is_correct = predicted == answer
            correct += is_correct
            log_denom = float(np.logaddexp.reduce(scores_np))
            correct_log_probs.append(float(scores_np[answer] - log_denom))
            incorrect = np.delete(scores_np, answer)
            margins.append(float(scores_np[answer] - np.max(incorrect)))
            subject = row["subject"]
            subject_correct[subject] = subject_correct.get(subject, 0) + is_correct
            subject_count[subject] = subject_count.get(subject, 0) + 1
            mx.clear_cache()
        domains[domain] = {
            "samples": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "mean_correct_choice_logprob": float(np.mean(correct_log_probs)),
            "mean_correct_vs_best_incorrect_margin": float(np.mean(margins)),
            "subjects": {
                subject: {
                    "samples": subject_count[subject],
                    "accuracy": subject_correct[subject] / subject_count[subject],
                }
                for subject in sorted(subject_count)
            },
        }
        total_correct += correct
        total_questions += len(rows)
        all_log_probs.extend(correct_log_probs)
        all_margins.extend(margins)
    return {
        "samples": total_questions,
        "correct": total_correct,
        "accuracy": total_correct / total_questions,
        "mean_correct_choice_logprob": float(np.mean(all_log_probs)),
        "mean_correct_vs_best_incorrect_margin": float(np.mean(all_margins)),
        "domains": domains,
    }


def compact_quality(
    model: nn.Module,
    tokenizer: Any,
    domain_questions: dict[str, list[dict[str, Any]]],
    ppl_tokens: int,
):
    mmlu = evaluate_domains(model, tokenizer, domain_questions)
    wikitext = evaluate_wikitext(model, tokenizer, ppl_tokens)
    return {
        "mmlu_samples": mmlu["samples"],
        "mmlu_correct": mmlu["correct"],
        "mmlu_accuracy": mmlu["accuracy"],
        "mmlu_mean_correct_choice_logprob": mmlu["mean_correct_choice_logprob"],
        "mmlu_mean_correct_vs_best_incorrect_margin": mmlu[
            "mean_correct_vs_best_incorrect_margin"
        ],
        "domains": mmlu["domains"],
        "wikitext_scored_tokens": wikitext["scored_tokens"],
        "wikitext_mean_nll_nats": wikitext["mean_nll_nats"],
        "wikitext_perplexity": wikitext["perplexity"],
    }


def masks() -> dict[str, list[int]]:
    return {
        "intact": [],
        "uniform_6": list(range(0, N_LAYERS, 8)),
        "uniform_12": list(range(0, N_LAYERS, 4)),
        "alternating_even_24": list(range(0, N_LAYERS, 2)),
        "alternating_odd_24": list(range(1, N_LAYERS, 2)),
        "first_24": list(range(24)),
        "last_24": list(range(24, 48)),
        "uniform_36": [i for i in range(N_LAYERS) if i % 4 != 3],
        "all_48": list(range(N_LAYERS)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--domain-samples", type=int, default=12)
    parser.add_argument("--single-ppl-tokens", type=int, default=128)
    parser.add_argument("--group-ppl-tokens", type=int, default=512)
    parser.add_argument("--trace-prompt-tokens", type=int, default=128)
    parser.add_argument("--speed-repeats", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()
    print(f"Loading {model_path}", flush=True)
    model, tokenizer = load(model_path, lazy=True)
    mlps, traces = install_controls(model)
    all_masks = masks()
    domain_questions = fixed_domain_questions(tokenizer, args.domain_samples)

    trace_prompt = fixed_prompt_tokens(tokenizer, args.trace_prompt_tokens)
    set_ablation(mlps, [])
    baseline_trace = capture_residuals(model, traces, trace_prompt)
    baseline_rms = [activation_rms(x) for x in baseline_trace]

    result: dict[str, Any] = {
        "experiment": {
            "model": str(model_path),
            "total_checkpoint_parameters": TOTAL_CHECKPOINT_PARAMETERS,
            "layers": N_LAYERS,
            "arguments": vars(args),
            "mask_definition": "Zero only the selected layer.mlp outputs; keep all attention, residual, and norm operations.",
            "rms_match_definition": "After each decoder layer, multiply the full residual stream by one scalar so its global RMS equals the intact calibration-prompt RMS at that depth.",
        },
        "single_layer": [],
        "groups": {},
        "traces": {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Single-layer sweep", flush=True)
    for index in range(N_LAYERS):
        set_ablation(mlps, [index])
        quality = compact_quality(
            model, tokenizer, domain_questions, args.single_ppl_tokens
        )
        quality["removed_layer"] = index
        result["single_layer"].append(quality)
        print(
            f"  layer {index:02d}: MMLU {quality['mmlu_accuracy']:.1%}, "
            f"NLL {quality['wikitext_mean_nll_nats']:.3f}",
            flush=True,
        )
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print("Grouped/progressive sweep", flush=True)
    for name, removed in all_masks.items():
        set_ablation(mlps, removed)
        set_rms_targets(traces, None)
        quality = compact_quality(
            model, tokenizer, domain_questions, args.group_ppl_tokens
        )
        result["groups"][name] = {
            "removed_layers": removed,
            "removed_count": len(removed),
            **quality,
        }
        print(
            f"  {name}: {len(removed)} layers, MMLU {quality['mmlu_accuracy']:.1%}, "
            f"NLL {quality['wikitext_mean_nll_nats']:.3f}",
            flush=True,
        )
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    # Directly test the scalar-statistics objection for the strongest ablation.
    set_ablation(mlps, all_masks["all_48"])
    set_rms_targets(traces, baseline_rms)
    matched = compact_quality(
        model, tokenizer, domain_questions, args.group_ppl_tokens
    )
    matched["samples"] = qualitative_samples(model, tokenizer, 40)
    result["groups"]["all_48_rms_matched"] = {
        "removed_layers": all_masks["all_48"],
        "removed_count": N_LAYERS,
        **matched,
    }
    set_rms_targets(traces, None)
    print(
        f"  all_48_rms_matched: MMLU {matched['mmlu_accuracy']:.1%}, "
        f"NLL {matched['wikitext_mean_nll_nats']:.3f}",
        flush=True,
    )

    # Trace key conditions plus the single layer with the largest NLL increase.
    worst_single = max(
        result["single_layer"], key=lambda row: row["wikitext_mean_nll_nats"]
    )["removed_layer"]
    trace_masks = {
        "intact": [],
        f"single_{worst_single}": [worst_single],
        "first_24": all_masks["first_24"],
        "last_24": all_masks["last_24"],
        "alternating_even_24": all_masks["alternating_even_24"],
        "all_48": all_masks["all_48"],
    }
    print("Residual-stream traces", flush=True)
    for name, removed in trace_masks.items():
        set_ablation(mlps, removed)
        set_rms_targets(traces, None)
        current = capture_residuals(model, traces, trace_prompt)
        result["traces"][name] = compare_trace(current, baseline_trace)
        print(f"  {name}", flush=True)

    set_ablation(mlps, all_masks["all_48"])
    set_rms_targets(traces, baseline_rms)
    current = capture_residuals(model, traces, trace_prompt)
    result["traces"]["all_48_rms_matched"] = compare_trace(current, baseline_trace)
    set_rms_targets(traces, None)

    # Speed only the conditions needed to tell the partial-ablation story.
    print("Speed sweep", flush=True)
    for name in ["intact", "first_24", "last_24", "alternating_even_24", "all_48"]:
        set_ablation(mlps, all_masks[name])
        result["groups"][name]["speed"] = benchmark_speed(
            model,
            tokenizer,
            [128, 512, 2048],
            args.speed_repeats,
            args.decode_tokens,
        )
        print(f"  {name}", flush=True)

    # Add representative generations for the conditions readers will compare.
    print("Qualitative samples", flush=True)
    for name in ["intact", "first_24", "last_24", "alternating_even_24", "all_48"]:
        set_ablation(mlps, all_masks[name])
        result["groups"][name]["samples"] = qualitative_samples(model, tokenizer, 60)
        print(f"  {name}", flush=True)

    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
