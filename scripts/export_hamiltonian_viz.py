"""Reshape the hamiltonian_net.py Kaggle results into public/hamiltonian-net/.

Reads scripts/results/kgl_blog-hamiltonian-v2/ (the downloaded run bundle) and
writes the JSON the explainer's live panels load:

  public/hamiltonian-net/pendulum.json   trained field weights (baseline MLP +
                                         HNN) and the reference rollout traces
  public/hamiltonian-net/depth.json      per-config accuracy/drift table, the
                                         seed-0 depth traces, and the seed-0
                                         trained nets + data samples for the
                                         live depth panels

Run locally (pure reshaping, no training): python3 scripts/export_hamiltonian_viz.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "results", "kgl_blog-hamiltonian-v2")
OUT = os.path.join(HERE, "..", "public", "hamiltonian-net")
os.makedirs(OUT, exist_ok=True)


def find(name):
    for root, _, files in os.walk(BUNDLE):
        if name in files:
            return os.path.join(root, name)
    raise SystemExit(f"missing {name} under {BUNDLE}")


def rnd(x, k=5):
    if isinstance(x, float):
        return round(x, k)
    if isinstance(x, list):
        return [rnd(v, k) for v in x]
    if isinstance(x, dict):
        return {a: rnd(b, k) for a, b in x.items()}
    return x


def main():
    phys = json.load(open(find("physics.json")))
    nets = json.load(open(find("networks.json")))

    pendulum = dict(
        E0=phys["E0"], dt=phys["dt"], n_steps=phys["n_steps"],
        q0=phys["q0"], p0=phys["p0"],
        baseline_energy_drift=phys["baseline_energy_drift"],
        hnn_energy_drift=phys["hnn_energy_drift"],
        baseline_field_mse=phys["baseline_field_mse"],
        hnn_field_mse=phys["hnn_field_mse"],
        trace=rnd(phys["trace"], 4),
        models=rnd(phys["models"], 6),
    )
    with open(os.path.join(OUT, "pendulum.json"), "w") as f:
        json.dump(pendulum, f)

    table, traces, models = [], {}, {}
    for r in nets:
        key = f"{r['dataset']}_{r['kind']}_L{r['L']}"
        table.append({k: r[k] for k in
                      ("dataset", "kind", "L", "acc_mean", "acc_std",
                       "acc_deep_mean", "energy_drift_mean", "rms_ratio_mean")})
        traces[key] = dict(energy=rnd(r["energy_trace_seed0"], 4),
                           rms=rnd(r["rms_trace_seed0"], 4))
        if r.get("model_seed0") and r["L"] == 16:   # live panels use the L=16 nets
            models[f"{r['dataset']}_{r['kind']}"] = dict(
                model=rnd(r["model_seed0"], 6),
                X=rnd(r["viz_data"]["X"], 4), y=r["viz_data"]["y"])
    with open(os.path.join(OUT, "depth.json"), "w") as f:
        json.dump(dict(table=rnd(table), traces=traces), f)
    with open(os.path.join(OUT, "models.json"), "w") as f:
        json.dump(models, f)

    for name in ("pendulum.json", "depth.json", "models.json"):
        kb = os.path.getsize(os.path.join(OUT, name)) / 1024
        print(f"  {name}: {kb:.0f} KB")


if __name__ == "__main__":
    main()
