"""The runs of the scaling study, in one place: what each trains on, for how many steps, what it costs, and the lines
that submit it. docs/budget.md says why the plan looks like this.

    python experiments/expanse/plan.py                                   # the table, at the rates measured on an MI300X
    python experiments/expanse/plan.py --rates results/expanse/throughput_h100.json --speedup 3.6
                                                                         # the same at the H100's measured rates, with
                                                                         # four GPUs 3.6 times as fast as one
    python experiments/expanse/plan.py --lr_check 500m                   # the short runs that pick a learning rate
    python experiments/expanse/plan.py --run exealm_500m_s0 --lr 6e-4    # the lines that submit one run

Every run takes steps of 262,144 tokens, on one GPU or four: --accum makes up the difference. A run on four GPUs that
is S times as fast as on one is charged 4 / S times the GPU-hours of the same run on one GPU.

--variant core is the fallback if the first job's `lead` part says the board and library slots cost the lead at
0.1B: ExeaLM without memory circuits, every row read alone.
"""
import argparse
import glob
import json
import os

SEQ = 512
STEP_TOKENS = 262_144
ALLOCATION = 2000                              # GPU-hours
TARGET = (9e9, 1e12)                           # what the study is evidence for: 9B parameters on 1T tokens
EVALUATION = 150                               # evaluation of every run, and the short skills phase of the ExeaLM runs
LR_STEPS = {"100m": 1000, "500m": 1000, "1b": 1000, "3b": 500}
LR_RATES = {"100m": 3, "500m": 3, "1b": 2, "3b": 2}          # how many learning rates the budget counts at each size

# Parameters in millions with the 32k tokenizer (tests/test_plan.py counts them from the code), rows per forward pass
# on one 80 GB GPU, and the GPUs a run uses. The 0.1B runs take one GPU each on the shared partition: four of them
# run side by side and none pays for the others' gradients. The rows are sized from two A40 runs (results/a40_check:
# 11.8 GB at 0.1B with 32 rows, 23.1 GB at 0.5B with 16), scaled by width x depth: about 40 GB at 0.5B and 1B, and
# about 59 GB at 3B, where 8 rows would come to about 71 GB. Estimates: throughput.py measures the real peaks.
SIZES = {
    "100m": dict(exealm=94.28, core=93.43, gated=95.47, batch=64, gpus=1, tokens=10.4e9, seeds=(0, 1)),
    "500m": dict(exealm=498.08, core=497.07, gated=510.14, batch=32, gpus=4, tokens=30e9, seeds=(0,)),
    "1b": dict(exealm=966.06, core=964.95, gated=989.81, batch=16, gpus=4, tokens=15e9, seeds=(0,)),
    "3b": dict(exealm=2902.88, core=2901.54, gated=2980.06, batch=4, gpus=4, tokens=4e9, seeds=(0,)),
}
# Effective TFLOP/s (6 x parameters x tokens per second) of one MI300X in bf16: results/funding/README.md, section 6.
# Measured for the memory-free hybrid and the plain Llama at 0.1B and 0.5B. Three assumptions until the first job on
# Expanse replaces them: ExeaLM with its memory slots runs at the hybrid's rate, the gated Llama at the plain Llama's,
# and 1B and 3B at the 0.5B rates.
MI300X = {"100m": dict(exealm=56.2, gated=85.1), "500m": dict(exealm=81.0, gated=124.8),
          "1b": dict(exealm=81.0, gated=124.8), "3b": dict(exealm=81.0, gated=124.8)}
SIZE_NAME = {"0.1B": "100m", "0.5B": "500m", "1B": "1b", "3B": "3b"}           # throughput.py's names for the presets


def runs():
    """Every training run of the plan: name, preset, model ("exealm" or "gated") and seed."""
    return [(f"{model}_{preset}_s{seed}", preset, model, seed)
            for preset, s in SIZES.items() for seed in s["seeds"] for model in ("exealm", "gated")]


def measured(rows):
    """Rates from the rows of a throughput_<gpu>.json: the last bf16 row without torch.compile or TF32 for each size
    and model. "exealm" and "gated" rows are the plan's own two models; without them, "hybrid" and "llama" stand in."""
    rates = {p: dict(r) for p, r in MI300X.items()}
    source = {p: dict(exealm="MI300X", gated="MI300X") for p in MI300X}
    for want, kinds in (("exealm", ("hybrid", "exealm")), ("gated", ("llama", "gated"))):
        for kind in kinds:                                                # the later kind is the closer match
            for r in rows:
                preset = SIZE_NAME.get(r.get("size"))
                if (preset and r.get("model") == kind and r.get("precision") == "bf16"
                        and r.get("effective_TFLOPs", 0) > 0 and not r.get("compile") and not r.get("tf32")):
                    rates[preset][want], source[preset][want] = r["effective_TFLOPs"], kind
    return rates, source


def gpu_hours(params_m, tokens, tflops, gpus=1, speedup=4.0):
    one = 6 * params_m * 1e6 * tokens / (tflops * 1e12 * 3600)
    return one * (gpus / speedup if gpus > 1 else 1)


def cost(preset, model, rates, a, tokens=None):
    s = SIZES[preset]
    params = s["core" if (model == "exealm" and a.variant == "core") else model]
    gpus = a.gpus or s["gpus"]
    return gpu_hours(params, s["tokens"] if tokens is None else tokens, rates[preset][model], gpus, a.speedup)


def lr_check_hours(rates, a):
    return sum(cost(p, m, rates, a, LR_STEPS[p] * STEP_TOKENS) * LR_RATES[p] for p in SIZES for m in ("exealm", "gated"))


def table(rates, source, a):
    print(f"Target: {TARGET[0] / 1e9:.0f}B parameters on {TARGET[1] / 1e12:.0f}T tokens, "
          f"{TARGET[1] / TARGET[0]:.0f} tokens per parameter.\n")
    print("| Run | Parameters | Tokens | Tokens per parameter | Steps | GPUs | GPU-hours |")
    print("|---|---|---|---|---|---|---|")
    total = 0.0
    for name, preset, model, seed in runs():
        s = SIZES[preset]
        params = s["core" if (model == "exealm" and a.variant == "core") else model]
        h = cost(preset, model, rates, a)
        total += h
        print(f"| `{name}` | {params:,.1f}M | {s['tokens'] / 1e9:g}B | {s['tokens'] / params / 1e6:.1f} | "
              f"{round(s['tokens'] / STEP_TOKENS):,} | {a.gpus or s['gpus']} | {h:,.0f} |")
    lr = lr_check_hours(rates, a)
    print(f"| **The ten runs** | | | | | | **{total:,.0f}** |")
    print(f"| Evaluation and the skills phase | | | | | | {EVALUATION} |")
    print(f"| Learning-rate checks | | | | | | {lr:,.0f} |")
    left = ALLOCATION - total - EVALUATION - lr
    print(f"| **Reserve of the {ALLOCATION:,}** | | | | | | **{left:,.0f}** |")
    if left < 0:
        print(f"\nOver the allocation by {-left:,.0f} GPU-hours. Shorten the largest runs. Do not drop a baseline.")
    used = sorted({f"{p} {m}: {source[p][m]}" for p in SIZES for m in ("exealm", "gated") if source[p][m] != "MI300X"})
    print("\nRates: " + ("; ".join(used) + "; the rest as on the MI300X" if used else "as measured on one MI300X")
          + (f". Four GPUs taken as {a.speedup:g} times one." if a.speedup != 4 else
             ". Four GPUs taken as 4 times one: pass --speedup with the measured figure."))


def pack_tokens(data):
    """Tokens in the packs a mix .json names, or None if they cannot be read from here."""
    try:
        total = 0
        for src in json.load(open(data))["sources"].values():
            paths = src["paths"] if isinstance(src["paths"], list) else [src["paths"]]
            for p in paths:
                for d in (glob.glob(p) if any(c in p for c in "*?") else [p]):
                    total += json.load(open(os.path.join(d, "meta.json")))["n_tokens"]
        return total
    except (OSError, KeyError, ValueError):
        return None


def command(run, preset, model, seed, a, steps=None, lr=None):
    s = SIZES[preset]
    gpus, batch = a.gpus or s["gpus"], a.batch or s["batch"]
    accum, rest = divmod(STEP_TOKENS, gpus * batch * SEQ)
    if rest or batch % 4:
        raise SystemExit(f"--batch {batch} on {gpus} GPU(s) does not divide a step of {STEP_TOKENS:,} tokens into "
                         f"whole swarms of 4")
    args = [f"--preset {preset}"]
    if model == "gated":
        args.append("--model llama --config experiments/expanse/gated_llama.json")
    else:
        args.append("--search 0" + (" --config experiments/expanse/exealm_core.json" if a.variant == "core" else ""))
    args.append(f"--data {a.data} --val {a.val} --seq {SEQ} --batch {batch} --accum {accum}")
    if steps:                                       # a learning-rate check: a tenth of it is warm-up
        args.append(f"--steps {steps} --warmup {steps // 10} --lr {lr:g} --seed {seed} --eval_every 100")
    else:
        args.append(f"--steps {round(s['tokens'] / STEP_TOKENS)} --lr {a.lr:g} --seed {seed} "
                    f"--eval_every {a.eval_every}")
    args.append(f"--eval_batches {512 // batch}")   # 512 rows of 512 tokens per evaluation at every size
    lines = [f"RUN={run} \\", 'ARGS="' + " ".join(args) + '" \\']
    flags = ""
    if gpus == 1:                                   # one GPU: the shared partition, and one task's worth of cores
        flags = "--ntasks-per-node=1 "
        lines.append(f'SBATCH_PARTITION=nairr-gpu-shared SBATCH_GPUS=h100:1 SBATCH_EXTRA="{flags.strip()}" \\')
    lines.append(f"sbatch {flags}experiments/expanse/train.sb")
    return "\n".join(lines)


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default=None, help="a throughput_<gpu>.json from experiments/throughput.py")
    ap.add_argument("--speedup", type=float, default=4.0, help="how many times as fast four GPUs are as one "
                    "(speedup_on_4_gpus in results/expanse/scaling_500m.json)")
    ap.add_argument("--variant", default="memory", choices=["memory", "core"],
                    help="ExeaLM with its board and library slots, in swarms (memory), or without them (core)")
    ap.add_argument("--run", default=None, help="print the lines that submit this run (needs --lr)")
    ap.add_argument("--lr_check", default=None, help="print the learning-rate check of this size: " + ", ".join(SIZES))
    ap.add_argument("--lr", type=float, default=None, help="the learning rate the check picked for this model and size")
    ap.add_argument("--lrs", default="3e-4,6e-4,1e-3", help="the learning rates --lr_check tries")
    ap.add_argument("--gpus", type=int, default=0, choices=[0, 1, 4], help="GPUs per run (default: 1 at 0.1B, else 4)")
    ap.add_argument("--batch", type=int, default=0, help="rows per forward pass on one GPU, if the default does not fit")
    ap.add_argument("--data", default="experiments/expanse/mix_tokpack.json")
    ap.add_argument("--val", default="data/tokpack/bpe32k/val_en")
    ap.add_argument("--eval_every", type=int, default=500)
    return ap


def defaults():
    return parser().parse_args([])


def main(argv=None):
    a = parser().parse_args(argv)
    rates, source = measured(json.load(open(a.rates)) if a.rates else [])
    by_name = {name: (preset, model, seed) for name, preset, model, seed in runs()}
    if a.lr_check:
        if a.lr_check not in SIZES:
            raise SystemExit(f"--lr_check takes one of {', '.join(SIZES)}")
        steps, lrs = LR_STEPS[a.lr_check], [float(x) for x in a.lrs.split(",")]
        each = sum(cost(a.lr_check, m, rates, a, steps * STEP_TOKENS) for m in ("exealm", "gated"))
        print(f"# {len(lrs)} learning rates x 2 models x {steps:,} steps at {a.lr_check}: "
              f"about {each * len(lrs):,.0f} GPU-hours. Each model keeps its own best rate.\n")
        for lr in lrs:
            for model in ("exealm", "gated"):
                print(command(f"lr_{model}_{a.lr_check}_{lr:g}", a.lr_check, model, 0, a, steps, lr) + "\n")
    elif a.run:
        if a.run not in by_name:
            raise SystemExit(f"no run called {a.run}. The runs: {', '.join(by_name)}")
        if a.lr is None:
            raise SystemExit("--run needs --lr: the rate the learning-rate check picked for this model at this size "
                             f"(python experiments/expanse/plan.py --lr_check {by_name[a.run][0]})")
        preset, model, seed = by_name[a.run]
        tokens, have = SIZES[preset]["tokens"], pack_tokens(a.data)
        if have is None:
            print(f"# Could not read the size of the packs in {a.data} from here. {a.run} reads {tokens / 1e9:g}B "
                  f"tokens:\n# they should hold at least a quarter of that.\n")
        elif tokens > 4 * have:
            print(f"# NOT READY: {a.run} reads {tokens / 1e9:g}B tokens and {a.data} holds {have / 1e9:.2f}B, so it "
                  f"would read them {tokens / have:.0f} times.\n# Pack the full data first and pass it with --data.\n")
        print(f"# about {cost(preset, model, rates, a):,.0f} GPU-hours")
        print(command(a.run, preset, model, seed, a))
    else:
        table(rates, source, a)


if __name__ == "__main__":
    main()
