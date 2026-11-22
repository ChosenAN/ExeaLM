"""The compute plan (experiments/expanse/plan.py): its sizes are the code's, its hours are the document's, and every
run it prints is one the training script can start."""
import contextlib
import io
import json
import os
import sys

import numpy as np
import pytest
import torch

from exealm import ExeaLM, Llama, count_params, get_config
from exealm.train import pretrain

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path[:0] = [os.path.join(ROOT, "experiments", "expanse"), os.path.join(ROOT, "experiments")]
import eval_text  # noqa: E402
import plan  # noqa: E402
import throughput  # noqa: E402


def printed(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        plan.main(argv)
    return out.getvalue()


def overrides(name):
    return json.load(open(os.path.join(ROOT, "experiments", "expanse", name)))


def test_the_plans_parameter_counts_are_the_codes():
    core, gated = overrides("exealm_core.json"), overrides("gated_llama.json")
    for preset, s in plan.SIZES.items():
        with torch.device("meta"):
            counts = dict(exealm=count_params(ExeaLM(get_config(preset, vocab_size=32768))),
                          core=count_params(ExeaLM(get_config(preset, vocab_size=32768, **core))),
                          gated=count_params(Llama(get_config(preset, vocab_size=32768, **gated))))
        for model, n in counts.items():
            assert round(n / 1e6, 2) == s[model], (preset, model, n)
    cfg = get_config("500m", **core)
    assert "RG" not in cfg.menu and "AG" not in cfg.menu and cfg.swarm_size == 1 and not cfg.pinned


def test_the_table_fits_the_allocation_and_is_the_one_in_the_budget_document():
    text = printed([])
    rows = [l for l in text.splitlines() if l.startswith("| `")]
    hours = {l.split("`")[1]: int(l.split("|")[-2].replace(",", "")) for l in rows}
    assert hours["exealm_500m_s0"] == 307 and hours["gated_500m_s0"] == 204        # 30B tokens at 81 and 124.8 TFLOP/s
    assert hours["exealm_3b_s0"] == 239 and hours["gated_3b_s0"] == 159
    assert len(hours) == 10 and all(f"gated_{n[7:]}" in hours for n in hours if n.startswith("exealm_"))
    reserve = int(text.split("**Reserve of the 2,000** | | | | | | **")[1].split("**")[0].replace(",", ""))
    assert 200 < reserve < 400
    budget = open(os.path.join(ROOT, "docs", "budget.md")).read()
    for line in text.splitlines():
        if line.startswith("|"):
            assert line in budget, f"docs/budget.md is missing this line of plan.py's table:\n{line}"


def test_measured_rates_and_the_four_gpu_speed_up_rescale_the_plan(tmp_path):
    rows = [dict(size="0.5B", model="hybrid", precision="bf16", effective_TFLOPs=100.0, compile=0, tf32=0),
            dict(size="0.5B", model="exealm", precision="bf16", effective_TFLOPs=162.0, compile=0, tf32=0),
            dict(size="0.5B", model="exealm", precision="bf16", effective_TFLOPs=500.0, compile=1, tf32=0),
            dict(size="0.5B", model="llama", precision="fp32", effective_TFLOPs=999.0, compile=0, tf32=0),
            dict(size="1B", model="exealm", precision="bf16", effective_TFLOPs=0.0, compile=0, tf32=0),
            dict(size="3B", model="exealm", precision="bf16", error="out of memory at batch 1")]
    path = tmp_path / "throughput_h100.json"
    path.write_text(json.dumps(rows))
    rates, source = plan.measured(rows)
    assert rates["500m"] == dict(exealm=162.0, gated=124.8) and source["500m"] == dict(exealm="exealm", gated="MI300X")
    assert rates["1b"] == plan.MI300X["1b"]                               # nothing measured: the MI300X figure stays
    text = printed(["--rates", str(path), "--speedup", "3.2"])
    row = next(l for l in text.splitlines() if l.startswith("| `exealm_500m_s0`"))
    assert int(row.split("|")[-2]) == round(307.5 / 2 * 4 / 3.2)           # twice the rate, 4 / 3.2 the charge
    one = next(l for l in text.splitlines() if l.startswith("| `exealm_100m_s0`"))
    assert int(one.split("|")[-2]) == 29                                  # a one-GPU run pays nothing for the others


def test_every_run_is_a_whole_number_of_swarms_and_passes_per_step():
    for name, preset, model, seed in plan.runs():
        text = printed(["--run", name, "--lr", "1e-3"])
        args = text.split('ARGS="')[1].split('"')[0].split()
        get = lambda k: args[args.index(k) + 1]                           # noqa: E731
        gpus = 1 if "nairr-gpu-shared" in text else 4
        assert gpus == plan.SIZES[preset]["gpus"]
        assert int(get("--batch")) * int(get("--accum")) * gpus * int(get("--seq")) == plan.STEP_TOKENS
        assert int(get("--batch")) % 4 == 0 and int(get("--batch")) * int(get("--eval_batches")) == 512
        assert int(get("--steps")) * plan.STEP_TOKENS == pytest.approx(plan.SIZES[preset]["tokens"], rel=1e-4)
        assert get("--seed") == str(seed) and get("--preset") == preset
        if model == "gated":
            assert get("--model") == "llama" and os.path.exists(os.path.join(ROOT, get("--config")))
        else:
            assert get("--search") == "0" and "--config" not in args
        assert ("--ntasks-per-node=1" in text) == (gpus == 1)
    core = printed(["--run", "exealm_1b_s0", "--lr", "1e-3", "--variant", "core"])
    assert "--config experiments/expanse/exealm_core.json" in core
    with pytest.raises(SystemExit, match="needs --lr"):
        plan.main(["--run", "exealm_1b_s0"])
    with pytest.raises(SystemExit, match="does not divide"):
        plan.main(["--run", "exealm_3b_s0", "--lr", "1e-3", "--batch", "12"])
    check = printed(["--lr_check", "1b", "--lrs", "3e-4,6e-4"])
    assert check.count("sbatch ") == 4 and "--steps 1000 --warmup 100" in check


def test_a_run_too_long_for_its_data_is_flagged(tmp_path):
    pack = tmp_path / "en"
    pack.mkdir()
    (pack / "meta.json").write_text(json.dumps(dict(n_tokens=400_000_000)))
    mix = tmp_path / "mix.json"
    mix.write_text(json.dumps({"sources": {"en": {"paths": str(pack), "weight": 1.0}}}))
    assert plan.pack_tokens(str(mix)) == 400_000_000
    assert "NOT READY" in printed(["--run", "exealm_500m_s0", "--lr", "1e-3", "--data", str(mix)])
    (pack / "meta.json").write_text(json.dumps(dict(n_tokens=8_000_000_000)))
    assert "NOT READY" not in printed(["--run", "exealm_500m_s0", "--lr", "1e-3", "--data", str(mix)])


def test_throughput_times_the_plans_two_models():
    dev = torch.device("cpu")
    built = {k: throughput.build("tiny", k, 256, dev) for k in throughput.KINDS}
    n = {k: count_params(m) for k, (m, _) in built.items()}
    assert n["gated"] > n["llama"]                                        # the gate and QK-norm are parameters
    assert built["exealm"][1].swarm_size == get_config("tiny").swarm_size and built["hybrid"][1].swarm_size == 1
    kinds = lambda m: {s["tmpl"] for lay in m.specs for s in lay}          # noqa: E731
    assert {"RG", "AG"} <= kinds(built["exealm"][0]) and not {"RG", "AG"} & kinds(built["hybrid"][0])
    with pytest.raises(SystemExit):
        throughput.build("tiny", "qwen", 256, dev)


def pack(path, n_tokens=6000, vocab=256, seed=0):
    path.mkdir(parents=True)
    np.random.default_rng(seed).integers(0, vocab, n_tokens).astype(np.uint16).tofile(path / "tokens.bin")
    offsets = np.arange(0, n_tokens, 100, dtype=np.int64)
    np.save(path / "offsets.npy", offsets)
    (path / "meta.json").write_text(json.dumps(dict(dtype="uint16", vocab_size=vocab, n_tokens=n_tokens,
                                                    n_docs=len(offsets), bytes_per_token=2.0)))
    return str(path)


def test_eval_text_scores_swarm_trained_and_plain_models_alike_and_groups_seeds(tmp_path, capsys):
    train = pack(tmp_path / "data" / "train")
    for i, name in enumerate(("val_a", "val_b")):
        pack(tmp_path / "data" / name, seed=i + 1)
    common = ["--preset", "tiny", "--data", train, "--seq", "16", "--batch", "8", "--steps", "4", "--eval_every", "4",
              "--search", "0", "--device", "cpu"]
    runs = []
    for name, extra in (("gated_s0", ["--model", "llama", "--config", os.path.join(ROOT, "experiments", "expanse",
                                                                                    "gated_llama.json")]),
                        ("exealm_s0", []), ("exealm_s1", ["--seed", "1"])):
        runs.append(str(tmp_path / name))
        pretrain.main(common + extra + ["--out", runs[-1]])
    capsys.readouterr()
    out = tmp_path / "text.json"
    rows, means = eval_text.main(runs + ["--root", str(tmp_path / "data"), "--sets", "val_a,val_b", "--batch", "4",
                                         "--seq", "16", "--eval_batches", "2", "--device", "cpu", "--out", str(out)])
    assert [r["model"] for r in rows] == ["llama", "tonn", "tonn"] and all(r["step"] == 4 for r in rows)
    assert set(rows[0]["bits_per_byte"]) == {"a", "b"}
    assert all(3.5 < r["mean"] < 4.5 for r in rows)                       # random tokens: 8 bits each, 2 bytes each
    assert list(means) == ["gated", "exealm"]
    assert means["exealm"] == pytest.approx((rows[1]["mean"] + rows[2]["mean"]) / 2, abs=1e-4)
    assert json.load(open(out))["means"] == means
    again, _ = eval_text.main(runs[:1] + ["--root", str(tmp_path / "data"), "--sets", "val_a,val_b", "--batch", "4",
                                          "--seq", "16", "--eval_batches", "2", "--device", "cpu"])
    assert again[0]["bits_per_byte"] == rows[0]["bits_per_byte"]          # the same windows every time
