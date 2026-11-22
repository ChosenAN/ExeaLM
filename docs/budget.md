# Compute budget

What the planned runs cost, from measured throughput where we have it and stated assumptions where we don't.
Update this page when a GPU benchmark replaces an assumption.

## The main run: 0.1B parameters on 100B tokens

Training compute is about 6 × parameters × tokens = 6 × 0.1e9 × 100e9 = **6 × 10¹⁹ FLOPs**. Retrieval on adds
about half again (measured on the Mac: the 2:1 hybrid costs 1.01× a same-size Llama per token with the memory circuits
idle, 1.49× with retrieval on).

| Where | Throughput | Time for 100B tokens | Source |
|---|---|---|---|
| Mac Studio (M3 Ultra, MPS, fp32) | ~12,000 tokens/s, retrieval off; ~8,000 on | ~97 days; ~144 days | measured, `results/speed_hybrid21/` |
| One H100 or MI300X at ~300 TFLOPs sustained | ~500,000 tokens/s | ~56 GPU-hours; ~85 with retrieval | assumed, DATA.md |
| 8 of the same | | ~7 hours; ~11 hours | |

The 300 TFLOPs figure is an assumption. Attention runs as fused kernels on NVIDIA and AMD GPUs and TONN's circuits do
not yet, so real throughput could be up to 2× lower. **A one-hour benchmark on the 0.1B preset should come first**
(about $3–4 on one GPU).

## The scaling study: 2,000 GPU-hours on SDSC Expanse

The NAIRR Pilot approved 2,000 GPU-hours for ExeaLM on the SDSC Expanse AI Resource (nodes of four NVIDIA H100s,
80 GB each). The proposal planned ExeaLM at 0.5B, 1B and 3B with a plain Llama at 0.5B and 1B. On 18 November 2026 we
changed the plan, before any of it had run. This section is the changed plan. The proposal's table is at its end.

### What the runs are for

They are evidence for one later run: ExeaLM at 9B parameters on 1T tokens or more. At the rates measured so far that
run costs about 185,000 GPU-hours, against about 120,000 for a Llama of the same size. The allocation is 1% of it.
The study cannot prove that run. It can show whether ExeaLM's lead holds under the two things that change on the way
there: size and training length.

The proposal's plan left four things open.

- **Training length.** The target is 111 tokens per parameter. The lead we have at 0.1B (1.413 against 1.446 bits
  per byte for a gated Llama, 3 seeds, [results/funding](../results/funding/README.md) section 3) was measured at 0.5
  tokens per parameter. The proposal's runs reached 60, 21 and 2.8: the larger the model, the less trained. A lead
  between architectures often shrinks as training gets longer. Nothing we have measures ours past 0.5.
- **No baseline at 3B.** A run without its baseline shows speed and stability. It cannot show a lead.
- **The baseline.** The proposal used a plain Llama. A Llama with gated attention and QK-norm is the stronger model
  at 0.1B (1.446 against 1.509) and the same-parameter control, since ExeaLM's attention layers have both parts.
- **Cost.** ExeaLM takes about 1.5 times the GPU-hours of that baseline for the same tokens. At 9B on 1T tokens the
  difference is 65,000 GPU-hours, so the lead has to hold at equal GPU-hours, not only at equal tokens.

### The runs

Every ExeaLM run has a gated Llama of the same shape beside it, on the same tokens. A new pair at 0.1B trains to the
target's tokens per parameter, on two seeds. The 1B pair has fewer tokens than the proposal gave it. The 3B run
gave half its tokens for a baseline.

| Run | Parameters | Tokens | Tokens per parameter | Steps | GPUs | GPU-hours |
|---|---|---|---|---|---|---|
| `exealm_100m_s0` | 94.3M | 10.4B | 110.3 | 39,673 | 1 | 29 |
| `gated_100m_s0` | 95.5M | 10.4B | 108.9 | 39,673 | 1 | 19 |
| `exealm_100m_s1` | 94.3M | 10.4B | 110.3 | 39,673 | 1 | 29 |
| `gated_100m_s1` | 95.5M | 10.4B | 108.9 | 39,673 | 1 | 19 |
| `exealm_500m_s0` | 498.1M | 30B | 60.2 | 114,441 | 4 | 307 |
| `gated_500m_s0` | 510.1M | 30B | 58.8 | 114,441 | 4 | 204 |
| `exealm_1b_s0` | 966.1M | 15B | 15.5 | 57,220 | 4 | 298 |
| `gated_1b_s0` | 989.8M | 15B | 15.2 | 57,220 | 4 | 198 |
| `exealm_3b_s0` | 2,902.9M | 4B | 1.4 | 15,259 | 4 | 239 |
| `gated_3b_s0` | 2,980.1M | 4B | 1.3 | 15,259 | 4 | 159 |
| **The ten runs** | | | | | | **1,503** |
| Evaluation and the skills phase | | | | | | 150 |
| Learning-rate checks | | | | | | 61 |
| **Reserve of the 2,000** | | | | | | **286** |

`python experiments/expanse/plan.py` prints this table and the lines that submit each run
([experiments/expanse](../experiments/expanse/README.md)). The numbers behind it:

- **Rates.** One AMD MI300X in bf16 ([results/funding](../results/funding/README.md), section 6): 56 effective
  TFLOP/s for ExeaLM and 85 for a Llama at 0.1B, 81 and 125 at 0.5B.
  GPU-hours = 6 × parameters × tokens / (TFLOP/s × 3600 × 10¹²). Three assumptions stand in until the first job on
  Expanse measures them: ExeaLM with its memory slots runs at the rate of the memory-free hybrid, the gated Llama at
  the plain Llama's, and 1B and 3B at the 0.5B rates.
- **Four GPUs.** A whole node is charged 4 GPU-hours per hour. A run on four GPUs that is S times as fast as on one
  costs 4 / S times the hours above. `plan.py --rates results/expanse/throughput_h100.json --speedup S` prints the
  table at the H100's rates. If S is well under 4, one GPU per run on the shared partition is the cheaper way to
  train (`--gpus 1`): each run takes four times as long. Several can run side by side.
- **One GPU at 0.1B.** The four 0.1B runs take one GPU each, so none of them pays for the others' gradients.
- **Steps.** Every run takes steps of 262,144 tokens and evaluates every 500 steps on 512 held-out rows.
- **Evaluation and the skills phase** (150): every finished model on the four held-out sets, and a short second
  phase that teaches each ExeaLM the relay and the lookup.
- **Learning-rate checks.** Both models at every size get a short run at each of several learning rates (1,000
  steps; 500 at 3B) and keep their own best. Every number we have is at 1e-3 with steps of 16,384 tokens for 3,000
  steps. None of it says what rate a 0.5B model wants at steps sixteen times as large, or a 3B model at all. A short
  run can rule out a rate that diverges or lags badly. It cannot find the best rate for a run 30 to 100 times as
  long.

### Which ExeaLM

- The hybrid presets as they are (`100m`, `500m`, `1b`, `3b`): a TONN layer every third layer, gated attention with
  QK-norm between, no rotary attention in the starting circuits.
- **The circuit search is off** (`--search 0`). At 13M it gained 0.001 and 0.002 bits per byte on the two seeds it
  ran on and replaced one circuit in a whole run. On four GPUs, three wait during each round.
- **The board and library slots stay in the model**, one of each per TONN layer. The rows train in swarms of four
  that share a board. There is no datastore during the text run: retrieval for the whole run costs about 1.5 times
  per token (measured on the Mac), which this budget does not cover. The skills phase attaches it.
- **No weight-alter slots.** On text one costs 0.003 to 0.005 bits per byte at 13M. It stays a side experiment.
- Not the recipe that teaches all three skills with the text from the first step. It trained in 2 of 5 runs at 0.1B
  ([results/funding](../results/funding/README.md), section 5).

The 0.1B lead was measured on the hybrid with no memory slots at all. Whether it survives the slots is not known at
that size. The first job on Expanse checks it (its `lead` part: three seeds of 3,000 steps for ExeaLM as above, for
the memory-free hybrid and for the gated Llama). If ExeaLM with the slots is ahead of the gated Llama on every seed
and no more than 0.01 bits per byte behind the memory-free hybrid, the long runs train it. Otherwise they train the
memory-free hybrid (`plan.py --variant core`) and the skills become a separate study. The 0.01 is our judgment: it
is a third of the lead and twice the spread between ExeaLM's seeds.

### Before the long runs

1. **The first job** ([experiments/expanse](../experiments/expanse/README.md)): correctness on the H100, the rates
   of both models at all four sizes, the speed-up of four GPUs, and the lead check above. About 6 GPU-hours.
2. **The data.** The packs we have hold 413M tokens. The 0.5B pair reads 30B. Read 73 times over, a 0.5B model
   memorizes them and the comparison means nothing. [DATA.md](../DATA.md) allows a source to repeat about 4 times, so
   the study needs at least 7.5B packed tokens and is cleaner with 30B. Nothing long starts before that.
   `plan.py --run` marks a run NOT READY when its packs are too small.
3. **The learning-rate checks**, size by size, each before its pair.

Then the pairs, smallest first. The 0.1B pair is the cheapest and answers the question nothing else does.

### Reading the results

- **Equal tokens.** `python experiments/eval_text.py runs/gated_500m_s0 runs/exealm_500m_s0`: bits per byte on the
  four held-out sets, the same windows for both, every row read alone. It is the number of the 0.1B comparison.
- **Equal GPU-hours.** Each run's `log.jsonl` has the tokens seen and the validation loss every 500 steps. From the
  end of the warm-up to 80% of a run the learning rate is constant, so the two curves can be compared at the same
  GPU-hours there.
  Use `val_bits_per_token_board_cut` for ExeaLM (every row alone) and `val_bits_per_token` for the Llama.
- **Across sizes.** The same logs give every pair's gap at any earlier tokens per parameter: all four sizes at 1.3,
  three at 15, two at 59.

What would count as confidence in the 9B run: the gap at matched tokens per parameter does not shrink from 0.1B to
3B, and at 0.1B and 0.5B it is still there at 60 to 110 tokens per parameter and at equal GPU-hours. Even then it is
evidence, not proof. 9B is three times the largest model here. One seed per pair above 0.1B cannot tell a small lead from
noise: the spread between seeds at 0.1B is 0.005 bits per byte for ExeaLM and 0.003 for the gated Llama.

### If fewer hours fit

Shorten the largest runs. Do not drop a baseline.

1. The 1B pair from 15B to 10B tokens (saves about 165 GPU-hours).
2. The 3B pair from 4B to 3B tokens (about 100).
3. The second seed at 0.1B (about 49).

### The plan in the proposal

For the record, at the same MI300X rates. Nothing of it ran.

| Run | Width, layers | Training tokens | GPU-hours |
|---|---|---|---|
| ExeaLM 0.5B (`500m`) | 1280, 22 | 30B | 307 |
| Baseline 0.5B (plain Llama) | 1280, 22 | 30B | 190 |
| ExeaLM 1B (`1b`) | 1664, 26 | 20B | 397 |
| Baseline 1B (plain Llama) | 1664, 26 | 20B | 248 |
| ExeaLM 3B (`3b`) | 2560, 34 | 8B | 477 |
| Evaluation and short post-training of all five | | | 150 |
| Library builds, restarts and reserve | | | 231 |
| **Total** | | | **2,000** |

## What it costs to rent

RunPod on-demand list prices on 2026-11-11, per GPU-hour:

| GPU | Price | GPU-hours for the main run (retrieval on) | Cost |
|---|---|---|---|
| MI300X | $2.39 | ~85 | ~$200 |
| H100 SXM | $2.69 community, $3.99 secure | ~85 | ~$230–340 |
| A100 80GB | $1.39–1.79 | ~250 | ~$350–450 |
| RTX 4090 (8, community) | $0.34 | ~500 | ~$170, about 2.6 days on 8 |

Add 30–50% for restarts, evals and data loading. If the GPU benchmark comes in 2× slower, double these.

## The other runs

- **Data-mix pilots** (DATA.md: a 0.1B model on 20B tokens, three or four mixes): each about a fifth of the main run,
  so ~$40–80 each, ~$100–300 in all.
- **Experiments at `small` size (17M)** run on the Mac Studio: about 40,000–75,000 tokens/s, so 400M tokens in 2–3 hours.
- **Post-training** (SFT on 1–2B tokens, then preference tuning and RL): small next to pretraining, roughly 1–2% of
  its compute for SFT.

## Checkpoints

- Save every 2,000 steps and at the end (`--save_every`); keep the last three and every tenth.
- Exact resume works (`--resume auto`, PR #1), so community or spot GPUs that can be interrupted are usable.
