# Release notes — LFR code release

Companion release for the paper *Locally Fitted References for Training-Free
Industrial Anomaly Localization*. It is **code only**: no weights, no datasets,
no built banks, no run outputs. It is nonetheless complete in the sense that
matters — from this archive, plus the public inputs listed below, every number in
the main table can be regenerated.

Release date: 2026-10-10.

---

## 1. What is in the archive

| group | count | where |
|---|---|---|
| runtime modules | 12 | repository root |
| equivalence harnesses | 4 | repository root (`test_*.py`, `bench_*.py`) |
| table generators and audits | 11 | `tools/` |
| portable shell drivers | 8 | `scripts/` |
| measured reference data | 1 | `refs/` |
| figure and table sources | 2 | `figures/`, `docs/` |
| packaging files | 4 | `README.md`, `requirements.txt`, `.gitignore`, this file |

## 2. What is deliberately absent, and why

| absent | reason |
|---|---|
| DINOv3 ViT-B/16 checkpoint | gated on Hugging Face; we do not hold redistribution rights. `README.md` §3.2 gives the access request and the expected filename. |
| MVTec-AD, VisA, Real-IAD | each carries a non-commercial licence and the three together are ~800 GB. `README.md` §3.3 gives the download and the exact directory tree each loader expects. |
| built normal-memory banks | 33 GiB / 78 GiB / 352 GB. They are derived artefacts: rebuilding them is the first step of §4, takes ~20 min / ~45 min / ~40 min, and is deterministic given the same encoder. Shipping them would also make the "same memory for every arm" claim unverifiable by inspection. |
| `saved_curvot/`, `exp/tables/`, `exp/results_*.txt` | run outputs, ~60 h of GPU time, and `.gitignore` excludes them by design. §4 lists the exact paths and the expected contents, so a reproduction can be diffed field by field instead of eyeballed. |

A note on the empty directories: `data/`, `models/`, `bank/`, `exp/` and
`saved_curvot/` are present with a `.gitkeep` so that the default paths in
`scripts/common.sh` exist on a fresh clone. They hold nothing.

## 3. Code identity — the claim, and how to check it

The release claims that *the released runtime code is the code that produced the
reported numbers*. That claim is only worth its verifiability, so it is made
mechanically:

```bash
cd LFR
md5sum -c docs/md5sums.txt          # every listed file, one command
```

`docs/CODE_FINGERPRINTS.txt` is the human-readable form of the same thing, and
adds a status per file. **24 of the 40 listed files are `FROZEN`**: their md5 is
simultaneously the hash of the shipped file and the hash recorded on the training
box before this release was assembled. The remaining 16 are not runtime code —
they are the new portable drivers (`DRIVER`), packaging files (`GLUE`), the
figure and table sources (`ASSET`), the transcribed reference column (`DATA`),
one packaging edit (`EDIT`), and two harnesses that existed only on the
development machine (`HARNESS`, see §8).

Exactly one file was edited for packaging, and the edit is marked in the source
with a `PACKAGING EDIT` banner:

* `tools/dump_defaults.py` — carried the training box's absolute path
  (`/home/amax/project/rad/RAD-main/curvot_rad.py`). It now resolves the source
  relative to the repository, overridable with `--src` / `--out`. Its md5 before
  the edit is recorded at the end of `CODE_FINGERPRINTS.txt`. The edit changes
  where the tool reads and writes, nothing about what it computes, and its output
  was verified identical to the training-box dump (105 flags).

## 4. Terminology: the code says `lonm`

The paper's method is called **LFR** and its readout **PNR**. The code was
written under the working name *CurvOT-RAD* with the operator called *LONM*, so
the flags, the helper names and the entry-point filename keep the older spelling.
This is deliberate: the flag names are recorded inside the `args` block of every
`metrics.json`, and every command line in the paper's reproducibility notes uses
them. Renaming would silently invalidate both. The mapping is tabulated in
`README.md` §2.1.

The same section notes that `curvot_rad.py` still exposes switches from an
earlier research line (`--use_aptr`, partial-transport re-ranking). They default
to `0`, no script uses them, and no reported number depends on them.

## 5. Environment expectations

`requirements.txt` pins the versions the reported runs actually imported
(Python 3.11.7, PyTorch 2.4.1, torchvision 0.19.1, NumPy 1.26.4, …). Note that
**`dinov3` is not on PyPI** — it is the Meta encoder package, and `README.md`
§3.2 covers obtaining it and setting `PYTHONPATH`.

Two environment facts are load-bearing and easy to trip over:

* **Real-IAD bank loading.** A 352 GB checkpoint cannot be mapped privately on a
  host with `vm.overcommit_memory=0`; a plain `python curvot_rad.py` fails with
  `mmap ENOMEM` and then dies at ~125 GB resident. The `MAP_SHARED` read-only
  launcher `ri_launch_curvot_rad.py` is required, and
  `scripts/run_realiad_rho4.sh` already uses it.
* **GPU memory.** With `CACHE_MB=14000` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, peak observed allocation
  was 18.84 GiB on a 24 GiB card. Raising the cache to fit MVTec-AD's largest
  class was measured at 23.27 GiB, i.e. over budget — the shipped defaults are
  the working ones.

The drivers are `bash`. On Windows they need WSL or Git Bash; `zip` does not
preserve the executable bit, so `chmod +x scripts/*.sh` after unzipping on Unix
(or use the `.tar.gz`, which does).

## 6. Numbers this release does not claim

These are settled results and are stated here so that the release is not read as
claiming more than was measured.

* **Image-level superiority is not claimed, in either direction.** The advantage
  is a localization (pixel-level) effect. On MVTec-AD the image-AP margin against
  ANoCo is 0.19 pp and half the categories tie below the 0.005 pp print
  resolution; on VisA the image-level difference does not have a consistent sign.
* **Real-IAD image level is where LFR does not improve.** Image AUROC / F1 are
  flat against RAD on the same copy and image-AP is slightly lower; the gain is
  pixel-level (+0.84 pp, 24 of 30 categories).
* **The Real-IAD table prints three columns, not two.** The reproduction gap
  against the published RAD numbers is larger than LFR's gain, so the table shows
  our RAD rerun beside the paper's RAD column rather than claiming to beat the
  paper. The gap is confined to pixel-AP / pixel-F1 and is attributed to a
  downscaled local copy of the images (see `refs/rad_realiad_ab.json`, which is
  reference data, transcribed and cited — never mixed into our own columns).
* **The radius is a sensitivity bound, not an accuracy knob.** Nine β values move
  pixel-AP by 0.09 pp in total with no interior optimum; the unbounded arm is the
  maximum. Its role is to bound the projection error, not to be tuned.
* **AUPRO is not reproducible to four decimals** across implementations, because
  it is a quantile-averaged curve and a 1e-7 score change can flip a threshold
  rank. Image-level metrics and pixel AUROC / AP / F1 are stable to four
  decimals. The README prints AUPRO to two decimals for that reason.
* **The measured noise floor is 0.16 pp.** Paired MVTec seed sd is 0.08 pp, so
  differences below 2 sd should not be read as effects.

## 7. To settle before this repository goes public

1. **Licence — this is the one real blocker, and it is not a formality.** No
   licence is granted by this release, because part of the tree is derived from
   the upstream RAD codebase (<https://github.com/longkukuhi/RAD>, *Is
   Task-Specific Training Necessary for Anomaly Detection?*, ICML 2026) and that
   repository currently ships **without a licence file** — GitHub reports
   `"license": null` for it, and both `LICENSE` and `master:LICENSE` return 404
   even though its README asserts a licence exists. Absent a licence, the
   default is all rights reserved, so we cannot unilaterally place MIT or
   Apache-2.0 over the derived files.

   Measured overlap with upstream (non-comment lines, `difflib`):

   | shipped file | upstream | ours | shared | similarity |
   |---|---|---|---|---|
   | `utils.py` | 780 | 780 | 779 | **99.9 %** — effectively verbatim |
   | `dataset.py` | 453 | 557 | 453 | 89.7 % — upstream plus our additions |
   | `build_bank_multilayer.py` | 156 | 225 | 140 | 73.5 % |
   | `rad_mvtec_visa_3dadam.py` | 425 | 790 | 358 | 58.9 % — heavily rewritten |

   The remaining modules (`curvot_rad.py`, `curvot_utils.py`, `cu_ref.py`,
   `baseline_scores.py`, `bank_builder_disk.py`, `build_realiad_bank_multilayer.py`,
   `ri_launch_curvot_rad.py`, `ri_sharedbank.py`, the tests, the tools and the
   drivers) are ours.

   Options, in order of preference:

   * **Ask the RAD authors.** Open an issue on their repository and/or email the
     corresponding author requesting a licence (or written permission to
     redistribute the derived files). This is the normal path, it costs one
     message, and it fixes the problem at the root instead of working around it.
   * **Patch-only release.** Ship our modules plus a `git apply`-able diff
     against upstream, and have users clone upstream themselves. This removes
     the redistribution question entirely, at the cost of a two-step install.
   * **Reviewer-only archive.** If the venue allows, distribute to reviewers
     privately during review and defer the public repository until the upstream
     licence is settled. Since the paper is under review anyway, this costs
     nothing.
   * Shipping the tree publicly under a permissive licence *without* permission
     is not among the options: `utils.py` at 99.9 % upstream is not defensible
     as an independent work.

   Do not add a `LICENSE` file over the whole tree before this is resolved.
2. **Author list, affiliation and corresponding author.** The citation block in
   `README.md` §9 uses placeholders.
3. **Anonymity.** If the venue is running a double-blind round, do not publish
   author names, a personal repository URL or the acknowledgements before
   acceptance. The placeholders conveniently make this the default state.
4. **Repository URL and release tag.** `README.md` currently has no self-URL; add
   one once the remote exists, plus a `CITATION.cff` if the venue wants it.
5. **Whether to ship `refs/rad_realiad_ab.json`.** It contains our transcription
   of the published RAD per-class numbers, with its provenance recorded inside
   the file. It is reference data, not a measurement of ours. Publishing it is
   what makes the three-column table auditable, so the recommendation is to keep
   it — but it is a judgement call about redistributing another paper's table.
6. **Scope question: the PaDiM-on-MVTec numbers.** They are on disk (`bd_pad_m`)
   but were left out of the agreed panel, which is SubspaceAD + ANoCo on both
   datasets. `baseline_scores.py` implements PaDiM and PatchCore, so the runs are
   possible; shipping the *numbers* would change the paper's comparison table, so
   it was not done silently. Decide whether they belong.
7. **Issues/contact policy.** If the repository will not be actively maintained,
   say so in the README — it is kinder than silence.

## 8. Known gaps in this packaging

Stated plainly, because they are the things most likely to cost someone an hour:

* **Nothing was smoke-tested end-to-end.** The release was assembled on a machine
  with no GPU, no datasets and no bank files, so the drivers were built from the
  recorded command lines and checked by reading, not by running. The first thing
  to do on a fresh clone is run §4.7's tests, which need no data:
  `python test_unbounded_equiv.py`, `python test_baselines.py`. The β-sweep
  analyser has its own self-test (`python tools/test_beta_table.py`).
* **Two harnesses have no historical hash.** `test_window_fast.py` and
  `bench_window_fast.py` existed only on the development machine and are not on
  the training box, so there is no md5 to compare against. They are marked
  `HARNESS` rather than `FROZEN` for that reason, even though they exercise the
  frozen `curvot_utils.py` shipped here. Verified results from those harnesses
  are quoted in `README.md` §7.
* **The Real-IAD protocol value differs from the upstream README.** Verified
  against the upstream README on 2026-10-10: it passes `--pos_radius 0` for
  Real-IAD (`1` for MVTec-AD, `2` for VisA, `k_image 900` throughout). The value
  that reproduces is ρ = 4. The measurements behind that statement are in the
  header of `scripts/run_realiad_rho4.sh`, and `README.md` §4.2 points at it
  rather than restating the claim.
* **Runtime figures are single-machine observations**, one RTX 4090, not
  benchmarks. The Real-IAD wall times in particular (≈32 h / ≈20 h) come from an
  unloaded machine; a shared host has been measured to slow these runs by more
  than 2×.
