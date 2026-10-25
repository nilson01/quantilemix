# Scenario-5 Gaussian implementation pilot

This is the 90-case pilot for the reviewed
[scenario-5 protocol](PAPER_SCENARIO5_PROTOCOL.md), **not** a test of the
paper's statistical finding. The primary `ncor=2` translation and the
two-correlated-column `ncor=1` sensitivity each have five datasets in all
nine coefficient/correlation cells. All 90 Python fits succeeded with zero
captured warnings. An independent QR least-squares checker regenerated all
90 inputs and verified every effect, variance, interval and p-value; its
largest absolute difference was `9.55e-15`. Four deliberate corruptions
(input hash, effect, failed count and duplicate identity) were rejected.

The fitting loop took 0.079 seconds; the complete command, including Python
startup, took 0.64 seconds on the local M3 Pro with one numerical thread.
Maximum resident set size was 100,761,600 bytes. The saved result was 63,637
bytes and its flushed case journal was 33,738 bytes. These figures are a
resource pilot, not a server runtime promise. Scaling the entire 0.64-second
command by 200 gives 128 seconds for 18,000 cases; allow at least threefold
headroom for different hardware, file sync and independent checks. The planned server request was one CPU, 4 GiB, 30 minutes and a
100-MiB output ceiling.

The pilot also confirms why the generator ambiguity matters. At requested
ρ=0.75, mean realized X1–X2 correlation is about 0.74–0.76 in both variants.
Mean X1–X3 correlation is about 0.75 with `ncor=2` and near zero with
`ncor=1`. Five datasets per cell are too few to assess bias or coverage.

For one saved input in each of the 18 variant/cell combinations, both
isolated installed R qgcomp 1.3.0 and 2.19.6 agree with Python effects and
variances. The largest Python/R difference is `3.93e-15`. All 36 saved R
values are equal across versions at recorded precision. The optional R
checks record package MD5 values, dependencies and case input hashes. These
are same-input implementation checks, separate from the paper's Figure 1
and from the planned repeated-dataset study. R is not a Python runtime
dependency.

Private retained evidence is under
`validation-local/scenario5-20261023/`. The retained v2 run checks the
revised provenance and correlation-spread code; generated inputs and fit
values are unchanged from v1. SHA-256 values are:

| File | SHA-256 |
| --- | --- |
| `pilot-v2.json` | `bb491f59b77ddcf4b14fb2c1fd24010738c8a5013a65ddd130a50799555ce8a4` |
| `pilot-v2.json.jsonl` | `93e660bfa8ed7bc59ea6534e64a6b473694fb366104af4de8b5f3a307dc14716` |
| `pilot-v2-check.json` | `886f395567d780961faea195da553dfd5492d0a239aa73fc5f17eb5cbf811bef` |
| `r13-pilot-v2.json` | `ce82d4e84cad2fc19c74aaf2c4662f3cc40048b10ae99c916c4427114b776cf5` |
| `r219-pilot-v2.json` | `8a34c0b48c788fd86c2f57d9a9c03c294f4545713bc1b13f5e91db097b6d974c` |

The pilot used an uncommitted development checkout, which the record marks
dirty. The full study requires an exact clean revision and a separately
verified source archive. If a run fails, retain it and use a new run ID.
The 4.5.1 R package libraries must be used with the framework R 4.5.1
executable (`/usr/local/bin/Rscript` here). Homebrew R 4.5.0 crashed while
loading those compiled packages; this was an environment mismatch, not a
numerical disagreement.
