# Marginal Poisson row-bootstrap calibration protocol

Protocol `poisson-bootstrap-known-scores-v1`, prepared before study outcomes.

## Question and target

How often does the existing fixed-cut row-bootstrap normal interval with
200 resamples cover the true *marginal* log mean ratio for four prequantized,
known exposure scores (`q=None`)? This does not test estimated cutpoints,
raw exposures or ties.
We generate independent rows and fit the correctly specified additive
Poisson-log model. Four exposure scores Q_j are independent uniform draws from
{0,1,2,3}; S is their sum. A binary adjustment covariate Z is independent
Bernoulli(0.5). For each row,

    E[Y | Q,Z] = exp(log(m) + b*(S-6)/4 + 0.4*Z),
    Y | Q,Z ~ Poisson(E[Y | Q,Z]).

Here m is the mean at the hypothetical score center with Z=0, not the
population average. The reported model-based contrast sets every score to
0 versus 1, averaging predicted counts over the observed Z values. Its log
ratio is exactly b. The additive model gives the same ratio for supported
adjacent pairs 1 versus 2 and 2 versus 3. The Z factor cancels from the log
ratio: this design exercises the standardized-mean calculation but does not
test uncertainty unique to a noncancelling marginal contrast. It does not
test a nonlinear or effect-modified marginal contrast. Observation periods are equal;
there is no offset, overdispersion, clustering or causal identification claim.

| Factor | Fixed values |
| --- | --- |
| Rows per independent dataset | 500 |
| Center-point mean m | 0.20, 2.00 |
| True joint log mean ratio b | 0 (null), 0.40 (nonzero) |
| Row-bootstrap draws B | 200, fixed-cut, independently sampled with replacement |
| Independent datasets per cell | pilot 5; study 1,000 |
| Total | pilot 20; study 4,000 datasets and 800,000 bootstrap replicates |
| RNG | NumPy PCG64, distinct SeedSequence namespaces for inputs and bootstrap rows |

The cells are ordered by m then b. For each dataset fit q=None with Z as one
numeric covariate and alpha=0.05. Use the existing public
`fit_poisson_bootstrap` with its default B=200 and a saved 64-bit seed.
Separately fit the model-based additive `fit_poisson` on the same inputs as a
descriptive baseline. These two intervals share point estimand b but use
different variance calculations. Do not pool them or treat one as truth.

## Checks and planned table

The existing two fixed-input installed-R qgcomp 2.19.6 comparisons are a
separate software-agreement check. The repeated-data checker regenerates
every planned input and bootstrap schedule from fixed seeds. It independently
solves the Poisson score equations for the original and each resampled design,
uses inverse observed information for the model-based baseline, and recomputes
bootstrap standard deviation, normal interval and p-value. It audits every
case, including package failures, and records input/schedule identities,
numerical mismatches and failed independent solves. Estimator equations are
calculated independently; generation, seed mapping and summary code are
shared with the runner, so this is not an independent audit of the entire
design. A nonzero exit follows any warning, failure or mismatch, after
preserving all available case audits. Only top-level unusable plan/source
identity prevents a full audit. Exact random-stream replay requires the
same NumPy version; other versions are not verified. The checker source is
bound to the study commit.

For each cell and method, report planned, attempted, successful, failed and
missing counts; realized count mean and zero fraction; bias and its Monte
Carlo standard error; empirical estimate SD; root mean reported variance;
95% interval coverage with Wilson 95% interval; rejection fraction at 0.05
with Wilson interval. Label null rejection false-positive rate and nonzero
rejection descriptive power. Give covered/planned and all-planned worst/best
coverage bounds. Retain each failed case and its reason; no replacement draw.
At 1,000 successful fits, nominal 95% coverage has approximate Monte Carlo
SE 0.0069. Wilson intervals are cellwise and do not adjust for four cells or
two methods. A null test and its interval use the same cutoff, so their rates
are complementary, not independent evidence.

Investigate any bias larger than three Monte Carlo SEs, coverage Wilson
interval wholly outside [0.93,0.97], nominal 0.95 outside a cellwise Wilson
interval, or any warning, failure or numerical mismatch. These are review
flags, not pass criteria or an equivalence test. Failure to trigger a flag
does not establish 95% coverage. Report all cells and any adverse results without
changing the design after seeing study outcomes. A high fit success rate does
not prove valid bootstrap coverage.

## Execution boundary

Run only the 20-case timing pilot locally after a fresh space and worker check,
with one numerical thread, less than 2 GiB memory, ten minutes and 100 MiB
new output. The full study is SERVER_CPU. Before submission, require formal
protocol review, a clean committed source, passing pilot, fresh server
quota/queue and job-spec checks, bounded output and memory, and a no-duplicate
run identity. Preserve a flushed case journal; a partial study is not resumed
under the same identity. The original conditional Poisson study and its
possible sparse-null undercoverage remain unchanged. Python fitting must not
require R at runtime.
