# Unit tests for the likelihoods

The frozen snapshot `frozen/` and its manifest `manifest_sha256.json`
live in the parent `tests/` directory; the paths below are relative to
it.

These tests catch two kinds of silent breakage: a $\chi^2$ that drifted
because code or data changed by accident, and a race condition (a bug
where evaluating several points in a row corrupts a later result
through leftover internal state or colliding OpenMP threads).

Every model build runs in its own worker subprocess. In this project
both examples share one data set, so the isolation is preventive: it
keeps the tests immune to the process abort that different data-set
dimensions trigger inside cosmolike, and every project keeps one
architecture. The commands below stay the same.

The mechanism of every shared check family - the frozen-reference and
race contract, the zero-baseline construction, the sweeps and their
`--high`/`--mask` options, the accuracy settings and their meanings -
is documented once in
[the shared data-vector checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_checks)
of the CosmoLike core. This file records what roman_fourier runs and
what it measured.

# Table of contents

1. [Running the tests](#run_tests)
2. [The tests](#the_tests)
    1. [The CFASTPT vs FASTPT comparison](#cfastpt_fastpt)
    2. [The Halofit vs EE2 checks](#halofit_ee2)
    3. [The EE2 race test](#ee2_race)
    4. [Accuracy checks](#accuracy_checks)
    5. [Baryonic feedback checks](#baryon_checks)
    6. [The photo-z convention checks](#photoz_conventions)
    7. [The non-Limber checks](#nonlimber_checks)
    8. [The sector-ladder cache check](#cache_ladder)
    9. [The non-Limber early-exit tolerance checks](#nonlimber_tolerance)
    10. [The non-Limber radial grid](#nonlimber_grid)
3. [Hybrid-emulator accuracy test](#hybrid-emulator-accuracy-test)
4. [FoM convergence](#fom-convergence)
5. [Appendix](#appendix)
    1. [FAQ: What does this project's snapshot hold?](#frozen_copy)
    2. [FAQ: How can maintainers refresh the snapshot?](#refreeze)

## Running the tests <a name="run_tests"></a>

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`. Every command
below assumes `start_cocoa.sh` has been sourced once:

    source start_cocoa.sh

Run the full suite with

    python -m pytest ./projects/roman_fourier/tests/data_vector

or, without pytest,

    python -m unittest discover -s ./projects/roman_fourier/tests -v

The tests change no project files. Each test prints a progress line per
model build and per evaluation, then a report with the computed $\chi^2$,
the stored reference, the difference, and the pass limit. A full run
performs about 135 likelihood evaluations and takes a few minutes. The test
files set their OpenMP worker count before imports.

The shared likelihood tests default to four OpenMP workers. For six workers,
set `COCOA_TEST_OMP_THREADS=6` and `OMP_THREAD_LIMIT=6` before
starting Python. The entry points copy the test setting to
`OMP_NUM_THREADS` before importing native libraries. Keep BLAS at
one worker. A smaller OpenMP thread limit can reduce the actual team
below the requested count.

## The tests <a name="the_tests"></a>

The standard configurations get four tests each - the frozen-reference
drift check and the race check, in NLA and in TATT
([mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_frozen)).
The TATT variants set

    IA_model: 1
    roman_A2_1: 0.05
    roman_BTA_1: 0.05
    roman_A2_2: -1.51541

The suite follows the shared
[test-file naming convention](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_naming): a file name
means the same thing in every project. This project's
configurations and test numbers:

| file | configuration | tests |
|---|---|---|
| `test_example1.py` | cosmic shear | 1-4 |
| `test_example2.py` | 3x2pt | 5-8 |
| `test_example2_2x2pt.py` | 2x2pt (`roman_fourier.combo_2x2pt`) | 11-14 |
| `test_fastpt.py` | the sweep: shear; 3x2pt (`roman_fourier.combo_3x2pt`); 2x2pt (`roman_fourier.combo_2x2pt`) | 15, 16, 17 |
| `test_ee2.py` | race | 18 |

Tests 9 and 10 are not used: the numbering follows the shared
convention, where they are the frozen FAST-PT reference checks
this project does not ship.

### The CFASTPT vs FASTPT comparison (`test_fastpt.py`, tests 15-17) <a name="cfastpt_fastpt"></a>

The sweep mechanism, the zero-baseline construction and the
`--high`/`--mask` options are in
[the shared checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_fastpt).
Here, test 15 runs it on cosmic shear, test 16 on the 3x2pt likelihood
and test 17 on the 2x2pt likelihood. In test 16 the frozen
configuration fixes the one-loop bias amplitudes (`roman_B2_*`) at
zero, so the sweep scores the intrinsic-alignment tables alone on the
wider data vector; in test 17 the TATT tables are scored through
galaxy-galaxy lensing alone.

This project's precision needs `accuracyboost: 2`, the doubled output
density. The tests assert there with the 0.2 band of the other checks
as the pass limit, doubling the configuration again as an advisory.

The roman_fourier scale-cut masks `--mask` accepts: `frozen` (the
default) keeps the contract mask (`roman_example.mask`, 1,359 of the
1,485 data points); `ones` keeps every point. The 3x2pt sweep does
not run under `ones` (the Warning below).

> [!NOTE]
> Before the two-grid upgrade of the fastpt theory block (2026-09)
> there was no upsampling and the difference reached
> $\Delta\chi^2 = 87184$ across the prior.

The point values, the design, and the decision record live with the
lsst_y1 project (its tests/README.md carries the full discussion);
the table below is this project's own measurement:

| output table (points) | internal grid (points) | max $\Delta\chi^2$ | Runtime / first row |
|---|---|---|---|
| 1,100 | 1,100 (shared) | 87184 | 1.00 |
| 1,024,900 (`accuracyboost: 1`) | 1,100 | 0.209 | 1.27 |
| 2,048,900 (`accuracyboost: 2`, the pass configuration) | 1,100 | 0.1120 | 1.53 |
| 4,096,900 (`accuracyboost: 4`) | 1,300 (`internal_accuracyboost: 2`) | 0.0828 | 2.00 |
| 8,192,900 | 1,100 | 0.073 | 2.80 |

Runtime ratios use the first row of this table as the baseline, with the
same timed scope and machine. They compare sampling costs within this
project; absolute runtimes depend on the machine.

![The 30 comparison points, colored by the per-point difference](cfastpt_vs_fastpt_points.png)

> [!NOTE]
> This project's precision needs `accuracyboost: 2`, the value its
> example yamls recommend and the tests assert on; the defaults
> land at 0.209, just above the band. cfastpt (`IA_code: 0`)
> remains the reference implementation.

Measured on 2026-09-23:

- Test 16 (3x2pt): max $\Delta\chi^2 = 0.1475$ at the defaults and
  $0.1092$ at the pushed camb/cosmolike settings, above cosmic
  shear's 0.1120 under the 3x2pt masked covariance.
- Test 17 (2x2pt): $0.0021$ and $0.0013$, with the TATT tables
  entering through galaxy-galaxy lensing alone.
- `--mask=ones` (no scale cuts, all 1,485 points weighted): cosmic
  shear repeats its frozen-mask values exactly - the contract mask
  keeps every shear point, so the two masks coincide there - and
  measures 0.0903 at the pushed camb/cosmolike settings; the 2x2pt
  sweep measures max $\Delta\chi^2 = 0.0039$ at the defaults and
  0.0025 pushed.

> [!Warning]
> `--mask=ones` with the 3x2pt likelihood (test 16) fails at model
> build: with every data point kept the shipped
> covariance is not positive definite, and cosmolike aborts
> (`IP::set_inv_cov`). The mask runs with tests 15 and 17.

Run the comparison at the default and at the pushed camb/cosmolike
settings (the full comparison is both invocations):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_fastpt.py
    python -m pytest ./projects/roman_fourier/tests/data_vector/test_fastpt.py --high=1

### The Halofit vs EE2 checks (`test_nonlinear.py`, NL1-NL2) <a name="halofit_ee2"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_halofit_ee2):
NL1 evaluates the cosmic-shear data vector and NL2 the 3x2pt one, each
with both nonlinear-power sources at the ten shared cosmologies
(halofit, `non_linear_emul: 2`, is the frozen contract's setting).
Measured on 2026-09-23 (the figure below, frozen mask):

- NL1 (cosmic shear): per-cosmology $\Delta\chi^2$ between 29.1
  and 864.7 (median 99.9), largest at the high-omegam draws; under
  `--mask=ones` it repeats these values digit for digit - the
  contract mask keeps every shear point, so the two masks coincide
  there.
- NL2 (3x2pt): between 358.4 and 3876.0 (median 879.8), largest at
  the high-omegam draws; it does not run under `ones` - with every
  data point kept the shipped covariance is not positive definite
  and cosmolike aborts at model build (`IP::set_inv_cov`, the
  Warning above).

![The ten cosmologies, colored by the Halofit-vs-EE2 difference](halofit_vs_ee2_points.png)

Run them on their own (`--mask=ones` reruns them with every data
point kept; NL2 aborts there, the Warning above):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_nonlinear.py

### The EE2 race test (`test_ee2.py`, test 18) <a name="ee2_race"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_ee2):
test 18 is the race check with EE2 sourcing the nonlinear $P(k)$
(`non_linear_emul: 1`) instead of the frozen contract's halofit. The
modification gate, which compiles the pre-modification build (commit
`ff59f66`) at test time and scores the installed build against it,
runs as the lsst_y1 project's test 18 (its `tests/test_ee2.py`); the
emulator build is project-independent, so that comparison is not
repeated here. Measured on 2026-09-23:

- Test 18: the fresh and 10th-in-a-row fiducial agree to all eight
  printed decimals.
- The modified-vs-original comparison of the lsst_y1 gate, run on
  this project's cosmic shear under its frozen mask (ten
  cosmologies), measured max $\Delta\chi^2 = 4.3\times10^{-4}$ (max
  fractional data-vector difference $3.5\times10^{-5}$): the largest
  of the six projects, still far below the 0.2 band.

### Accuracy checks (`test_accuracy.py`, A1-A6) <a name="accuracy_checks"></a>

[Mechanism, setting meanings and cost order](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_accuracy):
the scan changes one accuracy parameter at a time on the 3x2pt NLA
configuration (the `KNOB` lines of the test output; the measured
values are in the project README under "Minimum accuracy
parameters"); checks A1-A6 then re-evaluate cosmic shear, 3x2pt, and
2x2pt, with NLA and TATT, with every setting pushed far beyond the
defaults at once. This project's pushed values:

| setting | raised to |
|---------|-----------|
| `accuracyboost` (cosmolike) | 3 |
| `integration_accuracy` (cosmolike) | 10 |
| `internal_accuracyboost` (cosmolike) | 2 |
| `kmax_boltzmann` (cosmolike) | 40 |
| `AccuracyBoost` (CAMB) | 2 |
| `k_per_logint` (CAMB) | 50 |
| `kmax` (CAMB) | 50 |

There is no `lmax` entry here: the ell range lives in the dataset.

The nested z-grid construction `accuracyboost` refines is commented in
`likelihood/_cosmolike_prototype_base.py`. The likelihood yamls set
`internal_accuracyboost: 1`, which keeps the convolution grid equal to
the output table: the exact single-grid path. Coarser grids also hold:
the lsst_y1 scan of this setting measured
$`\Delta^T C^{-1} \Delta \le 10^{-9}`$ against the single-grid path
down to `internal_accuracyboost: 0.27`.

#### Why these accuracy defaults? <a name="accuracy_defaults"></a>

The default is `accuracyboost: 2.0` with `pk_z_refinement: 2`. Two grids
of this project need more than `accuracyboost: 1` (measured 2026-10-01):

- The z nodes of the power-spectrum tables. cosmolike interpolates the
  tables linearly in z between their nodes, and at the band centers of
  this Fourier-space data vector that residual does not average out:
  the 3x2pt $\chi^2$ at the synthetic test point moves by 0.25, 0.030
  and 0.002 from 1 to 2, 4 and 8 times the nodes of `accuracyboost: 1`.
  The default has 4 times the nodes: `accuracyboost: 2` doubles them
  and `pk_z_refinement: 2` doubles them again.
- The C-FAST-PT tables of the TATT and one-loop galaxy-bias terms. At
  `accuracyboost: 1` the cfastpt vs FAST-PT comparison of the tests
  (tests 15 and 16) reaches $\Delta\chi^2$ = 0.25 (cosmic shear) and
  0.31 (3x2pt), above the 0.2 band; at `accuracyboost: 2` it is 0.11
  and 0.14.

`accuracyboost` stays global: it multiplies every grid, these included,
and `pk_z_refinement` refines the z nodes alone. Under NLA alone,
`accuracyboost: 1` with `pk_z_refinement: 4` has the same z nodes and
gives a data vector within $\delta^T C^{-1} \delta = 0.005$ of the
default at a lower cost; the TATT tables are what need the boost. The
CAMB request grid stays at 140 redshifts (under CAMB's 256-node cap), so
only the cosmolike tables pay.

The project tag v5.01 shipped `accuracyboost: 1` with
`pk_z_refinement: 4`, which fails the TATT checks above; the default
returned to `accuracyboost: 2` on 2026-10-01.

> [!Warning]
> Do not lower these settings to save computing time: with fewer than 4
> times the z nodes of `accuracyboost: 1`, or with `accuracyboost` below
> 2 under TATT, the error exceeds the 0.2 accuracy budget of the
> reference tests. The likelihood yaml files repeat this next to the
> knobs.

Run them on their own, or skip them (high-accuracy evaluations take
minutes):

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_accuracy.py
    python -m pytest ./projects/roman_fourier/tests --ignore ./projects/roman_fourier/tests/data_vector/test_accuracy.py

### Baryonic feedback checks (`test_accuracy_baryons.py` BF0-BF7, `test_baryons.py` BD1-BD7) <a name="baryon_checks"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_baryons):
the accuracy checks (BF) report the default-versus-pushed
$\Delta\chi^2$ with each feedback method on; check BF0 also runs the
one-setting-at-a-time scan with the Akino SP(k) feedback on, so a
large delta names the setting causing it. The drift tests (BD) pin
each method's frozen prediction. The BACCOemu point evaluates with
`omegab: 0.049`, inside that emulator's baryon-density training box,
whose floor sits exactly above the fiducial `omegab: 0.04`. Run them
with

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_accuracy_baryons.py
    python -m pytest ./projects/roman_fourier/tests/data_vector/test_baryons.py

### The photo-z convention checks (`test_photoz_conventions.py`) <a name="photoz_conventions"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_photoz):
the frozen cosmic-shear fiducial under the default, each alternative
convention, and the default again. Measured on 2026-09-24:

- default: $\chi^2 = 0.040700$ (the frozen reference exactly).
- linear: $\Delta\chi^2 = 9.4\times10^{-4}$; Steffen:
  $\Delta\chi^2 = 2.9\times10^{-6}$. Either interpolant change is far below
  the survey's statistical precision.
- Z_MID: $\Delta\chi^2 = 3.60$ - the half-bin z-column reading
  is the one photo-z convention that matters at this precision.

The figures below are regenerated by
`generate_photoz_convention_figure.py`:

![The Z_LOW vs Z_MID reading of the n(z) z column](photoz_zmid_dcl.png)

![Linear and Steffen n(z) interpolation vs cubic spline](photoz_interp_dcl.png)

### The non-Limber checks (`test_nonlimber_ggl.py`, `test_nonlimber_gg.py`) <a name="nonlimber_checks"></a>

[Mechanism](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_nonlimber):
`adopt_limber_gs` and `adopt_limber_gg` are both `0` (non-Limber
below the cap `lmax_nonlimber: 200`) in this project, because the
measured differences below are too large to absorb. In this project's Fourier-space data
vector each band center takes its Limber value plus the non-Limber
correction interpolated linearly between the two integer multipoles
around it.

Galaxy-galaxy lensing (the test pins $\Delta\chi^2 = 1.2376$ to 5%):

- $\Delta\chi^2 = 1.24$ for the 3x2pt data vector.
- The pairs with lens bin = source bin carry it (measured 2026-10-01):
  (3,3), (2,2), (4,4), (1,1), (5,5), (6,6) with 0.72, 0.70, 0.59,
  0.49, 0.40, 0.21.
- 1.30 on 2026-09-27, before the band-center correction was computed
  without cosmolike's 1% early exit, at the two integer multipoles
  around each band center only, and before the z nodes of the
  power-spectrum tables were doubled (`pk_z_refinement: 2` on top of
  `accuracyboost: 2`).

On 2026-09-28 the galaxy-galaxy lensing default switched from Limber
to the exact projection: the measured $\Delta\chi^2$ was judged too
large to absorb. The shipped data vector was regenerated with the
non-Limber default and the frozen references were refrozen from it.

Galaxy clustering (the test pins $\Delta\chi^2 = 3.4764$ to 5%):

- $\Delta\chi^2 = 3.48$ for the 3x2pt data vector, against 1.24 for
  the same comparison in galaxy-galaxy lensing.
- The largest contributions (measured 2026-10-01) are lens bins 5, 6,
  4, 3 with 1.04, 0.90, 0.82, 0.49 (each bin's block alone).

On 2026-10-01 the clustering default switched from Limber to the exact
projection: the measured $\Delta\chi^2$ (3.34 on 2026-09-28) exceeds
the 1.30 that switched galaxy-galaxy lensing, and clustering had
stayed Limber only because the key preserved the old behavior when it
was introduced. The shipped data vector was regenerated with the
non-Limber default (and `accuracyboost: 2` with `pk_z_refinement: 2`)
and the frozen references were refrozen from it.

### The sector-ladder cache check (`test_cache_consistency.py`) <a name="cache_ladder"></a>

The deterministic mixed-update ladder and its bitwise assertions are
described in
[the shared checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_cache_ladder).
In this project the ladder takes three cosmology-only steps, then
IA-only, source-photo-z and shear-calibration steps - the lenses are
the source sample, so there is no separate lens-photo-z rung - and
both intrinsic-alignment models run.

### The non-Limber early-exit tolerance checks (`test_nonlimber_tolerance.py`) <a name="nonlimber_tolerance"></a>

The non-Limber $C_\ell$ of $w(\theta)$ (and of $\gamma_t$ where
`adopt_limber_gs: 0`) are computed in blocks of 16 multipoles, and a
lens bin (clustering) or lens-source pair (lensing) whose $C_\ell$
agrees with the Limber $C_\ell$ within a tolerance at the end of a
block takes the Limber values from there: the early exit. The switch
multipole is an integer that moves with the parameters, so a
finite-difference derivative whose stencil straddles a switch sees a
step in the model, which a Fisher matrix reads as information. The
likelihood keys `nonlimber_tolerance_gg: 0.002` and
`nonlimber_tolerance_gs: 0.01` set the tolerances; `0` removes the
exit, the setting Fisher derivatives use, and `accuracyboost` $> 1$
divides the tolerances in use by $1 + 2\,(\mathrm{boost} - 1)$. The
Fourier data vectors (`C_gg_tomo_ells`, `C_gs_tomo_ells`) never take
the exit, so in this project the keys keep the core defaults and the
Fisher switch to tolerance 0 changes nothing (test 3 checks that bit
for bit).

The production cap is `lmax_nonlimber: 200` (test 1 asserts it),
judged on 25 models drawn across the examples' prior box ($\Omega_m$
in $[0.2, 0.5]$, $w_0 + w_a$ in $[-3, -0.01]$; seed base 137) with the
examples' amplitude cut, which keeps the ratio
$(A_s/A_s^{\rm fid})(b_i/b_i^{\rm fid})^2 \le 8$ over the sampled
lens bins ($A_s$ the primordial amplitude, $b_i$ a bin's linear galaxy
bias), and on 5 held models with $\Omega_m$, $A_s$ and the lens biases
at the fiducial.

Cap 150 differs from cap 200 by $\Delta\chi^2$ up to $1.32\times10^3$
over the drawn models (15 of 25 at or above 0.2, almost all in the
galaxy-shear block) and up to 65.7 over the held models (1 of 5),
measured at `nonlimber_accuracyboost: 1` and grid end 4.0; at the
production grid settings the extreme-IA held model falls from 66 to
43. The models above 0.2 sit at the edges of the $[-5, 5]$ NLA priors
(the worst at $10^9 A_s = 4.86$ with the IA amplitude at $-1.53$ and
the redshift power at $+4.9$) and are accepted there: the galaxy-shear
intrinsic-alignment (GI) term carries a Limber error that decays like
$1/\ell^2$ and is 2 to 3 times the clustering one, because the
intrinsic shear follows the tidal field. At those models the GI term
makes the highest-bin $C_{gs}$ about 100 times its fiducial value
while the covariance stays fiducial, so a fraction-of-a-percent error
scores as many $\sigma$ per band.

`test_nonlimber_tolerance.py` builds the `example2` (3x2pt, NLA,
Fourier) model once and runs the shared checks of
`nonlimber_tolerance_suite` (`cosmolike_core/cocoa_testing_suites.py`):
the frozen $\chi^2$, the documented defaults and a bit-for-bit tolerance
round trip (tests 1-2); that tolerance 0 changes nothing, because this
data vector is Fourier (test 3; test 4 skips); the stencil noise of the
$w_0$, $w_a$ and `roman_B1_8` derivative columns and the FKEM hand-off
at the caps `lmax_nonlimber` = 150 and 200 (test 5: a step at the
hand-off would raise the noise at the lower cap, where the FKEM-Limber
difference is largest); the low-rank (SVD) engine next to FKEM at caps
150 and 200, with the galaxy-shear cap matched between the engines in
each pair (test 6); that a negative or nonfinite tolerance is refused
(test 7); the accuracy-boost scaling of the tolerances in use (test 8);
and that the production state returns bit for bit (test 9). The noise is
$\sigma$ of `get_stencil_noise` (`cosmolike_notebook_utils/fisher.py`):
the whitened fourth difference of five evaluations spaced $h = 0.025$
apart, in $\sqrt{\chi^2}$ units, the jitter one evaluation adds to a
derivative column.

Measured on 2026-11-18 (OMP 4), $\sigma$ in $\sqrt{\chi^2}$ units:

| stencil noise $\sigma$ | $w_0$ | $w_a$ | `roman_B1_8` |
|---|---|---|---|
| cap 150 (no early exit) | 3.88e-3 | 5.00e-4 | 3e-13 |
| cap 200 (production) | 4.09e-3 | 5.03e-4 | - |

The frozen configuration runs at `accuracyboost: 2`. The $w_0$ noise
floor of about 4e-3 is the same at both caps and in the SVD engine,
so it does not come from the non-Limber path; the bound at cap 150 is
5e-3, and test 5 allows cap 150 at most 1.1 times cap 200. The
largest $|C_\ell^{\rm FKEM}/C_\ell^{\rm Limber} - 1|$ over the
lens bins is 4.1e-3 / 2.6e-3 at $\ell$ = 149 / 199. Test 6 allows the
SVD noise at most 1.1 times FKEM with the galaxy-shear cap matched
(the 2026-11-19 run is recorded in
`tests/nonlimber_matched_caps_20261119.json`).

Run it alone with

    python -m pytest ./projects/roman_fourier/tests/data_vector/test_nonlimber_tolerance.py -s

### The non-Limber radial grid (`test_nonlimber_endpoint.py`) <a name="nonlimber_grid"></a>

The non-Limber engines integrate on a radial grid uniform in $\ln\chi$
whose end is the redshift `nonlimber_zmax` (core default 4.0; this
project sets 5.0) and whose node count scales with
`nonlimber_accuracyboost` (this project sets 4). The highest bin's
$n(z)$ (one file serves lens and source) ends at $z = 3.99$, at 10% of
its peak, right at the core default grid end. Two numerical effects
follow for that bin's pairs:

- the FKEM wavenumber sum starts at $k = (\ell + 1)/\chi(z_{\rm max})$:
  a grid that ends at $z = 4$ truncates the low-$k$ part of the bin's
  transform, and the loss shrinks only slowly with $\ell$;
- the $n(z)$ end falls between radial nodes ($\Delta z = 0.067$ apart
  there at boost 1, 0.017 at boost 4), so the grid sees a kernel that
  differs from the one the Limber quadrature integrates.

Measured on an extreme-IA test model (NLA amplitude $-3.87$, redshift
power $+4.84$): the relative non-Limber correction of the highest-bin
pair at $\ell = 180$ is $-1.04\times10^{-2}$ of $C_{gs}$ at boost 1
and grid end 4.0, against $-6.95\times10^{-3}$ from a direct Bessel
integration, off by 50%. At boost 4 with the grid end past the $n(z)$
end, the same correction follows the direct integration to
$7\times10^{-5}$ of $C_{gs}$ from $\ell = 180$ to 1000, and grid ends
4.5 and 5.0 agree to $\Delta\chi^2 = 0.23$ on the whole vector at that
model. The grid end 5.0 also clears the low-$k$ truncation at
$\ell = 30$ (a residual $1.2\times10^{-3}$ of $C_{gs}$ there, against
$4\times10^{-3}$ at 4.5), at no measured cost in time; boost 4 costs
133 ms against 109 ms per cosmolike evaluation at cap 150 (4 OpenMP
threads).

At the fiducial, moving the grid end from 4.0 to 5.0 changes the 3x2pt
vector by $\Delta\chi^2 = 1.1\times10^{-3}$, only in the non-Limber
$C_{gs}$ and $C_{gg}$ entries ($C_{ss}$ is bitwise unchanged); the
frozen references were regenerated at 5.0, and
`tests/refreeze_record_20261119.json` records the comparison.
`test_nonlimber_endpoint.py` checks the preset (5.0), the resets,
invalid values, and that the endpoint is independent of the radial
node count.

## Hybrid-emulator accuracy test

[compare_emulator_camb.py](../compare_emulator_camb.py) compares the candidate
matter-spectrum emulator with fresh CAMB predictions at the saved 50
broad and 25 focused cosmologies. Its fixed [benchmark](../fisher_reference/)
supplies the nuisance point, mask and covariance, enforces $0.2<\Omega_m<0.5$,
and fixes
$\sum m_\nu=0.06$ eV. Both routes use plain `mead2020` and CAMB distances;
the production YAML is not changed. The selected example has HOD galaxy
power and halo intrinsic alignments disabled. Its prediction therefore does
not test the EMUL2 approximation to cold-matter power used by those models.
The two routes retain their own power-spectrum coverage and extrapolation,
so the distance includes those differences as well as network error.

**Step :one:**: from activated Cocoa, set `EMULATOR_CODE_DIR` and
`EMULATOR_DATA_DIR` to the candidate code and model checkouts. The model checkout must contain
`emulmps/w0wa_mead2020/emulator_mead2020.keras` and its `metadata.joblib`.
**Step :two:**: set the OpenMP thread count. The driver fixes BLAS to one
thread before importing numerical libraries.

```bash
export OMP_NUM_THREADS=4
```

**Step :three:**: run the comparison.

```bash
python projects/roman_fourier/tests/compare_emulator_camb.py
```

The script uses the existing shared `cocoa_testing.py` machinery. It saves
points, vectors, failures and the prediction distance
$\Delta\chi^2=\delta m^T C^{-1}\delta m$ under
a timestamped directory below `chains/matter_emulator_points/`. A value below 1 passes the
requested criterion; failures are retained, never replaced by new draws.
The project prior can extend beyond the emulator training range. Separate
those extrapolation failures from interpolation errors when interpreting
the saved points; this test does not narrow the prior to favor the emulator.
`--models`, `--seed` and `--limit` change the test size, reproducible seed and
threshold. BLAS stays at one thread. This comparison does not establish
that posterior importance weights have adequate effective sample size;
that must be checked on the chain used for reweighting.

<!-- hybrid-emulator figure -->
**Emulator training range**

| Parameter | Range |
| --- | --- |
| Total matter density | $0.2<\Omega_m<0.5$ |
| Primordial amplitude | $1<10^9 A_s\le 3.5$ |
| Hubble constant | $H_0<85$ km/s/Mpc |
| Present dark-energy equation of state in CPL | $w_0>-2$ |
| Physical baryon density | $\Omega_b(H_0/100)^2<0.035$ |

The fixed tables intersect this range with the benchmark priors. CAMB uses
one massive species with $\sum m_\nu=0.06$ eV. Both model routes evaluate
the same 50 broad points (open symbols) and 25 focused points (filled symbols).
The solid and dashed curves show the saved 68% and 95%
nuisance-marginalized Fisher regions.

CAMB uses `AccuracyBoost: 1.05`, `k_per_logint: 15` and `kmax: 7.5`.

![Roman Fourier hybrid-emulator error against CAMB](figures/emulator_precision.png)

Counts use the thresholds 0.2, 0.5, 1 and 4 in that order.

| Case | Broad below threshold (50) | Focused below threshold (25) | Nonfinite broad/focused |
| --- | --- | --- | --- |
| Roman Fourier | 7/16/21/27 | 8/20/21/23 | 0/0 |

<!-- end hybrid-emulator figure -->

### Full 3×2pt

The [3×2pt benchmark](../fisher_reference/3x2pt/) uses Evaluate 2's full
optical selection and fixed nuisance values. Its matched Fisher forecast
uses the same retained data vector and marginalizes over the sampled
nuisance parameters. The 50 broad cosmologies are the same as in the
shear comparison. The 25 focused cosmologies come from the 3×2pt Fisher
distribution.

CAMB uses `AccuracyBoost: 1.1`, `k_per_logint: 15`, and `kmax: 10.0`.

![Roman Fourier - 3×2pt](figures/emulator_precision_3x2pt.png)

Counts use the thresholds 0.2, 0.5, 1 and 4 in that order.

| Probe | Broad below threshold (50) | Focused below threshold (25) | Nonfinite broad/focused |
| --- | --- | --- | --- |
| 3×2pt | 2/3/3/15 | 0/0/3/22 | 0/0 |

Add `--probe 3x2pt` to the standalone comparison command below.


E5 reads the saved `benchmark.yaml` in [fisher_reference/](../fisher_reference/).
The benchmark fixes the likelihood, data selection, sampled parameters,
fiducial and normal numerical controls.

The accuracy test evaluates CAMB and the candidate at the fixed
[50 broad and 25 focused points](../fisher_reference/) and stores both
vectors, the covariance-weighted error, and point status. Nuisance inputs are
held at the evaluate fiducial. Missing CAMB predictions or a comparison
with no valid vectors fails E5. The E5 adapter treats large prediction errors
as advisory. The strict standalone command also checks the error threshold.

```bash
python -m pytest projects/roman_fourier/tests/data_vector/test_emul2.py -k e5
```

To distribute the numerical comparisons across two MPI ranks with four
OpenMP threads each, use the same driver directly from activated Cocoa.
Keep BLAS at one thread and pass `--mpi` so the ranks own different points.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 mpirun -n 2 --bind-to none python -m mpi4py \
  projects/roman_fourier/tests/compare_emulator_camb.py --mpi --advisory
```

The shared plot uses saved 68% and 95% bounded Fisher contours with
nuisance parameters marginalized. The reference records the matrix, priors,
fiducial and data selection. Use `--fisher-reference` to select a saved
reference directory or `--no-fisher` to show only the comparison points.
Use `EMULATOR_CODE_DIR` and `EMULATOR_DATA_DIR` to select the candidate
network installation.

```bash
python projects/roman_fourier/tests/compare_emulator_camb.py --plot-only /path/to/run
```

## FoM convergence

This test estimates how numerical settings affect the Fisher figure of merit
(FoM), which is proportional to the inverse area of the marginalized
parameter ellipse. The pair is $w_0,w_a$ for CPL and $10^9 A_s,\Omega_m$
for ΛCDM. The test builds one full CAMB Fisher matrix at the benchmark
settings. Fourth differences of its model vectors rank contributions from
numerical noise.
The higher-accuracy calculation reevaluates only the selected derivative
columns. The estimate uses the same baseline Fisher weights at both settings.

Each nonzero fiducial uses `h = 0.025 * abs(fiducial)`, capped as needed
to keep both `±2h` stencil points within its prior. Zero-centered
coordinates use their configured absolute step or one tenth of their proposal
width. The output records the actual steps, numerical settings, measured and
omitted columns, and the estimated FoM shift. Contributions from omitted
columns remain unbounded, so the estimate measures sensitivity to the tested
refinement.

Run from activated Cocoa with two MPI ranks, four OpenMP threads per rank
and one BLAS thread. Add `--probe 3x2pt` for the full selection.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 mpirun -n 2 --bind-to none python -m mpi4py \
  projects/roman_fourier/tests/check_fom_convergence.py --mpi
```

## Appendix <a name="appendix"></a>

## :interrobang: FAQ: What does this project's snapshot hold? <a name="frozen_copy"></a>

The snapshot-protection mechanism is described in
[the shared checks](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_frozen).
This project's `frozen/` holds:

| `frozen/` entry | holds |
|---|---|
| `frozen_config_*.py` | one module per configuration (cosmic shear, 3x2pt, and 2x2pt): the complete cobaya configuration as a yaml string, plus the exact evaluation point |
| `data/` | the tests' own copy of the data vectors, covariance, n(z), and masks |
| `reference_chi2.json` | the six reference $`\chi^2`$ values: cosmic shear, 3x2pt, and 2x2pt, each with NLA and TATT |
| `EXAMPLE_EVALUATE{1,2}.yaml` | snapshots kept only so a human can diff how the live examples drifted since the freeze |

All TATT variants evaluate against `frozen/data/tatt_roman_fourier.dataset`,
the synthetic vector generated at freeze time
([why](https://github.com/CosmoLike/cocoa-cosmolike-core/blob/main/README.md#dv_frozen)).

## :interrobang: FAQ: How can maintainers refresh the snapshot? <a name="refreeze"></a>

A deliberate change to the data vectors, n(z), covariance, examples,
or likelihood defaults requires a re-freeze.

We assume users are in the Conda cocoa environment from a previous
`conda activate cocoa` command, that the shell is bash, and that the
current folder is the cocoa main folder `cocoa/Cocoa`.

**Step :one:**: activate the private Python environment by sourcing
the script `start_cocoa.sh`

    source start_cocoa.sh

**Step :two:**: rebuild the snapshot

    python ./projects/roman_fourier/tests/generate_frozen_reference.py --overwrite

**Step :three:**: add the feedback vectors of the drift tests

    python ./projects/roman_fourier/tests/generate_frozen_reference.py --baryons

The first command deletes and rebuilds `frozen/` from the current
project, prints the six new reference $\chi^2$ values, and rewrites the
manifest. It does not write the per-method feedback vectors of
`test_baryons.py`; the second command adds them and rewrites the
manifest again. Review the printed $\chi^2$ values against the old
references before committing: they define what every later test run
compares against.
