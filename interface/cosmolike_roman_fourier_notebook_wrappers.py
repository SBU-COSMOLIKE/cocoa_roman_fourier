"""Notebook wrappers for the roman_fourier Cosmolike interface.

roman_fourier forecasts a Roman Space Telescope 3x2pt analysis in
Fourier space: the data vector holds angular power spectra C_l
averaged over n_cl = 15 logarithmic multipole bands between l = 30
and l = 4000, evaluated at the logarithmic center of each band (the
band centers). Three probes share 8 tomographic bins (redshift
slices) of one n(z) file, used both as lens and as source sample:

  C_ss  cosmic shear, the E-mode spectra of the 36 source-bin pairs;
  C_gs  galaxy-galaxy lensing, the 55 lens-source pairs left after
        ggl_exclude;
  C_gg  galaxy clustering, the 8 lens-bin auto spectra.

The EXAMPLE_EVALUATE notebooks drive the compiled interface
(cosmolike_roman_fourier_interface, imported as ci) through the steps
of one likelihood evaluation: run CAMB once (cnu.get_camb_cosmology;
cnu is the shared package cosmolike_notebook_utils), push the power
spectra, growth and distances into the interface with set_cosmology,
set the nuisance parameters, and read off a spectrum, the masked data
vector or its chi2. This module holds those steps once:

    import cosmolike_roman_fourier_notebook_wrappers as nw
    nw.init_cosmolike(CLprobe="xi", with_data=True)
    nw.get_chi2(omegam=0.31)

Three layers of state matter here:

- The compiled interface is a C library with global state: every
  init_* and set_* call replaces part of it, and the spectrum
  functions read whatever was set last. Each wrapper therefore
  rebuilds everything it depends on (accuracy, cosmology, nuisances)
  on every call, so no call depends on which wrapper ran before it.
- The project fiducial point lives in this module as plain constants
  (roman_B1_1, ...), shared by every notebook; a notebook overrides
  any of them per call (nw.C_ss_tomo_limber(omegam=0.31)) or imports
  the names for its own sweeps.
- The options that mirror the likelihood yaml files (dataset, excluded
  pairs, model choices, the CAMB settings) live in _CONFIG and change
  once per notebook with configure(), before init_cosmolike.

Every wrapper accepts the same accuracy arguments and combines them as
the roman_real wrappers do (_set_state): CLAccuracyBoost is multiplied
by AccuracyBoost, and the integration accuracy grows by
|3 (CLAccuracyBoost - 1)|. The defaults (every boost 1) are lighter
than the likelihood yaml files, which run accuracyboost 2 and
pk_z_refinement 2; the accuracy sections of the notebook measure what
that costs.

Arguments shared by the spectrum wrappers (C_ss_tomo_limber ...
get_chi2 and the scale-cut wrappers):

  ell   = 1D array of multipoles, or None for the band centers of the
          dataset (band_centers()).
  omegam, omegab, H0, ns, As_1e9, w, w0pwa = the cosmology (H0 in
          km/s/Mpc, As_1e9 = 10^9 A_s, w0pwa = w0 + wa); defaults: the
          fiducial constants below. mnu is always the module constant.
  A1, A2, BTA, shear_photoz_bias, M = source nuisance vectors, one
          entry per source bin (IA parameter slots, photo-z shifts,
          shear calibration); None = the fiducial vector.
  lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
  galaxy_bias_b3nl, galaxy_bias_bk = lens nuisance vectors, one entry
          per lens bin (the B1, B2, B_MAG, B3nl and BK of _set_state);
          None = the fiducial vector.
  baryon_sims = name of a hydrodynamical simulation whose baryonic
          suppression multiplies the matter power, or None for none.
  allsims = HDF5 file of the hydro simulations, or None for the file
          init_cosmolike read from the dataset.
  AccuracyBoost, kmax (1/Mpc), k_per_logint, CAMBAccuracyBoost,
  CLAccuracyBoost, CLIntegrationAccuracy = accuracy settings
          (_set_state); kmax, k_per_logint and CAMBAccuracyBoost
          default (None) to the CAMB settings of _CONFIG.
  non_linear_emul = 1 (EuclidEmulator2) or 2 (CAMB halofit); None =
          the configure()d value.
"""

import os
import sys

import numpy as np
from getdist import IniFile

# The shared notebook utilities live in cosmolike_core, so that folder goes
# first in Python's module search list (sys.path). ROOTDIR, the cocoa/Cocoa
# folder, is set by start_cocoa.sh: a KeyError here means Cocoa was not
# started. The compiled interface is on the path already (each project's
# interface/ directory is part of the Cocoa PYTHONPATH).
sys.path.insert(0, os.environ["ROOTDIR"] + "/external_modules/code/cosmolike_core")
import cosmolike_notebook_utils as cnu
import cosmolike_roman_fourier_interface as ci

# OpenMP team size of the compiled loops, read once at import from
# OMP_NUM_THREADS (1 when unset). Some Python libraries lower the
# process-wide thread count without saying so, so _set_state restores it
# before every evaluation, as the likelihood does.
COSMOLIKE_OMP_THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))


# ----------------------------------------------------------------------
# Project fiducial point
# ----------------------------------------------------------------------
# The cosmology and the source nuisances: the sampler: evaluate: override
# block of EXAMPLE_EVALUATE1.yaml (cosmic shear). The linear galaxy bias
# of the lens bins: the same block of EXAMPLE_EVALUATE2.yaml (3x2pt),
# since the cosmic-shear yaml has no lens sample. The lens photo-z shifts
# equal the source ones (likelihood/params_lens.yaml ties them: one n(z)
# file serves both samples).
As_1e9 = 2.1
ns = 0.96605
H0 = 67.32
omegab = 0.04
omegam = 0.3
mnu = 0.06
w = -1.0
w0pwa = -1.0
roman_A1_1 = 0.6      # NLA amplitude A_IA
roman_A1_2 = -1.5     # NLA redshift power-law index eta_IA
roman_DZ_S1 = 0.0
roman_DZ_S2 = 0.0
roman_DZ_S3 = 0.0
roman_DZ_S4 = 0.0
roman_DZ_S5 = 0.0
roman_DZ_S6 = 0.0
roman_DZ_S7 = 0.0
roman_DZ_S8 = 0.0
roman_M1 = 0.0
roman_M2 = 0.0
roman_M3 = 0.0
roman_M4 = 0.0
roman_M5 = 0.0
roman_M6 = 0.0
roman_M7 = 0.0
roman_M8 = 0.0
roman_DZ_L1 = 0.0
roman_DZ_L2 = 0.0
roman_DZ_L3 = 0.0
roman_DZ_L4 = 0.0
roman_DZ_L5 = 0.0
roman_DZ_L6 = 0.0
roman_DZ_L7 = 0.0
roman_DZ_L8 = 0.0
roman_B1_1 = 1.4
roman_B1_2 = 1.5
roman_B1_3 = 1.6
roman_B1_4 = 1.7
roman_B1_5 = 1.8
roman_B1_6 = 1.9
roman_B1_7 = 2.0
roman_B1_8 = 2.1

# Default nuisance vectors built from the constants above; wrappers take
# None and fall back to these, so a call overrides one vector without
# retyping the rest. With IA_redshift_evolution = 3 the IA vectors hold
# parameter slots, not bins: slot 0 = amplitude, slot 1 = redshift index.
A1_FID = np.array([roman_A1_1, roman_A1_2, 0, 0, 0, 0, 0, 0], dtype=float)
A2_FID = np.zeros(8)
BTA_FID = np.zeros(8)
SHEAR_PHOTOZ_FID = np.array([roman_DZ_S1, roman_DZ_S2, roman_DZ_S3,
                             roman_DZ_S4, roman_DZ_S5, roman_DZ_S6,
                             roman_DZ_S7, roman_DZ_S8], dtype=float)
M_FID = np.array([roman_M1, roman_M2, roman_M3, roman_M4,
                  roman_M5, roman_M6, roman_M7, roman_M8], dtype=float)
LENS_PHOTOZ_FID = np.array([roman_DZ_L1, roman_DZ_L2, roman_DZ_L3,
                            roman_DZ_L4, roman_DZ_L5, roman_DZ_L6,
                            roman_DZ_L7, roman_DZ_L8], dtype=float)
B1_FID = np.array([roman_B1_1, roman_B1_2, roman_B1_3, roman_B1_4,
                   roman_B1_5, roman_B1_6, roman_B1_7, roman_B1_8],
                  dtype=float)
ZEROS8 = np.zeros(8)

# ----------------------------------------------------------------------
# Per-notebook configuration
# ----------------------------------------------------------------------
# The likelihood options of the roman_fourier likelihood blocks
# (likelihood/cosmic_shear.yaml and combo_3x2pt.yaml defaults, with the
# overrides of EXAMPLE_EVALUATE1.yaml), and the camb block of that yaml.
_CONFIG = {
    # Upper redshift, independent of sampling; matches this project YAML.
    "nonlimber_zmax": 5.0,
    # external_modules/data/roman_fourier is a link to this project's data/
    "path": os.environ["ROOTDIR"] + "/external_modules/data/roman_fourier",
    "data_file": "roman_example.dataset",
    # (lens, source) pairs left out of galaxy-galaxy lensing, counted
    # from 0: the source bin sits in front of or inside the lens bin
    "ggl_exclude": [[4, 0], [5, 0], [6, 0], [6, 1], [6, 2],
                    [7, 0], [7, 1], [7, 2], [7, 3]],
    "IA_model": 0,              # 0 = NLA, 1 = TATT
    "IA_redshift_evolution": 3,  # power law in (1 + z)
    "IA_code": 0,               # 0 = C FASTPT (NLA always uses 0)
    # one entry per bias term (b1, b2, bs2, b3, bmag, bK): 0 = a value
    # per lens bin, 1 in the b3 slot = b3 from b1 (bias.c)
    "bias_model": [0, 0, 0, 1, 0, 0],
    # n(z) conventions: interpolation 0 = cubic spline, z column 0 =
    # left bin edges; histogram_akima 0 = the midpoint curve
    "photoz_histogram_akima": 0,
    "photoz_histogram_akima_factor": 3,
    "photoz_interpolation_type": 0,
    "photoz_zmid_convention": 0,
    # C-FAST-PT convolution grid / output grid; 1.0 = equal
    "internal_accuracyboost": 1.0,
    # the comoving-distance grid of the non-Limber FFTLog integrals
    "nonlimber_accuracyboost": 1.0,
    # 0 = exact (non-Limber) projection below l = 150, 1 = Limber
    "adopt_limber_gs": 0,
    "adopt_limber_gg": 0,
    # non-Limber gg engine: 0 = FKEM, 1 = the low-rank engine
    "nonlimber_lowrank_gg": 0,
    "nonlimber_lowrank_rank": 5,
    "nonlimber_lowrank_lmax": 150,
    "include_HOD_GX": 0,        # 0 = perturbative galaxy bias
    "include_halo_IA": 0,       # 0 = the init_IA model
    "non_linear_emul": 2,       # 1 = EuclidEmulator2, 2 = halofit
    # the camb block of EXAMPLE_EVALUATE1.yaml: AccuracyBoost 1.05 and
    # k_per_logint 15; kmax = 10/Mpc is the likelihood's kmax_boltzmann
    # (the yaml's camb kmax of 7.5 sits below it). cnu.get_camb_cosmology
    # scales all three with its CAMB boost.
    "camb_accuracy_boost": 1.05,
    "camb_kmax": 10.0,
    "camb_k_per_logint": 15,
}

# filled by init_cosmolike: the HDF5 file with every hydro simulation of
# the dataset (used by init_baryons_contamination), and whether the
# covariance, mask and data vector are loaded (compute_chi2 stops the
# Python process without them)
allsims = None
_DATA_LOADED = False


def configure(**overrides):
    """Sets this notebook's yaml-mirroring options, once per notebook.

    Every keyword must already exist in _CONFIG; an unknown name is
    almost always a typo, so it raises instead of being stored
    silently. Options read by init_cosmolike (dataset, excluded pairs,
    model choices) take effect at the next init_cosmolike call.

    The endpoint can also change after initialization. Updating the compiled
    setting here makes the next prediction rebuild the affected projections,
    including helpers that prepare CAMB directly instead of using _set_state.
    It must be finite and at least 4.0; invalid values leave _CONFIG intact.

    Arguments:
      overrides = keyword form of any _CONFIG entry, e.g.
                  configure(non_linear_emul=1).

    Returns:
      nothing; later wrapper calls read the stored values.

    Raises:
      KeyError naming the unknown keyword and the valid names;
      ValueError for a photoz_histogram_akima other than 0 or 1,
      or a nonfinite nonlimber_zmax or one below 4.0.
    """
    for name in overrides:
        if name not in _CONFIG:
            raise KeyError(
                f"configure() got unknown option '{name}'; valid options: "
                + ", ".join(sorted(_CONFIG)))
    if "nonlimber_zmax" in overrides:
        endpoint = float(overrides["nonlimber_zmax"])
        if not np.isfinite(endpoint) or endpoint < 4.0:
            raise ValueError(
                f"nonlimber_zmax = {endpoint}; choose a finite value >= 4.0")

    if overrides.get("photoz_histogram_akima", 0) not in (0, 1):
        raise ValueError("configure(): photoz_histogram_akima must be 0 or 1")
    for name, value in overrides.items():
        _CONFIG[name] = value
    if "nonlimber_zmax" in overrides:
        ci.init_nonlimber_zmax(zmax=endpoint)


def current_config():
    """Return a copy of the yaml-mirroring options every wrapper reads.

    A copy, so a notebook can print or inspect the options without
    changing them (configure() is the one way to change them).

    Returns:
      dict {option name: value}, the _CONFIG entries.
    """
    return dict(_CONFIG)


def init_cosmolike(CLprobe=None, with_data=False):
    """Set up the compiled interface once per notebook session.

    Reads the dataset file (the small text file of key = value lines
    naming the n(z), covariance, mask and data-vector files and the
    multipole binning), then runs the init sequence of
    likelihood/_cosmolike_prototype_base.py in its order: the probes,
    the bands and the excluded pairs (they fix the data-vector
    layout), the run-time options, the n(z), the data and the IA
    model. The chi2 machinery (covariance, mask, data vector) only
    loads when asked, because the plotting cells never need it.

    Arguments:
      CLprobe   = "xi" (cosmic shear: the shared code's name, from the
                  real-space xi_+/-, selects C_ss here), "2x2pt",
                  "3x2pt", ..., or None to skip init_probes for a
                  plotting-only session.
      with_data = True also loads covariance, mask and data vector,
                  which get_chi2 and get_dv need; it requires CLprobe.

    Returns:
      the parsed IniFile, for notebooks that read extra entries.

    Raises:
      ValueError when with_data is True and CLprobe is None: the mask
      and the data-vector layout follow the probe selection.

    Side effects:
      replaces the compiled interface's global state and sets this
      module's allsims to the hydro-simulation file of the dataset
      (None when the dataset names none).
    """
    global allsims
    global _DATA_LOADED
    if with_data and CLprobe is None:
        raise ValueError(
            "init_cosmolike(with_data=True) needs CLprobe: the mask and the "
            "data-vector layout follow the probe selection; pass e.g. "
            "CLprobe='xi' for cosmic shear")
    dataset = os.path.normpath(os.path.join(_CONFIG["path"],
                                            _CONFIG["data_file"]))
    ini = IniFile(dataset)
    if "all_sims_hdf5_file" in ini.params:
        allsims = ini.relativeFileName("all_sims_hdf5_file")
    else:
        allsims = None

    ci.initial_setup()
    ci.init_nonlimber_zmax(zmax=float(_CONFIG["nonlimber_zmax"]))
    if CLprobe is not None:
        ci.init_probes(possible_probes=CLprobe)
    ci.init_binning(int(ini.int("n_cl")),
                    int(ini.float("l_min")),
                    int(ini.float("l_max")),
                    int(ini.float("l_max_shear")))
    # flatten() turns the [[lens, source], ...] pair list into the
    # flat [l0, s0, l1, s1, ...] array the C layer expects
    ggl_exclude = np.array(_CONFIG["ggl_exclude"], dtype=int).flatten()
    ci.init_ggl_exclude(ggl_exclude)
    ci.set_log_level_info()
    ci.init_photoz_histogram_akima(
        enabled=int(_CONFIG["photoz_histogram_akima"]),
        preparation_factor=int(_CONFIG["photoz_histogram_akima_factor"]))
    ci.init_photoz_conventions(
        interpolation_type=int(_CONFIG["photoz_interpolation_type"]),
        zmid_convention=int(_CONFIG["photoz_zmid_convention"]))
    ci.init_fpt_internal_boost(
        internal_boost=float(_CONFIG["internal_accuracyboost"]))
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=float(_CONFIG["nonlimber_accuracyboost"]))
    ci.init_adopt_limber_gs(adopt_limber_gs=int(_CONFIG["adopt_limber_gs"]))
    ci.init_adopt_limber_gg(adopt_limber_gg=int(_CONFIG["adopt_limber_gg"]))
    ci.init_include_HOD_GX(include_HOD_GX=int(_CONFIG["include_HOD_GX"]))
    ci.init_include_halo_IA(include_halo_IA=int(_CONFIG["include_halo_IA"]))
    ci.init_lowrank_nonlimber_gg(
        flag=int(_CONFIG["nonlimber_lowrank_gg"]),
        nrank=int(_CONFIG["nonlimber_lowrank_rank"]),
        lmax=int(_CONFIG["nonlimber_lowrank_lmax"]))
    ci.init_accuracy_boost(accuracy_boost=1.0, integration_accuracy=0)
    # is_linear=False: cosmolike uses the nonlinear P(k) it is handed
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=ini.relativeFileName("nz_lens_file"),
        lens_ntomo=int(ini.int("lens_ntomo")),
        source_multihisto_file=ini.relativeFileName("nz_source_file"),
        source_ntomo=int(ini.int("source_ntomo")))
    if with_data:
        ci.init_data_fourier(ini.relativeFileName("cov_file"),
                             ini.relativeFileName("mask_file"),
                             ini.relativeFileName("data_file"))
    _DATA_LOADED = with_data
    ci.init_IA(ia_model=int(_CONFIG["IA_model"]),
               ia_redshift_evolution=int(_CONFIG["IA_redshift_evolution"]),
               ia_code=int(_CONFIG["IA_code"]))
    if CLprobe is not None and CLprobe != "xi":
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    return ini


def band_centers():
    """Return the band centers of the dataset's multipole binning.

    The data vector evaluates every spectrum at the logarithmic center
    of each of the n_cl bands between l_min and l_max; init_binning
    (called by init_cosmolike) computes them.

    Returns:
      1D float array of the n_cl band centers (multipoles).
    """
    return np.array(ci.get_binning_fourier_space(), dtype=float)


def _resolve_ell(ell):
    """Return ell as a float array, or the band centers for None."""
    if ell is None:
        return band_centers()
    return np.asarray(ell, dtype=float)


def _set_state(omegam, omegab, H0, ns, As_1e9, w, w0pwa,
               AccuracyBoost, kmax, k_per_logint, CAMBAccuracyBoost,
               CLAccuracyBoost, CLIntegrationAccuracy, non_linear_emul,
               M=None, shear_photoz_bias=None, A1=None, A2=None, BTA=None,
               lens_photoz_bias=None, B1=None, B2=None, B_MAG=None,
               B3nl=None, BK=None, baryon_sims=None, allsims_file=None):
    """Runs CAMB and pushes one complete state into the interface.

    This is the body every wrapper shares. The compiled interface
    keeps global state, so the sequence rebuilds everything a
    spectrum call reads: the OpenMP team, the accuracy settings, the
    cosmology (power spectra, growth, distances from one CAMB run),
    and each nuisance group whose vectors were passed. A group passed
    as None is skipped, which leaves that part of the state at
    whatever the interface holds (a cosmic-shear wrapper never touches
    galaxy bias).

    Arguments:
      omegam, omegab, H0, ns, As_1e9, w, w0pwa, AccuracyBoost, kmax,
      k_per_logint, CAMBAccuracyBoost, CLAccuracyBoost, non_linear_emul
               = the cosmology and accuracy arguments, forwarded to
                 cnu.get_camb_cosmology together with the module
                 constant mnu (kmax in 1/Mpc, before the CAMB boost;
                 None entries take the _CONFIG CAMB settings).
      CLIntegrationAccuracy = cosmolike's integration accuracy before
                 the boost rule below.
      M, shear_photoz_bias, A1, A2, BTA = shear nuisance vectors.
      lens_photoz_bias, B1, B2, B_MAG, B3nl, BK = clustering nuisance
                 vectors; passing B1 also selects the configured
                 galaxy-bias model.
      baryon_sims = a hydro simulation name to contaminate the matter
                 power with, or None to reset that state.
      allsims_file = HDF5 file for baryon_sims, or None for the one
                 init_cosmolike recorded.

    Returns:
      nothing; the interface state is the result.

    Raises:
      ValueError when baryon_sims is given but no HDF5 file is known.
    """
    ci.init_nonlimber_zmax(zmax=float(_CONFIG["nonlimber_zmax"]))
    if kmax is None:
        kmax = _CONFIG["camb_kmax"]
    if k_per_logint is None:
        k_per_logint = _CONFIG["camb_k_per_logint"]
    if CAMBAccuracyBoost is None:
        CAMBAccuracyBoost = _CONFIG["camb_accuracy_boost"]
    if non_linear_emul is None:
        non_linear_emul = _CONFIG["non_linear_emul"]
    if allsims_file is None:
        allsims_file = allsims
    if baryon_sims is not None and allsims_file is None:
        raise ValueError(
            f"baryon_sims = '{baryon_sims}' needs the HDF5 file of the hydro "
            f"simulations, but the dataset {_CONFIG['data_file']} names no "
            "all_sims_hdf5_file: pass allsims=<path to baryons_logPkR.h5>")

    (log10k_interp_2D, z_interp_2D, lnPL, lnPNL,
     G_growth, z_growth, z_interp_1D, chi,
     omegan2, lnPL_cb) = cnu.get_camb_cosmology(
        omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
        w=w, w0pwa=w0pwa, mnu=mnu, AccuracyBoost=AccuracyBoost,
        kmax=kmax, k_per_logint=k_per_logint,
        CAMBAccuracyBoost=CAMBAccuracyBoost,
        CLAccuracyBoost=CLAccuracyBoost,
        non_linear_emul=non_linear_emul)
    ci.set_omp_threads(n=COSMOLIKE_OMP_THREADS)
    # the accuracy rule shared by every wrapper: the overall boost
    # multiplies the cosmolike boost, and the integration accuracy grows
    # with it
    CLAccuracyBoost = CLAccuracyBoost*AccuracyBoost
    CLIntegrationAccuracy = max(
        0, CLIntegrationAccuracy + abs(3*(CLAccuracyBoost - 1.0)))
    ci.init_photoz_histogram_akima(
        enabled=int(_CONFIG["photoz_histogram_akima"]),
        preparation_factor=int(_CONFIG["photoz_histogram_akima_factor"]))
    ci.init_photoz_conventions(
        interpolation_type=int(_CONFIG["photoz_interpolation_type"]),
        zmid_convention=int(_CONFIG["photoz_zmid_convention"]))
    # init_fpt_internal_boost comes first, as in the likelihood:
    # init_accuracy_boost sets the C-FAST-PT internal grid fraction to
    # base x CLAccuracyBoost, where base is the fraction it found at its
    # first call in the process; a fraction set after init_accuracy_boost
    # would discard the boost
    ci.init_fpt_internal_boost(
        internal_boost=float(_CONFIG["internal_accuracyboost"]))
    ci.init_accuracy_boost(accuracy_boost=CLAccuracyBoost,
                           integration_accuracy=int(CLIntegrationAccuracy))
    if B1 is not None:
        ci.init_bias(bias_model=_CONFIG["bias_model"])
    # the growth table has its own z grid (z_growth, the dense 1D grid
    # cut at the last z_2D node), handed over as z_G, as the likelihood
    # does
    ci.set_cosmology(omegam=omegam,
                     H0=H0,
                     log10k_2D=log10k_interp_2D,
                     z_2D=z_interp_2D,
                     lnP_linear=lnPL,
                     lnP_nonlinear=lnPNL,
                     G=G_growth,
                     z_1D=z_interp_1D,
                     chi=chi,
                     omegab=omegab,
                     z_G=z_growth,
                     omegan2=omegan2,
                     lnP_linear_cb=lnPL_cb)
    # numpy arrays, not Python lists: the binding converts long lists
    # element by element and a numpy array in one block
    if M is not None:
        ci.set_nuisance_shear_calib(M=np.asarray(M, dtype=float))
    if shear_photoz_bias is not None:
        ci.set_nuisance_shear_photoz(
            bias=np.asarray(shear_photoz_bias, dtype=float))
    if A1 is not None:
        ci.set_nuisance_ia(A1=np.asarray(A1, dtype=float),
                           A2=np.asarray(A2, dtype=float),
                           B_TA=np.asarray(BTA, dtype=float))
    if lens_photoz_bias is not None:
        ci.set_nuisance_clustering_photoz(
            bias=np.asarray(lens_photoz_bias, dtype=float))
    if B1 is not None:
        ci.set_nuisance_bias(B1=np.asarray(B1, dtype=float),
                             B2=np.asarray(B2, dtype=float),
                             B_MAG=np.asarray(B_MAG, dtype=float),
                             B3nl=np.asarray(B3nl, dtype=float),
                             BK=np.asarray(BK, dtype=float))
    if baryon_sims is None:
        ci.reset_bary_struct()
    else:
        ci.init_baryons_contamination(sim=baryon_sims, allsims=allsims_file)


def _shear_defaults(M, shear_photoz_bias, A1, A2, BTA):
    """Replaces None shear vectors with the fiducial ones.

    Returns:
      (M, shear_photoz_bias, A1, A2, BTA), in the argument order.
    """
    if M is None:
        M = M_FID
    if shear_photoz_bias is None:
        shear_photoz_bias = SHEAR_PHOTOZ_FID
    if A1 is None:
        A1 = A1_FID
    if A2 is None:
        A2 = A2_FID
    if BTA is None:
        BTA = BTA_FID
    return M, shear_photoz_bias, A1, A2, BTA


def _clustering_defaults(lens_photoz_bias, B1, B2, B_MAG, B3nl, BK):
    """Replaces None clustering vectors with the fiducial ones.

    B2, B_MAG, B3nl and BK default to zeros (ZEROS8).

    Returns:
      (lens_photoz_bias, B1, B2, B_MAG, B3nl, BK), in the argument order.
    """
    if lens_photoz_bias is None:
        lens_photoz_bias = LENS_PHOTOZ_FID
    if B1 is None:
        B1 = B1_FID
    if B2 is None:
        B2 = ZEROS8
    if B_MAG is None:
        B_MAG = ZEROS8
    if B3nl is None:
        B3nl = ZEROS8
    if BK is None:
        BK = ZEROS8
    return lens_photoz_bias, B1, B2, B_MAG, B3nl, BK


def C_ss_tomo_limber(ell=None, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                     As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                     A1=None, A2=None, BTA=None,
                     shear_photoz_bias=None, M=None,
                     baryon_sims=None, AccuracyBoost=1.0, kmax=None,
                     k_per_logint=None, CAMBAccuracyBoost=None,
                     CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                     non_linear_emul=None, allsims=None):
    """Cosmic-shear angular power spectra (EE, BB) at multipoles ell.

    Rebuilds the shear state (see _set_state) and evaluates
    ci.C_ss_tomo_limber, the Limber projection the data vector uses.

    Arguments:
      ell = 1D array of multipoles, or None for the band centers; the
      rest as listed in the module docstring, with the shear group only
      (this wrapper never touches clustering).

    Returns:
      (EE, BB): two 3D arrays (n_ell, n_source, n_source), the E- and
      B-mode spectra of every source-bin pair; B modes vanish for NLA.
    """
    ell = _resolve_ell(ell)
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    _set_state(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
               w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost, kmax=kmax,
               k_per_logint=k_per_logint,
               CAMBAccuracyBoost=CAMBAccuracyBoost,
               CLAccuracyBoost=CLAccuracyBoost,
               CLIntegrationAccuracy=CLIntegrationAccuracy,
               non_linear_emul=non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.C_ss_tomo_limber(l=ell)


def C_gs_tomo_limber(ell=None, omegam=omegam, omegab=omegab, H0=H0, ns=ns,
                     As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                     A1=None, A2=None, BTA=None,
                     shear_photoz_bias=None, M=None,
                     lens_photoz_bias=None, galaxy_bias_b1=None,
                     galaxy_bias_b2=None, galaxy_bias_bmag=None,
                     galaxy_bias_b3nl=None, galaxy_bias_bk=None,
                     baryon_sims=None, AccuracyBoost=1.0, kmax=None,
                     k_per_logint=None, CAMBAccuracyBoost=None,
                     CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                     non_linear_emul=None, allsims=None):
    """Galaxy-galaxy lensing spectra C_gs at multipoles ell.

    Builds both the shear and the clustering state (the source and the
    lens samples both enter galaxy-galaxy lensing) and evaluates
    ci.C_gs_tomo_limber, the Limber projection. Pairs dropped at init
    time via ggl_exclude come back as identically zero.

    Arguments:
      ell = 1D array of multipoles, or None for the band centers; the
      rest as listed in the module docstring.

    Returns:
      3D array (n_ell, n_lens, n_source).
    """
    ell = _resolve_ell(ell)
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
               w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost, kmax=kmax,
               k_per_logint=k_per_logint,
               CAMBAccuracyBoost=CAMBAccuracyBoost,
               CLAccuracyBoost=CLAccuracyBoost,
               CLIntegrationAccuracy=CLIntegrationAccuracy,
               non_linear_emul=non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    return ci.C_gs_tomo_limber(l=ell)


def C_gg_tomo(ell=None, limber=0, omegam=omegam, omegab=omegab, H0=H0,
              ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa,
              lens_photoz_bias=None, galaxy_bias_b1=None,
              galaxy_bias_b2=None, galaxy_bias_bmag=None,
              galaxy_bias_b3nl=None, galaxy_bias_bk=None,
              baryon_sims=None, AccuracyBoost=1.0, kmax=None,
              k_per_logint=None, CAMBAccuracyBoost=None,
              CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
              non_linear_emul=None, allsims=None):
    """Galaxy-clustering spectra C_gg at multipoles ell.

    Clustering state only. limber = 1 evaluates the Limber
    approximation at every multipole (ci.C_gg_tomo_limber); anything
    else evaluates ci.C_gg_tomo, the computation of the data vector
    (non-Limber below l = 150 with adopt_limber_gg = 0), so the two can
    be compared on the same state.

    Arguments:
      ell = 1D array of multipoles, or None for the band centers;
      limber = 1 or another value (see above); the rest as listed in
      the module docstring.

    Returns:
      3D array (n_ell, n_lens, n_lens).
    """
    ell = _resolve_ell(ell)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
               w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost, kmax=kmax,
               k_per_logint=k_per_logint,
               CAMBAccuracyBoost=CAMBAccuracyBoost,
               CLAccuracyBoost=CLAccuracyBoost,
               CLIntegrationAccuracy=CLIntegrationAccuracy,
               non_linear_emul=non_linear_emul,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    if limber == 1:
        return ci.C_gg_tomo_limber(l=ell)
    return ci.C_gg_tomo(l=ell)


def get_chi2(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
             w=w, w0pwa=w0pwa, A1=None, A2=None, BTA=None,
             shear_photoz_bias=None, M=None,
             lens_photoz_bias=None, galaxy_bias_b1=None,
             galaxy_bias_b2=None, galaxy_bias_bmag=None,
             galaxy_bias_b3nl=None, galaxy_bias_bk=None,
             baryon_sims=None, AccuracyBoost=1.0, kmax=None,
             k_per_logint=None, CAMBAccuracyBoost=None,
             CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
             non_linear_emul=None, allsims=None):
    """Return the chi2 of the masked theory vector against the loaded data.

    Requires init_cosmolike(CLprobe=..., with_data=True) first: the
    probe selection fixes which blocks enter the masked vector, and
    with_data loads the covariance, mask and data vector this chi2
    compares against. The full 3x2pt nuisance state is set every call;
    blocks outside the selected probes never read theirs (a "xi" run
    ignores the clustering state).

    Arguments:
      as listed in the module docstring.

    Returns:
      float chi2 = (d - t)^T C^-1 (d - t) over the entries the mask
      keeps, d = the stored data vector, t = the theory vector.

    Raises:
      RuntimeError when no data vector is loaded: compute_chi2 would
      otherwise stop the whole Python process.
    """
    if not _DATA_LOADED:
        raise RuntimeError(
            "get_chi2: no covariance, mask and data vector are loaded; run "
            "nw.init_cosmolike(CLprobe='xi', with_data=True) first")
    M, shear_photoz_bias, A1, A2, BTA = _shear_defaults(
        M, shear_photoz_bias, A1, A2, BTA)
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
               w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost, kmax=kmax,
               k_per_logint=k_per_logint,
               CAMBAccuracyBoost=CAMBAccuracyBoost,
               CLAccuracyBoost=CLAccuracyBoost,
               CLIntegrationAccuracy=CLIntegrationAccuracy,
               non_linear_emul=non_linear_emul,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2, BTA=BTA,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk,
               baryon_sims=baryon_sims, allsims_file=allsims)
    datavector = np.array(ci.compute_data_vector_masked(), dtype=float)
    return ci.compute_chi2(datavector=datavector)


# ----------------------------------------------------------------------
# UV-fraction scale cuts of C_gg (arXiv:2603.13031)
# ----------------------------------------------------------------------
# Each C_gg band power splits into C_IR, computed with the 3D power
# spectrum multiplied by a step that keeps only k <= k_max/D(z), and
# C_UV = C - C_IR. k_max is in h/Mpc at z = 0 and D(z) is the linear
# growth factor with D(0) = 1, so the cut moves to larger k at higher
# redshift, where structure is less nonlinear. A band is kept while
# |C_UV|/(|C_UV| + |C_IR|) < alpha; the kept bands of each spectrum are a
# prefix in l that only shortens as k_max decreases. The compiled
# interface aborts the process on a bad k_max array, so the wrappers
# check it first.
def _check_kmax_grid(kmax):
    """Return kmax as a validated 1D float array.

    Raises:
      ValueError unless kmax is a non-empty 1D array of finite,
      positive, strictly ascending values (h/Mpc at z = 0).
    """
    kmax = np.asarray(kmax, dtype=float)
    if kmax.ndim != 1 or len(kmax) < 1:
        raise ValueError(
            f"kmax has shape {kmax.shape}: need a non-empty 1D array of "
            "k_max values in h/Mpc at z = 0")
    if not np.all(np.isfinite(kmax)) or np.any(kmax <= 0):
        raise ValueError(
            f"kmax = {kmax}: every k_max must be finite and > 0 (h/Mpc)")
    if np.any(np.diff(kmax) <= 0):
        raise ValueError(
            f"kmax = {kmax}: the values must be strictly ascending; sort "
            "the grid (np.sort) and drop duplicates")
    return kmax


def gg_scale_cuts_tomo(kmax, alpha=0.05, omegam=omegam, omegab=omegab,
                       H0=H0, ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                       lens_photoz_bias=None, galaxy_bias_b1=None,
                       galaxy_bias_b2=None, galaxy_bias_bmag=None,
                       galaxy_bias_b3nl=None, galaxy_bias_bk=None,
                       AccuracyBoost=1.0, kmax_camb=None,
                       k_per_logint=None, CAMBAccuracyBoost=None,
                       CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                       non_linear_emul=None):
    """Return the UV-fraction scale cuts of C_gg at the band centers.

    Evaluates the model at the given point (the clustering state of
    C_gg_tomo, built by _set_state) and calls ci.gg_scale_cuts_tomo,
    which splits every C_gg band power into its IR and UV parts at each
    k_max and keeps a band while |C_UV|/(|C_UV| + |C_IR|) < alpha. The
    step multiplies P(k) inside every gg engine of the data vector, the
    non-Limber projection below l = 150 included, so no Limber relation
    between l and k enters the cut.

    Arguments:
      kmax  = 1D array of cuts in h/Mpc at z = 0, strictly ascending;
              bin i is cut at kmax/D(z) inside its redshift range.
      alpha = allowed UV fraction, in (0, 1); 0.05 is the baseline of
              arXiv:2603.13031.
      kmax_camb = CAMB's maximum k in 1/Mpc (the kmax argument of the
              other wrappers, renamed here), or None for _CONFIG.
      the rest as listed in the module docstring (clustering group).

    Returns:
      dict of numpy arrays, rows s = the lens-bin autos (then the gg
      cross pairs, none in this project):
        kmax        (n_k,)              the cuts, as passed
        alpha       float
        ell         (n_ell,)            the band centers
        nautos      int                 rows 0 .. nautos - 1 are autos
        lmax        (n_spec, n_k)       center of the last band kept
                                        (0 when no band survives)
        mask        (n_spec, n_ell, n_k) 1 = band kept, 0 = cut
        uv_fraction (n_spec, n_ell, n_k) |C_UV|/(|C_UV| + |C_IR|)
        full        (n_spec, n_ell)     the full spectra C_gg
        ir          (n_spec, n_ell, n_k) the IR parts C_IR
        uv          (n_spec, n_ell, n_k) the UV parts, full - ir

    Raises:
      ValueError for a malformed kmax grid or an alpha outside (0, 1),
      before any computation.
    """
    kmax = _check_kmax_grid(kmax)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha = {alpha}: the allowed UV fraction must "
                         "lie in (0, 1); the baseline is 0.05")
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
               w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost,
               kmax=kmax_camb, k_per_logint=k_per_logint,
               CAMBAccuracyBoost=CAMBAccuracyBoost,
               CLAccuracyBoost=CLAccuracyBoost,
               CLIntegrationAccuracy=CLIntegrationAccuracy,
               non_linear_emul=non_linear_emul,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk)
    raw = ci.gg_scale_cuts_tomo(kmax=kmax, alpha=float(alpha))
    cuts = {}
    # the binding returns the two vectors as (n, 1) columns: ravel
    # flattens them to the 1D arrays the docstring states
    for name in ("kmax", "ell"):
        cuts[name] = np.array(raw[name], dtype=float).ravel()
    for name in ("lmax", "mask", "uv_fraction", "full", "ir"):
        cuts[name] = np.array(raw[name], dtype=float)
    cuts["alpha"] = float(raw["alpha"])
    cuts["nautos"] = int(raw["nautos"])
    # full has no k_max axis: [:, :, None] adds a length-1 axis so the
    # subtraction broadcasts over every k_max column of ir
    cuts["uv"] = cuts["full"][:, :, None] - cuts["ir"]
    return cuts


def C_gg_kwindow_tomo(kmax, keep_ir=1, omegam=omegam, omegab=omegab,
                      H0=H0, ns=ns, As_1e9=As_1e9, w=w, w0pwa=w0pwa,
                      lens_photoz_bias=None, galaxy_bias_b1=None,
                      galaxy_bias_b2=None, galaxy_bias_bmag=None,
                      galaxy_bias_b3nl=None, galaxy_bias_bk=None,
                      AccuracyBoost=1.0, kmax_camb=None,
                      k_per_logint=None, CAMBAccuracyBoost=None,
                      CLAccuracyBoost=1.0, CLIntegrationAccuracy=0,
                      non_linear_emul=None):
    """Return the IR or UV part of C_gg at one k_max, at the band centers.

    Same state as gg_scale_cuts_tomo, then ci.C_gg_kwindow_tomo: with
    keep_ir = 1 the spectra computed from modes k <= kmax/D(z) only,
    with keep_ir = 0 the rest; kmax <= 0 returns the full spectra. The
    IR and UV parts add up to the full spectra, which is the check this
    function exists for.

    Arguments:
      kmax    = the cut in h/Mpc at z = 0 (a single number).
      keep_ir = 1 for the IR part, 0 for the UV part.
      kmax_camb and the rest as in gg_scale_cuts_tomo.

    Returns:
      (autos, pairs): 2D arrays (n_lens, n_ell) and (n_pairs, n_ell),
      n_pairs = 0 in this project.

    Raises:
      ValueError for keep_ir other than 0 or 1 (the compiled interface
      would stop the process).
    """
    if keep_ir not in (0, 1):
        raise ValueError(f"keep_ir = {keep_ir}: use 1 (IR part) or 0 "
                         "(UV part)")
    (lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2, galaxy_bias_bmag,
     galaxy_bias_b3nl, galaxy_bias_bk) = _clustering_defaults(
        lens_photoz_bias, galaxy_bias_b1, galaxy_bias_b2,
        galaxy_bias_bmag, galaxy_bias_b3nl, galaxy_bias_bk)
    _set_state(omegam=omegam, omegab=omegab, H0=H0, ns=ns, As_1e9=As_1e9,
               w=w, w0pwa=w0pwa, AccuracyBoost=AccuracyBoost,
               kmax=kmax_camb, k_per_logint=k_per_logint,
               CAMBAccuracyBoost=CAMBAccuracyBoost,
               CLAccuracyBoost=CLAccuracyBoost,
               CLIntegrationAccuracy=CLIntegrationAccuracy,
               non_linear_emul=non_linear_emul,
               lens_photoz_bias=lens_photoz_bias, B1=galaxy_bias_b1,
               B2=galaxy_bias_b2, B_MAG=galaxy_bias_bmag,
               B3nl=galaxy_bias_b3nl, BK=galaxy_bias_bk)
    (autos, pairs) = ci.C_gg_kwindow_tomo(kmax=float(kmax),
                                          keep_ir=int(keep_ir))
    return (np.array(autos, dtype=float), np.array(pairs, dtype=float))


def C_gg_kwindow_white_noise(kmax):
    """Return the white-noise correction of the low-rank IR C_gg at kmax.

    A diagnostic of the model of the last evaluation (no new state is
    built): the correction the low-rank non-Limber engine adds to the IR
    part when it runs with one-loop galaxies. With this project's
    defaults (the FKEM engine, linear bias) it is zero.

    Arguments:
      kmax = the cut in h/Mpc at z = 0, > 0.

    Returns:
      (autos, pairs): 2D arrays (n_lens, n_ell) and (n_pairs, n_ell).

    Raises:
      ValueError for kmax <= 0 (the compiled interface would stop the
      process).
    """
    if not (kmax > 0):
        raise ValueError(f"kmax = {kmax}: must be > 0 (h/Mpc at z = 0)")
    (autos, pairs) = ci.C_gg_kwindow_white_noise(kmax=float(kmax))
    return (np.array(autos, dtype=float), np.array(pairs, dtype=float))


# ----------------------------------------------------------------------
# Fisher forecasting (cosmic shear)
# ----------------------------------------------------------------------
# One flat parameter vector drives the Fisher machinery: the sampled
# cosmology (with the dark-energy pair w0 and wa), the two NLA numbers,
# the eight source photo-z shifts and the eight shear calibrations, in
# this order. The names, LaTeX labels and priors below index into the
# same vector.
FISHER_PARAM_FID = np.array(
    [As_1e9, ns, H0, omegab, omegam, w, w0pwa - w,
     roman_A1_1, roman_A1_2,
     roman_DZ_S1, roman_DZ_S2, roman_DZ_S3, roman_DZ_S4,
     roman_DZ_S5, roman_DZ_S6, roman_DZ_S7, roman_DZ_S8,
     roman_M1, roman_M2, roman_M3, roman_M4,
     roman_M5, roman_M6, roman_M7, roman_M8], dtype="float64")

# positions of the dark-energy parameters w0 and wa in the vector (the
# pair of the dark-energy figure of merit)
FISHER_INDEX_W0 = 5
FISHER_INDEX_WA = 6

FISHER_PARAM_NAMES = [
    "As_1e9", "ns", "H0", "omegab", "omegam", "w0", "wa",
    "roman_A1_1", "roman_A1_2",
    "zshift_1", "zshift_2", "zshift_3", "zshift_4", "zshift_5",
    "zshift_6", "zshift_7", "zshift_8",
    "M_1", "M_2", "M_3", "M_4", "M_5", "M_6", "M_7", "M_8",
]

FISHER_PARAM_LABELS = [
    r"10^{9} A_s", r"n_s", r"H_0", r"\Omega_b", r"\Omega_m", r"w_0", r"w_a",
    r"A_\mathrm{1IA,roman}^1", r"A_\mathrm{1IA,roman}^2",
    r"\Delta z_\mathrm{s,roman}^1", r"\Delta z_\mathrm{s,roman}^2",
    r"\Delta z_\mathrm{s,roman}^3", r"\Delta z_\mathrm{s,roman}^4",
    r"\Delta z_\mathrm{s,roman}^5", r"\Delta z_\mathrm{s,roman}^6",
    r"\Delta z_\mathrm{s,roman}^7", r"\Delta z_\mathrm{s,roman}^8",
    r"m_\mathrm{roman}^1", r"m_\mathrm{roman}^2", r"m_\mathrm{roman}^3",
    r"m_\mathrm{roman}^4", r"m_\mathrm{roman}^5", r"m_\mathrm{roman}^6",
    r"m_\mathrm{roman}^7", r"m_\mathrm{roman}^8",
]

# Flat priors {index: (min, max)} of EXAMPLE_EVALUATE1.yaml on the
# parameters without Gaussian priors (As_1e9, ns, H0, omegab, omegam, w0,
# wa and the two NLA numbers); they truncate the getdist sample clouds, so
# the contours stay inside them. The yaml samples w0 in [-3, -0.01] and
# w0 + wa in [-5, -0.01]; at the fiducial w0 = -1 the second range is wa
# in [-4, 0.99].
FISHER_FLAT_PRIORS = {
    0: (0.5, 5),
    1: (0.87, 1.07),
    2: (55.0, 91.0),
    3: (0.03, 0.07),
    4: (0.1, 0.9),
    5: (-3.0, -0.01),
    6: (-4.0, 0.99),
    7: (-5.0, 5.0),
    8: (-5.0, 5.0),
}

# Gaussian priors {index: (mean, sigma)} of EXAMPLE_EVALUATE1.yaml:
# sigma = 0.005 on the eight shear calibrations M (indices 17..24) and
# 0.002 on the eight source photo-z shifts (indices 9..16)
FISHER_GAUSSIAN_PRIORS = {
    17: (0.0, 0.005), 18: (0.0, 0.005), 19: (0.0, 0.005),
    20: (0.0, 0.005), 21: (0.0, 0.005), 22: (0.0, 0.005),
    23: (0.0, 0.005), 24: (0.0, 0.005),
    9: (0.0, 0.002), 10: (0.0, 0.002), 11: (0.0, 0.002),
    12: (0.0, 0.002), 13: (0.0, 0.002), 14: (0.0, 0.002),
    15: (0.0, 0.002), 16: (0.0, 0.002),
}

# Random-number generator of the getdist sample clouds of the Fisher
# contours. Seed 0 makes a fresh kernel reproduce the same figures; being
# module-level, it makes repeated plot calls draw successive parts of one
# stream instead of identical sample noise.
_FISHER_RNG = np.random.default_rng(0)


def fisher_fiducial_point():
    """Return a fresh copy of the Fisher fiducial vector.

    A copy lets a notebook shift entries for a forecast without editing
    the module's fiducial in place.

    Returns:
      1D float64 array in the FISHER_PARAM_NAMES layout.
    """
    return FISHER_PARAM_FID.copy()


def get_dv(param=None, AccuracyBoost=1.0):
    """Return the masked cosmic-shear data vector at a flat parameter vector.

    The Fisher derivatives evaluate this at shifted copies of the
    fiducial vector; the layout is the one FISHER_PARAM_NAMES
    documents; w0 and wa enter CAMB as w = w0 and w0pwa = w0 + wa.
    Requires init_cosmolike(CLprobe="xi", with_data=True).

    Arguments:
      param = 1D float array in the FISHER_PARAM_NAMES layout, or None
              for the fiducial vector.
      AccuracyBoost = the overall accuracy boost, forwarded to
              _set_state as AccuracyBoost (it also multiplies the
              cosmolike boost and a third of it reaches CAMB).

    Returns:
      1D float64 array: the masked data vector (zeros where the mask
      removes an entry).

    Raises:
      RuntimeError when no data vector is loaded.
    """
    if not _DATA_LOADED:
        raise RuntimeError(
            "get_dv: no mask is loaded; run "
            "nw.init_cosmolike(CLprobe='xi', with_data=True) first")
    if param is None:
        param = FISHER_PARAM_FID
    w0 = param[FISHER_INDEX_W0]
    wa = param[FISHER_INDEX_WA]
    A1 = np.array([param[7], param[8], 0, 0, 0, 0, 0, 0], dtype=float)
    shear_photoz_bias = np.array(param[9:17], dtype=float)
    M = np.array(param[17:25], dtype=float)
    _set_state(omegam=param[4], omegab=param[3], H0=param[2], ns=param[1],
               As_1e9=param[0], w=w0, w0pwa=w0 + wa,
               AccuracyBoost=AccuracyBoost, kmax=None, k_per_logint=None,
               CAMBAccuracyBoost=None, CLAccuracyBoost=1.0,
               CLIntegrationAccuracy=0, non_linear_emul=None,
               M=M, shear_photoz_bias=shear_photoz_bias,
               A1=A1, A2=A2_FID, BTA=BTA_FID)
    return np.array(ci.compute_data_vector_masked(), dtype=np.float64)


def get_ddv(index=0, h=0.02, CV=None, AccuracyBoost=1.0):
    """Return the derivative of get_dv along one parameter (cnu.get_ddv).

    Five-point finite-difference stencil with relative step h around CV.

    Arguments:
      index = position of the parameter in the FISHER_PARAM_NAMES layout.
      h     = relative step (0.02 = two percent; absolute when the
              fiducial value is zero).
      CV    = fiducial parameter vector, or None for FISHER_PARAM_FID.
      AccuracyBoost = forwarded to every get_dv evaluation.

    Returns:
      1D array, d(masked data vector)/d(parameter).
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    return cnu.get_ddv(get_dv, index=index, h=h, CV=CV,
                       AccuracyBoost=AccuracyBoost)


def get_Fisher(CV=None, h=0.02, AccuracyBoost=3.1, ddv=None,
               priors=None, invcov=None):
    """Return the Fisher matrix from the five-point-stencil derivatives.

    Thin binding of cnu.get_Fisher to this project's data vector,
    priors and masked inverse covariance (fetched from the interface
    when not passed, so init_cosmolike with data must have run).

    Arguments:
      CV     = fiducial parameter vector, or None for FISHER_PARAM_FID.
      h      = relative derivative step (see get_ddv).
      AccuracyBoost = forwarded to every data-vector evaluation.
      ddv    = derivative function, or None for get_ddv.
      priors = {index: (mean, sigma)}, or None for
               FISHER_GAUSSIAN_PRIORS.
      invcov = inverse covariance of the masked data vector, or None
               for ci.get_inv_cov_masked().

    Returns:
      the Fisher matrix, an (n_params, n_params) array.
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    if ddv is None:
        ddv = get_ddv
    if priors is None:
        priors = FISHER_GAUSSIAN_PRIORS
    if invcov is None:
        invcov = ci.get_inv_cov_masked()
    return cnu.get_Fisher(ddv, CV=CV, h=h, AccuracyBoost=AccuracyBoost,
                          priors=priors, invcov=invcov)


def get_ddv_dkit(index=0, CV=None, AccuracyBoost=1.0, min_samples=7,
                 fallback_mode="poly_at_floor"):
    """Return the derivative of get_dv via derivkit (cnu.get_ddv_dkit).

    derivkit fits polynomials through adaptively chosen samples and
    differentiates the fit, so numerical noise averages out.

    Arguments:
      index, CV, AccuracyBoost = as in get_ddv.
      min_samples, fallback_mode = forwarded to derivkit (see
               cnu.get_ddv_dkit).

    Returns:
      1D array, d(masked data vector)/d(parameter).
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    return cnu.get_ddv_dkit(get_dv, index=index, CV=CV,
                            AccuracyBoost=AccuracyBoost,
                            min_samples=min_samples,
                            fallback_mode=fallback_mode)


def get_Fisher2(CV=None, AccuracyBoost=3.0, priors=None, invcov=None,
                min_samples=7, fallback_mode="poly_at_floor"):
    """Return the Fisher matrix from the derivkit derivatives (cnu.get_Fisher2).

    Arguments:
      CV, AccuracyBoost, priors, invcov = as in get_Fisher.
      min_samples, fallback_mode = as in get_ddv_dkit.

    Returns:
      the Fisher matrix, an (n_params, n_params) array.
    """
    if CV is None:
        CV = FISHER_PARAM_FID
    if priors is None:
        priors = FISHER_GAUSSIAN_PRIORS
    if invcov is None:
        invcov = ci.get_inv_cov_masked()
    return cnu.get_Fisher2(get_dv, CV=CV, AccuracyBoost=AccuracyBoost,
                           priors=priors, invcov=invcov,
                           min_samples=min_samples,
                           fallback_mode=fallback_mode)


def plot_Fisher(F, mu, F2=None, root=None, select=None, labels=None,
                names=None, filled=True, flat_priors=None,
                chain_names=None):
    """Draw the Fisher contour triangle via getdist (cnu.plot_Fisher).

    labels, names and flat_priors default to this module's
    FISHER_PARAM_LABELS, FISHER_PARAM_NAMES and FISHER_FLAT_PRIORS; the
    sample clouds come from _FISHER_RNG. The other arguments are those
    of cnu.plot_Fisher.

    Returns:
      the getdist subplot plotter holding the figure.
    """
    if labels is None:
        labels = FISHER_PARAM_LABELS
    if names is None:
        names = FISHER_PARAM_NAMES
    if flat_priors is None:
        flat_priors = FISHER_FLAT_PRIORS
    return cnu.plot_Fisher(F, mu, F2=F2, root=root, select=select,
                           labels=labels, names=names, filled=filled,
                           flat_priors=flat_priors,
                           chain_names=chain_names, rng=_FISHER_RNG)
