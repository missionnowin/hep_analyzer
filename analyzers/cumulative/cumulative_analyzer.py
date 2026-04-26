"""
Cumulative-effect analyzer for heavy-ion collisions.

Headline observable
-------------------
The cumulative variable
        x_cum  =  (E - p_z) / m_N
in the **target rest frame** (input data is already in that frame).

Physical meaning
    x_cum < 1   : kinematically allowed in a single N+N collision -- "ordinary" particle
    x_cum > 1   : cumulative -- only producible if the projectile sees an
                  effective target with mass > m_N
                  (short-range correlations, fluctons, multi-nucleon clusters)
    1 < x_cum < 2 : 1-nucleon cumulative
    2 < x_cum < 3 : 2-nucleon cumulative   ...

The plots produced here are designed to make the cumulative effect
*visually obvious*:

    1. (x_cum, p_perp) density heatmaps for modified vs. unmodified
       (logarithmic colour scale).
    2. A *ratio* panel  density_mod / density_unm  -- shows directly where
       the modification produces extra particles.
    3. 1D x_cum spectrum (log-y), mod vs. unm overlaid.
    4. Per-event distribution of N_cum  =  number of particles with x_cum > 1
       in a single event, mod vs. unm.
    5. Backward-only k_z vs k_perp density heatmap (sanity check, comparable
       to the older scatter plot).
    6. Exponential-slope fit of the cumulative tail
            dN/dx_cum  ~  exp(-x_cum / x_0)
       overlaid on plot (2).  The Leksin/Baldin slope x_0 ~ 0.13-0.18 is the
       textbook signature of cumulative production.
    7. Composition fractions vs x_cum (three diagnostic plots):
         (a) Proton fraction:  N_p / N_charged
         (b) Baryon-counting fraction:  N_(p+flucton) / N_charged
             A flucton is treated as a multi-proton cluster, so this plot
             measures the *true* baryon content delivered by the modification.
         (c) Stacked species composition vs x_cum -- pions, muons, protons,
             fluctons -- mod and unm side by side.
       Real cumulative data shows the proton fraction GROW with x_cum because
       flucton breakup favours baryons over mesons.

All plots are produced separately for charged hadrons and for protons.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.ions import CHARGED_PDGIDS, PROTONS, FLUCTONS, PIONS, MUONS

try:
    from models.particle import Particle
except ImportError as e:  # pragma: no cover
    print(f"Error importing modules: {e}")
    sys.exit(1)


# Nucleon mass in GeV.  Used for the cumulative variable definition.
M_N = 0.9382720813


# ----------------------------------------------------------------------
# Internal container for one (species, dataset) bucket.
# ----------------------------------------------------------------------
class _SpeciesBucket:
    """Accumulator for one particle species in one dataset (mod or unm)."""

    __slots__ = ("xcum", "pperp", "pz", "n_cum_per_event")

    def __init__(self) -> None:
        self.xcum: List[float] = []
        self.pperp: List[float] = []
        self.pz: List[float] = []
        # one entry per event:  number of particles with x_cum > 1
        self.n_cum_per_event: List[int] = []


# ----------------------------------------------------------------------
class CumulativeAnalyzer:
    """
    Build cumulative-effect plots and statistics from a stream of events.

    Public API (unchanged):
        process_batch(batch_mod, batch_unm)
        plot_distributions(output_dir) -> List[Path]
        get_statistics() -> Dict
        reset()
    """

    # Cuts applied when filling the buckets (target frame, "going backward").
    _E_MIN = 0.3        # GeV  -- drop very-soft particles, same as old code

    # Binning for the headline 2D heatmap.
    _XCUM_EDGES = np.linspace(0.0, 3.0, 61)        # 60 bins, 0..3
    _PPERP_EDGES = np.linspace(0.0, 1.5, 31)       # 30 bins, 0..1.5 GeV/c

    # Binning for the backward k_z, k_perp heatmap (sanity check vs. legacy).
    _KZ_EDGES = np.linspace(-1.4, 0.0, 57)
    _KPERP_EDGES = np.linspace(0.0, 1.5, 31)

    # Cumulative threshold.
    _XCUM_THRESHOLD = 1.0

    # Exponential-slope fit window (x_cum).  Avoid the soft shoulder near
    # x_cum ~ 1 and the noisy far tail above ~2.2.
    _FIT_XMIN = 1.2
    _FIT_XMAX = 2.2

    def __init__(self) -> None:
        # Event / particle counters (kept for backward-compat in stats)
        self.total_events_mod = 0
        self.total_events_unm = 0
        self.total_particles_mod = 0
        self.total_particles_unm = 0
        self.multiplicity_mod: List[int] = []
        self.multiplicity_unm: List[int] = []

        # species: { "charged", "protons", "fluctons", "pions", "muons" }
        # dataset: { "mod", "unm" }
        # Note: "charged" is the headline species (used for ratios denominator);
        # "fluctons", "pions", "muons" are auxiliary, only filled to support
        # composition / fraction plots.
        self._buckets: Dict[str, Dict[str, _SpeciesBucket]] = {
            "charged":  {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
            "protons":  {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
            "fluctons": {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
            "pions":    {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
            "muons":    {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
        }

        # Slope fits filled by the 1D plotter; consumed by get_statistics().
        # Keyed by (species, tag) -> dict(x0, x0_err, amplitude, n_points).
        self._fit_results: Dict[Tuple[str, str], Dict[str, float]] = {}

        self._finalized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_batch(self, batch_mod: List, batch_unm: List) -> None:
        """Consume one batch of events from the modified and unmodified runs."""
        if not batch_mod or not batch_unm:
            return

        # Bookkeeping: per-event totals.
        for particles_mod, particles_unm in zip(batch_mod, batch_unm):
            self.total_events_mod += 1
            self.total_events_unm += 1
            n_mod = len(particles_mod) if particles_mod else 0
            n_unm = len(particles_unm) if particles_unm else 0
            self.total_particles_mod += n_mod
            self.total_particles_unm += n_unm
            self.multiplicity_mod.append(n_mod)
            self.multiplicity_unm.append(n_unm)

        # Physics observables.
        self._fill_dataset(batch_mod, "mod")
        self._fill_dataset(batch_unm, "unm")

    # ------------------------------------------------------------------
    def _fill_dataset(self, batch: List[List[Particle]], tag: str) -> None:
        for event in batch:
            if not event:
                # Still count "0 cumulative particles" for empty events so
                # the per-event histogram averages correctly.
                for sp in self._buckets:
                    self._buckets[sp][tag].n_cum_per_event.append(0)
                continue

            n_cum = {sp: 0 for sp in self._buckets}

            for p in event:
                # input data is already in target rest frame
                if p.E <= self._E_MIN:
                    continue
                pid = p.particle_id

                if pid not in CHARGED_PDGIDS:
                    continue

                pperp = float(np.hypot(p.px, p.py))
                pz = float(p.pz)
                xcum = (p.E - pz) / M_N

                # Always fill the headline "charged" bucket
                tags_for_this_particle = ["charged"]
                if pid in PROTONS:
                    tags_for_this_particle.append("protons")
                if pid in FLUCTONS:
                    tags_for_this_particle.append("fluctons")
                if pid in PIONS:
                    tags_for_this_particle.append("pions")
                if pid in MUONS:
                    tags_for_this_particle.append("muons")
                for sp in tags_for_this_particle:
                    bkt = self._buckets[sp][tag]

                for sp in tags_for_this_particle:
                    bkt = self._buckets[sp][tag]
                    bkt.xcum.append(xcum)
                    bkt.pperp.append(pperp)
                    bkt.pz.append(pz)
                    if xcum > self._XCUM_THRESHOLD:
                        n_cum[sp] += 1

            for sp in self._buckets:
                self._buckets[sp][tag].n_cum_per_event.append(n_cum[sp])

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_distributions(self, output_dir) -> List[Path]:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
        except ImportError:
            print("Matplotlib not available - skipping plots")
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out: List[Path] = []

        for species in ("charged", "protons"):
            tag_mod = self._buckets[species]["mod"]
            tag_unm = self._buckets[species]["unm"]
            if not tag_mod.xcum or not tag_unm.xcum:
                continue

            label_species = {"charged": "charged hadrons", "protons": "protons"}[species]
            prefix = {"charged": "charged", "protons": "protons"}[species]

            # 1) (x_cum, p_perp) density: mod | unm | ratio
            out.append(self._plot_xcum_pperp_panels(
                tag_mod, tag_unm, label_species,
                output_dir / f"01_{prefix}_xcum_pperp_panels.png", plt, LogNorm,
            ))

            # 2) 1D x_cum spectrum, mod vs unm overlay (log-y, normalized per event)
            out.append(self._plot_xcum_1d(
                tag_mod, tag_unm, label_species,
                output_dir / f"02_{prefix}_xcum_spectrum.png", plt,
            ))

            # 3) Per-event N_cum distribution
            out.append(self._plot_ncum_per_event(
                tag_mod, tag_unm, label_species,
                output_dir / f"03_{prefix}_ncum_per_event.png", plt,
            ))

            # 4) Backward k_z, k_perp density (sanity / legacy)
            out.append(self._plot_kz_kperp_density(
                tag_mod, tag_unm, label_species,
                output_dir / f"04_{prefix}_kz_kperp_density.png", plt, LogNorm,
            ))

        # 5) Proton fraction vs x_cum (N_p / N_charged).
        frac_path = self._plot_proton_fraction(
            output_dir / "05_proton_fraction_vs_xcum.png", plt,
        )
        if frac_path is not None:
            out.append(frac_path)
        
        # 6) Baryon-counting fraction vs x_cum (N_(p+flucton) / N_charged).
        bf_path = self._plot_baryon_fraction(
            output_dir / "06_baryon_fraction_vs_xcum.png", plt,
        )
        if bf_path is not None:
            out.append(bf_path)
            
        # 7) Stacked species composition vs x_cum (mod | unm side-by-side).
        comp_path = self._plot_composition(
            output_dir / "07_composition_vs_xcum.png", plt,
        )
        if comp_path is not None:
            out.append(comp_path)

        return [p for p in out if p is not None]

    # ------------------------------------------------------------------
    # Individual plotters
    # ------------------------------------------------------------------
    def _plot_xcum_pperp_panels(self, mod, unm, species, path, plt, LogNorm):
        H_mod, _, _ = np.histogram2d(
            mod.xcum, mod.pperp, bins=[self._XCUM_EDGES, self._PPERP_EDGES])
        H_unm, _, _ = np.histogram2d(
            unm.xcum, unm.pperp, bins=[self._XCUM_EDGES, self._PPERP_EDGES])

        # Normalise per event so the two datasets are directly comparable.
        n_ev_mod = max(self.total_events_mod, 1)
        n_ev_unm = max(self.total_events_unm, 1)
        D_mod = H_mod / n_ev_mod
        D_unm = H_unm / n_ev_unm

        # Ratio: avoid /0 by floor on denominator.
        eps = 1.0 / max(n_ev_unm, 1)        # one count per event = floor
        ratio = np.where(D_unm > 0, D_mod / np.maximum(D_unm, eps), np.nan)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                                 gridspec_kw={"wspace": 0.35})

        # Common color range for the two density panels (log).
        vmax = max(D_mod.max(), D_unm.max(), 1e-6)
        vmin = max(min(D_mod[D_mod > 0].min() if (D_mod > 0).any() else vmax,
                       D_unm[D_unm > 0].min() if (D_unm > 0).any() else vmax),
                   vmax * 1e-5)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        ext = [self._XCUM_EDGES[0], self._XCUM_EDGES[-1],
               self._PPERP_EDGES[0], self._PPERP_EDGES[-1]]

        for ax, D, title in (
            (axes[0], D_mod, f"Modified: {species}"),
            (axes[1], D_unm, f"Unmodified: {species}"),
        ):
            im = ax.imshow(D.T, origin="lower", aspect="auto", extent=ext,
                           cmap="viridis", norm=norm)
            ax.axvline(1.0, color="white", lw=1, ls="--", alpha=0.7)
            ax.set_xlabel(r"$x_{\mathrm{cum}} = (E - p_z)/m_N$")
            ax.set_ylabel(r"$p_\perp$  (GeV/c)")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="particles / event / bin")

        # Ratio panel: diverging around 1.
        im2 = axes[2].imshow(
            ratio.T, origin="lower", aspect="auto", extent=ext,
            cmap="RdBu_r", vmin=0.0, vmax=2.0,
        )
        axes[2].axvline(1.0, color="black", lw=1, ls="--", alpha=0.5)
        axes[2].set_xlabel(r"$x_{\mathrm{cum}}$")
        axes[2].set_ylabel(r"$p_\perp$  (GeV/c)")
        axes[2].set_title(f"Ratio mod / unm: {species}")
        fig.colorbar(im2, ax=axes[2], label="density ratio")

        fig.suptitle(
            r"Cumulative phase space  ($x_{\mathrm{cum}}$ vs $p_\perp$)"
            f" -- {species}, target frame",
            fontsize=14, fontweight="bold",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    def _plot_xcum_1d(self, mod, unm, species, path, plt):
        n_ev_mod = max(self.total_events_mod, 1)
        n_ev_unm = max(self.total_events_unm, 1)

        h_mod, edges = np.histogram(mod.xcum, bins=self._XCUM_EDGES)
        h_unm, _ = np.histogram(unm.xcum, bins=self._XCUM_EDGES)
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        # dN/dx_cum per event
        y_mod = h_mod / (n_ev_mod * widths)
        y_unm = h_unm / (n_ev_unm * widths)

        # ----- Exponential-slope fit in the cumulative window ------------
        fit_mod = self._fit_exponential_tail(centers, y_mod, h_mod)
        fit_unm = self._fit_exponential_tail(centers, y_unm, h_unm)
        # cache for get_statistics()
        species_key = self._species_key(species)
        if fit_mod is not None:
            self._fit_results[(species_key, "mod")] = fit_mod
        if fit_unm is not None:
            self._fit_results[(species_key, "unm")] = fit_unm

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        )

        ax_top.step(centers, y_unm, where="mid", color="tab:blue",
                    label="Unmodified", lw=2)
        ax_top.step(centers, y_mod, where="mid", color="tab:red",
                    label="Modified", lw=2)

        # Overlay exponential fits.
        x_fit = np.linspace(self._FIT_XMIN, self._FIT_XMAX, 50)
        if fit_unm is not None:
            y_fit = fit_unm["amplitude"] * np.exp(-x_fit / fit_unm["x0"])
            ax_top.plot(x_fit, y_fit, color="tab:blue", ls="--", lw=1.8,
                        label=fr"unm fit: $x_0$ = {fit_unm['x0']:.3f}"
                              fr" $\pm$ {fit_unm['x0_err']:.3f}")
        if fit_mod is not None:
            y_fit = fit_mod["amplitude"] * np.exp(-x_fit / fit_mod["x0"])
            ax_top.plot(x_fit, y_fit, color="tab:red", ls="--", lw=1.8,
                        label=fr"mod fit: $x_0$ = {fit_mod['x0']:.3f}"
                              fr" $\pm$ {fit_mod['x0_err']:.3f}")
        # Shade the fit window
        ax_top.axvspan(self._FIT_XMIN, self._FIT_XMAX, color="gold", alpha=0.10,
                       label=fr"fit window [{self._FIT_XMIN}, {self._FIT_XMAX}]")

        ax_top.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5)
        ax_top.set_yscale("log")
        ax_top.set_ylabel(r"$\frac{1}{N_{\mathrm{ev}}}\,dN/dx_{\mathrm{cum}}$")
        ax_top.set_title(
            f"Cumulative spectrum with exponential fit -- {species}, target frame",
            fontweight="bold",
        )
        ax_top.legend(fontsize=9, loc="upper right")
        ax_top.grid(True, which="both", alpha=0.3)
        ax_top.text(1.02, ax_top.get_ylim()[1] * 0.5,
                    "cumulative\nregion", color="black",
                    fontsize=9, alpha=0.7)

        # Ratio subplot
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(y_unm > 0, y_mod / y_unm, np.nan)
        ax_bot.step(centers, ratio, where="mid", color="tab:purple", lw=2)
        ax_bot.axhline(1.0, color="k", ls="-", lw=0.7)
        ax_bot.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5)
        ax_bot.set_xlabel(r"$x_{\mathrm{cum}} = (E - p_z)/m_N$")
        ax_bot.set_ylabel("mod / unm")
        ax_bot.set_ylim(0, 4)
        ax_bot.grid(True, alpha=0.3)

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    @staticmethod
    def _species_key(species_label: str) -> str:
        # Convert pretty label back to bucket key.
        return "protons" if "proton" in species_label.lower() else "charged"

    # ------------------------------------------------------------------
    def _fit_exponential_tail(self, centers, y, counts):
        """
        Fit  log(y) = log(A) - x/x0  on  x in [_FIT_XMIN, _FIT_XMAX].

        Bins are weighted by sqrt(N) (Poisson) so empty / single-count bins
        in the far tail don't dominate the slope.
        Returns dict(x0, x0_err, amplitude, n_points) or None if fit fails.
        """
        centers = np.asarray(centers, dtype=float)
        y = np.asarray(y, dtype=float)
        counts = np.asarray(counts, dtype=float)

        mask = (centers >= self._FIT_XMIN) & (centers <= self._FIT_XMAX) & (y > 0) & (counts > 0)
        if mask.sum() < 3:
            return None

        x = centers[mask]
        ly = np.log(y[mask])
        # Weight: sqrt(N) -> log-error sigma_log = 1/sqrt(N).  Use w = sqrt(N).
        w = np.sqrt(counts[mask])

        # Weighted linear fit:  ly = a + b * x   ;  x0 = -1/b
        # numpy.polyfit supports weights.
        try:
            coeffs, cov = np.polyfit(x, ly, 1, w=w, cov=True)
        except (ValueError, np.linalg.LinAlgError):
            return None

        b, a = coeffs[0], coeffs[1]
        if b >= 0:                       # not a falling exponential -> drop
            return None
        x0 = -1.0 / b
        # error on x0 from error on b via 1-sigma propagation: dx0 = |x0^2 * db|
        try:
            db = float(np.sqrt(cov[0, 0]))
        except (IndexError, ValueError):
            db = 0.0
        x0_err = abs(x0 * x0 * db)
        amplitude = float(np.exp(a))
        return {
            "x0":        float(x0),
            "x0_err":    float(x0_err),
            "amplitude": amplitude,
            "n_points":  int(mask.sum()),
        }

    # ------------------------------------------------------------------
    def _bin_xcum(self, species: str, tag: str) -> np.ndarray:
        """Return histogram of x_cum for one (species, tag) bucket."""
        return np.histogram(
            self._buckets[species][tag].xcum, bins=self._XCUM_EDGES,
        )[0]
    @staticmethod
    def _binomial_fraction(num: np.ndarray, den: np.ndarray, n_min: int = 5):
        """
        Return (fraction, error) per bin with binomial errors.
        Bins with denominator < n_min are returned as NaN.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ok = den >= n_min
            f = np.where(ok, num / np.maximum(den, 1), np.nan)
            sig = np.where(
                ok,
                np.sqrt(np.clip(f * (1.0 - f), 0, None) / np.maximum(den, 1)),
                np.nan,
            )
        return f, sig
    
    # ------------------------------------------------------------------
    def _plot_proton_fraction(self, path, plt):
        """
        Per-bin proton fraction  N_p(x_cum) / N_charged(x_cum)  for mod and unm.
        """
        # Sanity
        ch_mod = self._buckets["charged"]["mod"]
        ch_unm = self._buckets["charged"]["unm"]
        if not ch_mod.xcum or not ch_unm.xcum:
            return None

        h_ch_mod = self._bin_xcum("charged", "mod")
        h_ch_unm = self._bin_xcum("charged", "unm")
        h_pr_mod = self._bin_xcum("protons", "mod")
        h_pr_unm = self._bin_xcum("protons", "unm")
        centers = 0.5 * (self._XCUM_EDGES[:-1] + self._XCUM_EDGES[1:])

        f_mod, e_mod = self._binomial_fraction(h_pr_mod, h_ch_mod)
        f_unm, e_unm = self._binomial_fraction(h_pr_unm, h_ch_unm)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(centers, f_unm, yerr=e_unm, fmt="o-", color="tab:blue",
                    ms=4, lw=1.5, capsize=2, label="Unmodified")
        ax.errorbar(centers, f_mod, yerr=e_mod, fmt="s-", color="tab:red",
                    ms=4, lw=1.5, capsize=2, label="Modified")
        ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5,
                   label=r"$x_{\mathrm{cum}} = 1$")
        ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7,
                   label="naive equipartition (p:pi+:pi- 1:1:1)")
        ax.set_xlabel(r"$x_{\mathrm{cum}} = (E - p_z)/m_N$")
        ax.set_ylabel(r"$N_{\mathrm{p}}\,/\,N_{\mathrm{charged}}$")
        ax.set_title(
            "Proton fraction vs cumulative variable (target frame)",
            fontweight="bold",
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 3.0)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.text(1.05, 0.05,
                "baryon enrichment expected here\nfor flucton-driven yield",
                fontsize=9, alpha=0.7)

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    
    # ------------------------------------------------------------------
    def _plot_baryon_fraction(self, path, plt):
        """
        Per-bin baryon-counting fraction  N_(p+flucton)(x_cum) / N_charged(x_cum).
        A flucton in our event record is a multi-proton cluster, so this
        plot estimates the *true* baryon content of the charged sample
        at each x_cum.  Compared with plot 05, the only difference is the
        flucton contribution: in mod we should see a clear lift in the
        cumulative region wherever fluctons populate.
        """
        ch_mod = self._buckets["charged"]["mod"]
        ch_unm = self._buckets["charged"]["unm"]
        if not ch_mod.xcum or not ch_unm.xcum:
            return None
        h_ch_mod = self._bin_xcum("charged", "mod")
        h_ch_unm = self._bin_xcum("charged", "unm")
        h_pr_mod = self._bin_xcum("protons", "mod")
        h_pr_unm = self._bin_xcum("protons", "unm")
        h_fl_mod = self._bin_xcum("fluctons", "mod")
        h_fl_unm = self._bin_xcum("fluctons", "unm")
        centers = 0.5 * (self._XCUM_EDGES[:-1] + self._XCUM_EDGES[1:])
        # Numerator counts a flucton as one baryon-like object; we could
        # instead weight by some integer (constituent count), but our event
        # record exposes one entry per flucton with its own four-momentum,
        # so a one-for-one count is the natural choice for a phase-space
        # fraction plot.
        num_mod = h_pr_mod + h_fl_mod
        num_unm = h_pr_unm + h_fl_unm
        f_mod, e_mod = self._binomial_fraction(num_mod, h_ch_mod)
        f_unm, e_unm = self._binomial_fraction(num_unm, h_ch_unm)
        # Also overlay protons-only fractions in faint colour for direct
        # comparison with plot 05 (so the flucton lift is visible).
        fp_mod, _ = self._binomial_fraction(h_pr_mod, h_ch_mod)
        fp_unm, _ = self._binomial_fraction(h_pr_unm, h_ch_unm)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(centers, f_unm, yerr=e_unm, fmt="o-", color="tab:blue",
                    ms=4, lw=1.5, capsize=2,
                    label=r"Unmodified  $(p+\mathrm{flucton})/\mathrm{charged}$")
        ax.errorbar(centers, f_mod, yerr=e_mod, fmt="s-", color="tab:red",
                    ms=4, lw=1.5, capsize=2,
                    label=r"Modified  $(p+\mathrm{flucton})/\mathrm{charged}$")
        ax.plot(centers, fp_unm, color="tab:blue", ls=":", lw=1.2, alpha=0.6,
                label=r"Unmodified  $p/\mathrm{charged}$ (ref.)")
        ax.plot(centers, fp_mod, color="tab:red", ls=":", lw=1.2, alpha=0.6,
                label=r"Modified  $p/\mathrm{charged}$ (ref.)")
        ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.5,
                   label=r"$x_{\mathrm{cum}} = 1$")
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel(r"$x_{\mathrm{cum}} = (E - p_z)/m_N$")
        ax.set_ylabel(r"$N_{\mathrm{p}+\mathrm{flucton}}\,/\,N_{\mathrm{charged}}$")
        ax.set_title(
            "Baryon-counting fraction vs cumulative variable (target frame)",
            fontweight="bold",
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 3.0)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.text(1.05, 0.05,
                "flucton lift = mod (solid)  -  mod (dotted)\n"
                "this is the effect of the patch on baryon content",
                fontsize=9, alpha=0.7)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    # ------------------------------------------------------------------
    def _plot_composition(self, path, plt):
        """
        Stacked species composition vs x_cum.  Two side-by-side panels:
        modified | unmodified.  At each x_cum bin, the bar is split into
        fractional contributions of:  pions, muons, protons, fluctons.
        """
        ch_mod = self._buckets["charged"]["mod"]
        ch_unm = self._buckets["charged"]["unm"]
        if not ch_mod.xcum or not ch_unm.xcum:
            return None
        species_order = [
            ("pions",    r"$\pi^{\pm}$",  "#5fa8d3"),
            ("muons",    r"$\mu^{\pm}$",  "#a06ab8"),
            ("protons",  r"$p$",          "#e07a5f"),
            ("fluctons", r"flucton",      "#3d405b"),
        ]
        centers = 0.5 * (self._XCUM_EDGES[:-1] + self._XCUM_EDGES[1:])
        widths  = np.diff(self._XCUM_EDGES)
        # Suppress bins with very low statistics in either dataset (looks
        # noisy and is not meaningful).
        N_MIN = 5
        def _frac_table(tag: str):
            """Return (h_charged, dict[species]->fraction array)."""
            h_ch = self._bin_xcum("charged", tag).astype(float)
            ok = h_ch >= N_MIN
            fracs = {}
            for sp, _, _ in species_order:
                h = self._bin_xcum(sp, tag).astype(float)
                fracs[sp] = np.where(ok, h / np.maximum(h_ch, 1), np.nan)
            return h_ch, fracs, ok
        h_ch_mod, fr_mod, ok_mod = _frac_table("mod")
        h_ch_unm, fr_unm, ok_unm = _frac_table("unm")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True,
                                 gridspec_kw={"wspace": 0.05})
        for ax, fracs, ok, title in (
            (axes[0], fr_mod, ok_mod, "Modified"),
            (axes[1], fr_unm, ok_unm, "Unmodified"),
        ):
            bottom = np.zeros_like(centers, dtype=float)
            for sp, label, color in species_order:
                vals = np.where(ok, fracs[sp], 0.0)
                ax.bar(centers, vals, width=widths * 0.95,
                       bottom=bottom, color=color, label=label,
                       edgecolor="none")
                bottom = bottom + np.nan_to_num(vals, nan=0.0)
            ax.axvline(1.0, color="white", ls="--", lw=1.2, alpha=0.85)
            ax.set_xlim(0, 3.0)
            ax.set_ylim(0, 1.0)
            ax.set_xlabel(r"$x_{\mathrm{cum}} = (E - p_z)/m_N$")
            ax.set_title(title, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel("fraction of charged sample")
        axes[1].legend(loc="upper right", fontsize=9, framealpha=0.95)
        fig.suptitle(
            "Species composition of charged hadrons vs $x_{\\mathrm{cum}}$"
            "  (target frame)",
            fontsize=13, fontweight="bold",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    # ------------------------------------------------------------------
    def _plot_ncum_per_event(self, mod, unm, species, path, plt):
        nmod = np.asarray(mod.n_cum_per_event, dtype=int)
        nunm = np.asarray(unm.n_cum_per_event, dtype=int)
        if nmod.size == 0 and nunm.size == 0:
            return None

        hi = max(int(nmod.max()) if nmod.size else 0,
                 int(nunm.max()) if nunm.size else 0,
                 1)
        edges = np.arange(-0.5, hi + 1.5, 1.0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(nunm, bins=edges, alpha=0.55, color="tab:blue",
                label=f"Unmodified  ⟨N⟩ = {nunm.mean():.2f}")
        ax.hist(nmod, bins=edges, alpha=0.55, color="tab:red",
                label=f"Modified    ⟨N⟩ = {nmod.mean():.2f}")
        ax.set_xlabel(r"$N_{\mathrm{cum}}$  (particles with $x_{\mathrm{cum}} > 1$ per event)")
        ax.set_ylabel("Number of events")
        ax.set_title(
            f"Cumulative-particle count per event -- {species}",
            fontweight="bold",
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    def _plot_kz_kperp_density(self, mod, unm, species, path, plt, LogNorm):
        # Same axes as the older scatter plot but properly density-binned.
        # Restrict to the backward hemisphere (k_z < 0) for the cumulative
        # interpretation.
        def _kz_kperp(b):
            kz = np.asarray(b.pz)
            kperp = np.asarray(b.pperp)
            mask = kz < 0
            return kz[mask], kperp[mask]

        kz_m, kp_m = _kz_kperp(mod)
        kz_u, kp_u = _kz_kperp(unm)
        if kz_m.size == 0 and kz_u.size == 0:
            return None

        H_m, _, _ = np.histogram2d(kz_m, kp_m,
                                   bins=[self._KZ_EDGES, self._KPERP_EDGES])
        H_u, _, _ = np.histogram2d(kz_u, kp_u,
                                   bins=[self._KZ_EDGES, self._KPERP_EDGES])

        n_ev_mod = max(self.total_events_mod, 1)
        n_ev_unm = max(self.total_events_unm, 1)
        D_m = H_m / n_ev_mod
        D_u = H_u / n_ev_unm

        vmax = max(D_m.max(), D_u.max(), 1e-6)
        vmin = max(vmax * 1e-5,
                   min(D_m[D_m > 0].min() if (D_m > 0).any() else vmax,
                       D_u[D_u > 0].min() if (D_u > 0).any() else vmax))
        norm = LogNorm(vmin=vmin, vmax=vmax)

        ext = [self._KZ_EDGES[0], self._KZ_EDGES[-1],
               self._KPERP_EDGES[0], self._KPERP_EDGES[-1]]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, D, title in (
            (axes[0], D_m, f"Modified: {species}"),
            (axes[1], D_u, f"Unmodified: {species}"),
        ):
            im = ax.imshow(D.T, origin="lower", aspect="auto", extent=ext,
                           cmap="viridis", norm=norm)
            ax.set_xlabel(r"$k_z$  (GeV/c)")
            ax.set_ylabel(r"$k_\perp$  (GeV/c)")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label="particles / event / bin")

        fig.suptitle(
            f"Backward phase-space density ($k_z < 0$) -- {species}, target frame",
            fontsize=13, fontweight="bold",
        )
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Statistics & lifecycle
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict:
        self._finalize_and_detect()

        n_ev_mod = max(self.total_events_mod, 1)
        n_ev_unm = max(self.total_events_unm, 1)

        stats: Dict = {
            "total_events_modified": int(self.total_events_mod),
            "total_events_unmodified": int(self.total_events_unm),
            "total_particles_modified": int(self.total_particles_mod),
            "total_particles_unmodified": int(self.total_particles_unm),
            "avg_multiplicity_modified": self.total_particles_mod / n_ev_mod,
            "avg_multiplicity_unmodified": self.total_particles_unm / n_ev_unm,
            "cumulative_threshold_xcum": self._XCUM_THRESHOLD,
            "frame": "target rest frame",
        }

        for species in ("charged", "protons"):
            mod = self._buckets[species]["mod"]
            unm = self._buckets[species]["unm"]

            xcum_mod = np.asarray(mod.xcum)
            xcum_unm = np.asarray(unm.xcum)

            n_total_mod = xcum_mod.size
            n_total_unm = xcum_unm.size
            n_cum_mod = int(np.sum(xcum_mod > self._XCUM_THRESHOLD))
            n_cum_unm = int(np.sum(xcum_unm > self._XCUM_THRESHOLD))

            ev_nmod = np.asarray(mod.n_cum_per_event, dtype=float)
            ev_nunm = np.asarray(unm.n_cum_per_event, dtype=float)

            stats[species] = {
                "n_total_mod": int(n_total_mod),
                "n_total_unm": int(n_total_unm),
                "n_cum_mod": n_cum_mod,
                "n_cum_unm": n_cum_unm,
                "frac_cum_mod": (n_cum_mod / n_total_mod) if n_total_mod else 0.0,
                "frac_cum_unm": (n_cum_unm / n_total_unm) if n_total_unm else 0.0,
                "n_cum_per_event_mod_mean": float(ev_nmod.mean()) if ev_nmod.size else 0.0,
                "n_cum_per_event_unm_mean": float(ev_nunm.mean()) if ev_nunm.size else 0.0,
                "n_cum_per_event_mod_std":  float(ev_nmod.std())  if ev_nmod.size else 0.0,
                "n_cum_per_event_unm_std":  float(ev_nunm.std())  if ev_nunm.size else 0.0,
                "xcum_max_mod": float(xcum_mod.max()) if n_total_mod else 0.0,
                "xcum_max_unm": float(xcum_unm.max()) if n_total_unm else 0.0,
            }

            # Exponential slope fits (filled by plot_distributions). Only
            # present after plotting; absent if plot_distributions wasn't
            # called or fit failed.
            fit_mod = self._fit_results.get((species, "mod"))
            fit_unm = self._fit_results.get((species, "unm"))
            if fit_mod is not None:
                stats[species]["xcum_slope_x0_mod"]     = fit_mod["x0"]
                stats[species]["xcum_slope_x0_err_mod"] = fit_mod["x0_err"]
            if fit_unm is not None:
                stats[species]["xcum_slope_x0_unm"]     = fit_unm["x0"]
                stats[species]["xcum_slope_x0_err_unm"] = fit_unm["x0_err"]
            stats[species]["xcum_slope_fit_window"] = [self._FIT_XMIN, self._FIT_XMAX]

        return stats

    def _finalize_and_detect(self) -> None:
        if self._finalized:
            return
        self._finalized = True

    def reset(self) -> None:
        self.total_events_mod = 0
        self.total_events_unm = 0
        self.total_particles_mod = 0
        self.total_particles_unm = 0
        self.multiplicity_mod = []
        self.multiplicity_unm = []
        self._buckets = {
            "charged":  {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
            "protons":  {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
            "fluctons": {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
            "pions":    {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
            "muons":    {"mod": _SpeciesBucket(), "unm": _SpeciesBucket()},
        }
        self._fit_results = {}
        self._finalized = False
