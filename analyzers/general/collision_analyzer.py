import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    from analyzers.general.angle_analyzer import AngleAnalyzer
    from models.particle import Particle
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


class CollisionAnalyzer:
    """High-level analyzer for a single event using lazy computation."""

    def __init__(self, particles: List[Particle], system_label: str = "Au+Au @ 10 GeV"):
        self.particles = particles
        self.system_label = system_label
        self.angle = AngleAnalyzer() 
        self._cache = {}

    def _get_px_py_pz_E(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get momentum arrays efficiently (cached as a tuple)."""
        if "_momentum" not in self._cache:
            px = np.array([p.px for p in self.particles], dtype=np.float32)
            py = np.array([p.py for p in self.particles], dtype=np.float32)
            pz = np.array([p.pz for p in self.particles], dtype=np.float32)
            E = np.array([p.E for p in self.particles], dtype=np.float32)
            self._cache["_momentum"] = (px, py, pz, E)
        return self._cache["_momentum"]

    @property
    def px(self) -> np.ndarray:
        """Longitudinal momentum (X component)."""
        if "px" not in self._cache:
            px, _, _, _ = self._get_px_py_pz_E()
            self._cache["px"] = px
        return self._cache["px"]

    @property
    def py(self) -> np.ndarray:
        """Transverse momentum (Y component)."""
        if "py" not in self._cache:
            _, py, _, _ = self._get_px_py_pz_E()
            self._cache["py"] = py
        return self._cache["py"]

    @property
    def pz(self) -> np.ndarray:
        """Longitudinal momentum (Z component, beam direction)."""
        if "pz" not in self._cache:
            _, _, pz, _ = self._get_px_py_pz_E()
            self._cache["pz"] = pz
        return self._cache["pz"]

    @property
    def E(self) -> np.ndarray:
        """Total energy."""
        if "E" not in self._cache:
            _, _, _, E = self._get_px_py_pz_E()
            self._cache["E"] = E
        return self._cache["E"]

    @property
    def pt(self) -> np.ndarray:
        """Transverse momentum magnitude (computed from px, py)."""
        if "pt" not in self._cache:
            px = self.px
            py = self.py
            self._cache["pt"] = np.sqrt(px**2 + py**2).astype(np.float32)
        return self._cache["pt"]

    @property
    def p(self) -> np.ndarray:
        """Total momentum magnitude."""
        if "p" not in self._cache:
            px = self.px
            py = self.py
            pz = self.pz
            self._cache["p"] = np.sqrt(px**2 + py**2 + pz**2).astype(np.float32)
        return self._cache["p"]

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angle (phi) in radians."""
        if "phi" not in self._cache:
            phi_vals = np.array(
                [self.angle.azimuthal_angle(p) for p in self.particles],
                dtype=np.float32,
            )
            self._cache["phi"] = phi_vals
        return self._cache["phi"]

    @property
    def theta(self) -> np.ndarray:
        """Polar angle (theta) in radians."""
        if "theta" not in self._cache:
            theta_vals = np.array(
                [self.angle.polar_angle(p) for p in self.particles],
                dtype=np.float32,
            )
            self._cache["theta"] = theta_vals
        return self._cache["theta"]

    @property
    def phi_deg(self) -> np.ndarray:
        """Azimuthal angle in degrees."""
        if "phi_deg" not in self._cache:
            self._cache["phi_deg"] = np.degrees(self.phi).astype(np.float32)
        return self._cache["phi_deg"]

    @property
    def theta_deg(self) -> np.ndarray:
        """Polar angle in degrees."""
        if "theta_deg" not in self._cache:
            self._cache["theta_deg"] = np.degrees(self.theta).astype(np.float32)
        return self._cache["theta_deg"]

    @property
    def eta(self) -> np.ndarray:
        """Pseudorapidity (eta = -ln(tan(theta/2)))."""
        if "eta" not in self._cache:
            theta = self.theta
            small = np.float32(1e-8)
            # Avoid singularities at theta ≈ 0, π
            theta_clipped = np.clip(theta, small, np.pi - small)
            self._cache["eta"] = (-np.log(np.tan(theta_clipped / 2.0))).astype(np.float32)
        return self._cache["eta"]

    @property
    def y(self) -> np.ndarray:
        """Rapidity (y = 0.5 * ln((E + pz) / (E - pz)))."""
        if "y" not in self._cache:
            E = self.E
            pz = self.pz
            
            y_vals = np.zeros_like(E, dtype=np.float32)
            
            # Compute rapidity where denominator is valid
            mask = np.abs(E - pz) > 1e-8
            numerator = E[mask] + pz[mask]
            denominator = E[mask] - pz[mask]
            
            with np.errstate(divide='ignore', invalid='ignore'):
                y_vals[mask] = (0.5 * np.log(numerator / denominator)).astype(np.float32)
            
            self._cache["y"] = y_vals
        return self._cache["y"]

    @property
    def pdg(self) -> Optional[np.ndarray]:
        """PDG particle IDs (if available)."""
        if "pdg" not in self._cache:
            try:
                pdg_vals = np.array(
                    [p.particle_id for p in self.particles],
                    dtype=np.int32,
                )
                self._cache["pdg"] = pdg_vals
            except (AttributeError, ValueError):
                self._cache["pdg"] = None
        return self._cache["pdg"]

    def _pid_mask(self, species: str) -> np.ndarray:
        """Get particle ID mask for specified species."""
        if self.pdg is None:
            return np.ones_like(self.pt, dtype=bool)
        
        species = species.lower()
        if species == "pi+":
            return self.pdg == 211
        elif species == "pi-":
            return self.pdg == -211
        elif species == "p":
            return self.pdg == 2212
        elif species == "pbar":
            return self.pdg == -2212
        elif species == "k+":
            return self.pdg == 321
        elif species == "k-":
            return self.pdg == -321
        else:
            return np.ones_like(self.pt, dtype=bool)

    def multiplicity_charged_midrap(self, eta_cut: float = 1.0) -> Dict[str, float]:
        """Get charged multiplicity in midrapidity region."""
        if self.pdg is None:
            return {"Nch": float(len(self.pt))}
        
        charged_mask = np.isin(np.abs(self.pdg), [211, 321, 2212])  # π, K, p
        eta_mask = np.abs(self.eta) < eta_cut
        mask = charged_mask & eta_mask
        Nch = int(mask.sum())
        return {"Nch": float(Nch)}

    # -------- Plotting methods - General Kinematics --------

    def plot_rapidity(
        self,
        out: Optional[str] = None,
        y_range: Tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        """Plot rapidity distribution."""
        plt.figure(figsize=(7, 5))
        mask = (self.y > -100) & (self.y < 100)
        plt.hist(self.y[mask], bins=40, range=y_range, color="steelblue", edgecolor="black", alpha=0.75)
        plt.xlabel("Rapidity y")
        plt.ylabel("dN/dy (arb.)")
        plt.title(f"Rapidity distribution\n{self.system_label}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_pseudorapidity(
        self,
        out: Optional[str] = None,
        eta_range: Tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        """Plot pseudorapidity distribution."""
        plt.figure(figsize=(7, 5))
        mask = np.isfinite(self.eta)
        plt.hist(self.eta[mask], bins=40, range=eta_range, color="darkgreen", edgecolor="black", alpha=0.75)
        plt.xlabel("Pseudorapidity η")
        plt.ylabel("dN/dη (arb.)")
        plt.title(f"Pseudorapidity distribution\n{self.system_label}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_pt(self, out: Optional[str] = None, pt_max: float = 3.0) -> None:
        """Plot transverse momentum spectrum."""
        plt.figure(figsize=(7, 5))
        mask = (self.pt >= 0) & (self.pt < pt_max)
        plt.hist(
            self.pt[mask],
            bins=50,
            range=(0, pt_max),
            color="teal",
            edgecolor="black",
            alpha=0.75,
        )
        plt.xlabel(r"$p_T$ (GeV)")
        plt.ylabel("dN/dpT (arb.)")
        plt.title(f"Transverse momentum spectrum\n{self.system_label}")
        plt.yscale("log")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_y_pt(
        self,
        out: Optional[str] = None,
        y_range: Tuple[float, float] = (-2.0, 2.0),
        pt_max: float = 3.0,
    ) -> None:
        """Plot 2D correlation: y vs pT."""
        plt.figure(figsize=(7, 5))
        y = np.clip(self.y, y_range, y_range)
        pt = np.clip(self.pt, 0, pt_max)
        h, xedges, yedges, im = plt.hist2d(
            y,
            pt,
            bins=[40, 40],
            range=[[y_range, y_range], [0, pt_max]],
            cmap="viridis",
        )
        plt.xlabel("y")
        plt.ylabel(r"$p_T$ (GeV)")
        plt.title(f"2D correlation: y vs pT\n{self.system_label}")
        cbar = plt.colorbar(im)
        cbar.set_label("Counts")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_eta_phi(
        self,
        out: Optional[str] = None,
        eta_range: Tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        """Plot 2D correlation: η vs φ."""
        plt.figure(figsize=(7, 5))
        eta = np.clip(self.eta, eta_range, eta_range)
        phi = self.phi
        h, xedges, yedges, im = plt.hist2d(
            eta,
            phi,
            bins=[40, 40],
            range=[[eta_range, eta_range], [0, 2 * np.pi]],
            cmap="plasma",
        )
        plt.xlabel("η")
        plt.ylabel("φ (rad)")
        plt.title(f"2D correlation: η vs φ\n{self.system_label}")
        cbar = plt.colorbar(im)
        cbar.set_label("Counts")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    # -------- Identified Particle Spectra --------

    def plot_pid_pt(self, species: str, out: Optional[str] = None,
                    pt_max: float = 3.0, y_window: Tuple[float, float] = (-0.5, 0.5)) -> None:
        """Plot identified particle pT spectrum."""
        mask = self._pid_mask(species)
        mask &= (self.y >= y_window) & (self.y <= y_window)
        if not mask.any():
            return

        plt.figure(figsize=(7, 5))
        plt.hist(self.pt[mask], bins=40, range=(0, pt_max),
                 color="steelblue", edgecolor="black", alpha=0.8)
        plt.xlabel(r"$p_T$ (GeV)")
        plt.ylabel("dN/dpT (arb.)")
        plt.yscale("log")
        plt.title(f"{species} pT spectrum, {y_window}<y<{y_window}\n{self.system_label}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_pid_rapidity(self, species: str, out: Optional[str] = None,
                          y_range: Tuple[float, float] = (-2.0, 2.0)) -> None:
        """Plot identified particle rapidity distribution."""
        mask = self._pid_mask(species)
        if not mask.any():
            return

        plt.figure(figsize=(7, 5))
        plt.hist(self.y[mask], bins=40, range=y_range,
                 color="darkorange", edgecolor="black", alpha=0.8)
        plt.xlabel("Rapidity y")
        plt.ylabel("dN/dy (arb.)")
        plt.title(f"{species} rapidity distribution\n{self.system_label}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    # -------- Elliptic Flow (v2) --------

    def estimate_v2(self, pt_bins: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Basic elliptic flow (v2) estimator."""
        if pt_bins is None:
            pt_bins = np.linspace(0, 3.0, 7)  # 0–3 GeV, 6 bins

        Qx = np.sum(self.pt * np.cos(2 * self.phi))
        Qy = np.sum(self.pt * np.sin(2 * self.phi))
        psi2 = 0.5 * np.arctan2(Qy, Qx)

        v2_vals = []
        pt_centers = []

        for i in range(len(pt_bins) - 1):
            lo, hi = pt_bins[i], pt_bins[i+1]
            mask = (self.pt >= lo) & (self.pt < hi)
            
            if not mask.any():
                v2_vals.append(0.0)
                pt_centers.append(0.5 * (lo + hi))
                continue

            v2_bin = np.mean(np.cos(2 * (self.phi[mask] - psi2)))
            v2_vals.append(v2_bin)
            pt_centers.append(0.5 * (lo + hi))

        return {
            "pt_centers": np.array(pt_centers),
            "v2": np.array(v2_vals),
            "psi2": psi2,
        }

    def plot_v2(self, out: Optional[str] = None) -> None:
        """Plot elliptic flow v2."""
        res = self.estimate_v2()

        plt.figure(figsize=(7, 5))
        plt.plot(res["pt_centers"], res["v2"], "o-", color="purple")
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel(r"$p_T$ (GeV)")
        plt.ylabel(r"$v_2(p_T)$")
        plt.title(f"Basic elliptic flow estimate v2(pT)\n{self.system_label}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    def clear_cache(self) -> None:
        """Explicitly clear cache to free memory after use."""
        self._cache.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.clear_cache()