"""
TLI GUI — Translunar Injection Educational Tool  v2
====================================================
2D coplanar Earth-centred two-body model.
Educational goal: show that a TLI intercepts the Moon's FUTURE position.

v2 changes vs v1
----------------
* Departure angle driven by "Transfer Arc" slider instead of fixed antipode.
  Default 160° → Δv ≈ 3.2 km/s (near-Hohmann, matches published Artemis imagery).
* "Min Δv" button sweeps arc angles and picks the cheapest solution.
* Kepler propagator uses proper Newton-Raphson (guaranteed convergence).
* Second subplot: Δv vs Transfer-Arc curve with current point highlighted.
* Info bar shows actual geocentric arc angle, apogee, Δv.

Physics note on the arc slider
-------------------------------
Transfer arc = geocentric angle (departure → Moon-future).
  arc ≈ 180°  →  near-Hohmann, apogee ≈ Moon distance, lowest Δv (~3.1 km/s)
  arc < 180°  →  Moon intercepted on ascending leg, higher Δv, shorter flight path
The minimum-Δv solution is always close to 180° for a direct transfer.
"""

import sys
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QGridLayout,
    QSizePolicy, QFrame, QTabWidget, QCheckBox, QFileDialog,
    QTextBrowser, QSplitter
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPalette, QColor, QFont

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.patches as mpa
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU_EARTH   = 3.986004418e14   # m³/s²
R_EARTH    = 6_371_000.0      # m
R_MOON_ORB = 384_400_000.0    # m
T_MOON     = 27.3217 * 86400  # s

# ---------------------------------------------------------------------------
# Stumpff functions (universal-variable Lambert)
# ---------------------------------------------------------------------------

def _C(psi):
    if psi > 1e-6:  return (1.0 - np.cos(np.sqrt(psi))) / psi
    if psi < -1e-6: return (np.cosh(np.sqrt(-psi)) - 1.0) / (-psi)
    return 0.5

def _S(psi):
    if psi > 1e-6:
        sp = np.sqrt(psi); return (sp - np.sin(sp)) / (psi * sp)
    if psi < -1e-6:
        sp = np.sqrt(-psi); return (np.sinh(sp) - sp) / ((-psi) * sp)
    return 1.0 / 6.0

# ---------------------------------------------------------------------------
# Lambert solver
# ---------------------------------------------------------------------------

def lambert_uv(r1_vec, r2_vec, tof, mu=MU_EARTH, prograde=True):
    """Universal-variable Lambert solver (Battin/Bate).
    Returns v1, v2 [m/s] as 2-D arrays. Raises on degenerate geometry."""
    r1 = np.append(np.asarray(r1_vec, float), 0.0)
    r2 = np.append(np.asarray(r2_vec, float), 0.0)
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    cos_dnu = np.clip(np.dot(r1, r2) / (n1 * n2), -1.0, 1.0)
    cz = r1[0]*r2[1] - r1[1]*r2[0]
    dnu = np.arccos(cos_dnu) if (cz >= 0) == prograde else 2*np.pi - np.arccos(cos_dnu)
    if abs(np.sin(dnu)) < 1e-7:
        raise ValueError(f"Degenerate geometry Δν={np.degrees(dnu):.1f}°")
    A = np.sin(dnu) * np.sqrt(n1 * n2 / (1.0 - cos_dnu))

    def F(psi):
        C, S = _C(psi), _S(psi)
        if C <= 0:
            return -1e12 - tof   # guard: C approaches 0 only at psi→-∞, safe to reject
        y = n1 + n2 + A*(psi*S - 1.0)/np.sqrt(C)
        if y < 0: return -1e12 - tof
        x = np.sqrt(y/C)
        return (x**3*S + A*np.sqrt(y)) / np.sqrt(mu) - tof

    lo, hi = -4*np.pi**2, 4*np.pi**2
    for _ in range(400):
        if F(lo)*F(hi) < 0: break
        hi *= 1.5
    else:
        raise RuntimeError("Lambert: cannot bracket root.")
    psi_sol = brentq(F, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=1000)

    C, S = _C(psi_sol), _S(psi_sol)
    y  = n1 + n2 + A*(psi_sol*S - 1.0)/np.sqrt(C)
    f  = 1.0 - y/n1
    g  = A * np.sqrt(y/mu)
    gd = 1.0 - y/n2
    v1 = (r2 - f*r1) / g
    v2 = (gd*r2 - r1) / g
    return v1[:2], v2[:2]

# ---------------------------------------------------------------------------
# Orbital helpers
# ---------------------------------------------------------------------------

def circular_velocity(r, mu=MU_EARTH):
    return np.sqrt(mu / r)

def orbit_points(r, n=400):
    th = np.linspace(0, 2*np.pi, n)
    return r*np.cos(th), r*np.sin(th)

def moon_now(phase_deg):
    phi = np.radians(phase_deg)
    return R_MOON_ORB * np.array([np.cos(phi), np.sin(phi)])

def moon_future(phase_deg, tof_days):
    dphi = (2*np.pi / T_MOON) * tof_days * 86400.0
    phi  = np.radians(phase_deg) + dphi
    return R_MOON_ORB * np.array([np.cos(phi), np.sin(phi)])

def departure_point(r_leo, future_moon_angle_deg, arc_deg):
    """
    Departure on LEO circle such that geocentric angle
    departure → Moon-future = arc_deg (prograde).
    """
    angle = np.radians(future_moon_angle_deg - arc_deg)
    return r_leo * np.array([np.cos(angle), np.sin(angle)])

def kepler_nr(M, ecc, tol=1e-12, maxiter=80):
    """Newton-Raphson solution of M = E − e sin E."""
    E = M + ecc*np.sin(M)
    for _ in range(maxiter):
        dE = (M - E + ecc*np.sin(E)) / (1.0 - ecc*np.cos(E))
        E += dE
        if abs(dE) < tol: break
    return E

def propagate_keplerian(r0, v0, tof, mu=MU_EARTH, n=500):
    """
    Propagate Keplerian 2-D arc from (r0, v0) for time tof [s].
    Returns x_arr, y_arr in [m].
    """
    r0, v0 = np.asarray(r0, float), np.asarray(v0, float)
    nr0 = np.linalg.norm(r0)
    nv0 = np.linalg.norm(v0)
    rdv = np.dot(r0, v0)
    hz  = r0[0]*v0[1] - r0[1]*v0[0]

    energy = nv0**2/2.0 - mu/nr0
    if abs(energy) < 1e-3:
        raise ValueError("Near-parabolic, skip.")
    a = -mu / (2.0*energy)

    ecc_vec = ((nv0**2 - mu/nr0)*r0 - rdv*v0) / mu
    ecc     = max(np.linalg.norm(ecc_vec), 1e-10)
    omega   = np.arctan2(ecc_vec[1], ecc_vec[0])

    cos_nu0 = np.clip(np.dot(ecc_vec/ecc, r0/nr0), -1.0, 1.0)
    nu0 = np.arccos(cos_nu0)
    if rdv < 0: nu0 = 2*np.pi - nu0

    if ecc < 1.0:
        n_mean = np.sqrt(mu / a**3)
        E0 = 2*np.arctan2(np.sqrt(1-ecc)*np.sin(nu0/2),
                           np.sqrt(1+ecc)*np.cos(nu0/2))
        M0 = E0 - ecc*np.sin(E0)
        E1 = kepler_nr(M0 + n_mean*tof, ecc)
        nu1 = 2*np.arctan2(np.sqrt(1+ecc)*np.sin(E1/2),
                            np.sqrt(1-ecc)*np.cos(E1/2))
        nu_arr = np.linspace(nu0, nu1, n)
        p = a*(1 - ecc**2)
    else:
        nu_max = np.arccos(-1.0/ecc) - 0.02
        nu1    = min(nu0 + np.pi*0.8, nu_max*0.9)
        nu_arr = np.linspace(nu0, nu1, n)
        p = a*(ecc**2 - 1)

    r_arr = p / (1.0 + ecc*np.cos(nu_arr))
    mask  = r_arr > 0
    nu_arr, r_arr = nu_arr[mask], r_arr[mask]
    x_arr = r_arr * np.cos(nu_arr + omega)
    y_arr = r_arr * np.sin(nu_arr + omega)
    if hz < 0: y_arr = -y_arr
    return x_arr, y_arr

# ---------------------------------------------------------------------------
# Core TLI computation
# ---------------------------------------------------------------------------

def compute_tli(alt_km, tof_days, phase_deg, arc_deg):
    """
    Compute TLI quantities.
    Returns result dict or raises on failure.
    """
    r_leo = R_EARTH + alt_km * 1e3
    tof_s = tof_days * 86400.0

    r2  = moon_future(phase_deg, tof_days)
    fa  = np.degrees(np.arctan2(r2[1], r2[0]))
    r1  = departure_point(r_leo, fa, arc_deg)

    v1, v2 = lambert_uv(r1, r2, tof_s)

    dep_angle = np.arctan2(r1[1], r1[0])
    v_leo_c   = circular_velocity(r_leo)
    v_leo_v   = v_leo_c * np.array([-np.sin(dep_angle), np.cos(dep_angle)])

    dv    = np.linalg.norm(v1 - v_leo_v)
    en    = np.linalg.norm(v1)**2/2.0 - MU_EARTH/r_leo
    a_tr  = -MU_EARTH/(2*en) if en < 0 else None
    ecc_v = ((np.linalg.norm(v1)**2 - MU_EARTH/r_leo)*r1
             - np.dot(r1, v1)*v1) / MU_EARTH
    ecc   = np.linalg.norm(ecc_v)
    r_apo = (a_tr*(1+ecc)/1e3) if a_tr else np.linalg.norm(r2)/1e3

    cos_arc  = np.clip(np.dot(r1, r2)/(np.linalg.norm(r1)*np.linalg.norm(r2)), -1, 1)
    arc_true = np.degrees(np.arccos(cos_arc))

    return dict(
        dv        = dv / 1e3,
        v_leo     = v_leo_c / 1e3,
        v_dep     = np.linalg.norm(v1) / 1e3,
        v_arr     = np.linalg.norm(v2) / 1e3,
        r_apo_km  = r_apo,
        r1        = r1, r2        = r2,
        v1        = v1, v_leo_vec = v_leo_v,
        arc_true  = arc_true,
    )

def sweep_dv_vs_arc(alt_km, tof_days, phase_deg, n=160):
    """Sweep arc 91°–179° and return (arcs, dvs) arrays."""
    arcs = np.linspace(91, 179, n)
    dvs  = np.full(n, np.nan)
    for i, arc in enumerate(arcs):
        try:
            dvs[i] = compute_tli(alt_km, tof_days, phase_deg, arc)["dv"]
        except Exception:
            pass
    return arcs, dvs

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
DARK_BG      = "#0d1117"
PANEL_BG     = "#161b22"
ACCENT       = "#58a6ff"
ACCENT2      = "#3fb950"
ACCENT3      = "#f78166"
ACCENT4      = "#d2a8ff"
TEXT_PRIMARY = "#e6edf3"
TEXT_DIM     = "#8b949e"
BORDER       = "#30363d"

STYLE = f"""
QMainWindow,QWidget{{background:{DARK_BG};color:{TEXT_PRIMARY};
    font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif;
    font-size:13px;}}
QGroupBox{{background:{PANEL_BG};border:1px solid {BORDER};border-radius:6px;
    margin-top:14px;padding:14px 8px 8px 8px;font-weight:bold;font-size:13px;color:{ACCENT};}}
QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 5px;font-size:13px;}}
QLabel{{color:{TEXT_PRIMARY};font-size:13px;}}
QSlider::groove:horizontal{{border:none;height:5px;background:{BORDER};border-radius:3px;}}
QSlider::handle:horizontal{{background:{ACCENT};border:none;
    width:16px;height:16px;margin:-6px 0;border-radius:8px;}}
QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:3px;}}
QPushButton{{background:{ACCENT};color:#0d1117;border:none;border-radius:6px;
    padding:9px 14px;font-weight:bold;font-size:13px;}}
QPushButton:hover{{background:#79c0ff;}}
QPushButton:pressed{{background:#388bfd;}}
QPushButton#opt{{background:{ACCENT4};color:#0d1117;}}
QPushButton#opt:hover{{background:#e0b9ff;}}
"""

# ---------------------------------------------------------------------------
# CR3BP physics  — planar Earth-Moon rotating frame
# ---------------------------------------------------------------------------
# Conventions (all quantities dimensionless unless stated):
#   μ  = M_Moon / (M_Earth + M_Moon) ≈ 0.01215
#   Length unit  L* = Earth-Moon distance
#   Time unit    T* = 1 / ω  where ω = mean motion of Earth-Moon system
#   Earth at (-μ, 0),  Moon at (1-μ, 0)  in rotating frame
#
# Equations of motion (planar CR3BP):
#   ẍ - 2ẏ = ∂Ω/∂x
#   ÿ + 2ẋ = ∂Ω/∂y
# where Ω = ½(x²+y²) + (1-μ)/r1 + μ/r2   (effective potential)

MU_MOON    = 4.902800066e12    # m³/s²  (Moon gravitational parameter)
MU_SYS     = MU_EARTH + MU_MOON
MU_CR3BP   = MU_MOON / MU_SYS           # dimensionless mass ratio ≈ 0.01215
L_STAR     = R_MOON_ORB                  # length unit [m]
T_STAR     = T_MOON / (2 * np.pi)       # time unit [s]  (= 1/ω)
V_STAR     = L_STAR / T_STAR            # velocity unit [m/s]


def cr3bp_eom(t, state, mu=MU_CR3BP):
    """
    CR3BP equations of motion in the rotating frame (dimensionless).
    state = [x, y, vx, vy]
    Returns [vx, vy, ax, ay].
    """
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2      + y**2)   # distance to Earth
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)  # distance to Moon

    # Partial derivatives of effective potential
    dOdx = x - (1-mu)*(x+mu)/r1**3 - mu*(x-(1-mu))/r2**3
    dOdy = y - (1-mu)*y/r1**3       - mu*y/r2**3

    ax = dOdx + 2*vy     # Coriolis: +2ẏ
    ay = dOdy - 2*vx     # Coriolis: -2ẋ

    return [vx, vy, ax, ay]


def jacobi_constant(state, mu=MU_CR3BP):
    """Jacobi constant C = 2Ω - v²  (conserved in CR3BP)."""
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2      + y**2)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2)
    Omega = 0.5*(x**2 + y**2) + (1-mu)/r1 + mu/r2
    v2    = vx**2 + vy**2
    return 2*Omega - v2


def lagrange_points(mu=MU_CR3BP):
    """
    Compute collinear Lagrange points L1, L2, L3 and equilateral L4, L5.
    Returns dict {name: (x, y)} in dimensionless units (barycentre origin).
    """
    from scipy.optimize import brentq as _brent

    def dOdx_collinear(x):
        # Derivative of effective potential on y=0 axis
        # Sign of r1, r2 matters: use signed distances
        sgn1 = 1.0 if (x + mu) >= 0 else -1.0
        sgn2 = 1.0 if (x - (1-mu)) >= 0 else -1.0
        r1 = abs(x + mu)
        r2 = abs(x - (1-mu))
        if r1 < 1e-10 or r2 < 1e-10: return 1e10
        return x - (1-mu)*sgn1/r1**2 - mu*sgn2/r2**2

    # L1: between Earth and Moon, 0 < x < 1-μ
    L1x = _brent(dOdx_collinear, -mu + 0.01, 1-mu - 0.01)
    # L2: beyond Moon, x > 1-μ
    L2x = _brent(dOdx_collinear, 1-mu + 0.01, 2.0)
    # L3: opposite Moon, x < -μ
    L3x = _brent(dOdx_collinear, -2.0, -mu - 0.01)

    return {
        "L1": (L1x, 0.0),
        "L2": (L2x, 0.0),
        "L3": (L3x, 0.0),
        "L4": (0.5 - mu,  np.sqrt(3)/2),
        "L5": (0.5 - mu, -np.sqrt(3)/2),
    }


def inertial_to_rotating(r_i, v_i, t, omega=1.0):
    """
    Convert inertial-frame state (r, v) to rotating-frame state.
    t in dimensionless time units, omega = 1 (rotating frame angular rate).
    All quantities dimensionless.
    """
    ct, st = np.cos(omega*t), np.sin(omega*t)
    # Position: rotate by -ωt
    x =  ct*r_i[0] + st*r_i[1]
    y = -st*r_i[0] + ct*r_i[1]
    # Velocity: d/dt of rotated position
    vx_rot =  ct*v_i[0] + st*v_i[1] + omega*y
    vy_rot = -st*v_i[0] + ct*v_i[1] - omega*x
    return np.array([x, y, vx_rot, vy_rot])


def lambert_to_cr3bp_ic(r1_m, v1_ms, phase_deg):
    """
    Convert Lambert departure state (inertial, SI, Earth-centred) to
    CR3BP initial conditions (rotating frame, dimensionless, barycentre-centred).

    At t=0 the rotating frame is aligned so the Moon is on the +x axis.
    Earth is at (-μ, 0), Moon at (1-μ, 0) in the rotating frame.

    The rotating frame angular velocity ω = 1 (dimensionless).
    """
    mu = MU_CR3BP

    # --- Step 1: SI → dimensionless ---
    r_nd = np.asarray(r1_m,   float) / L_STAR
    v_nd = np.asarray(v1_ms,  float) / V_STAR

    # --- Step 2: shift from Earth-centred to barycentre-centred inertial ---
    # Moon in inertial frame (dimensionless, Earth-centred) at t=0:
    phi0 = np.radians(phase_deg)
    r_moon_nd = np.array([np.cos(phi0), np.sin(phi0)])   # |r_moon| = 1
    # Barycentre in Earth-centred inertial = μ * r_moon
    r_bary_nd = mu * r_moon_nd

    r_bc = r_nd - r_bary_nd    # position in barycentre-centred inertial
    v_bc = v_nd                  # barycentre is inertial → same velocity

    # --- Step 3: rotate inertial barycentre frame → rotating frame ---
    # Rotating frame is aligned with Moon at t=0, i.e. rotated by +phi0.
    # To go from inertial to rotating: rotate by -phi0.
    ct, st = np.cos(-phi0), np.sin(-phi0)
    x  =  ct*r_bc[0] - st*r_bc[1]
    y  =  st*r_bc[0] + ct*r_bc[1]

    # Velocity in rotating frame: v_rot = R(-φ0)·v_bc − ω × r_rot
    # ω = ẑ → ω × r_rot = (-y, x) in 2-D  →  v_rot = R·v_bc + (y, -x)
    # Note sign: v_rot = v_inertial_rotated + ω × r   (Coriolis correction)
    vx =  ct*v_bc[0] - st*v_bc[1] + y    # +ωy
    vy =  st*v_bc[0] + ct*v_bc[1] - x    # -ωx

    return np.array([x, y, vx, vy])


def rotating_to_inertial_traj(x_rot, y_rot, t_nd, phase_deg):
    """
    Convert rotating-frame trajectory back to inertial frame.
    t_nd : dimensionless time array
    Returns x_i, y_i arrays (dimensionless, barycentre origin).
    """
    theta = np.radians(phase_deg) + t_nd   # rotating frame angle in inertial
    ct, st = np.cos(theta), np.sin(theta)
    x_i = ct*x_rot - st*y_rot
    y_i = st*x_rot + ct*y_rot
    # Shift from barycentre to Earth origin:
    # Earth in barycentre frame is at -μ on rotating x-axis → inertial position
    x_earth_i = -MU_CR3BP * ct
    y_earth_i = -MU_CR3BP * st
    x_i += x_earth_i  # this is bary→Earth correction, but we want Earth-centred
    y_i += y_earth_i
    return x_i, y_i


def run_cr3bp(r1_m, v1_ms, phase_deg, tof_days, n_points=3000):
    """
    Integrate CR3BP trajectory.

    Parameters
    ----------
    r1_m      : departure position [m], inertial, Earth-centred
    v1_ms     : departure velocity [m/s], inertial
    phase_deg : Moon angle at departure [°] (inertial)
    tof_days  : integration time [days]

    Returns
    -------
    dict with:
      t_nd          : dimensionless time array
      xy_rot        : (2, N) rotating-frame positions (dimensionless)
      xy_inertial   : (2, N) inertial positions [m], Earth-centred
      state0        : initial CR3BP state
      closest_moon_km : closest approach to Moon [km]
      jacobi_0      : Jacobi constant at t=0
      jacobi_f      : Jacobi constant at t=final (should be ≈ jacobi_0)
      success       : bool
    """
    tof_nd = tof_days * 86400.0 / T_STAR   # dimensionless TOF

    # Initial conditions in rotating frame
    state0 = lambert_to_cr3bp_ic(r1_m, v1_ms, phase_deg)

    # Moon position in rotating frame is always at (1-μ, 0) by definition
    moon_rot = np.array([1.0 - MU_CR3BP, 0.0])

    # Event: stop if spacecraft crashes into Earth or Moon
    def hit_earth(t, s, mu=MU_CR3BP):
        return np.sqrt((s[0]+mu)**2 + s[1]**2) - R_EARTH/L_STAR
    hit_earth.terminal  = True
    hit_earth.direction = -1

    def hit_moon(t, s, mu=MU_CR3BP):
        return np.sqrt((s[0]-(1-mu))**2 + s[1]**2) - 1737e3/L_STAR
    hit_moon.terminal  = True
    hit_moon.direction = -1

    t_eval = np.linspace(0, tof_nd, n_points)

    sol = solve_ivp(
        cr3bp_eom, [0, tof_nd], state0,
        method="DOP853",
        t_eval=t_eval,
        events=[hit_earth, hit_moon],
        rtol=1e-10, atol=1e-12,
        dense_output=False,
    )

    t_nd  = sol.t
    x_rot = sol.y[0]
    y_rot = sol.y[1]

    # Closest approach to Moon (rotating frame)
    dist_moon_nd = np.sqrt((x_rot - moon_rot[0])**2 + y_rot**2)
    closest_moon_km = np.min(dist_moon_nd) * L_STAR / 1e3

    # Jacobi constant conservation check
    j0 = jacobi_constant(state0)
    jf = jacobi_constant(sol.y[:4, -1])

    # Convert rotating→inertial (Earth-centred, SI)
    # Angle of rotating frame at each time step = phase_deg + t_nd (in rad)
    theta = np.radians(phase_deg) + t_nd
    ct, st = np.cos(theta), np.sin(theta)
    # Barycentre origin: shift by Earth offset from barycentre
    # Earth in rotating frame is at (-μ, 0), so in inertial barycentre frame:
    # r_Earth_i = (-μ cos θ, -μ sin θ)  → to get Earth-centred, add μ·(cosθ, sinθ)
    x_i = (ct*x_rot - st*y_rot + MU_CR3BP*ct) * L_STAR
    y_i = (st*x_rot + ct*y_rot + MU_CR3BP*st) * L_STAR

    return dict(
        t_nd            = t_nd,
        xy_rot          = np.vstack([x_rot, y_rot]),
        xy_inertial     = np.vstack([x_i, y_i]),
        state0          = state0,
        closest_moon_km = closest_moon_km,
        jacobi_0        = j0,
        jacobi_f        = jf,
        success         = sol.success,
        msg             = sol.message,
    )


# ---------------------------------------------------------------------------
# Value label
# ---------------------------------------------------------------------------

class VLabel(QLabel):
    def __init__(self, accent=ACCENT):
        super().__init__("—")
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setStyleSheet(f"color:{accent};font-size:18px;font-weight:bold;")


# ---------------------------------------------------------------------------
# CR3BP physics helpers — Zero Velocity Curves & free-return
# ---------------------------------------------------------------------------

def effective_potential(x, y, mu=MU_CR3BP):
    """Ω(x,y) = ½(x²+y²) + (1-μ)/r1 + μ/r2"""
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - (1-mu))**2 + y**2)
    return 0.5*(x**2 + y**2) + (1-mu)/r1 + mu/r2


def run_cr3bp_full(r1_m, v1_ms, phase_deg,
                   tof_days=8.0, n_points=6000,
                   v1_rotate_deg=0.0):
    """
    Integrate the CR3BP trajectory starting from the Lambert IC,
    optionally rotating v1 by v1_rotate_deg degrees in the orbital plane
    (B-plane targeting: changes Moon closest-approach distance without
    changing departure speed).

    Integration runs for tof_days (default 8 d) to capture the full
    figure-8 free-return arc (TLI at t=0, lunar flyby ~t=2.8 d,
    Earth return ~t=6.5 d for the Artemis-like baseline).

    Returns
    -------
    dict with keys:
      t_nd, xy_rot, xy_inertial,
      state0, jacobi_0, jacobi_f,
      closest_moon_km, closest_earth_return_km,
      t_flyby_d, t_return_d,
      is_free_return (bool),
      success, msg, v1_rotate_deg
    """
    mu     = MU_CR3BP
    tof_nd = tof_days * 86400.0 / T_STAR

    # --- B-plane targeting: rotate v1 ---
    if abs(v1_rotate_deg) > 1e-9:
        angle = np.radians(v1_rotate_deg)
        c, s  = np.cos(angle), np.sin(angle)
        v1_use = np.array([c*v1_ms[0] - s*v1_ms[1],
                           s*v1_ms[0] + c*v1_ms[1]])
    else:
        v1_use = np.asarray(v1_ms, float)

    state0 = lambert_to_cr3bp_ic(r1_m, v1_use, phase_deg)
    j0     = jacobi_constant(state0)

    # --- Termination events ---
    r_earth_surface = R_EARTH / L_STAR

    def hit_earth(t, s):
        return np.sqrt((s[0]+mu)**2 + s[1]**2) - r_earth_surface * 1.02
    hit_earth.terminal = True; hit_earth.direction = -1

    def hit_moon(t, s):
        return np.sqrt((s[0]-(1-mu))**2 + s[1]**2) - 1737e3/L_STAR
    hit_moon.terminal = True; hit_moon.direction = -1

    t_eval = np.linspace(0, tof_nd, n_points)

    sol = solve_ivp(
        cr3bp_eom, [0, tof_nd], state0,
        method="DOP853",
        t_eval=t_eval,
        events=[hit_earth, hit_moon],
        rtol=1e-10, atol=1e-12,
        dense_output=False,
    )

    t_nd  = sol.t
    xr    = sol.y[0]; yr = sol.y[1]
    t_d   = t_nd * T_STAR / 86400.0

    # Distances (km)
    dist_moon_km  = np.sqrt((xr-(1-mu))**2 + yr**2) * L_STAR / 1e3
    dist_earth_km = np.sqrt((xr+mu)**2    + yr**2) * L_STAR / 1e3

    # Moon flyby
    idx_flyby = np.argmin(dist_moon_km)
    ca_moon_km = dist_moon_km[idx_flyby]
    t_flyby_d  = t_d[idx_flyby]

    # Earth return: only look AFTER the flyby (skip first half-day after departure)
    skip = max(idx_flyby + 5, int(0.5 * 86400 / T_STAR / (tof_nd/n_points)))
    if skip < len(dist_earth_km):
        earth_after     = dist_earth_km[skip:]
        idx_ret         = np.argmin(earth_after)
        ca_earth_ret_km = earth_after[idx_ret]
        t_return_d      = t_d[skip + idx_ret]
    else:
        ca_earth_ret_km = dist_earth_km[-1]
        t_return_d      = t_d[-1]

    # Free-return: passed Moon AND returned within ~100 000 km of Earth
    passed_moon   = ca_moon_km < 100_000
    returned_earth = ca_earth_ret_km < 100_000
    is_fr         = passed_moon and returned_earth

    # Rotating → inertial (Earth-centred, SI)
    theta = np.radians(phase_deg) + t_nd
    ct, st = np.cos(theta), np.sin(theta)
    xi = (ct*xr - st*yr + mu*ct) * L_STAR
    yi = (st*xr + ct*yr + mu*st) * L_STAR

    jf = jacobi_constant(sol.y[:4, -1])

    return dict(
        t_nd                = t_nd,
        xy_rot              = np.vstack([xr, yr]),
        xy_inertial         = np.vstack([xi, yi]),
        state0              = state0,
        jacobi_0            = j0,
        jacobi_f            = jf,
        closest_moon_km     = ca_moon_km,
        closest_earth_return_km = ca_earth_ret_km,
        reentry_alt_km      = max(ca_earth_ret_km - R_EARTH/1e3, 0.0),
        t_flyby_d           = t_flyby_d,
        t_return_d          = t_return_d,
        is_free_return      = is_fr,
        success             = sol.success,
        msg                 = sol.message,
        v1_rotate_deg       = v1_rotate_deg,
    )


def scan_bplane(r1_m, v1_ms, phase_deg, tof_days=8.0,
                delta_range=(-5.0, 5.0), n_scan=40):
    """
    Quick B-plane scan: rotate v1 across delta_range degrees,
    return arrays (deltas, ca_moon_km, ca_earth_km, is_fr).
    Used to build the B-plane map in the GUI.
    """
    deltas = np.linspace(delta_range[0], delta_range[1], n_scan)
    ca_moon  = np.full(n_scan, np.nan)
    ca_earth = np.full(n_scan, np.nan)
    is_fr    = np.zeros(n_scan, dtype=bool)

    tof_nd = tof_days * 86400.0 / T_STAR
    mu     = MU_CR3BP

    for i, d in enumerate(deltas):
        try:
            angle = np.radians(d)
            c, s  = np.cos(angle), np.sin(angle)
            v1r   = np.array([c*v1_ms[0]-s*v1_ms[1],
                               s*v1_ms[0]+c*v1_ms[1]])
            s0 = lambert_to_cr3bp_ic(r1_m, v1r, phase_deg)
            sol = solve_ivp(cr3bp_eom, [0, tof_nd], s0,
                            method="DOP853", rtol=1e-9, atol=1e-11,
                            t_eval=np.linspace(0, tof_nd, 2000),
                            dense_output=False)
            xr, yr = sol.y[0], sol.y[1]
            dm = np.sqrt((xr-(1-mu))**2+yr**2)*L_STAR/1e3
            de = np.sqrt((xr+mu)**2+yr**2)*L_STAR/1e3
            idx_m = np.argmin(dm)
            ca_moon[i]  = dm[idx_m]
            ca_earth[i] = np.min(de[idx_m+5:]) if idx_m+5 < len(de) else de[-1]
            is_fr[i]    = (ca_moon[i] < 100_000) and (ca_earth[i] < 100_000)
        except Exception:
            pass

    return deltas, ca_moon, ca_earth, is_fr


# ---------------------------------------------------------------------------
# CR3BP Tab widget
# ---------------------------------------------------------------------------

class CR3BPTab(QWidget):
    """
    Second tab: CR3BP dynamics in the Earth-Moon rotating frame.
    Features:
    - Direct CR3BP integration from Lambert IC
    - Free-return trajectory search (figure-8)
    - Zero Velocity Curves overlay
    - Correct visual scale: markers proportional to plot, not physical size
    - Lambert overlay in rotating frame
    - Lagrange points L1, L2
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lambert    = None
        self._result     = None
        self._fr_result  = None
        self._bplane_scan = None
        self._lagrange   = lagrange_points()
        self._show_fr    = False
        self._build_ui()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left = QWidget(); left.setFixedWidth(390)
        lv = QVBoxLayout(left); lv.setContentsMargins(0,0,0,0); lv.setSpacing(8)
        lv.addWidget(self._build_source_box())
        lv.addWidget(self._build_mode_box())
        lv.addWidget(self._build_options_box())
        lv.addWidget(self._build_metrics_box())
        lv.addStretch()

        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(0,0,0,0); rv.setSpacing(4)

        # Header row with save button
        hdr_row = QWidget()
        hdr_h = QHBoxLayout(hdr_row); hdr_h.setContentsMargins(0,0,0,0)
        hdr = QLabel("🌑  CR3BP Dynamics — Earth-Moon Rotating Frame")
        hdr.setStyleSheet(f"font-size:16px;font-weight:bold;color:{TEXT_PRIMARY};")
        self.btn_save = QPushButton("💾  Save image")
        self.btn_save.setFixedWidth(130)
        self.btn_save.setStyleSheet(
            f"background:{PANEL_BG};color:{TEXT_DIM};border:1px solid {BORDER};"
            f"border-radius:5px;padding:5px 10px;font-size:12px;")
        hdr_h.addWidget(hdr); hdr_h.addStretch(); hdr_h.addWidget(self.btn_save)
        rv.addWidget(hdr_row)

        # Canvas + matplotlib navigation toolbar
        self._canvas, self._toolbar = self._build_canvas()
        # Style the toolbar to match dark theme
        self._toolbar.setStyleSheet(
            f"background:{PANEL_BG};color:{TEXT_DIM};"
            f"border-bottom:1px solid {BORDER};")
        rv.addWidget(self._toolbar)
        rv.addWidget(self._canvas)

        sub = QLabel(
            "LEFT — Rotating frame (Earth+Moon fixed): the figure-8 topology is visible here. "
            "CENTRE — Flyby zoom (rotating frame). "
            "RIGHT — Inertial frame (geocentric, like Tab 1): shows the real sky trajectory — "
            "the figure-8 shape disappears because the Moon has moved during the 6-day trip."
        )
        sub.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
        sub.setWordWrap(True)
        rv.addWidget(sub)

        self.btn_save.clicked.connect(self._save_image)
        root.addWidget(left)
        root.addWidget(right, stretch=1)

    def _build_source_box(self):
        box = QGroupBox("Lambert Source  (Tab 1)")
        g = QGridLayout(box); g.setSpacing(3)
        def kv(label, accent=ACCENT):
            l = QLabel(label); l.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            v = QLabel("—"); v.setAlignment(Qt.AlignRight)
            v.setStyleSheet(f"color:{accent};font-size:12px;font-weight:bold;")
            return l, v
        (la, self._la),(lt, self._lt),(ld, self._ld),(lp, self._lp) = [
            kv("LEO alt"), kv("TOF"), kv("TLI Δv", ACCENT3), kv("Moon phase")]
        for i,(l,v) in enumerate([(la,self._la),(lt,self._lt),(ld,self._ld),(lp,self._lp)]):
            g.addWidget(l,i,0); g.addWidget(v,i,1)
        return box

    def _build_mode_box(self):
        box = QGroupBox("Simulation")
        g = QGridLayout(box); g.setSpacing(7)

        # B-plane angle slider  (−5° … +5°)
        lbl_bp = QLabel("B-plane angle δ  [°]")
        lbl_bp.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
        self.lbl_bp_val = QLabel("0.0°")
        self.lbl_bp_val.setAlignment(Qt.AlignRight)
        self.lbl_bp_val.setStyleSheet(f"color:{ACCENT};font-size:12px;font-weight:bold;")

        self.sl_bplane = QSlider(Qt.Horizontal)
        self.sl_bplane.setRange(-50, 50)    # ×0.1 → -5.0 … +5.0 deg
        self.sl_bplane.setValue(0)
        self.sl_bplane.valueChanged.connect(self._on_bplane_slide)

        note = QLabel(
            "δ rotates v₁ in the orbital plane.\n"
            "Changes Moon closest-approach (CA)\nwithout altering departure speed.")
        note.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;")
        note.setWordWrap(True)

        self.btn_run = QPushButton("▶  Integrate CR3BP  (8 days)")
        self.btn_run.setStyleSheet(
            f"background:{ACCENT};color:#0d1117;border:none;border-radius:6px;"
            f"padding:9px 8px;font-weight:bold;font-size:12px;")

        self.btn_scan = QPushButton("📊  B-plane scan")
        self.btn_scan.setStyleSheet(
            f"background:{PANEL_BG};color:{TEXT_DIM};border:1px solid {BORDER};"
            f"border-radius:6px;padding:7px 8px;font-size:11px;")

        self.lbl_mode = QLabel("Go to Tab 1, compute Lambert, then run here.")
        self.lbl_mode.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        self.lbl_mode.setWordWrap(True)

        g.addWidget(lbl_bp,          0, 0); g.addWidget(self.lbl_bp_val, 0, 1)
        g.addWidget(self.sl_bplane,  1, 0, 1, 2)
        g.addWidget(note,            2, 0, 1, 2)
        g.addWidget(self.btn_run,    3, 0, 1, 2)
        g.addWidget(self.btn_scan,   4, 0, 1, 2)
        g.addWidget(self.lbl_mode,   5, 0, 1, 2)

        self.btn_run.clicked.connect(self._run_direct)
        self.btn_scan.clicked.connect(self._run_scan)
        return box

    def _build_options_box(self):
        box = QGroupBox("Display Options")
        g = QGridLayout(box); g.setSpacing(5)

        def chk(label, desc, checked=True):
            w = QCheckBox(label)
            w.setChecked(checked)
            w.setStyleSheet(f"color:{TEXT_PRIMARY};font-size:12px;font-weight:bold;")
            d = QLabel(desc)
            d.setStyleSheet(f"color:{TEXT_DIM};font-size:10px;padding-left:20px;")
            d.setWordWrap(True)
            return w, d

        self.chk_overlay,  d1 = chk(
            "Lambert overlay",
            "Dashed: 2-body Keplerian arc in rotating frame. "
            "Shows where lunar gravity deviates the real path.")
        self.chk_zvc,      d2 = chk(
            "Zero Velocity Curves",
            "Forbidden regions (2Omega < C). The trajectory cannot "
            "cross into the shaded area for the current Jacobi constant.")
        self.chk_lagrange, d3 = chk(
            "Lagrange points L1, L2",
            "L1 (between Earth-Moon) and L2 (beyond Moon). "
            "The free-return corridor passes near L1.")

        row = 0
        for w, d in [(self.chk_overlay, d1), (self.chk_zvc, d2),
                     (self.chk_lagrange, d3)]:
            g.addWidget(w, row, 0, 1, 2);   row += 1
            g.addWidget(d, row, 0, 1, 2);   row += 1
            w.stateChanged.connect(self._replot)

        return box

    def _build_metrics_box(self):
        box = QGroupBox("Results")
        g = QGridLayout(box); g.setSpacing(2)
        g.setColumnStretch(0, 1); g.setColumnStretch(1, 1)

        def mrow(label, acc=ACCENT, big=False):
            fs_lbl = "11px"
            fs_val = "20px" if big else "16px"
            l = QLabel(label)
            l.setStyleSheet(f"color:{TEXT_DIM};font-size:{fs_lbl};padding-top:4px;")
            v = QLabel("—")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v.setStyleSheet(f"color:{acc};font-size:{fs_val};font-weight:bold;")
            return l, v

        rows = [
            ("Periselene  [km]",       ACCENT3,  True ),
            ("Periselene  [R☽]",       ACCENT3,  False),
            ("Perigee return  [km]",   ACCENT2,  True ),
            ("Return altitude  [km]",  ACCENT2,  False),
            ("Flyby time  [days]",     TEXT_PRIMARY, False),
            ("Return time  [days]",    TEXT_PRIMARY, False),
            ("Free-return?",           ACCENT4,  True ),
            ("Jacobi  C",             ACCENT4,  False),
            ("Jacobi  ΔC  (error)",   TEXT_DIM,  False),
        ]
        attrs = ["met_ca_km","met_ca_rm","met_earth","met_reentry",
                 "met_tflyby","met_treturn","met_fr","met_jacobi","met_dc"]

        row = 0
        for (lbl, acc, big), attr in zip(rows, attrs):
            l, v = mrow(lbl, acc, big)
            g.addWidget(l, row, 0);   g.addWidget(v, row, 1)
            setattr(self, attr, v)
            row += 1

        return box

    def _build_canvas(self):
        self._fig = Figure(facecolor=DARK_BG)
        gs = gridspec.GridSpec(1, 3, figure=self._fig,
                               width_ratios=[1.4, 0.85, 1.4],
                               wspace=0.10,
                               left=0.05, right=0.98,
                               top=0.93, bottom=0.09)
        self._ax        = self._fig.add_subplot(gs[0])   # rotating frame
        self._ax_flyby  = self._fig.add_subplot(gs[1])   # flyby zoom
        self._ax_inert  = self._fig.add_subplot(gs[2])   # inertial frame
        for ax in (self._ax, self._ax_flyby, self._ax_inert):
            self._style_ax(ax)
        c = FigureCanvas(self._fig)
        c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tb = NavToolbar(c, None)
        return c, tb

    def _style_ax(self, ax):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.3, alpha=0.35)

    # ---------------------------------------------------------- public API --

    def set_lambert_result(self, ldict, alt_km, tof_days, phase_deg, arc_deg):
        self._lambert    = ldict
        self._alt_km     = alt_km
        self._tof_days   = tof_days
        self._phase_deg  = phase_deg
        self._result     = None
        self._fr_result  = None
        self._bplane_scan = None
        self._la.setText(f"{alt_km} km")
        self._lt.setText(f"{tof_days:.1f} d")
        self._ld.setText(f"{ldict['dv']:.3f} km/s")
        self._lp.setText(f"{phase_deg}°")
        self._draw_empty()

    def _save_image(self):
        """Save current figure to PNG/PDF/SVG chosen by user."""
        path, filt = QFileDialog.getSaveFileName(
            self, "Save CR3BP Figure", "cr3bp_trajectory.png",
            "PNG image (*.png);;PDF vector (*.pdf);;SVG vector (*.svg)")
        if not path:
            return
        ext = path.rsplit(".", 1)[-1].lower()
        dpi = 200 if ext == "png" else 150
        try:
            self._fig.savefig(path, dpi=dpi, facecolor=DARK_BG,
                              bbox_inches="tight")
            self.lbl_mode.setText(f"✓ Saved → {path.split('/')[-1]}")
        except Exception as e:
            self.lbl_mode.setText(f"Save error: {e}")

    def _on_bplane_slide(self):
        v = self.sl_bplane.value() / 10.0
        self.lbl_bp_val.setText(f"{v:+.1f}°")

    def _busy(self, label):
        self.btn_run.setEnabled(False); self.btn_run.setText("⏳  Running…")
        self.btn_scan.setEnabled(False)
        self.lbl_mode.setText(label)
        QApplication.processEvents()

    def _done(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  Integrate CR3BP  (8 days)")
        self.btn_scan.setEnabled(True)

    def _run_direct(self):
        if self._lambert is None:
            self.lbl_mode.setText("⚠ Compute Lambert solution in Tab 1 first.")
            return
        delta = self.sl_bplane.value() / 10.0
        self._busy(f"Integrating 8 days (δ={delta:+.1f}°)…")
        try:
            self._result = run_cr3bp_full(
                self._lambert["r1"], self._lambert["v1"],
                self._phase_deg, tof_days=8.0,
                v1_rotate_deg=delta)
            self._show_fr = self._result["is_free_return"]
            self._update_metrics(self._result)
            fr_str = "✓ Free-return confirmed" if self._show_fr else "ℹ Direct trajectory (no return)"
            self.lbl_mode.setText(
                f"{fr_str}\n"
                f"Moon CA: {self._result['closest_moon_km']:,.0f} km  |  "
                f"Earth return: {self._result['closest_earth_return_km']:,.0f} km")
            self._replot()
        except Exception as e:
            self.lbl_mode.setText(f"Error: {e}")
        finally:
            self._done()

    def _run_scan(self):
        if self._lambert is None:
            self.lbl_mode.setText("⚠ Compute Lambert solution in Tab 1 first.")
            return
        self._busy("Running B-plane scan (−5° … +5°)…")
        try:
            deltas, ca_m, ca_e, is_fr = scan_bplane(
                self._lambert["r1"], self._lambert["v1"],
                self._phase_deg, tof_days=8.0)
            self._bplane_scan = (deltas, ca_m, ca_e, is_fr)
            n_fr = int(np.sum(is_fr))
            self.lbl_mode.setText(
                f"B-plane scan done.  Free-return corridor: {n_fr} of {len(deltas)} angles.\n"
                f"Use slider to pick δ, then Integrate.")
            self._replot()
        except Exception as e:
            self.lbl_mode.setText(f"Scan error: {e}")
        finally:
            self._done()

    def _update_metrics(self, r):
        if r is None: return
        ca_km = r['closest_moon_km']
        self.met_ca_km.setText(f"{ca_km:,.0f}")
        self.met_ca_rm.setText(f"{ca_km / 1737.4:.2f}")

        earth_dist_km = r['closest_earth_return_km']
        self.met_earth.setText(f"{earth_dist_km:,.0f}")
        reentry_alt_km = earth_dist_km - R_EARTH / 1e3
        if reentry_alt_km > 0:
            self.met_reentry.setText(f"{reentry_alt_km:,.0f}")
            col = ACCENT2 if reentry_alt_km < 1000 else ACCENT3
        else:
            self.met_reentry.setText("< surface!")
            col = ACCENT3
        self.met_reentry.setStyleSheet(f"color:{col};font-size:13px;font-weight:bold;")

        self.met_tflyby.setText(f"{r['t_flyby_d']:.2f}")
        self.met_treturn.setText(f"{r['t_return_d']:.2f}")

        fr = r["is_free_return"]
        self.met_fr.setText("✓  YES" if fr else "✗  no")
        self.met_fr.setStyleSheet(
            f"color:{'#3fb950' if fr else ACCENT3};font-size:14px;font-weight:bold;")

        dc = abs(r['jacobi_f'] - r['jacobi_0'])
        self.met_jacobi.setText(f"{r['jacobi_0']:.5f}")
        self.met_dc.setText(f"{dc:.2e}")

    def _conic_deviation(self, r):
        if r is None or self._lambert is None: return None
        try:
            xl, yl = propagate_keplerian(
                self._lambert["r1"], self._lambert["v1"],
                r["t_nd"][-1]*T_STAR, n_points=len(r["t_nd"]))
            xc = r["xy_inertial"][0]; yc = r["xy_inertial"][1]
            n = min(len(xl), len(xc))
            return np.max(np.sqrt((xl[:n]-xc[:n])**2+(yl[:n]-yc[:n])**2))/1e3
        except Exception:
            return None

    # ------------------------------------------------------------- plotting --

    def _replot(self):
        r = self._result
        if r is None:
            self._draw_empty(); return
        self._draw_cr3bp(r)

    def _draw_empty(self):
        for ax in (self._ax, self._ax_flyby, self._ax_inert):
            ax.clear(); self._style_ax(ax)
        # Rotating frame placeholder
        self._draw_bodies(self._ax, marker_frac=0.025)
        if self.chk_lagrange.isChecked():
            self._draw_lagrange(self._ax)
        self._ax.set_xlim(-1.6, 1.6); self._ax.set_ylim(-1.4, 1.4)
        self._ax.set_aspect("equal")
        self._ax.set_xlabel("x  [L*]", color=TEXT_DIM, fontsize=10)
        self._ax.set_ylabel("y  [L*]", color=TEXT_DIM, fontsize=10)
        self._ax.set_title("Rotating frame", color=TEXT_DIM, fontsize=11)
        # Flyby placeholder
        self._ax_flyby.text(0.5, 0.5, "Run simulation\nto see flyby zoom",
                            transform=self._ax_flyby.transAxes,
                            color=TEXT_DIM, ha="center", va="center", fontsize=10)
        self._ax_flyby.set_title("Lunar flyby — zoom", color=TEXT_DIM, fontsize=11)
        # Inertial placeholder
        SC = 1e6
        th = np.linspace(0, 2*np.pi, 300)
        self._ax_inert.plot(R_MOON_ORB/SC*np.cos(th), R_MOON_ORB/SC*np.sin(th),
                            color="#f0e68c", lw=0.7, ls="--", alpha=0.35)
        self._ax_inert.add_patch(mpa.Circle((0, 0), R_EARTH/SC*6,
                                            color="#1a6fbf", zorder=5))
        self._ax_inert.text(0, 0, "Earth", color="white", ha="center",
                            va="center", fontsize=8, fontweight="bold", zorder=6)
        self._ax_inert.set_aspect("equal")
        lim = R_MOON_ORB / SC * 1.2
        self._ax_inert.set_xlim(-lim, lim); self._ax_inert.set_ylim(-lim, lim)
        self._ax_inert.set_xlabel("x  [×10³ km]", color=TEXT_DIM, fontsize=10)
        self._ax_inert.set_ylabel("y  [×10³ km]", color=TEXT_DIM, fontsize=10)
        self._ax_inert.set_title("Inertial frame (geocentric)", color=TEXT_DIM, fontsize=11)
        self._ax_inert.text(0.5, 0.5, "Run simulation\nto see inertial view",
                            transform=self._ax_inert.transAxes,
                            color=TEXT_DIM, ha="center", va="center", fontsize=10)
        self._canvas.draw_idle()

    def _draw_cr3bp(self, r):
        mu  = MU_CR3BP
        rot = r["xy_rot"]
        traj_color = ACCENT4 if r.get("is_free_return") else ACCENT3
        label      = ("∞ Free-return (CR3BP)"
                      if r.get("is_free_return") else "Direct (CR3BP)")

        # ---- Clear all three axes ----
        for ax in (self._ax, self._ax_flyby, self._ax_inert):
            ax.clear(); self._style_ax(ax)

        # ========================================================
        # LEFT panel — full Earth-Moon system view
        # ========================================================
        ax = self._ax

        # Determine plot extent — centre on Earth/Moon axis midpoint,
        # use the LARGER of (x-extent, y-extent) symmetrically so the
        # figure-8 always looks like an eight and not a squashed ellipse.
        x_all = np.concatenate([rot[0], [-mu, 1-mu]])
        y_all = np.concatenate([rot[1], [0.0, 0.0]])
        pad   = 0.10
        xlo, xhi = x_all.min()-pad, x_all.max()+pad
        # Force y to be symmetric around 0 so both lobes are equally visible
        y_extent = max(abs(y_all.min()), abs(y_all.max())) + pad
        ylo, yhi = -y_extent, y_extent
        span = max(xhi-xlo, yhi-ylo) / 2 * 1.02
        xc   = (xlo+xhi)/2; yc = 0.0   # always centre vertically on Earth-Moon axis
        xlim = (xc-span, xc+span); ylim = (yc-span, yc+span)
        plot_span = 2*span

        if self.chk_zvc.isChecked():
            self._draw_zvc(ax, r["jacobi_0"], xlim, ylim)

        th = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(th), np.sin(th), color="#f0e68c",
                lw=0.5, ls="--", alpha=0.2, zorder=1)

        if self.chk_overlay.isChecked() and self._lambert is not None:
            self._draw_lambert_rot(ax, r)

        ax.plot(rot[0], rot[1], color=traj_color, lw=2.0,
                alpha=0.93, zorder=4, label=label)

        # Direction arrows
        n = len(rot[0])
        for idx in [n//5, 2*n//5, 3*n//5, 4*n//5]:
            if idx+1 < n:
                dx = rot[0,idx+1]-rot[0,idx]; dy = rot[1,idx+1]-rot[1,idx]
                ax.annotate("", xy=(rot[0,idx]+dx*4, rot[1,idx]+dy*4),
                            xytext=(rot[0,idx], rot[1,idx]),
                            arrowprops=dict(arrowstyle="-|>", color=traj_color,
                                            lw=1.1, mutation_scale=11), zorder=5)

        ms = plot_span * 0.022
        self._draw_bodies(ax, marker_frac=ms)

        if self.chk_lagrange.isChecked():
            self._draw_lagrange(ax)

        ax.plot(rot[0,0], rot[1,0], "o", color=ACCENT2, ms=9, zorder=9,
                label="Departure (LEO)")
        ax.plot(rot[0,-1], rot[1,-1], "s", color=traj_color, ms=8, zorder=9)

        # ---- Return perigee marker (rotating frame) ----
        if r.get("is_free_return"):
            dist_earth_rot = np.sqrt((rot[0]+mu)**2 + rot[1]**2)
            idx_flyby_r = np.argmax(dist_earth_rot)
            after = dist_earth_rot[idx_flyby_r:]
            if len(after) > 5:
                idx_p = idx_flyby_r + np.argmin(after)
                ax.plot(rot[0, idx_p], rot[1, idx_p],
                        "v", color=ACCENT2, ms=10, zorder=9,
                        label=f"Perigee return ({r.get('reentry_alt_km',0):,.0f} km)")
                ax.annotate(
                    f"Perigee\n{r.get('reentry_alt_km',0):,.0f} km",
                    xy=(rot[0,idx_p], rot[1,idx_p]),
                    xytext=(rot[0,idx_p] - 0.18, rot[1,idx_p] - 0.10),
                    color=ACCENT2, fontsize=8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.1),
                    zorder=10)

        dx_lbl = 0.06 if rot[0,0] > -mu else -0.14
        ax.text(rot[0,0]+dx_lbl, rot[1,0]+0.04, "Departure",
                color=ACCENT2, fontsize=9, va="bottom", fontweight="bold")

        dist_moon = np.sqrt((rot[0]-(1-mu))**2 + rot[1]**2)
        idx_c = np.argmin(dist_moon)
        offset_x = 0.12 if rot[0,idx_c] < 1-mu else -0.22
        ax.annotate(
            f"{r['closest_moon_km']:,.0f} km",
            xy=(rot[0,idx_c], rot[1,idx_c]),
            xytext=(rot[0,idx_c]+offset_x, rot[1,idx_c]+0.14),
            color=traj_color, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=traj_color, lw=1.2), zorder=10)

        # Draw a rectangle on the full view showing the flyby zoom region
        flyby_hw = 0.16   # half-width of flyby box in L* units
        moon_x, moon_y = 1-mu, 0.0
        rect = mpa.Rectangle(
            (moon_x - flyby_hw, moon_y - flyby_hw),
            2*flyby_hw, 2*flyby_hw,
            linewidth=1.2, edgecolor=ACCENT4, facecolor="none",
            ls="--", zorder=11, alpha=0.8)
        ax.add_patch(rect)
        ax.text(moon_x + flyby_hw + 0.02, moon_y + flyby_hw,
                "flyby\nzoom →", color=ACCENT4, fontsize=8,
                va="top", alpha=0.8)

        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        ax.set_xlabel("x  [L* = 384 400 km]", color=TEXT_DIM, fontsize=10)
        ax.set_ylabel("y  [L*]", color=TEXT_DIM, fontsize=10)
        mode_str = "∞ Free-Return" if r.get("is_free_return") else "Direct"
        delta_str = f"  |  δ={r.get('v1_rotate_deg',0):+.1f}°" if abs(r.get('v1_rotate_deg',0)) > 0.05 else ""
        ax.set_title(
            f"{mode_str}{delta_str}  |  C = {r['jacobi_0']:.4f}  |  "
            f"T = {r['t_nd'][-1]*T_STAR/86400:.1f} d",
            color=TEXT_PRIMARY, fontsize=11, pad=4)
        ax.legend(loc="upper left", fontsize=8,
                  facecolor=PANEL_BG, edgecolor=BORDER,
                  labelcolor=TEXT_PRIMARY, framealpha=0.85)

        # ========================================================
        # CENTRE panel — flyby zoom
        # ========================================================
        self._draw_flyby_zoom(r, traj_color, label)

        # ========================================================
        # RIGHT panel — inertial frame (geocentric, like Tab 1)
        # ========================================================
        self._draw_inertial(r, traj_color)

        self._canvas.draw_idle()

    def _draw_inertial(self, r, traj_color):
        """
        Right panel: trajectory in the geocentric inertial frame —
        same reference as Tab 1. Earth fixed at centre, Moon moves along
        its circular orbit. Shows the iconic figure-8 shape of Artemis.
        """
        ax = self._ax_inert
        SC = 1e6          # m → ×10³ km (axis unit)
        mu = MU_CR3BP

        xi = r["xy_inertial"][0]   # [m], Earth-centred
        yi = r["xy_inertial"][1]
        t_nd = r["t_nd"]
        t_d  = t_nd * T_STAR / 86400.0

        # ---- Moon orbit (circular reference) ----
        th = np.linspace(0, 2*np.pi, 400)
        ax.plot(R_MOON_ORB/SC*np.cos(th), R_MOON_ORB/SC*np.sin(th),
                color="#f0e68c", lw=0.8, ls="--", alpha=0.35, zorder=1,
                label="Moon orbit")

        # ---- Lambert 2-body overlay ----
        if self.chk_overlay.isChecked() and self._lambert is not None:
            try:
                tof_s = t_nd[-1] * T_STAR
                xl, yl = propagate_keplerian(
                    self._lambert["r1"], self._lambert["v1"],
                    tof_s, n_points=len(t_nd))
                ax.plot(xl/SC, yl/SC, color=ACCENT, lw=1.2, ls="--",
                        alpha=0.55, zorder=3, label="Lambert 2-body")
            except Exception:
                pass

        # ---- CR3BP inertial trajectory ----
        ax.plot(xi/SC, yi/SC, color=traj_color, lw=2.1,
                alpha=0.93, zorder=4,
                label="CR3BP trajectory")

        # Direction arrows at ~20% intervals
        n = len(xi)
        for idx in [n//6, n//3, n//2, 2*n//3, 5*n//6]:
            if idx+1 < n:
                dx = xi[idx+1]-xi[idx]; dy = yi[idx+1]-yi[idx]
                norm = max(np.hypot(dx, dy), 1e-12)
                scale = R_MOON_ORB * 0.04 / norm
                ax.annotate("",
                    xy=((xi[idx]+dx*scale)/SC, (yi[idx]+dy*scale)/SC),
                    xytext=(xi[idx]/SC, yi[idx]/SC),
                    arrowprops=dict(arrowstyle="-|>", color=traj_color,
                                    lw=1.2, mutation_scale=12), zorder=5)

        # ---- Earth ----
        r_earth_vis = R_EARTH / SC * 8   # exaggerated for visibility
        ax.add_patch(mpa.Circle((0, 0), r_earth_vis,
                                color="#1a6fbf", zorder=6, ec="#4a9fff", lw=0.8))
        ax.text(0, -r_earth_vis*2.5, "Earth", color="#4a9fff",
                ha="center", va="top", fontsize=9, fontweight="bold", zorder=7)

        # ---- Moon positions at key events ----
        phi0 = np.radians(self._phase_deg)
        omega_moon = 2*np.pi / T_MOON

        def moon_pos_at(t_s):
            phi = phi0 + omega_moon * t_s
            return np.array([R_MOON_ORB*np.cos(phi), R_MOON_ORB*np.sin(phi)])

        # Moon at departure (t=0)
        mp0 = moon_pos_at(0)
        ax.plot(mp0[0]/SC, mp0[1]/SC, "o", color="#b0c4de", ms=10, zorder=6)
        ax.text(mp0[0]/SC, mp0[1]/SC + R_MOON_ORB/SC*0.07,
                "Moon\n(t=0)", color="#b0c4de", ha="center",
                fontsize=8, fontweight="bold")

        # Moon at flyby
        t_flyby_s = r["t_flyby_d"] * 86400.0
        mp_flyby  = moon_pos_at(t_flyby_s)
        ax.plot(mp_flyby[0]/SC, mp_flyby[1]/SC, "o",
                color="#ffd700", ms=12, zorder=7)
        ax.text(mp_flyby[0]/SC, mp_flyby[1]/SC + R_MOON_ORB/SC*0.07,
                f"Moon\n(flyby t={r['t_flyby_d']:.1f}d)",
                color="#ffd700", ha="center", fontsize=8, fontweight="bold")

        # Moon at Earth return (if free-return)
        if r.get("is_free_return"):
            t_ret_s  = r["t_return_d"] * 86400.0
            mp_ret   = moon_pos_at(t_ret_s)
            ax.plot(mp_ret[0]/SC, mp_ret[1]/SC, "o",
                    color="#3fb950", ms=9, zorder=6, alpha=0.7)
            ax.text(mp_ret[0]/SC, mp_ret[1]/SC - R_MOON_ORB/SC*0.10,
                    f"Moon\n(return t={r['t_return_d']:.1f}d)",
                    color="#3fb950", ha="center", fontsize=8, va="top")

        # ---- Earth return perigee ----
        if r.get("is_free_return"):
            # Find the return perigee point (min Earth distance after flyby)
            dist_earth_i = np.sqrt(xi**2 + yi**2)
            idx_flyby_i  = np.argmax(dist_earth_i)  # approx: max dist = near Moon
            after_flyby  = dist_earth_i[idx_flyby_i:]
            if len(after_flyby) > 5:
                idx_ret_local = np.argmin(after_flyby)
                idx_ret = idx_flyby_i + idx_ret_local
                reentry_km = r.get("reentry_alt_km", dist_earth_i[idx_ret]/1e3 - R_EARTH/1e3)
                ax.plot(xi[idx_ret]/SC, yi[idx_ret]/SC,
                        "v", color=ACCENT2, ms=11, zorder=9)
                off_x = xi[idx_ret]/SC + R_MOON_ORB/SC*0.12
                off_y = yi[idx_ret]/SC - R_MOON_ORB/SC*0.10
                ax.annotate(
                    f"Perigee return\n{reentry_km:,.0f} km alt\nt={r['t_return_d']:.1f} d",
                    xy=(xi[idx_ret]/SC, yi[idx_ret]/SC),
                    xytext=(off_x, off_y),
                    color=ACCENT2, fontsize=8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.2),
                    zorder=10)
                # Draw Earth atmosphere ring (entry interface ~120 km)
                r_entry = (R_EARTH + 120e3) / SC
                ax.add_patch(mpa.Circle((0, 0), r_entry,
                                        color="none", ec="#3fb950",
                                        lw=0.8, ls=":", alpha=0.5, zorder=4))
                ax.text(r_entry * 0.72, r_entry * 0.72,
                        "entry\ninterface\n120 km", color="#3fb950",
                        fontsize=7, alpha=0.7, ha="center")
        ax.plot(xi[0]/SC, yi[0]/SC, "o", color=ACCENT2, ms=9, zorder=8,
                label="Departure (LEO)")
        ax.plot(xi[-1]/SC, yi[-1]/SC, "s", color=traj_color, ms=8, zorder=8)
        ax.text(xi[0]/SC + R_MOON_ORB/SC*0.07, yi[0]/SC,
                "Departure", color=ACCENT2, fontsize=9,
                fontweight="bold", va="center")

        # ---- Closest Moon approach annotation on inertial ----
        dist_moon_i = np.sqrt((xi - mp_flyby[0])**2 + (yi - mp_flyby[1])**2)
        idx_c = np.argmin(dist_moon_i)
        off = R_MOON_ORB/SC * 0.18
        ax.annotate(
            f"CA: {r['closest_moon_km']:,.0f} km",
            xy=(xi[idx_c]/SC, yi[idx_c]/SC),
            xytext=(xi[idx_c]/SC + off, yi[idx_c]/SC + off),
            color=ACCENT3, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=1.3),
            zorder=10)

        # ---- Axes ----
        lim = R_MOON_ORB / SC * 1.22
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x  [×10³ km]", color=TEXT_DIM, fontsize=10)
        ax.set_ylabel("y  [×10³ km]", color=TEXT_DIM, fontsize=10)
        fr_str = "∞ Free-Return" if r.get("is_free_return") else "Direct"
        ax.set_title(f"Inertial frame — {fr_str}  (figure-8 is only visible in rotating frame)",
                     color=TEXT_PRIMARY, fontsize=10, pad=4)
        ax.legend(loc="upper left", fontsize=8,
                  facecolor=PANEL_BG, edgecolor=BORDER,
                  labelcolor=TEXT_PRIMARY, framealpha=0.85)

    def _draw_bodies(self, ax, marker_frac=0.03):
        """Draw Earth and Moon as visually appropriate markers (not physical scale)."""
        mu = MU_CR3BP
        ms = marker_frac  # used as radius in data coords
        # Earth
        ax.add_patch(mpa.Circle((-mu, 0), ms,
                                color="#1e5faa", zorder=6, ec="#4a9fff", lw=0.8))
        ax.text(-mu, -ms*1.8, "Earth", color="#4a9fff",
                ha="center", va="top", fontsize=10, fontweight="bold", zorder=7)
        # Moon
        r_moon_vis = ms * 0.55
        ax.add_patch(mpa.Circle((1-mu, 0), r_moon_vis,
                                color="#666666", zorder=6, ec="#aaaaaa", lw=0.8))
        ax.text(1-mu, -ms*1.8, "Moon", color="#aaaaaa",
                ha="center", va="top", fontsize=10, fontweight="bold", zorder=7)

    def _draw_zvc(self, ax, jacobi_C, xlim, ylim):
        """
        Draw Zero Velocity Curves.
        Forbidden region (2Ω < C): spacecraft cannot enter — shown as a
        deep-blue/purple filled area.
        ZVC boundary (2Ω = C): bright violet contour — this is the actual curve.
        """
        try:
            nx, ny = 420, 420
            x  = np.linspace(xlim[0], xlim[1], nx)
            y  = np.linspace(ylim[0], ylim[1], ny)
            X, Y = np.meshgrid(x, y)
            mu  = MU_CR3BP
            r1  = np.sqrt((X + mu)**2 + Y**2)
            r2  = np.sqrt((X - (1-mu))**2 + Y**2)
            # Mask singularities
            sing = (r1 < 0.025) | (r2 < 0.015)
            with np.errstate(divide='ignore', invalid='ignore'):
                Omega = np.where(sing, 1e9,
                                 0.5*(X**2+Y**2) + (1-mu)/r1 + mu/r2)
            two_omega = 2 * Omega

            # --- Forbidden fill: vivid but not overwhelming ---
            ax.contourf(X, Y, two_omega,
                        levels=[-1e6, jacobi_C],
                        colors=["#1e1060"],    # deep indigo
                        alpha=0.70, zorder=2)

            # --- ZVC boundary: bright violet, clearly visible ---
            cs = ax.contour(X, Y, two_omega,
                            levels=[jacobi_C],
                            colors=["#bf5fff"],
                            linewidths=[1.8],
                            alpha=0.90, zorder=4)
            # Label the contour
            try:
                ax.clabel(cs, fmt=f"C={jacobi_C:.3f}",
                          fontsize=8, colors="#bf5fff", inline=True)
            except Exception:
                pass
        except Exception:
            pass

    def _draw_lambert_rot(self, ax, r):
        """Convert Lambert Keplerian arc to rotating frame and overlay."""
        try:
            r1m = self._lambert["r1"]; v1m = self._lambert["v1"]
            t_nd = r["t_nd"]; n = len(t_nd)
            tof_s = t_nd[-1] * T_STAR
            xl, yl = propagate_keplerian(r1m, v1m, tof_s, n_points=n)
            mu = MU_CR3BP; phi0 = np.radians(self._phase_deg)
            theta = phi0 + t_nd
            ct, st = np.cos(-theta), np.sin(-theta)
            # SI → nd, shift to barycentre
            xnd = xl/L_STAR - mu*np.cos(phi0)
            ynd = yl/L_STAR - mu*np.sin(phi0)
            xr =  ct*xnd - st*ynd
            yr =  st*xnd + ct*ynd  # bug in previous: was +st*xnd
            ax.plot(xr, yr, color=ACCENT, lw=2.0, ls="--", alpha=0.80,
                    zorder=3, label="Lambert 2-body (rotating frame)")
        except Exception:
            pass

    def _draw_lagrange(self, ax):
        cols = {"L1": "#bf5fff", "L2": "#bf5fff"}
        for name, (lx, ly) in self._lagrange.items():
            if name not in ("L1", "L2"):
                continue
            c = cols[name]
            # Marker
            ax.plot(lx, ly, "+", color=c, ms=18, mew=2.5, zorder=8)
            ax.text(lx + 0.04, ly + 0.06, name,
                    color=c, fontsize=11, fontweight="bold", zorder=9)
        # Highlight L1 neck region — the free-return corridor gateway
        L1x = self._lagrange["L1"][0]
        neck_patch = mpa.FancyArrowPatch(
            (L1x, -0.12), (L1x, 0.12),
            arrowstyle="-", color="#bf5fff",
            linewidth=1.2, linestyle="dashed",
            alpha=0.45, zorder=3)
        ax.add_patch(neck_patch)
        ax.text(L1x + 0.03, 0.14, "L1 neck\n(FR corridor)",
                color="#bf5fff", fontsize=8, alpha=0.7, ha="left")

    def _draw_flyby_zoom(self, r, traj_color, label):
        """
        Right panel: high-resolution zoom on the lunar flyby region.
        Shows trajectory arc near Moon, physical Moon radius ring,
        L1/L2, Lambert overlay, closest-approach annotation, km scale bar.
        """
        ax  = self._ax_flyby
        mu  = MU_CR3BP
        rot = r["xy_rot"]
        moon_x, moon_y = 1-mu, 0.0

        dist_moon = np.sqrt((rot[0]-moon_x)**2 + rot[1]**2)
        closest_nd = np.min(dist_moon)

        # --- Zoom window: 5× closest approach, capped at 0.16 L* ---
        zoom_hw = np.clip(closest_nd * 5.5, 0.035, 0.16)
        xlim = (moon_x - zoom_hw, moon_x + zoom_hw)
        ylim = (moon_y - zoom_hw, moon_y + zoom_hw)

        # --- Check trajectory enters this window ---
        mask = (np.abs(rot[0]-moon_x) < zoom_hw*1.4) & \
               (np.abs(rot[1]-moon_y) < zoom_hw*1.4)
        if not np.any(mask):
            ax.text(0.5, 0.5, "Trajectory does not\npass near Moon",
                    transform=ax.transAxes, color=TEXT_DIM,
                    ha="center", va="center", fontsize=11)
            ax.set_title("Lunar flyby — zoom", color=TEXT_DIM, fontsize=11)
            return

        # --- ZVC ---
        if self.chk_zvc.isChecked():
            self._draw_zvc(ax, r["jacobi_0"], xlim, ylim)

        # --- Lambert overlay ---
        if self.chk_overlay.isChecked() and self._lambert is not None:
            self._draw_lambert_rot(ax, r)

        # --- Full trajectory (dimmed) + flyby arc (highlighted) ---
        ax.plot(rot[0], rot[1], color=traj_color, lw=0.9,
                alpha=0.18, zorder=3)
        x_fb = rot[0][mask]; y_fb = rot[1][mask]
        ax.plot(x_fb, y_fb, color=traj_color, lw=2.6,
                alpha=0.95, zorder=4)

        # Direction arrows
        nf = len(x_fb)
        for idx in [nf//4, nf//2, 3*nf//4]:
            if 0 < idx < nf-1:
                dx = x_fb[idx+1]-x_fb[idx]; dy = y_fb[idx+1]-y_fb[idx]
                norm = max(np.hypot(dx,dy), 1e-12)
                scale = zoom_hw * 0.06 / norm
                ax.annotate("", xy=(x_fb[idx]+dx*scale, y_fb[idx]+dy*scale),
                            xytext=(x_fb[idx], y_fb[idx]),
                            arrowprops=dict(arrowstyle="-|>", color=traj_color,
                                            lw=1.3, mutation_scale=14), zorder=6)

        # --- Moon body (visual, ~8% of zoom window) ---
        r_vis = zoom_hw * 0.08
        ax.add_patch(mpa.Circle((moon_x, moon_y), r_vis,
                                color="#444444", zorder=7, ec="#888888", lw=1.0))

        # --- Physical Moon radius (to scale, dotted ring) ---
        r_phys_nd = 1737e3 / L_STAR
        ax.add_patch(mpa.Circle((moon_x, moon_y), r_phys_nd,
                                color="none", ec="#888888", lw=0.8,
                                ls=":", zorder=6, alpha=0.7))
        if r_phys_nd < zoom_hw:
            ax.text(moon_x + r_phys_nd*0.72, moon_y + r_phys_nd*0.72,
                    "R☽", color="#888888", fontsize=8, alpha=0.8)

        ax.text(moon_x, moon_y - r_vis*1.9, "Moon",
                color="#aaaaaa", ha="center", va="top",
                fontsize=9, fontweight="bold", zorder=8)

        # --- L1, L2 if visible ---
        if self.chk_lagrange.isChecked():
            for name, (lx, ly) in self._lagrange.items():
                if name not in ("L1","L2"): continue
                if xlim[0] < lx < xlim[1] and ylim[0] < ly < ylim[1]:
                    ax.plot(lx, ly, "+", color=ACCENT4, ms=14, mew=2.0, zorder=9)
                    ax.text(lx + zoom_hw*0.04, ly + zoom_hw*0.06,
                            name, color=ACCENT4, fontsize=9, fontweight="bold")

        # --- Closest approach annotation ---
        idx_c = np.argmin(dist_moon)
        ca_km = r["closest_moon_km"]
        off = zoom_hw * 0.35
        ax.annotate(
            f"CA: {ca_km:,.0f} km\n({ca_km/1737:.1f} R☽)",
            xy=(rot[0,idx_c], rot[1,idx_c]),
            xytext=(rot[0,idx_c] + off, rot[1,idx_c] + off),
            color=ACCENT3, fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=1.3), zorder=10)

        # --- Km scale bar ---
        bar_km = max(1000, int(closest_nd * L_STAR / 1e3 * 3 / 1000) * 1000)
        bar_km = min(bar_km, 20_000)
        bar_nd = bar_km * 1e3 / L_STAR
        bar_x0 = xlim[0] + zoom_hw * 0.06
        bar_y0 = ylim[0] + zoom_hw * 0.14
        ax.plot([bar_x0, bar_x0+bar_nd], [bar_y0, bar_y0],
                color=TEXT_DIM, lw=2.2, solid_capstyle="butt", zorder=10)
        ax.plot([bar_x0, bar_x0], [bar_y0-zoom_hw*0.015, bar_y0+zoom_hw*0.015],
                color=TEXT_DIM, lw=1.5, zorder=10)
        ax.plot([bar_x0+bar_nd]*2, [bar_y0-zoom_hw*0.015, bar_y0+zoom_hw*0.015],
                color=TEXT_DIM, lw=1.5, zorder=10)
        ax.text(bar_x0+bar_nd/2, bar_y0+zoom_hw*0.05,
                f"{bar_km:,} km", color=TEXT_DIM, ha="center", fontsize=8)

        # --- Axes ---
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
        # Δkm labels on x axis relative to Moon centre
        xticks = np.linspace(xlim[0], xlim[1], 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [f"{(v-moon_x)*L_STAR/1e3:+.0f}" for v in xticks],
            color=TEXT_DIM, fontsize=8)
        yticks = np.linspace(ylim[0], ylim[1], 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(
            [f"{v*L_STAR/1e3:.0f}" for v in yticks],
            color=TEXT_DIM, fontsize=8)
        ax.set_xlabel("Δx from Moon  [km]", color=TEXT_DIM, fontsize=9)
        ax.set_ylabel("y  [km]", color=TEXT_DIM, fontsize=9)
        ax.set_title("Lunar flyby — zoom", color=TEXT_PRIMARY, fontsize=11, pad=4)


# ---------------------------------------------------------------------------
# Animation Tab  —  Tab 3
# ---------------------------------------------------------------------------

class AnimationTab(QWidget):
    """
    Tab 3: Animated playback of the CR3BP trajectory in the geocentric
    inertial frame.  Earth is fixed at centre; Moon moves along its
    circular orbit; the spacecraft follows xy_inertial computed by
    run_cr3bp_full().

    All data come from the CR3BP tab via set_cr3bp_result().
    No new physics is computed here — it is pure playback.
    """

    # ---- tuneable constants ----
    FPS        = 30          # timer fires at ~30 Hz
    TRAIL_LEN  = 120         # number of past frames shown as fading trail

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result   = None   # cr3bp result dict
        self._phase0   = 90.0   # Moon phase at t=0 [deg]
        self._idx      = 0      # current frame index
        self._playing  = False
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(1000 // self.FPS)
        self._anim_timer.timeout.connect(self._step)
        self._build_ui()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # ---- header row ----
        hdr_row = QWidget()
        hh = QHBoxLayout(hdr_row); hh.setContentsMargins(0,0,0,0)
        hdr = QLabel("Inertial Frame Animation — Geocentric View")
        hdr.setStyleSheet(f"font-size:16px;font-weight:bold;color:{TEXT_PRIMARY};")

        self.btn_save_anim = QPushButton("Save image")
        self.btn_save_anim.setFixedWidth(110)
        self.btn_save_anim.setStyleSheet(
            f"background:{PANEL_BG};color:{TEXT_DIM};border:1px solid {BORDER};"
            f"border-radius:5px;padding:5px 8px;font-size:12px;")
        hh.addWidget(hdr); hh.addStretch(); hh.addWidget(self.btn_save_anim)
        root.addWidget(hdr_row)

        # ---- canvas ----
        self._fig = Figure(facecolor=DARK_BG, tight_layout=True)
        self._ax  = self._fig.add_subplot(111)
        self._style_ax()
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._canvas, stretch=1)

        # ---- transport bar ----
        transport = QWidget()
        th = QHBoxLayout(transport); th.setContentsMargins(0,0,0,0); th.setSpacing(10)

        self.btn_play  = QPushButton("▶  Play")
        self.btn_play.setFixedWidth(90)
        self.btn_pause = QPushButton("⏸  Pause")
        self.btn_pause.setFixedWidth(90)
        self.btn_reset = QPushButton("⏮  Reset")
        self.btn_reset.setFixedWidth(90)
        for b in (self.btn_play, self.btn_pause, self.btn_reset):
            b.setStyleSheet(
                f"background:{PANEL_BG};color:{TEXT_PRIMARY};"
                f"border:1px solid {BORDER};border-radius:5px;"
                f"padding:6px 10px;font-size:13px;font-weight:bold;")

        self.lbl_speed = QLabel("Speed:")
        self.lbl_speed.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
        self.sl_speed = QSlider(Qt.Horizontal)
        self.sl_speed.setRange(1, 20); self.sl_speed.setValue(5)
        self.sl_speed.setFixedWidth(140)
        self.lbl_speed_val = QLabel("5×")
        self.lbl_speed_val.setStyleSheet(f"color:{ACCENT};font-size:12px;font-weight:bold;")
        self.lbl_speed_val.setFixedWidth(35)

        self.sl_frame = QSlider(Qt.Horizontal)
        self.sl_frame.setRange(0, 100); self.sl_frame.setValue(0)

        self.lbl_time = QLabel("t = 0.00 d")
        self.lbl_time.setStyleSheet(f"color:{ACCENT4};font-size:14px;font-weight:bold;")
        self.lbl_time.setFixedWidth(110)

        th.addWidget(self.btn_reset)
        th.addWidget(self.btn_play)
        th.addWidget(self.btn_pause)
        th.addWidget(self.lbl_speed)
        th.addWidget(self.sl_speed)
        th.addWidget(self.lbl_speed_val)
        th.addSpacing(10)
        th.addWidget(self.sl_frame, stretch=1)
        th.addWidget(self.lbl_time)
        root.addWidget(transport)

        # ---- info bar ----
        self.lbl_info = QLabel(
            "Run the CR3BP simulation in Tab 2 first, then come here to play the animation.")
        self.lbl_info.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
        self.lbl_info.setWordWrap(True)
        root.addWidget(self.lbl_info)

        # ---- wiring ----
        self.btn_play.clicked.connect(self._play)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_reset.clicked.connect(self._reset)
        self.sl_speed.valueChanged.connect(self._on_speed)
        self.sl_frame.valueChanged.connect(self._on_scrub)
        self.btn_save_anim.clicked.connect(self._save_frame)

        self._draw_empty()

    def _style_ax(self):
        ax = self._ax
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=10)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.4)

    # ---------------------------------------------------------- public API --

    def set_cr3bp_result(self, result, phase0_deg):
        """Called by main window when a CR3BP result is available."""
        self._anim_timer.stop()
        self._playing = False
        self._result  = result
        self._phase0  = phase0_deg
        self._idx     = 0
        n = len(result["t_nd"])
        self.sl_frame.setRange(0, n - 1)
        self.sl_frame.setValue(0)
        self.lbl_info.setText(
            f"Trajectory loaded: {n} frames, "
            f"T = {result['t_nd'][-1] * T_STAR / 86400:.1f} d  |  "
            f"Periselene = {result['closest_moon_km']:,.0f} km  |  "
            f"Free-return: {'YES' if result['is_free_return'] else 'no'}")
        self._draw_frame(0)

    # ---------------------------------------------------------- transport --

    def _play(self):
        if self._result is None:
            return
        if self._idx >= len(self._result["t_nd"]) - 1:
            self._idx = 0
        self._playing = True
        self._anim_timer.start()

    def _pause(self):
        self._playing = False
        self._anim_timer.stop()

    def _reset(self):
        self._pause()
        self._idx = 0
        self.sl_frame.setValue(0)
        if self._result is not None:
            self._draw_frame(0)

    def _on_speed(self, v):
        self.lbl_speed_val.setText(f"{v}×")

    def _on_scrub(self, v):
        if not self._playing and self._result is not None:
            self._idx = v
            self._draw_frame(v)

    def _step(self):
        if self._result is None:
            self._anim_timer.stop(); return
        skip = self.sl_speed.value()
        self._idx = min(self._idx + skip, len(self._result["t_nd"]) - 1)
        self.sl_frame.blockSignals(True)
        self.sl_frame.setValue(self._idx)
        self.sl_frame.blockSignals(False)
        self._draw_frame(self._idx)
        if self._idx >= len(self._result["t_nd"]) - 1:
            self._anim_timer.stop()
            self._playing = False

    # --------------------------------------------------------------- draw --

    def _draw_empty(self):
        ax = self._ax; ax.clear(); self._style_ax()
        SC = 1e6
        th = np.linspace(0, 2*np.pi, 300)
        ax.plot(R_MOON_ORB/SC*np.cos(th), R_MOON_ORB/SC*np.sin(th),
                color="#f0e68c", lw=0.7, ls="--", alpha=0.3)
        ax.add_patch(mpa.Circle((0,0), R_EARTH/SC*7, color="#1a6fbf", zorder=5))
        ax.text(0, 0, "Earth", color="white", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=6)
        ax.set_aspect("equal")
        lim = R_MOON_ORB/SC * 1.25
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("x  [×10³ km]", color=TEXT_DIM, fontsize=11)
        ax.set_ylabel("y  [×10³ km]", color=TEXT_DIM, fontsize=11)
        ax.set_title("Run CR3BP simulation (Tab 2) to load trajectory",
                     color=TEXT_DIM, fontsize=12)
        self._canvas.draw_idle()

    def _draw_frame(self, idx):
        r   = self._result
        SC  = 1e6
        ax  = self._ax
        ax.clear(); self._style_ax()

        xi  = r["xy_inertial"][0]
        yi  = r["xy_inertial"][1]
        t_d = r["t_nd"] * T_STAR / 86400.0
        n   = len(xi)

        t_now = t_d[idx]
        self.lbl_time.setText(f"t = {t_now:.2f} d")

        # ---- Moon orbit ring ----
        th = np.linspace(0, 2*np.pi, 300)
        ax.plot(R_MOON_ORB/SC*np.cos(th), R_MOON_ORB/SC*np.sin(th),
                color="#f0e68c", lw=0.7, ls="--", alpha=0.25, zorder=1)

        # ---- Full ghost trajectory — the figure-8, protagonist of the animation ----
        ghost_color = ACCENT4 if r.get("is_free_return") else ACCENT3
        ax.plot(xi/SC, yi/SC, color=ghost_color, lw=1.8,
                alpha=0.40, zorder=2, ls="--",
                label="Full trajectory (figure-8)")

        # ---- Fading trail ----
        trail_start = max(0, idx - self.TRAIL_LEN)
        trail_x = xi[trail_start:idx+1] / SC
        trail_y = yi[trail_start:idx+1] / SC
        ntr = len(trail_x)
        if ntr > 1:
            # Colour gradient: old → faint, recent → bright
            traj_color = ACCENT4 if r.get("is_free_return") else ACCENT3
            from matplotlib.colors import to_rgb
            rgb = to_rgb(traj_color)
            for k in range(ntr - 1):
                alpha = 0.15 + 0.85 * (k / (ntr - 1))
                lw    = 0.8  + 1.6  * (k / (ntr - 1))
                ax.plot(trail_x[k:k+2], trail_y[k:k+2],
                        color=rgb, alpha=alpha, lw=lw, zorder=3, solid_capstyle="round")

        # ---- Earth ----
        r_earth_vis = R_EARTH / SC * 7
        ax.add_patch(mpa.Circle((0, 0), r_earth_vis,
                                color="#1e5faa", zorder=6, ec="#4a9fff", lw=0.8))
        ax.text(0, 0, "Earth", color="white", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=7)

        # ---- Moon current position ----
        omega_moon = 2*np.pi / T_MOON
        phi_moon   = np.radians(self._phase0) + omega_moon * t_now * 86400.0
        mx = R_MOON_ORB/SC * np.cos(phi_moon)
        my = R_MOON_ORB/SC * np.sin(phi_moon)
        r_moon_vis = R_MOON_ORB/SC * 0.028
        ax.add_patch(mpa.Circle((mx, my), r_moon_vis,
                                color="#555555", zorder=6, ec="#aaaaaa", lw=0.8))
        ax.text(mx, my - r_moon_vis*1.9, "Moon",
                color="#aaaaaa", ha="center", va="top",
                fontsize=9, fontweight="bold", zorder=7)

        # ---- Spacecraft ----
        sc_x = xi[idx] / SC
        sc_y = yi[idx] / SC
        traj_color = ACCENT4 if r.get("is_free_return") else ACCENT3
        ax.plot(sc_x, sc_y, "o", color=traj_color, ms=6, zorder=10,
                markeredgecolor="white", markeredgewidth=0.7)

        # ---- Spacecraft–Moon line (dashed, when near) ----
        dist_moon_km = np.hypot(xi[idx] - mx*SC, yi[idx] - my*SC) / 1e3
        if dist_moon_km < 80_000:
            ax.plot([sc_x, mx], [sc_y, my], color=traj_color,
                    lw=0.8, ls=":", alpha=0.5, zorder=4)
            ax.text((sc_x+mx)/2, (sc_y+my)/2,
                    f" {dist_moon_km:,.0f} km",
                    color=traj_color, fontsize=9, alpha=0.85)

        # ---- Key event markers (static, shown throughout) ----
        # Periselene marker
        idx_peri = np.argmin(np.sqrt(
            (xi - R_MOON_ORB*np.cos(np.radians(self._phase0) +
             omega_moon*r["t_nd"]*T_STAR))**2 +
            (yi - R_MOON_ORB*np.sin(np.radians(self._phase0) +
             omega_moon*r["t_nd"]*T_STAR))**2))
        if idx >= idx_peri:
            ax.plot(xi[idx_peri]/SC, yi[idx_peri]/SC,
                    "*", color=ACCENT3, ms=13, zorder=9,
                    label=f"Periselene {r['closest_moon_km']:,.0f} km")

        # Return perigee marker
        if r.get("is_free_return"):
            dist_e = np.sqrt(xi**2 + yi**2)
            idx_p = np.argmax(dist_e)
            after  = dist_e[idx_p:]
            if len(after) > 5:
                idx_ret = idx_p + np.argmin(after)
                if idx >= idx_ret:
                    h_km = max(np.sqrt(xi[idx_ret]**2+yi[idx_ret]**2)/1e3
                               - R_EARTH/1e3, 0)
                    ax.plot(xi[idx_ret]/SC, yi[idx_ret]/SC,
                            "v", color=ACCENT2, ms=11, zorder=9,
                            label=f"Return perigee {h_km:,.0f} km")

        # ---- Phase annotations ----
        phase_label = self._phase_label(r, idx, t_now, dist_moon_km)
        ax.text(0.02, 0.97, phase_label, transform=ax.transAxes,
                color=TEXT_PRIMARY, fontsize=12, fontweight="bold",
                va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG,
                          edgecolor=BORDER, alpha=0.90))

        # ---- Time progress bar (drawn as axis annotation) ----
        frac = idx / max(n - 1, 1)
        ax.axhline(0, color=BORDER, lw=0.3, alpha=0.3)   # just keep grid clean

        # ---- Axes ----
        lim = R_MOON_ORB/SC * 1.25
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x  [×10³ km]", color=TEXT_DIM, fontsize=11)
        ax.set_ylabel("y  [×10³ km]", color=TEXT_DIM, fontsize=11)
        ax.set_title(
            f"Inertial frame  |  t = {t_now:.2f} d  |  "
            f"{'Free-Return' if r.get('is_free_return') else 'Direct'}  |  "
            f"Spacecraft dist from Earth: {np.hypot(sc_x,sc_y)*1e3:.0f} km",
            color=TEXT_PRIMARY, fontsize=11, pad=5)

        if idx_peri <= idx or (r.get("is_free_return") and idx > n//2):
            ax.legend(loc="lower right", fontsize=9,
                      facecolor=PANEL_BG, edgecolor=BORDER,
                      labelcolor=TEXT_PRIMARY, framealpha=0.85)

        self._canvas.draw_idle()

    def _phase_label(self, r, idx, t_now, dist_moon_km):
        """Return a short mission-phase string for the annotation box."""
        n = len(r["t_nd"])
        t_flyby = r["t_flyby_d"]
        t_return = r["t_return_d"]

        if t_now < 0.3:
            return "TLI burn complete\nTranslunar coast begins"
        elif t_now < t_flyby - 0.3:
            return f"Translunar coast\nApproaching Moon..."
        elif abs(t_now - t_flyby) < 0.4:
            return f"LUNAR FLYBY\nPeriselene: {r['closest_moon_km']:,.0f} km"
        elif t_now < t_return - 0.3 and r.get("is_free_return"):
            return "Post-flyby return arc\nGravity assist complete"
        elif r.get("is_free_return") and t_now >= t_return - 0.3:
            return f"EARTH RETURN\nReentry altitude: {r.get('reentry_alt_km',0):,.0f} km"
        else:
            return "Free-flight coast"

    def _save_frame(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Animation Frame", f"animation_frame_{self._idx:04d}.png",
            "PNG image (*.png);;PDF (*.pdf)")
        if path:
            self._fig.savefig(path, dpi=180, facecolor=DARK_BG, bbox_inches="tight")
            self.lbl_info.setText(f"Saved: {path.split('/')[-1].split(chr(92))[-1]}")


# ---------------------------------------------------------------------------
# Theory Tab  —  Tab 4
# ---------------------------------------------------------------------------

THEORY_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 15pt;
    line-height: 1.75;
    margin: 0;
    padding: 0;
  }
  .page { max-width: 920px; margin: 0 auto; padding: 28px 36px 60px 36px; }

  .subtitle { color: #8b949e; font-size: 13pt; margin-top: -10px;
              margin-bottom: 24px; }

  .box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 16px 0;
  }
  .box.accent  { border-left: 4px solid #58a6ff; }
  .box.green   { border-left: 4px solid #3fb950; }
  .box.purple  { border-left: 4px solid #d2a8ff; }
  .box.orange  { border-left: 4px solid #f78166; }
  .box.key     { border-left: 4px solid #ffd700; background: #1a1a0a; }

  code {
    background: #21262d;
    color: #f78166;
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13pt;
  }
  .eq {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px 20px;
    margin: 10px 0;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14pt;
    color: #f0e68c;
    text-align: center;
    letter-spacing: 0.5px;
  }
  .param-table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 13pt;
  }
  .param-table th {
    background: #21262d;
    color: #58a6ff;
    padding: 8px 12px;
    text-align: left;
    border-bottom: 2px solid #30363d;
    font-size: 13pt;
  }
  .param-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #21262d;
    color: #e6edf3;
    font-size: 13pt;
  }
  .param-table tr:hover td { background: #1f2937; }
  .param-table .val { color: #3fb950; font-family: monospace; }

  .ref-list { list-style: none; padding: 0; }
  .ref-list li {
    padding: 8px 0;
    border-bottom: 1px solid #21262d;
    color: #8b949e;
    font-size: 13pt;
  }
  .ref-list li b { color: #e6edf3; }
  .ref-list a { color: #58a6ff; }

  .tag {
    display: inline-block;
    background: #21262d;
    color: #8b949e;
    font-size: 11pt;
    padding: 2px 8px;
    border-radius: 10px;
    margin-right: 4px;
  }
  .tag.blue   { background: #0d2137; color: #58a6ff; }
  .tag.green  { background: #0d2015; color: #3fb950; }
  .tag.purple { background: #1a0d37; color: #d2a8ff; }

  hr { border: none; border-top: 1px solid #30363d; margin: 32px 0; }

  .highlight { color: #ffd700; font-weight: bold; }
  .dim { color: #8b949e; }
  .blue { color: #58a6ff; }
  .green { color: #3fb950; }
  .purple { color: #d2a8ff; }
  .orange { color: #f78166; }
</style>
</head>
<body>
<div class="page">

<h1 style="color:#58a6ff;font-size:22pt;font-weight:bold;border-bottom:2px solid #30363d;padding-bottom:8px;margin-top:0;">Theory Reference — Artemis 2 TLI &amp; CR3BP</h1>
<p class="subtitle">
  A concise guide to the orbital mechanics behind this tool.
  <span class="tag blue">2-body</span>
  <span class="tag purple">CR3BP</span>
  <span class="tag green">Educational</span>
</p>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 style="color:#d2a8ff;font-size:18pt;font-weight:bold;margin-top:28px;padding-left:10px;border-left:4px solid #d2a8ff;">1 · The Physical Setting</h2>

<p>
After launch, Artemis 2 parks in a <b>Low Earth Orbit (LEO)</b> at ~185 km altitude.
The <b>Trans-Lunar Injection (TLI)</b> burn — a single ~18-minute firing of the
ICPS upper stage — places the Orion capsule on a trajectory toward the Moon.
Two fundamental questions must be answered:
</p>

<div class="box accent">
  <b class="blue">Q1 (Tab 1 — Lambert):</b> What velocity vector must the spacecraft
  have at the end of the burn, so that it reaches the Moon's <em>future</em> position
  after a chosen time of flight?<br><br>
  <b class="purple">Q2 (Tab 2 — CR3BP):</b> Once en route, how does the Moon's
  gravity alter the trajectory, and does the spacecraft return to Earth if no
  further burns are executed?
</div>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 style="color:#d2a8ff;font-size:18pt;font-weight:bold;margin-top:28px;padding-left:10px;border-left:4px solid #d2a8ff;">2 · Two-Body Keplerian Mechanics</h2>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Equation of motion</h3>
<p>With only Earth's gravity, the spacecraft obeys:</p>
<div class="eq">r̈ = −(μ / r³) · r</div>
<p>
where <code>μ = GM⊕ = 3.986 × 10¹⁴ m³/s²</code>.
All solutions are conic sections — ellipses, parabolas, or hyperbolas.
</p>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Vis-viva equation</h3>
<div class="eq">v² = μ · (2/r − 1/a)</div>
<p>
<code>a</code> is the semi-major axis. At LEO (<code>r = 6 556 km</code>):
circular velocity <code>v<sub>circ</sub> ≈ 7.79 km/s</code>.
After TLI: <code>v<sub>dep</sub> ≈ 10.8 km/s</code>, giving
<span class="highlight">Δv<sub>TLI</sub> ≈ 3.1 km/s</span>.
</p>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Key insight</h3>
<div class="box key">
  TLI does <b>not</b> aim at the Moon's current position.
  It aims at an ellipse whose apogee intercepts the Moon's
  <em>future</em> position — after ~3 days of free flight.
  The Moon travels ~40° along its orbit during transit.
</div>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 style="color:#d2a8ff;font-size:18pt;font-weight:bold;margin-top:28px;padding-left:10px;border-left:4px solid #d2a8ff;">3 · Lambert's Problem</h2>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Statement</h3>
<div class="box accent">
  <b>Given:</b> position vectors <b>r</b><sub>1</sub> (departure) and
  <b>r</b><sub>2</sub> (arrival), and a time of flight Δt.<br>
  <b>Find:</b> the unique conic section connecting them, and the velocity
  vectors <b>v</b><sub>1</sub>, <b>v</b><sub>2</sub> at each endpoint.
</div>

<p>
Lambert (1761) proved that the time of flight depends only on three
geometric quantities: the semi-major axis <code>a</code>, the chord
<code>c = |r₂ − r₁|</code>, and the semi-perimeter
<code>s = (r₁ + r₂ + c)/2</code>.
</p>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">The Transfer Arc angle Δν</h3>
<p>
The geocentric angle between departure and arrival sets the geometry.
In this tool, the <b>Transfer Arc slider</b> indirectly chooses the
departure point on the LEO circle, such that the angle between
<b>r</b><sub>1</sub> and <b>r</b><sub>2</sub> equals Δν.
</p>
<ul>
  <li><b>Δν = 180°</b> → Hohmann transfer, minimum Δv, apogee = Moon distance</li>
  <li><b>Δν &lt; 180°</b> → Moon intercepted before apogee, higher Δv</li>
  <li><b>Optimal for Artemis 2:</b> Δν ≈ 172°, Δv ≈ 3.16 km/s</li>
</ul>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Universal-variable solver (Battin / Bate)</h3>
<p>
The solver introduces a scalar variable <code>ψ</code> (proportional to the
square of the universal variable χ) and Stumpff functions:
</p>
<div class="eq">
C(ψ) = (1 − cos √ψ) / ψ &nbsp;&nbsp;&nbsp; S(ψ) = (√ψ − sin √ψ) / ψ<sup>3/2</sup> &nbsp; [ψ &gt; 0]
</div>
<p>
These unify elliptic, parabolic, and hyperbolic solutions in a single
framework. The time-of-flight equation becomes a scalar function of ψ:
</p>
<div class="eq">
√μ · Δt = (y/C)<sup>3/2</sup> S + A√y
</div>
<p>
where <code>A = sin(Δν)√(r₁r₂ / (1−cos Δν))</code> and
<code>y = r₁ + r₂ + A(ψS − 1)/√C</code>.
This is solved numerically with Brent's method
(<code>scipy.optimize.brentq</code>). Once ψ* is found, velocities follow
from <b>Lagrange coefficients</b>:
</p>
<div class="eq">
v₁ = (r₂ − f·r₁) / g &nbsp;&nbsp;&nbsp;
f = 1 − y/r₁ &nbsp;&nbsp;
g = A√(y/μ) &nbsp;&nbsp;
g<sub>d</sub> = 1 − y/r₂
</div>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 style="color:#d2a8ff;font-size:18pt;font-weight:bold;margin-top:28px;padding-left:10px;border-left:4px solid #d2a8ff;">4 · Circular Restricted Three-Body Problem (CR3BP)</h2>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Hypotheses</h3>

<table class="param-table">
<tr><th>Assumption</th><th>Justification</th></tr>
<tr><td>Circular Earth–Moon orbit</td>
    <td>Eccentricity of Moon orbit ≈ 0.055, small enough for a first model</td></tr>
<tr><td>Spacecraft mass negligible</td>
    <td>m<sub>Orion</sub> / m<sub>Moon</sub> ≈ 10⁻¹⁸</td></tr>
<tr><td>Planar (2D) motion</td>
    <td>Moon orbit inclination ≈ 5.1° — omitted for clarity</td></tr>
<tr><td>No perturbations</td>
    <td>Solar gravity, J₂, SRP — second-order effects for 6-day arcs</td></tr>
</table>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Dimensionless units</h3>

<table class="param-table">
<tr><th>Quantity</th><th>Symbol</th><th class="val">Value</th><th>Unit</th></tr>
<tr><td>Mass ratio</td>
    <td>μ = M<sub>Moon</sub> / (M⊕ + M<sub>Moon</sub>)</td>
    <td class="val">0.01215</td><td>—</td></tr>
<tr><td>Length unit</td>
    <td>L* = Earth–Moon distance</td>
    <td class="val">384 400 km</td><td>m</td></tr>
<tr><td>Time unit</td>
    <td>T* = T<sub>Moon</sub> / 2π</td>
    <td class="val">4.343 days</td><td>s</td></tr>
<tr><td>Velocity unit</td>
    <td>V* = L* / T*</td>
    <td class="val">1.023 km/s</td><td>m/s</td></tr>
</table>

<p>In the rotating frame: <b>Earth at (−μ, 0)</b>, <b>Moon at (1−μ, 0)</b>.</p>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Equations of motion</h3>
<p>In the rotating frame the spacecraft obeys:</p>
<div class="eq">
ẍ − 2ẏ = ∂Ω/∂x<br>
ÿ + 2ẋ = ∂Ω/∂y
</div>
<p>
The terms <code>−2ẏ</code> and <code>+2ẋ</code> are the <b>Coriolis acceleration</b>.
The effective potential Ω combines gravity and centrifugal effects:
</p>
<div class="eq">Ω = ½(x² + y²) + (1−μ)/r₁ + μ/r₂</div>
<p>
where <code>r₁ = |r − r⊕|</code> (Earth distance) and
<code>r₂ = |r − r<sub>Moon</sub>|</code> (Moon distance).
</p>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">The Jacobi Constant</h3>
<div class="box purple">
  <b class="purple">C = 2Ω − v²</b>&nbsp;&nbsp; (conserved along any trajectory)
</div>
<p>
C is the only integral of motion in the CR3BP — analogous to energy.
It defines <b>Zero Velocity Curves (ZVC)</b>: regions where <code>v² = 0</code>,
i.e. <code>2Ω(x,y) = C</code>. The spacecraft cannot enter the <em>forbidden
regions</em> (where <code>2Ω &lt; C</code>) regardless of its trajectory.
</p>
<p>
As C decreases (more energy), the forbidden regions shrink and eventually
open "necks" around the Lagrange points, allowing transit between Earth,
Moon, and outer space.
</p>

<table class="param-table">
<tr><th>C value</th><th>Topology</th></tr>
<tr><td><code>C &gt; C<sub>L1</sub> = 3.1883</code></td>
    <td>Spacecraft confined near Earth or near Moon — no transit</td></tr>
<tr><td><code>C<sub>L2</sub> &lt; C &lt; C<sub>L1</sub></code></td>
    <td>L1 neck open — transit Earth ↔ Moon possible</td></tr>
<tr><td><code>C &lt; C<sub>L2</sub> = 3.1722</code></td>
    <td>Both necks open — free transit through L1 and L2</td></tr>
<tr><td><b>Artemis 2 baseline: C ≈ 1.71</b></td>
    <td>Both necks wide open — free-return corridor accessible</td></tr>
</table>

<p class="dim">
  Note: C ≈ 1.71 is well below both thresholds — the spacecraft is energetic
  enough to move freely through the entire Earth–Moon system.
  The free-return property is topological, not energetic: it depends on
  the precise initial direction (B-plane angle δ ≈ ±0.3°).
</p>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 style="color:#d2a8ff;font-size:18pt;font-weight:bold;margin-top:28px;padding-left:10px;border-left:4px solid #d2a8ff;">5 · Lagrange Points</h2>

<p>
Five equilibrium points where gravitational and centrifugal forces balance
(relative to the rotating frame):
</p>

<table class="param-table">
<tr><th>Point</th><th>Location</th><th>Stability</th><th>Relevance</th></tr>
<tr><td><b class="purple">L1</b></td>
    <td>Between Earth and Moon, x ≈ 0.837 L*</td>
    <td>Unstable</td>
    <td>Free-return corridor gateway; Halo orbits</td></tr>
<tr><td><b class="purple">L2</b></td>
    <td>Beyond Moon, x ≈ 1.156 L*</td>
    <td>Unstable</td>
    <td>NRHO (Near-Rectilinear Halo Orbit) — Gateway station</td></tr>
<tr><td><b class="dim">L3</b></td>
    <td>Opposite Moon, x ≈ −1.005 L*</td>
    <td>Unstable</td>
    <td>Rarely used in mission design</td></tr>
<tr><td><b class="green">L4, L5</b></td>
    <td>60° ahead/behind Moon, y = ±√3/2</td>
    <td>Stable</td>
    <td>Trojan asteroids; long-term stability</td></tr>
</table>

<div class="box green">
  <b>The free-return corridor</b> in this tool passes near L1.
  When the B-plane angle δ is tuned to ±0.3°, the trajectory's asymptotic
  manifold aligns with the stable manifold of L1, enabling the gravity-assist
  return without additional burns.
</div>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 style="color:#d2a8ff;font-size:18pt;font-weight:bold;margin-top:28px;padding-left:10px;border-left:4px solid #d2a8ff;">6 · Free-Return Trajectory</h2>

<p>
A free-return trajectory satisfies the constraint that, after the TLI burn
and a lunar flyby, the spacecraft returns to Earth's atmosphere without any
additional propulsion. This is a <b>safety requirement</b> for crewed lunar
missions (first used in Apollo 13).
</p>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">Mechanism</h3>
<ol>
  <li>TLI places the spacecraft on an ellipse toward the Moon (Lambert, 2-body)</li>
  <li>As it approaches within ~60 000 km, lunar gravity becomes significant</li>
  <li>The flyby deflects the trajectory — if the geometry is correct (δ ≈ 0°),
      the spacecraft is "slung" back toward Earth</li>
  <li>Earth return occurs ~3–4 days after periselene, at a perigee altitude
      compatible with atmospheric reentry</li>
</ol>

<h3 style="color:#3fb950;font-size:15pt;font-weight:bold;margin-top:18px;">B-plane targeting</h3>
<p>
The <b>B-plane</b> is the plane perpendicular to the incoming hyperbolic
asymptote at the Moon, containing the Moon's centre. The impact parameter
<b>B</b> (miss distance in this plane) determines whether the spacecraft
passes in front of or behind the Moon, and at what distance.
</p>
<p>
In this tool, the slider <b>δ</b> rotates the departure velocity vector
<b>v</b><sub>1</sub> by a small angle, effectively changing the B-plane
offset. The free-return corridor has a width of only <b>~0.6°</b> in δ.
</p>

<div class="box orange">
  <b>Model limitations vs Artemis 2 reality:</b><br>
  • <b>Periselene:</b> model gives ~2 600 km; real Artemis 2 ≈ 7 400 km<br>
  • <b>Return perigee altitude:</b> model ~315 km; real entry interface ~120 km<br>
  • <b>Cause:</b> 2D coplanar model omits 5.1° Moon inclination and 3D geometry<br>
  • <b>TOF:</b> model ~6 days; real Artemis 2 ~10 days (hybrid free-return)
</div>

<!-- ═══════════════════════════════════════════════════════════ -->
<h2 style="color:#d2a8ff;font-size:18pt;font-weight:bold;margin-top:28px;padding-left:10px;border-left:4px solid #d2a8ff;">7 · Numerical Integration</h2>

<p>
The CR3BP equations are integrated with <b>DOP853</b>
(Dormand–Prince 8th-order Runge–Kutta), via <code>scipy.integrate.solve_ivp</code>:
</p>
<ul>
  <li>Relative tolerance: <code>rtol = 10⁻¹⁰</code></li>
  <li>Absolute tolerance: <code>atol = 10⁻¹²</code></li>
  <li>Jacobi constant error after 8 days: <code>ΔC ≈ 10⁻⁸</code> — excellent</li>
  <li>~4 500 output points over 8 days → smooth animation at 30 fps</li>
</ul>

<p>
The Lambert solver uses <b>Brent's method</b> to find the root of the
scalar time-of-flight equation in ψ.
Convergence is guaranteed for <code>xtol = 10⁻¹²</code> in &lt;50 iterations.
</p>

<!-- ═══════════════════════════════════════════════════════════ -->
<hr>
<h2 style="color:#d2a8ff;font-size:18pt;font-weight:bold;margin-top:28px;padding-left:10px;border-left:4px solid #d2a8ff;">References &amp; Further Reading</h2>

<ul class="ref-list">
  <li>
    <b>Bate, Mueller &amp; White</b> — <em>Fundamentals of Astrodynamics</em>
    (Dover, 1971). <span class="tag">Lambert solver · Kepler propagator</span><br>
    <span class="dim">The classic reference. Chapter 5 covers the universal variable formulation used in this tool.</span>
  </li>
  <li>
    <b>Battin, R.H.</b> — <em>An Introduction to the Mathematics and Methods
    of Astrodynamics</em> (AIAA, 1987).
    <span class="tag">Lambert · Stumpff functions</span><br>
    <span class="dim">Rigorous treatment of Lambert's problem and the Stumpff function approach.</span>
  </li>
  <li>
    <b>Szebehely, V.</b> — <em>Theory of Orbits: The Restricted Problem of
    Three Bodies</em> (Academic Press, 1967).
    <span class="tag purple">CR3BP · Lagrange points · ZVC</span><br>
    <span class="dim">Definitive reference for the CR3BP. Chapters 3–4 cover equilibrium points and zero velocity curves.</span>
  </li>
  <li>
    <b>Koon, Lo, Marsden &amp; Ross</b> — <em>Dynamical Systems, the
    Three-Body Problem and Space Mission Design</em> (Springer, 2011).
    <span class="tag purple">Manifolds · Free-return</span><br>
    <span class="dim">Available free online. Essential for understanding the topological structure of the CR3BP and free-return trajectories.</span>
    <a href="http://www.cds.caltech.edu/~marsden/books/Mission_Design.html">[pdf]</a>
  </li>
  <li>
    <b>Vallado, D.A.</b> — <em>Fundamentals of Astrodynamics and Applications</em>
    (Microcosm, 4th ed. 2013). <span class="tag blue">Comprehensive reference</span><br>
    <span class="dim">Practical algorithms for all topics in this tool, with MATLAB/Python code examples.</span>
  </li>
  <li>
    <b>NASA TM-2004-213522</b> — <em>Trajectory Design Considerations for
    Exploration Missions</em>. <span class="tag green">Artemis context</span><br>
    <span class="dim">NASA technical memo covering free-return and hybrid trajectories relevant to Artemis.</span>
  </li>
  <li>
    <b>poliastro</b> — Open-source Python astrodynamics library.
    <a href="https://github.com/poliastro/poliastro">[GitHub]</a>
    <span class="tag blue">Python</span><br>
    <span class="dim">Higher-fidelity Lambert solvers, ephemeris support, and 3D trajectory tools.</span>
  </li>
</ul>

<hr>
<p class="dim" style="font-size:12pt; text-align:center;">
  This tool implements a simplified educational model (2D, coplanar, Earth-only for Lambert,
  CR3BP for 3-body dynamics). It is intended for conceptual understanding, not mission design.
</p>

</div>
</body>
</html>
"""


class TheoryTab(QWidget):
    """Tab 4 — Theory reference card rendered as styled HTML."""

    # Base font size in points — change this one constant to scale everything.
    BASE_PT = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        # ── Font fix ──────────────────────────────────────────────────────────
        # QTextBrowser on Windows ignores setFont() once HTML with <style> is
        # loaded.  The only reliable approach is:
        #   1. Set default font on the underlying QTextDocument BEFORE loading HTML
        #   2. Inject the pt size into the <body style="..."> attribute so the
        #      HTML engine cannot ignore it
        from PyQt5.QtGui import QFont as _QFont
        base_font = _QFont("Segoe UI", self.BASE_PT)
        browser.document().setDefaultFont(base_font)

        # Inject pt size directly into body tag — overrides any CSS cascade issue
        html = THEORY_HTML.replace(
            '<body>',
            f'<body style="font-size:{self.BASE_PT}pt; font-family: Segoe UI, Helvetica Neue, Arial, sans-serif;">'
        )
        browser.setHtml(html)
        # ─────────────────────────────────────────────────────────────────────

        browser.setStyleSheet(f"""
            QTextBrowser {{
                background: {DARK_BG};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {PANEL_BG};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {BORDER};
                border-radius: 6px;
                min-height: 40px;
            }}
            QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0; }}
        """)
        layout.addWidget(browser)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class TLIWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TLI + CR3BP Educational Tool  —  Artemis 2 Trajectory Analyser")
        # Adaptive size: use 90% of available screen, minimum 1280×780
        from PyQt5.QtWidgets import QDesktopWidget
        screen = QDesktopWidget().availableGeometry()
        w = max(1280, int(screen.width()  * 0.90))
        h = max(780,  int(screen.height() * 0.90))
        self.resize(w, h)
        self._build_ui()
        self._connect()
        self._update()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        root = QVBoxLayout(c); root.setContentsMargins(8,8,8,8); root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {BORDER};
                border-top: none;
                border-radius: 0 0 4px 4px;
            }}
            QTabBar {{
                font-size: 14px;
                font-weight: bold;
            }}
            QTabBar::tab {{
                background: {PANEL_BG};
                color: {TEXT_DIM};
                padding: 10px 28px;
                min-width: 220px;
                border: 1px solid {BORDER};
                border-bottom: none;
                border-radius: 6px 6px 0 0;
                margin-right: 4px;
                font-size: 14px;
                font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background: {DARK_BG};
                color: {TEXT_PRIMARY};
                border-bottom: 3px solid {ACCENT};
            }}
            QTabBar::tab:hover {{
                color: {TEXT_PRIMARY};
                background: #1f2937;
            }}
        """)

        # Tab 1: Lambert
        self._tab_lambert = QWidget()
        self._build_lambert_tab(self._tab_lambert)
        self._tabs.addTab(self._tab_lambert, "Tab 1  —  Lambert / TLI")
        # Tab 2: CR3BP
        self._tab_cr3bp = CR3BPTab()
        self._tabs.addTab(self._tab_cr3bp, "Tab 2  —  CR3BP Dynamics")
        # Tab 3: Animation
        self._tab_anim = AnimationTab()
        self._tabs.addTab(self._tab_anim, "Tab 3  —  Animation")
        # Tab 4: Theory
        self._tab_theory = TheoryTab()
        self._tabs.addTab(self._tab_theory, "Tab 4  —  Theory")

        self._tabs.currentChanged.connect(self._on_tab_change)
        root.addWidget(self._tabs)

    def _build_lambert_tab(self, parent):
        root = QHBoxLayout(parent)
        root.setContentsMargins(12,12,12,12); root.setSpacing(12)

        left = QWidget(); left.setFixedWidth(360)
        lv = QVBoxLayout(left); lv.setContentsMargins(0,0,0,0); lv.setSpacing(8)
        lv.addWidget(self._ctrl_box())
        lv.addWidget(self._out_box())
        lv.addWidget(self._legend_box())
        lv.addStretch()

        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(0,0,0,0); rv.setSpacing(4)
        hdr = QLabel("🌕  Translunar Injection — Earth-Centered Inertial View")
        hdr.setStyleSheet(f"font-size:16px;font-weight:bold;color:{TEXT_PRIMARY};")
        rv.addWidget(hdr)
        self.canvas = self._build_canvas()
        rv.addWidget(self.canvas)
        sub = QLabel("Key insight: TLI does not aim at the Moon — it aims at where the Moon will be after TOF.  "
                     "The transfer arc angle controls which Lambert solution is selected.")
        sub.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
        sub.setWordWrap(True)
        rv.addWidget(sub)

        root.addWidget(left)
        root.addWidget(right, stretch=1)

    def _ctrl_box(self):
        box = QGroupBox("Mission Parameters")
        g = QGridLayout(box); g.setSpacing(6); g.setColumnStretch(0,1)

        def sl(row, label, lo, hi, val, unit, scale=1):
            ln = QLabel(label); ln.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            lv = QLabel(f"{val/scale:.1f}{unit}" if scale != 1 else f"{val}{unit}")
            lv.setAlignment(Qt.AlignRight); lv.setStyleSheet(f"color:{ACCENT};font-size:12px;font-weight:bold;")
            s  = QSlider(Qt.Horizontal); s.setRange(lo,hi); s.setValue(val)
            g.addWidget(ln, row,   0); g.addWidget(lv, row,   1)
            g.addWidget(s,  row+1, 0, 1, 2)
            return s, lv

        self.sl_alt,   self.lbl_alt   = sl(0, "LEO Altitude [km]",   150, 1000, 185, " km")
        self.sl_tof,   self.lbl_tof   = sl(2, "Time of Flight",        20,   50,  30, " d", scale=10)
        self.sl_phase, self.lbl_phase = sl(4, "Moon Phase [°]",          0,  360,  90, "°")
        self.sl_arc,   self.lbl_arc   = sl(6, "Transfer Arc [°]  ←key", 91,  179, 172, "°")

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};"); g.addWidget(sep, 8, 0, 1, 2)

        bw = QWidget(); bh = QHBoxLayout(bw)
        bh.setContentsMargins(0,0,0,0); bh.setSpacing(6)
        self.btn_run = QPushButton("⟳  Recompute")
        self.btn_opt = QPushButton("★  Min Δv")
        self.btn_opt.setObjectName("opt")
        bh.addWidget(self.btn_run); bh.addWidget(self.btn_opt)
        g.addWidget(bw, 9, 0, 1, 2)
        return box

    def _out_box(self):
        box = QGroupBox("Computed Results")
        g = QGridLayout(box); g.setSpacing(4)

        def row(txt, acc=ACCENT):
            l = QLabel(txt); l.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            v = VLabel(acc); return l, v

        rows = [("Δv  TLI  [km/s]", ACCENT3), ("v  LEO circular  [km/s]", TEXT_PRIMARY),
                ("v  departure  [km/s]", ACCENT2), ("v  arrival  [km/s]", TEXT_DIM),
                ("Apogee radius  [km]", TEXT_DIM), ("Transfer arc  [°]", ACCENT4)]
        attrs = ["val_dv","val_vleo","val_vdep","val_varr","val_ra","val_arc"]
        for i, ((txt,acc),attr) in enumerate(zip(rows,attrs)):
            l,v = row(txt,acc)
            g.addWidget(l, i*2, 0, 1, 2); g.addWidget(v, i*2+1, 0, 1, 2)
            setattr(self, attr, v)
        return box

    def _legend_box(self):
        box = QGroupBox("Legend")
        vl = QVBoxLayout(box); vl.setSpacing(3)
        items = [("●",ACCENT,"LEO orbit"), ("●","#f0e68c","Moon orbit"),
                 ("●","#b0c4de","Moon — now"), ("●","#ffd700","Moon — arrival"),
                 ("●",ACCENT3,"Transfer trajectory"),
                 ("→",ACCENT2,"TLI Δv (post-burn vel.)"),
                 ("→",ACCENT,"v_LEO (pre-burn vel.)"),
                 ("[X]","#ff6e6e","Wrong: aim at Moon now")]
        for sym,col,txt in items:
            w=QWidget(); hb=QHBoxLayout(w); hb.setContentsMargins(0,0,0,0)
            s=QLabel(sym); s.setStyleSheet(f"color:{col};font-size:15px;"); s.setFixedWidth(20)
            t=QLabel(txt); t.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            hb.addWidget(s); hb.addWidget(t); hb.addStretch()
            vl.addWidget(w)
        return box

    def _build_canvas(self):
        self.fig = Figure(facecolor=DARK_BG)
        gs  = gridspec.GridSpec(1, 2, figure=self.fig, width_ratios=[2.6, 1],
                                wspace=0.08, left=0.05, right=0.97,
                                top=0.95, bottom=0.09)
        self.ax_map  = self.fig.add_subplot(gs[0])
        self.ax_dv   = self.fig.add_subplot(gs[1])
        self._style(self.ax_map); self._style(self.ax_dv)
        c = FigureCanvas(self.fig)
        c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return c

    def _style(self, ax):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_DIM, labelsize=10)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        ax.grid(True, color=BORDER, linewidth=0.4, alpha=0.5)

    def _on_tab_change(self, idx):
        if idx == 1:   # CR3BP tab
            self._push_to_cr3bp()
        elif idx == 2:  # Animation tab
            self._push_to_anim()

    def _push_to_anim(self):
        """Push latest CR3BP result to animation tab."""
        r = getattr(self._tab_cr3bp, "_result", None)
        if r is not None:
            self._tab_anim.set_cr3bp_result(
                r, self._tab_cr3bp._phase_deg)

    def _push_to_cr3bp(self):
        """Send current Lambert result to the CR3BP tab."""
        if hasattr(self, "_last_result") and self._last_result is not None:
            self._tab_cr3bp.set_lambert_result(
                self._last_result,
                self.sl_alt.value(),
                self.sl_tof.value() / 10.0,
                self.sl_phase.value(),
                self.sl_arc.value(),
            )

    # ------------------------------------------------------------ signals --

    def _connect(self):
        for sl in (self.sl_alt, self.sl_tof, self.sl_phase, self.sl_arc):
            sl.valueChanged.connect(self._on_slide)
        self.btn_run.clicked.connect(self._update)
        self.btn_opt.clicked.connect(self._optimize)

    def _on_slide(self):
        self.lbl_alt.setText(f"{self.sl_alt.value()} km")
        self.lbl_tof.setText(f"{self.sl_tof.value()/10:.1f} d")
        self.lbl_phase.setText(f"{self.sl_phase.value()}°")
        self.lbl_arc.setText(f"{self.sl_arc.value()}°")
        try:    self._timer.stop()
        except AttributeError:
            self._timer = QTimer(); self._timer.setSingleShot(True)
            self._timer.timeout.connect(self._update)
        self._timer.start(170)

    def _optimize(self):
        alt = self.sl_alt.value()
        tof = self.sl_tof.value()/10.0
        ph  = self.sl_phase.value()
        arcs, dvs = sweep_dv_vs_arc(alt, tof, ph)
        valid = ~np.isnan(dvs)
        if valid.any():
            idx = np.nanargmin(dvs)
            self.sl_arc.setValue(int(round(arcs[idx])))
        self._update()

    # ---------------------------------------------------------- computation --

    def _update(self):
        alt  = self.sl_alt.value()
        tof  = self.sl_tof.value()/10.0
        ph   = self.sl_phase.value()
        arc  = self.sl_arc.value()

        r_now = moon_now(ph)
        res = None; err = None
        try:
            res = compute_tli(alt, tof, ph, arc)
        except Exception as e:
            err = str(e)

        tx = ty = None
        if res is not None:
            try:
                tx, ty = propagate_keplerian(res["r1"], res["v1"], tof*86400.0)
            except Exception as e:
                err = (err or "") + f" | prop: {e}"

        # Δv vs arc curve (recomputed on each update for current params)
        arcs, dvs = sweep_dv_vs_arc(alt, tof, ph)

        self._draw_map(alt, tof, r_now, res, tx, ty, err)
        self._draw_dvcurve(arcs, dvs, arc, res)
        self._update_labels(res)

    def _update_labels(self, res):
        self._last_result = res   # store for CR3BP tab
        if res is None:
            for a in ("val_dv","val_vdep","val_varr","val_ra","val_arc"):
                getattr(self, a).setText("— err —")
            return
        self.val_dv.setText(f"{res['dv']:.3f}")
        self.val_vleo.setText(f"{res['v_leo']:.3f}")
        self.val_vdep.setText(f"{res['v_dep']:.3f}")
        self.val_varr.setText(f"{res['v_arr']:.3f}")
        self.val_ra.setText(f"{res['r_apo_km']:,.0f}")
        self.val_arc.setText(f"{res['arc_true']:.1f}")

    # --------------------------------------------------------------- plots --

    def _draw_map(self, alt_km, tof_days, r_now, res, tx, ty, err):
        ax = self.ax_map
        ax.clear(); self._style(ax)
        SC = 1e6  # m → ×10³ km

        # Moon orbit
        mx, my = orbit_points(R_MOON_ORB)
        ax.plot(mx/SC, my/SC, color="#f0e68c", lw=0.7, ls="--", alpha=0.4, zorder=2)

        # LEO
        r_leo = R_EARTH + alt_km*1e3
        lx, ly = orbit_points(r_leo)
        ax.plot(lx/SC, ly/SC, color=ACCENT, lw=1.1, alpha=0.7, zorder=3)

        # Earth
        ax.add_patch(mpa.Circle((0,0), R_EARTH/SC, color="#1a6fbf", zorder=6))
        ax.text(0, 0, "Earth", color="white", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=7)

        # Moon now
        ax.plot(*r_now/SC, "o", color="#b0c4de", ms=11, zorder=6)
        self._lbl(ax, r_now/SC, "Moon\n(now)", "#b0c4de", dy=+0.022)

        if res:
            r2 = res["r2"]; r1 = res["r1"]
            v1 = res["v1"]; vlv = res["v_leo_vec"]

            # Moon at arrival
            ax.plot(*r2/SC, "o", color="#ffd700", ms=13, zorder=7)
            self._lbl(ax, r2/SC, "Moon\n(arrival)", "#ffd700", dy=+0.022)

            # Departure
            ax.plot(*r1/SC, "o", color=ACCENT3, ms=8, zorder=8)
            self._lbl(ax, r1/SC, "Departure", ACCENT3, dy=-0.027, va="top")

            # Transfer trajectory
            if tx is not None and len(tx) > 2:
                ax.plot(tx/SC, ty/SC, color=ACCENT3, lw=2.3, alpha=0.92, zorder=4)

            # Arc annotation on LEO
            a1 = np.degrees(np.arctan2(r1[1], r1[0]))
            a2 = np.degrees(np.arctan2(r2[1], r2[0]))
            # Always draw the shorter arc
            da = (a2 - a1) % 360
            if da > 180: a1, a2 = a2, a1+360
            else: a2 = a1 + da
            ax.add_patch(mpa.Arc((0,0), 2*r_leo/SC, 2*r_leo/SC,
                                 angle=0, theta1=min(a1,a2), theta2=max(a1,a2),
                                 color=ACCENT4, lw=1.5, ls=":", zorder=5, alpha=0.8))

            # Velocity arrows — fixed visual scale = 8% of Moon distance
            sc_arr = R_MOON_ORB * 0.08 / max(np.linalg.norm(v1), 1.0)
            ax.annotate("", zorder=9,
                xy  =((r1[0]+v1[0]*sc_arr)/SC,  (r1[1]+v1[1]*sc_arr)/SC),
                xytext=(r1[0]/SC, r1[1]/SC),
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=2.3))
            ax.annotate("", zorder=9,
                xy  =((r1[0]+vlv[0]*sc_arr)/SC, (r1[1]+vlv[1]*sc_arr)/SC),
                xytext=(r1[0]/SC, r1[1]/SC),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6, linestyle="dashed"))

            tip = (r1 + v1*sc_arr*0.68)/SC
            ax.text(tip[0], tip[1], f"  Δv={res['dv']:.2f} km/s",
                    color=ACCENT2, fontsize=11, fontweight="bold", zorder=10)

            # "Wrong" arrow
            tgt = r_now/SC * 0.50
            ax.annotate("", zorder=3, xy=tgt, xytext=r1/SC,
                arrowprops=dict(arrowstyle="->", color="#ff6e6e",
                                lw=1.0, linestyle="dotted", alpha=0.5))
            mid = (r1/SC + tgt)/2
            ax.text(mid[0], mid[1], "[X] Wrong:\naim at Moon now",
                    color="#ff6e6e", fontsize=10, alpha=0.7, ha="center")

            # Info bar
            info = (f"LEO {alt_km} km  |  TOF {tof_days:.1f} d  |  "
                    f"Arc {res['arc_true']:.1f}°  |  Apogee {res['r_apo_km']:,.0f} km")
            ax.text(0.5, 0.01, info, transform=ax.transAxes,
                    color=TEXT_DIM, fontsize=10, ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                              edgecolor=BORDER, alpha=0.85))

        if err:
            ax.text(0.5, 0.5, f"[!]  {err}", transform=ax.transAxes,
                    color=ACCENT3, ha="center", va="center", fontsize=9,
                    wrap=True, zorder=15,
                    bbox=dict(boxstyle="round", facecolor=DARK_BG, alpha=0.8))

        lim = R_MOON_ORB/SC * 1.18
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("x  [×10³ km]", color=TEXT_DIM, fontsize=11)
        ax.set_ylabel("y  [×10³ km]", color=TEXT_DIM, fontsize=11)

    def _draw_dvcurve(self, arcs, dvs, current_arc, res):
        ax = self.ax_dv
        ax.clear(); self._style(ax)

        valid = ~np.isnan(dvs)
        if valid.any():
            ax.plot(arcs[valid], dvs[valid], color=ACCENT, lw=1.8, alpha=0.9)
            ax.fill_between(arcs[valid], dvs[valid],
                            dvs[valid].max()*1.05,
                            color=ACCENT, alpha=0.06)

        # Hohmann reference
        ax.axvline(180, color="#f0e68c", lw=0.8, ls="--", alpha=0.5)
        ax.text(178, ax.get_ylim()[1]*0.98 if ax.get_ylim()[1]>0 else 8,
                "Hohmann\n←", color="#f0e68c", fontsize=10,
                ha="right", va="top", alpha=0.8)

        # Current point
        if res:
            ax.axvline(current_arc, color=ACCENT3, lw=1.2, ls=":", alpha=0.8)
            ax.plot(current_arc, res["dv"], "o", color=ACCENT3, ms=9, zorder=6)
            ax.text(current_arc, res["dv"] + 0.05, f" {res['dv']:.2f}",
                    color=ACCENT3, fontsize=11, va="bottom", fontweight="bold")

        ax.set_xlabel("Transfer Arc [°]", color=TEXT_DIM, fontsize=11)
        ax.set_ylabel("Δv  [km/s]", color=TEXT_DIM, fontsize=11)
        ax.set_title("Δv vs Arc Angle", color=TEXT_PRIMARY, fontsize=12, pad=6, fontweight="bold")
        ax.set_xlim(88, 182)
        if valid.any() and not np.all(np.isnan(dvs[valid])):
            lo = max(0, np.nanmin(dvs[valid]) - 0.3)
            hi = min(np.nanmax(dvs[valid]) + 0.5, 12)
            ax.set_ylim(lo, hi)

        self.canvas.draw_idle()

    @staticmethod
    def _lbl(ax, pos, txt, col, dy, va="bottom"):
        ref = R_MOON_ORB/1e6
        ax.text(pos[0], pos[1]+dy*ref, txt,
                color=col, ha="center", va=va, fontsize=10, fontweight="bold")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    # Global base font — raises the floor for everything
    from PyQt5.QtGui import QFont
    base_font = QFont("Segoe UI", 14)
    app.setFont(base_font)
    p = QPalette()
    p.setColor(QPalette.Window,          QColor(DARK_BG))
    p.setColor(QPalette.WindowText,      QColor(TEXT_PRIMARY))
    p.setColor(QPalette.Base,            QColor(PANEL_BG))
    p.setColor(QPalette.AlternateBase,   QColor(DARK_BG))
    p.setColor(QPalette.Text,            QColor(TEXT_PRIMARY))
    p.setColor(QPalette.Button,          QColor(PANEL_BG))
    p.setColor(QPalette.ButtonText,      QColor(TEXT_PRIMARY))
    p.setColor(QPalette.Highlight,       QColor(ACCENT))
    p.setColor(QPalette.HighlightedText, QColor(DARK_BG))
    app.setPalette(p)
    win = TLIWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
