"""Shared helpers for data generation, training and evaluation of the LSTMKraus model."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from qutip import basis, mesolve, smesolve, qeye, sigmax, sigmay, sigmaz, sigmam

from constants import OMEGA, OMEGA_SPREAD, KAPPA, ETA, GAMMA_DECAY, dt, times, Nt


# ---------------------------------------------------------------------------
# Quantum constants -- Pauli operators, initial state, basis rotations
# ---------------------------------------------------------------------------

sigma_x     = np.array([[0,  1], [ 1, 0]], dtype=complex)
sigma_y     = np.array([[0,-1j], [1j, 0]], dtype=complex)
sigma_z     = np.array([[1,  0], [ 0,-1]], dtype=complex)
sigma_lower = np.array([[0,  0], [ 1, 0]], dtype=complex)

# task_id -> measured Pauli (matches data_generation.ipynb): 0=z, 1=x, 2=y.
sigma_task_map = {0: sigma_z, 1: sigma_x, 2: sigma_y}

# Unitaries that rotate rho into the eigenbasis of the measured observable;
# rho_basis[0,0] is then the population of the measured +eigenstate.
_ISQ2 = 1.0 / np.sqrt(2.0)
BASIS_U = {
    0: np.eye(2, dtype=complex),                                # z -> identity
    1: _ISQ2 * np.array([[1,  1], [ 1, -1]], dtype=complex),    # x -> Hadamard
    2: _ISQ2 * np.array([[1, -1j], [ 1,  1j]], dtype=complex),  # y -> sigma_y diagonaliser
}

rho0_torch = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64)
rho0_np    = np.array([[1, 0], [0, 0]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Tensor / dtype helpers
# ---------------------------------------------------------------------------

def normalize_J(J_raw, dt):
    """Scale the raw measurement current by sqrt(dt) so the NN input has O(1) variance."""
    return J_raw * np.sqrt(dt)


def t32(a):      return torch.tensor(a.astype(np.float32))
def t64_c(a):    return torch.tensor(a.astype(np.complex64))
def tlong(n, v): return torch.full((n,), v, dtype=torch.long)


# ---------------------------------------------------------------------------
# Kraus head -- real params -> CPTP map -> one step on rho
# ---------------------------------------------------------------------------

def to_complex_4x2x2(params):
    """(B, 32) real params -> (B, 4, 2, 2) complex Kraus candidates (Re | Im, 4 stacked 2x2)."""
    re = params[:, :16].reshape(-1, 4, 2, 2)
    im = params[:, 16:].reshape(-1, 4, 2, 2)
    return torch.complex(re, im)

#This step enforces the CPTP criteria using a QR algorithm from pytorch.
def normalize_kraus(K_raw):
    """Project 4 Kraus candidates onto the CPTP manifold via QR of the stacked 8x2 matrix.

    M = [K_0; ...; K_3] is 8x2; M = QR with Q 8x2 orthonormal gives sum K_n^dag K_n = I.
    A sign gauge on diag(R) makes the projection unique.
    """
    B = K_raw.shape[0]
    Q, R = torch.linalg.qr(K_raw.reshape(B, 8, 2)) 
    sign = torch.where(
        torch.diagonal(R, dim1=-2, dim2=-1).real >= 0,
        torch.ones (B, 2, device=K_raw.device),
        -torch.ones(B, 2, device=K_raw.device),
    )
    return (Q * sign.unsqueeze(-2)).reshape(B, 4, 2, 2)

#This step really just forwards rho by applying Kraus operators (naming could have been improved)
def apply_cptp(K, rho):
    """One CPTP step: rho_{t+1} = sum_n K_n rho_t K_n^dag, symmetrised to kill numerical drift."""
    rho_next = (K @ rho.unsqueeze(1) @ K.mH).sum(1)
    return (rho_next + rho_next.mH) * 0.5


def rho_to_real_vec(rho):
    """(B, 2, 2) complex rho -> (B, 8) real vector [Re flat | Im flat] for the LSTM input."""
    return torch.cat([rho.real.reshape(-1, 4), rho.imag.reshape(-1, 4)], dim=-1)


# ---------------------------------------------------------------------------
# Loss & model wrapper
# ---------------------------------------------------------------------------

def BCE_loss(P_pred, y_true):
    """Binary cross-entropy of the measured-basis end-population vs. the observed bit."""
    return F.binary_cross_entropy(P_pred.clamp(1e-6, 1 - 1e-6), y_true.float())


def bce_floor(P_true, y_true, eps=1e-6):
    """What would the average loss be, even if the model made every. prediction in agreement with rho_true (for this data)"""
    P = np.clip(np.asarray(P_true, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y_true, dtype=np.float64)
    return -(y * np.log(P) + (1 - y) * np.log(1 - P)).mean()


def predict_rho(model, J, task, rho0=None):
    """Forward the LSTMKraus model and return rho_NN as a (B, T, 2, 2) numpy array."""
    if rho0 is None:
        rho0 = rho0_torch
    rho0_b = rho0.unsqueeze(0).expand(len(J), -1, -1).clone()
    with torch.no_grad():
        rho_seq, _ = model(J, rho0_b, task)
    return rho_seq.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# SME forward integrator + parameter fit
# ---------------------------------------------------------------------------

_FIT_BOUNDS         = [(0, 15), (0, 2), (0, 1), (0, 2)]   # (Omega, kappa, eta, gamma_d)
_OVERFLOW_THRESHOLD = 1e3   # |rho| > this is unphysical -> the iteration is diverging
_OVERFLOW_SENTINEL  = 1e6   # finite stand-in written into the aborted tail


def sme_forward(rho0_arr, J_norm, theta, sigma_task, dt, abort_on_overflow=True):
    """Explicit Euler-Maruyama integrator for the conditional SME. Used to make fitted SME trajectory parameters.
    """
    omega, kappa, eta, gamma_d = theta
    H      = 0.5 * omega * sigma_x
    c_meas = np.sqrt(eta * kappa)         * sigma_task
    c_unm  = np.sqrt((1.0 - eta) * kappa) * sigma_task
    c_dec  = np.sqrt(gamma_d)             * sigma_lower

    c_meas_dag, c_unm_dag, c_dec_dag = c_meas.conj().T, c_unm.conj().T, c_dec.conj().T
    cmcm, cucu, cdcd = c_meas_dag @ c_meas, c_unm_dag @ c_unm, c_dec_dag @ c_dec
    c_m_sym = c_meas + c_meas_dag

    dt_sqrt = np.sqrt(dt)
    n_steps = len(J_norm)
    rho     = np.empty((n_steps + 1, 2, 2), dtype=np.complex128)
    rho[0]  = rho0_arr.astype(np.complex128)
    rho_t   = rho[0].copy()

    for t in range(n_steps):
        if abort_on_overflow and (not np.isfinite(rho_t).all()
                                  or np.abs(rho_t).max() > _OVERFLOW_THRESHOLD):
            rho[t:] = _OVERFLOW_SENTINEL
            break

        J_t  = J_norm[t] * dt_sqrt
        comm = H @ rho_t - rho_t @ H
        diss = ((c_meas @ rho_t @ c_meas_dag - 0.5 * (cmcm @ rho_t + rho_t @ cmcm))
              + (c_unm  @ rho_t @ c_unm_dag  - 0.5 * (cucu @ rho_t + rho_t @ cucu))
              + (c_dec  @ rho_t @ c_dec_dag  - 0.5 * (cdcd @ rho_t + rho_t @ cdcd)))
        expect_cm = np.trace(c_m_sym @ rho_t).real
        dW_t      = J_t - expect_cm * dt
        innov     = c_meas @ rho_t + rho_t @ c_meas_dag - expect_cm * rho_t

        rho_t = rho_t + (-1j * comm + diss) * dt + innov * dW_t
        rho[t + 1] = rho_t

    return rho


def fit_sme_params(rho_traj, J_norm, sigma_task, dt, theta0, maxiter=10000):
    """Fit theta=(Omega, kappa, eta, gamma_d) so SME(theta) matches rho_00 of rho_traj.

    Only rho_00 is supervised -- the LSTM's off-diagonals are not BCE-targeted.
    Uses the so-called L-BFGS-B algorithm.
    """
    def loss(theta):
        rho_fit = sme_forward(rho_traj[0], J_norm, tuple(theta), sigma_task, dt)
        diff    = rho_fit[:, 0, 0] - rho_traj[:, 0, 0]
        return float(np.sum(np.abs(diff) ** 2) / max(diff.shape[0], 1))

    #minimize is an imported scipy function
    res = minimize(loss, theta0, method='L-BFGS-B', bounds=_FIT_BOUNDS,
                   options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-9})
    return res.x, res


def fit_one(model, J, task, rho, idx, omega_i, theta_rates_true, dt):
    """Per-trajectory SME fit to the model's rho_NN. Returns dict for Plot 1."""
    sigma_task   = sigma_task_map[int(task[idx])]
    rho_true_i   = rho[idx].numpy()
    rho_NN       = predict_rho(model, J[idx:idx+1], task[idx:idx+1])[0]
    rho_NN_full  = np.concatenate([rho0_np[np.newaxis], rho_NN], axis=0).astype(np.complex128)
    J_i          = J[idx].numpy()
    theta_true_i = np.array([omega_i, *theta_rates_true])
    theta_hat, _ = fit_sme_params(rho_NN_full, J_i, sigma_task, dt, theta0=theta_true_i)
    return {
        'rho_true':     rho_true_i,
        'rho_NN':       rho_NN_full,
        'rho_fit':      sme_forward(rho0_np, J_i, tuple(theta_hat),    sigma_task, dt),
        'rho_sme_true': sme_forward(rho0_np, J_i, tuple(theta_true_i), sigma_task, dt),
        'theta_hat':    theta_hat,
        'theta_true':   theta_true_i,
    }

#lindblad evolution using the qutip package
def lindblad_ref(task_id, omega):
    """Unconditional Lindblad evolution for the measured basis -- ensemble reference.

    Eta drops out unconditionally: D[sqrt(eta k) sigma] + D[sqrt((1-eta) k) sigma] = k D[sigma].
    """
    sigma_qutip_map = {0: sigmaz(), 1: sigmax(), 2: sigmay()}
    H      = 0.5 * omega * sigmax()
    rho0_q = basis(2, 0) * basis(2, 0).dag()
    c_ops  = [np.sqrt(KAPPA) * sigma_qutip_map[task_id], np.sqrt(GAMMA_DECAY) * sigmam()]
    sol    = mesolve(H, rho0_q, times, c_ops=c_ops, e_ops=[])
    return np.stack([s.full() for s in sol.states])


# ---------------------------------------------------------------------------
# Density-matrix readouts
# ---------------------------------------------------------------------------

def to_meas_basis(rho_arr, task_id):
    """Rotate rho into the eigenbasis of the measured observable for task_id."""
    U = BASIS_U[task_id]
    return U @ rho_arr @ U.conj().T


def meas_pop(rho, tasks):
    """Population of the +eigenstate in the measured basis -- the BCE-supervised observable.

    Matches model.LSTMKraus's P_end:
        z -> rho00,  x -> (1 + 2 Re rho01)/2,  y -> (1 - 2 Im rho01)/2.
    `rho` is (N, T, 2, 2); `tasks` is (N,).
    """
    p  = rho[..., 0, 0].real.copy()
    px = (1.0 + 2.0 * rho[..., 0, 1].real) / 2.0
    py = (1.0 - 2.0 * rho[..., 0, 1].imag) / 2.0
    p  = np.where(tasks[:, None] == 1, px, p)
    p  = np.where(tasks[:, None] == 2, py, p)
    return np.clip(p, 0.0, 1.0)


def bloch_from_rho(arr):
    """Bloch components (bx, by, bz) of a 2x2 density matrix."""
    bx =  2 * arr[..., 0, 1].real
    by = -2 * arr[..., 0, 1].imag
    bz =  2 * arr[..., 0, 0].real - 1
    return bx, by, bz


# ---------------------------------------------------------------------------
# Distance & fidelity metrics  (Bernoulli forms -- BCE-supervised population)
# ---------------------------------------------------------------------------

#These two 'bernoulli' functions are not used for the final plots. We shifted to the full quantum fidelity.
def bernoulli_trace_distance(p, q):
    """Total-variation distance |p - q| between Bernoulli(p) and Bernoulli(q)."""
    return np.abs(p - q)

def bernoulli_fidelity(p, q):
    """Standard (Jozsa) fidelity (sqrt(pq)+sqrt((1-p)(1-q)))^2 between Bernoulli(p) and Bernoulli(q)."""
    return (np.sqrt(p * q) + np.sqrt((1.0 - p) * (1.0 - q))) ** 2


def _herm(a):
    """Symmetrise a batch of (..., n, n) matrices to their Hermitian part."""
    return 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))

#Due to the square roots, one must use the 'Hermitian eigendecomposition'. Takes a long time to run
def quantum_fidelity(rho_nn, rho_true):
    """Uhlmann fidelity  F = Tr sqrt( sqrt(rho_nn) rho_true sqrt(rho_nn) )  per (..., 2, 2) pair.

    Computed over the full density matrix (not just the measured population) and
    vectorised over the leading axes via Hermitian eigendecomposition. `rho_nn`
    (the network output) is symmetrised and its eigenvalues clipped to >=0, so a
    slightly non-physical prediction still yields a real fidelity in [0, 1].
    """
    w, v = np.linalg.eigh(_herm(rho_nn))
    sqrt_nn = (v * np.sqrt(np.clip(w.real, 0.0, None))[..., None, :]) @ np.conj(np.swapaxes(v, -1, -2))
    inner   = sqrt_nn @ rho_true @ sqrt_nn
    eig     = np.linalg.eigvalsh(_herm(inner)).real
    f       = np.sqrt(np.clip(eig, 0.0, None)).sum(axis=-1)
    return np.clip(f, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

#Why bootstrap instead of just using SEM? Because SEM assumes gaussianity (CLT) of the original distribution.
def bootstrap_mean_ci(arr, n_boot, rng, q_lo=16.0, q_hi=84.0):
    """Bootstrap the ensemble mean along axis 0; returns (mean, lo, hi) percentiles.

    Defaults give a ~68% interval (+/- 1 sigma for a Gaussian).
    """
    n    = arr.shape[0]
    boot = np.empty((n_boot, *arr.shape[1:]))
    for b in range(n_boot):
        boot[b] = arr[rng.integers(0, n, size=n)].mean(axis=0)
    lo, hi = np.percentile(boot, [q_lo, q_hi], axis=0)
    return boot.mean(0), lo, hi


def calibration_curve(P_pred, y, edges):
    """Bin (P_pred, y) by predicted probability; return (pred mean, hit rate, count) per bin."""
    idx    = np.clip(np.digitize(P_pred, edges) - 1, 0, len(edges) - 2)
    n_bins = len(edges) - 1
    p_mean = np.full(n_bins, np.nan)
    y_mean = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = idx == b
        counts[b] = int(sel.sum())
        if sel.any():
            p_mean[b] = P_pred[sel].mean()
            y_mean[b] = y[sel].mean()
    return p_mean, y_mean, counts


# ---------------------------------------------------------------------------
# Data generation -- conditional SME trajectories via qutip's smesolve
# ---------------------------------------------------------------------------

_MEAS_OP_QUTIP = {
    'z': (sigmaz, lambda: basis(2, 0) * basis(2, 0).dag()),
    'x': (sigmax, lambda: (qeye(2) + sigmax()) / 2),
    'y': (sigmay, lambda: (qeye(2) + sigmay()) / 2),
}

#Heart of all the data
def generate_trajectories(n_traj=1, meas_basis='z', purpose='train',
                          omega_mean=None, omega_rel_spread=OMEGA_SPREAD, eta=ETA):
    """Simulate `n_traj` conditional SME trajectories under continuous homodyne measurement.

    Each trajectory draws Omega uniformly in (1 +/- omega_rel_spread) * omega_mean.
    Returns (J, y, P, omegas) for purpose='train' and additionally rho when purpose='test'.
    """
    if meas_basis not in _MEAS_OP_QUTIP:
        raise ValueError(f'unknown meas_basis: {meas_basis}')
    if omega_mean is None:
        omega_mean = OMEGA

    sigma_factory, p_factory = _MEAS_OP_QUTIP[meas_basis]
    sigma_op = sigma_factory()
    P_op     = p_factory()

    C_mon   = np.sqrt(eta * KAPPA)         * sigma_op
    C_unmon = np.sqrt((1.0 - eta) * KAPPA) * sigma_op
    C_decay = np.sqrt(GAMMA_DECAY)         * sigmam()
    rho0_q  = basis(2, 0) * basis(2, 0).dag()

    is_test = (purpose == 'test')
    n_t     = Nt - 1
    opts    = {'dt': dt, 'store_measurement': True, 'store_states': is_test,
               'store_final_state': False, 'keep_runs_results': True, 'progress_bar': ''}

    omegas = np.random.uniform(low =(1.0 - omega_rel_spread) * omega_mean,
                               high=(1.0 + omega_rel_spread) * omega_mean,
                               size=n_traj)

    J_all = np.empty((n_traj, n_t))
    y_all = np.empty(n_traj, dtype=np.int8)
    P_all = np.empty(n_traj)
    if is_test:
        rho_all = np.empty((n_traj, len(times), 2, 2), dtype=np.complex64)

    for i in range(n_traj):
        H_i = (omegas[i] / 2.0) * sigmax()
        sol = smesolve(H_i, rho0_q, times,
                       c_ops=[C_unmon, C_decay], sc_ops=[C_mon],
                       ntraj=1, e_ops=[P_op], options=opts)

        J_all[i] = np.asarray(sol.measurement).reshape(-1)
        P_end    = float(np.asarray(sol.expect[0]).reshape(-1)[-1].real)
        P_all[i] = P_end
        y_all[i] = int(np.random.rand() < P_end)

        if is_test:
            states = (sol.states[0]
                      if isinstance(sol.states, list) and sol.states
                      and isinstance(sol.states[0], list)
                      else sol.states)
            rho_all[i] = np.array([r.full() for r in states], dtype=np.complex64)

    if is_test:
        return J_all, y_all, P_all, rho_all, omegas
    return J_all, y_all, P_all, omegas


# ---------------------------------------------------------------------------
# Persistence -- loading saved datasets, models, training histories
# ---------------------------------------------------------------------------

def load_history(suffix, data_dir='data'):
    """Load a saved training-history file."""
    return torch.load(f'{data_dir}/history{suffix}.pt', weights_only=False)


def load_dataset_and_model(suffix, n_eval, data_dir='data'):
    """Load an evaluation dataset + its trained LSTMKraus model."""
    from model import LSTMKraus   # local import to avoid circular dependency
    dataset = torch.load(f'{data_dir}/evaluation_data{suffix}_N{n_eval}.pt', weights_only=True)
    model   = LSTMKraus()
    model.load_state_dict(torch.load(f'{data_dir}/model{suffix}.pt', weights_only=True))
    model.eval()
    return dataset, model


# ---------------------------------------------------------------------------
# Evaluation -- model rollout, per-trajectory fits, residual binning
# ---------------------------------------------------------------------------

def evaluate_dataset(model, dataset, compute_qfidelity=True):
    """Run the model on every trajectory; cache rho_NN, rho_true and measured-basis metrics.

    The full-state Uhlmann ``qfidelity`` is expensive; set ``compute_qfidelity=False`` to
    skip it (the cache then stores ``qfidelity=None``). Only Plot 2 needs it.
    """
    rho_nn   = predict_rho(model, dataset['J_eval'], dataset['task_eval'])  # (N, T,   2, 2)
    rho_true = dataset['rho_eval'].numpy()                                  # (N, T+1, 2, 2)
    tasks    = dataset['task_eval'].numpy()
    pop_nn   = meas_pop(rho_nn,         tasks)
    pop_true = meas_pop(rho_true[:, 1:], tasks)
    return dict(rho_nn=rho_nn, rho_true=rho_true, tasks=tasks,
                omegas=dataset['omegas_eval'].numpy(),
                pop_nn=pop_nn, pop_true=pop_true,
                trace_dist=bernoulli_trace_distance(pop_nn, pop_true),
                fidelity  =bernoulli_fidelity(pop_nn, pop_true),
                qfidelity =quantum_fidelity(rho_nn, rho_true[:, 1:]) if compute_qfidelity else None)


def fit_trajectory(model, dataset, traj_index, omega_true, theta_rates_true, dt):
    """Per-trajectory SME fit wrapper -- thin shim over `fit_one` keyed by dataset dict."""
    return fit_one(model, dataset['J_eval'], dataset['task_eval'], dataset['rho_eval'],
                   traj_index, omega_true, theta_rates_true, dt)


def population_metric_band(cache, n_boot, rng, task_id=0):
    """Bootstrap (mean, lo, hi) bands of trace distance and fidelity over `task_id` trajectories."""
    sel = cache['tasks'] == task_id
    td_mean, td_lo, td_hi = bootstrap_mean_ci(cache['trace_dist'][sel], n_boot, rng)
    fd_mean, fd_lo, fd_hi = bootstrap_mean_ci(cache['fidelity'  ][sel], n_boot, rng)
    return dict(n=int(sel.sum()),
                td_mean=td_mean, td_lo=td_lo, td_hi=td_hi,
                fd_mean=fd_mean, fd_lo=fd_lo, fd_hi=fd_hi)


def quantum_fidelity_band(cache, n_boot, rng, task_id=0):
    """Bootstrap (mean, lo, hi) band of the full-state Uhlmann fidelity over `task_id` trajectories."""
    if cache['qfidelity'] is None:
        raise ValueError("qfidelity was not computed -- rebuild eval_cache with "
                         "compute_qfidelity=True (COMPUTE_QFIDELITY = True).")
    sel = cache['tasks'] == task_id
    mean, lo, hi = bootstrap_mean_ci(cache['qfidelity'][sel], n_boot, rng)
    return dict(n=int(sel.sum()), mean=mean, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# Plot helpers -- shared by Plot 1 (rho components) and Plot 7 (basis labels)
# ---------------------------------------------------------------------------

import plot_style as ps

# (label, extractor, ylim) per rho component -- used by Plot 1.
RHO_COMPONENTS = [
    (r'$\rho_{00}$',              lambda r: r[:, 0, 0].real,    (-0.05, 1.05)),
    (r'$|\rho_{01}|$',            lambda r: np.abs(r[:, 0, 1]), (-0.02, 0.55)),
    (r'$\mathrm{Re}\,\rho_{01}$', lambda r: r[:, 0, 1].real,    (-0.55, 0.55)),
    (r'$\mathrm{Im}\,\rho_{01}$', lambda r: r[:, 0, 1].imag,    (-0.55, 0.55)),
]

TASK_BASIS_CHAR = {0: 'z', 1: 'x', 2: 'y'}


def draw_component(ax, fit, extract, ylim, show_fit):
    """Plot rho_true, rho_NN (and optionally rho_fit) of one component on a single axis."""
    ax.plot(times, extract(fit['rho_true']), color=ps.SEMANTIC['truth'], lw=1.4,
            label=r'$\rho_\text{true}$')
    ax.plot(times, extract(fit['rho_NN']),   color=ps.SEMANTIC['model'], lw=1.4,
            label=r'$\rho_\text{NN}$')
    if show_fit:
        ax.plot(times, extract(fit['rho_fit']), color=ps.SEMANTIC['fit'], lw=1.3,
                ls='--', label=r'$\rho_\text{fit}$')
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(ylim)


def print_theta(tag, fit):
    """Pretty-print the per-trajectory theta fit alongside the ground-truth rates."""
    est, true = fit['theta_hat'], fit['theta_true']
    print(f"  {tag:7s} Omega {est[0]:6.3f}/{true[0]:.3f}   kappa {est[1]:.3f}/{true[1]:.3f}   "
          f"eta {est[2]:.3f}/{true[2]:.3f}   gamma_d {est[3]:.3f}/{true[3]:.3f}")


def basis_components(b):
    """(label, extractor, ylim) per rho^(b) component in the measurement basis -- Plot 7."""
    return [
        (rf'$\rho^{{({b})}}_{{00}}$',                lambda r: r[:, 0, 0].real,    (-0.05, 1.05)),
        (rf'$|\rho^{{({b})}}_{{01}}|$',              lambda r: np.abs(r[:, 0, 1]), (-0.02, 0.55)),
        (rf'$\mathrm{{Re}}\,\rho^{{({b})}}_{{01}}$', lambda r: r[:, 0, 1].real,    (-0.55, 0.55)),
        (rf'$\mathrm{{Im}}\,\rho^{{({b})}}_{{01}}$', lambda r: r[:, 0, 1].imag,    (-0.55, 0.55)),
    ]


#This is the function that fits to the SME parameters. The stride parameter determines that every stride'nth trajectory is fitted on.
def fit_residuals(model, dataset, stride, maxiter, theta_rates_true, dt, task_id=0):
    """Fit (Omega, kappa, eta, gamma_d) on a strided subset; return (residuals, omegas)."""
    J     = dataset['J_eval']
    tasks = dataset['task_eval']
    omegas_full = dataset['omegas_eval'].numpy()
    fit_indices = np.where(tasks.numpy() == task_id)[0][::stride]

    residuals  = np.full((len(fit_indices), 4), np.nan)
    fit_omegas = np.empty(len(fit_indices))
    for slot, idx in enumerate(fit_indices):
        sigma_task   = sigma_task_map[int(tasks[idx])]
        rho_nn       = predict_rho(model, J[idx:idx+1], tasks[idx:idx+1])[0]
        rho_nn_full  = np.concatenate([rho0_np[None], rho_nn], 0).astype(np.complex128)
        theta_guess  = np.array([float(omegas_full[idx]), *theta_rates_true])
        theta_hat, _ = fit_sme_params(rho_nn_full, J[idx].numpy(), sigma_task,
                                      dt, theta0=theta_guess, maxiter=maxiter)
        residuals[slot]  = theta_hat - theta_guess
        fit_omegas[slot] = omegas_full[idx]
    return residuals, fit_omegas
