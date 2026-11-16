"""Field-sweep strain widths from given eigenvectors (perf B15-field).

resfields_batch passes the two eigenvectors of each transition from its
candidate diagonalization instead of letting compute_strain_widths_batch
diagonalize again at the resonance field.  Given the eigenvectors at B0 the
widths must equal the ones from its own eigendecomposition, for every strain
kind (H, g, g + A with a tilted A frame, D with a tilted D frame, two-electron g)."""
import pytest
import torch

from torchspin.ham import ham
from torchspin.spinsystem import SpinSystem
from torchspin.strainwidth import astrain_level_mI, compute_strain_widths_batch

F64 = torch.float64
SYSTEMS = {
    'H': dict(S=0.5, g=[2.05, 2.06, 2.25], Nucs='63Cu', A=[30, 40, 500], HStrain=[40, 40, 120]),
    'gA_tilt': dict(S=0.5, g=[2.05, 2.06, 2.25], Nucs='63Cu', A=[30, 40, 500], gStrain=[0.01, 0.01, 0.03],
                    AStrain=[5, 5, 20], AFrame=[0.3, 0.4, 0.0]),
    'D_tilt': dict(S=1.0, g=2.0, D=[900, 150], DStrain=[90, 30], DFrame=[0.2, 0.5, 0.1]),
    'two_e_g': dict(S=[0.5, 0.5], g=[[2.0, 2.0, 2.0], [2.05, 2.05, 2.2]], ee=[3000.0],
                    gStrain=[[0.005, 0.005, 0.01], [0.01, 0.01, 0.02]]),
}


def _inputs(kw):
    s = SpinSystem(**kw)
    H0, mux, muy, muz = ham(s)
    phi = torch.tensor([0.4, 1.3, 2.0], dtype=F64)
    th = torch.tensor([1.1, 0.6, 0.2], dtype=F64)
    st = torch.sin(th)
    z = torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(th)], dim=1).to(mux.dtype)
    muzL = z[:, 0, None, None] * mux + z[:, 1, None, None] * muy + z[:, 2, None, None] * muz
    B0 = torch.tensor([300.0, 320.0, 340.0], dtype=F64)
    n = H0.shape[0]
    pairs = torch.tensor([[0, n - 1], [0, 1], [1, n - 1]])
    mI = astrain_level_mI(s, H0, muz, 320.0) if 'AStrain' in kw else None
    return s, H0, muzL, phi, th, B0, pairs, mI


@pytest.mark.parametrize('name', list(SYSTEMS))
def test_given_eigenvectors_reproduce_widths(name):
    s, H0, muzL, phi, th, B0, pairs, mI = _inputs(SYSTEMS[name])
    ref = compute_strain_widths_batch(s, H0, muzL, phi, th, B0, None, 9.5, pairs_flat=pairs, mI_levels=mI)
    _, V = torch.linalg.eigh(H0.unsqueeze(0) - B0[:, None, None] * muzL)
    r = torch.arange(3)
    psi_u, psi_v = V[r, :, pairs[:, 0]], V[r, :, pairs[:, 1]]
    phase = torch.exp(1j * torch.tensor([0.3, -1.2, 2.5], dtype=F64)).to(V.dtype)   # gauge must not matter
    new = compute_strain_widths_batch(s, H0, muzL, phi, th, B0, None, 9.5, pairs_flat=pairs, mI_levels=mI,
                                      eigvecs_flat=(psi_u * phase[:, None], psi_v))
    assert (ref > 0).sum() >= 2
    torch.testing.assert_close(new, ref, rtol=1e-10, atol=0.0)   # 1e-12 except near-zero slopes (wide lines)


def test_given_eigenvectors_need_flat_pairs():
    s, H0, muzL, phi, th, B0, pairs, mI = _inputs(SYSTEMS['H'])
    v = torch.zeros(3, H0.shape[0], dtype=H0.dtype)
    with pytest.raises(ValueError, match='pairs_flat'):
        compute_strain_widths_batch(s, H0, muzL, phi, th, B0, [pairs[:1]] * 3, 9.5, eigvecs_flat=(v, v))
