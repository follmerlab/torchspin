"""Tests for torchspin.rotations.erot."""
import math

import torch

from torchspin.rotations import erot


def test_identity_rotation():
    """Zero angles give the identity matrix."""
    R = erot([0.0, 0.0, 0.0])
    assert (R - torch.eye(3, dtype=torch.float64)).abs().max().item() < 1e-14


def test_rotation_is_orthogonal():
    """R @ R.T == I for arbitrary angles."""
    angles = [0.3, 1.1, -0.7]
    R = erot(angles)
    I3 = torch.eye(3, dtype=torch.float64)
    assert (R @ R.T - I3).abs().max().item() < 1e-13


def test_rotation_determinant():
    """det(R) == +1 (proper rotation)."""
    angles = [0.5, 0.8, 1.2]
    R = erot(angles)
    det = torch.linalg.det(R).item()
    assert abs(det - 1.0) < 1e-13


def test_rotation_about_z():
    """Passive z-rotation: alpha=pi/2 maps old-x vector to [0,-1,0] in new frame.

    erot is a *passive* rotation (coordinate transformation).  Rotating the
    frame CCW by pi/2 around z means the new y-axis points along old -x, so
    a vector along old-x has coordinate [0,-1,0] in the new frame.
    """
    R = erot([math.pi / 2, 0.0, 0.0])
    x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    result = R @ x
    expected = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64)
    assert (result - expected).abs().max().item() < 1e-14


def test_rotation_about_y():
    """Passive y-rotation: beta=pi/2 maps old-z vector to [-1,0,0] in new frame.

    Rotating the frame CCW by pi/2 around y maps the old z-axis onto the
    new -x direction.
    """
    R = erot([0.0, math.pi / 2, 0.0])
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    result = R @ z
    expected = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float64)
    assert (result - expected).abs().max().item() < 1e-14


def test_sequence_composition():
    """R(alpha,beta,gamma) == Rz(gamma) @ Ry(beta) @ Rz(alpha)."""
    alpha, beta, gamma = 0.4, 1.0, -0.6

    def Rz(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=torch.float64)

    def Ry(b):
        c, s = math.cos(b), math.sin(b)
        return torch.tensor([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=torch.float64)

    expected = Rz(gamma) @ Ry(beta) @ Rz(alpha)
    result = erot([alpha, beta, gamma])
    assert (result - expected).abs().max().item() < 1e-13
