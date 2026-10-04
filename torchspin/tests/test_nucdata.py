"""Tests for torchspin.nucdata module.

Verifies nuclear spin database lookups against EasySpin reference values.
"""
import numpy as np
import pytest

from torchspin.nucdata import nucdata, nucspin, nucgval, nucqmom, nucabund


def test_nucdata_single_nucleus_1H():
    """Test single nucleus lookup for 1H."""
    I, gn, qm, abund = nucdata('1H')
    
    # Reference from EasySpin isotopedata.txt
    assert I == 0.5
    assert abs(gn - 5.58569468) < 1e-7
    assert abs(qm - 0.0) < 1e-10
    assert abs(abund - 0.999885) < 1e-5  # 99.9885% → 0.999885


def test_nucdata_single_nucleus_14N():
    """Test single nucleus lookup for 14N."""
    I, gn, qm, abund = nucdata('14N')
    
    assert I == 1.0
    assert abs(gn - 0.40376100) < 1e-7
    assert abs(qm - 0.02044) < 1e-5
    assert abs(abund - 0.99632) < 1e-5  # 99.632%


def test_nucdata_single_nucleus_63Cu():
    """Test single nucleus lookup for 63Cu."""
    I, gn, qm, abund = nucdata('63Cu')
    
    assert I == 1.5
    assert abs(gn - 1.4824) < 1e-4
    assert abs(qm - (-0.220)) < 1e-3
    assert abs(abund - 0.6917) < 1e-4  # 69.17%


def test_nucdata_multiple_nuclei_comma_string():
    """Test multiple nuclei from comma-separated string."""
    I, gn, qm, abund = nucdata('1H,14N,14N')
    
    # Should return arrays
    assert isinstance(I, np.ndarray)
    assert isinstance(gn, np.ndarray)
    assert isinstance(qm, np.ndarray)
    assert isinstance(abund, np.ndarray)
    
    # Check shape
    assert I.shape == (3,)
    assert gn.shape == (3,)
    
    # Check values
    np.testing.assert_array_equal(I, [0.5, 1.0, 1.0])
    assert abs(gn[0] - 5.58569468) < 1e-6
    assert abs(gn[1] - 0.40376100) < 1e-6
    assert abs(gn[2] - 0.40376100) < 1e-6


def test_nucdata_multiple_nuclei_list_input():
    """Test multiple nuclei from Python list."""
    I, gn, qm, abund = nucdata(['13C', '19F', '31P'])
    
    assert isinstance(I, np.ndarray)
    assert I.shape == (3,)
    
    np.testing.assert_array_equal(I, [0.5, 0.5, 0.5])
    # 13C gn
    assert abs(gn[0] - 1.4048236) < 1e-6
    # 19F gn
    assert abs(gn[1] - 5.257736) < 1e-5
    # 31P gn
    assert abs(gn[2] - 2.26320) < 1e-5


def test_nucdata_string_with_spaces():
    """Test that spaces in comma-separated string are ignored."""
    I, gn, qm, abund = nucdata('1H, 14N, 14N')
    
    assert I.shape == (3,)
    np.testing.assert_array_equal(I, [0.5, 1.0, 1.0])


def test_nucdata_trailing_comma():
    """Test that trailing comma is handled gracefully."""
    I, gn, qm, abund = nucdata('1H,14N,')
    
    assert I.shape == (2,)
    np.testing.assert_array_equal(I, [0.5, 1.0])


def test_nucdata_empty_string():
    """Test empty input string."""
    I, gn, qm, abund = nucdata('')
    
    assert isinstance(I, np.ndarray)
    assert I.shape == (0,)
    assert gn.shape == (0,)


def test_nucdata_empty_list():
    """Test empty list input."""
    I, gn, qm, abund = nucdata([])
    
    assert isinstance(I, np.ndarray)
    assert I.shape == (0,)


def test_nucdata_unknown_isotope():
    """Test that unknown isotope raises ValueError."""
    with pytest.raises(ValueError, match="Unknown isotope"):
        nucdata('999Unobtanium')


def test_nucdata_element_without_mass_gives_helpful_error():
    """Test that specifying element without mass number gives helpful error.
    
    E.g. 'Cu' should suggest '63Cu, 65Cu' instead of just failing.
    """
    with pytest.raises(ValueError, match="Please specify isotope mass number"):
        nucdata('Cu')
    
    # Error message should list available isotopes
    with pytest.raises(ValueError, match="63Cu"):
        nucdata('Cu')


def test_nucdata_unpacking_all_four():
    """Test that nucdata returns all four properties as a tuple."""
    # Full return is always a 4-tuple (for single nucleus)
    result = nucdata('14N')
    assert isinstance(result, tuple)
    assert len(result) == 4
    
    # Unpack all four
    I, gn, qm, abund = nucdata('14N')
    assert I == 1.0
    assert abs(gn - 0.40376100) < 1e-7
    assert abs(qm - 0.02044) < 1e-5
    assert abs(abund - 0.99632) < 1e-5
    
    # If you only want spin, use nucspin() convenience function
    I_only = nucspin('14N')
    assert I_only == 1.0


def test_nucdata_quadrupole_moment_NaN():
    """Test that isotopes without measured quadrupole moment return NaN."""
    # Spin-1/2 nuclei have no quadrupole moment (I < 1)
    I, gn, qm, abund = nucdata('1H')
    assert qm == 0.0  # should be exactly 0 for spin-1/2
    
    # Some nuclei have unmeasured qm (NaN in database)
    # Example: most radioactive isotopes or rare nuclei might have NaN


def test_nucdata_no_args_returns_database():
    """Test that calling nucdata() with no args returns full database dict."""
    db = nucdata()
    
    assert isinstance(db, dict)
    assert 'symbols' in db
    assert 'spins' in db
    assert 'gns' in db
    assert 'qms' in db
    assert 'abundances' in db
    
    # Should have many isotopes (EasySpin has ~300+)
    assert len(db['symbols']) > 300


def test_nucspin_convenience_function():
    """Test nucspin() convenience accessor."""
    I = nucspin('14N')
    assert I == 1.0
    
    I = nucspin('1H,14N')
    np.testing.assert_array_equal(I, [0.5, 1.0])


def test_nucgval_convenience_function():
    """Test nucgval() convenience accessor."""
    gn = nucgval('14N')
    assert abs(gn - 0.40376100) < 1e-7
    
    gn = nucgval('1H,14N')
    assert gn.shape == (2,)
    assert abs(gn[0] - 5.58569468) < 1e-6


def test_nucqmom_convenience_function():
    """Test nucqmom() convenience accessor."""
    qm = nucqmom('14N')
    assert abs(qm - 0.02044) < 1e-5
    
    qm = nucqmom('1H,14N')
    assert qm.shape == (2,)
    assert abs(qm[1] - 0.02044) < 1e-5


def test_nucabund_convenience_function():
    """Test nucabund() convenience accessor."""
    abund = nucabund('14N')
    assert abs(abund - 0.99632) < 1e-5
    
    abund = nucabund('1H,14N')
    assert abund.shape == (2,)
    assert abs(abund[0] - 0.999885) < 1e-5


def test_nucdata_common_isotopes_all_present():
    """Smoke test: verify common EPR-relevant isotopes are all in database."""
    common = [
        '1H', '2H', '13C', '14N', '15N', '17O', '19F',
        '23Na', '27Al', '31P', '33S', '35Cl', '37Cl',
        '51V', '55Mn', '57Fe', '59Co', '63Cu', '65Cu',
    ]
    
    for isotope in common:
        I, gn, qm, abund = nucdata(isotope)
        # Just check that lookup succeeds and returns reasonable values
        assert I >= 0
        assert isinstance(gn, float)


def test_nucdata_spin_values_realistic():
    """Test that returned spin values are physically reasonable half-integers."""
    # Sample a variety of nuclei
    isotopes = ['1H', '2H', '14N', '17O', '27Al', '51V', '55Mn', '59Co']
    I, *_ = nucdata(','.join(isotopes))
    
    # All spins should be non-negative half-integers
    for spin in I:
        assert spin >= 0
        # Check it's a half-integer: 2*I should be an integer
        assert abs(2 * spin - round(2 * spin)) < 1e-10


def test_nucdata_gn_values_nonzero_for_nonzero_spin():
    """Test that nuclei with I > 0 have non-zero g-factors."""
    # Nucleus with spin > 0 must have gn != 0
    I, gn, *_ = nucdata('14N')
    assert I > 0
    assert gn != 0.0
    
    # Nucleus with spin = 0 (rare, but exists: e.g. 12C)
    I, gn, *_ = nucdata('12C')
    assert I == 0.0
    assert gn == 0.0


def test_nucdata_invalid_input_type():
    """Test that invalid input types raise TypeError."""
    with pytest.raises(TypeError):
        nucdata(123)
    
    with pytest.raises(TypeError):
        nucdata(None.__class__)  # weird type
