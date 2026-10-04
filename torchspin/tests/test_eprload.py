"""
Tests for eprload - EPR data file loading.

These tests verify that eprload can correctly read Bruker BES3T and ESP formats.
"""

import numpy as np
import pytest
import struct
from pathlib import Path
import tempfile
import shutil

from torchspin.eprload import eprload, _apply_scaling, _convert_value


class TestEprload:
    """Tests for eprload function."""
    
    def setup_method(self):
        """Create temporary directory for test files."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_bruker_bes3t_files(self, basename='test', npts=1024):
        """Create synthetic Bruker BES3T .DTA/.DSC files for testing."""
        dta_file = self.test_dir / f'{basename}.DTA'
        dsc_file = self.test_dir / f'{basename}.DSC'
        
        # Create synthetic spectrum data
        # Simple Gaussian derivative
        x = np.linspace(3400, 3500, npts)  # Gauss
        center = 3450
        width = 10
        y = -(x - center) / width**2 * np.exp(-0.5 * ((x - center) / width)**2)
        
        # Write binary data (.DTA) - big endian 32-bit float
        with open(dta_file, 'wb') as f:
            for val in y:
                f.write(struct.pack('>f', val))
        
        # Write parameter file (.DSC)
        with open(dsc_file, 'w') as f:
            f.write("#DESC\t1.0\n")
            f.write("XMIN\t3400.0\n")
            f.write("XWID\t100.0\n")
            f.write(f"XPTS\t{npts}\n")
            f.write("XUNI\tG\n")  # Gauss
            f.write("YUNI\tarb.u.\n")
            f.write("BSEQ\tBIG\n")
            f.write("IKKF\tREAL\n")
            f.write("IRFMT\tF\n")  # Float32 format
            f.write("MWFQ\t9.5e9\n")  # 9.5 GHz
            f.write("MWPW\t2.0\n")  # 2 mW
            f.write("RCAG\t60.0\n")  # Receiver gain 60 dB
            f.write("STMP\t298.0\n")  # 298 K
            f.write("JSD\t10\n")  # 10 scans
        
        return dta_file, dsc_file, x, y  # native units (Gauss), as EasySpin
    
    def create_bruker_esp_files(self, basename='test', npts=1024):
        """Create synthetic Bruker ESP .spc/.par files for testing."""
        spc_file = self.test_dir / f'{basename}.spc'
        par_file = self.test_dir / f'{basename}.par'

        # Create synthetic spectrum data
        x = np.linspace(3400, 3500, npts)  # Gauss
        center = 3450
        width = 10
        y = -(x - center) / width**2 * np.exp(-0.5 * ((x - center) / width)**2)

        # Write binary data (.spc) — WinEPR format: little-endian float32
        # (DOS key present in .par triggers little-endian + float32 reading)
        with open(spc_file, 'wb') as f:
            f.write(y.astype(np.float32).tobytes())

        # Write parameter file (.par)
        with open(par_file, 'w') as f:
            f.write("DOS \n")            # WinEPR flag (triggers float32 LE)
            f.write("HCF 3450.0\n")      # Center field (Gauss)
            f.write("HSW 100.0\n")       # Sweep width (Gauss)
            f.write(f"RES {npts}\n")     # Resolution
            f.write("MF 9.5e9\n")        # Microwave frequency (Hz)
            f.write("MP 2.0\n")          # Microwave power (mW)
            f.write("RRG 60.0\n")        # Receiver gain (dB)
            f.write("TE 298.0\n")        # Temperature (K)
            f.write("NbScansDone 10\n")  # Number of scans

        # y_expected uses float32 precision (round-trip through binary)
        y_expected = y.astype(np.float32).astype(np.float64)
        return spc_file, par_file, x, y_expected  # native units (Gauss), as EasySpin
    
    def test_load_bruker_bes3t_dta(self):
        """Test loading Bruker BES3T via .DTA file."""
        dta_file, dsc_file, x_expected, y_expected = self.create_bruker_bes3t_files()
        
        x, y, params = eprload(dta_file)
        
        # Check x-axis (native file units, as EasySpin)
        assert len(x) == len(x_expected)
        np.testing.assert_allclose(x, x_expected, rtol=1e-6)
        
        # Check y-axis
        assert len(y) == len(y_expected)
        np.testing.assert_allclose(y, y_expected, rtol=1e-5)
        
        # Check parameters (stored as read from file, in original units)
        assert 'MWFQ' in params
        assert params['MWFQ'] == 9.5e9
        assert params['XMIN'] == 3400.0  # Stored in Gauss as in file
        assert params['XWID'] == 100.0  # Stored in Gauss as in file
    
    def test_load_bruker_bes3t_dsc(self):
        """Test loading Bruker BES3T via .DSC file."""
        dta_file, dsc_file, x_expected, y_expected = self.create_bruker_bes3t_files()
        
        x, y, params = eprload(dsc_file)
        
        assert len(x) == len(x_expected)
        np.testing.assert_allclose(x, x_expected, rtol=1e-6)
        np.testing.assert_allclose(y, y_expected, rtol=1e-5)
    
    def test_load_bruker_esp_spc(self):
        """Test loading Bruker ESP via .spc file."""
        spc_file, par_file, x_expected, y_expected = self.create_bruker_esp_files()
        
        x, y, params = eprload(spc_file)
        
        # Check x-axis (native file units, as EasySpin)
        assert len(x) == len(x_expected)
        np.testing.assert_allclose(x, x_expected, rtol=1e-6)
        
        # Check y-axis
        assert len(y) == len(y_expected)
        np.testing.assert_allclose(y, y_expected, rtol=1e-5)
        
        # Check parameters
        assert 'MF' in params
        assert params['MF'] == 9.5e9
    
    def test_load_bruker_esp_par(self):
        """Test loading Bruker ESP via .par file."""
        spc_file, par_file, x_expected, y_expected = self.create_bruker_esp_files()
        
        x, y, params = eprload(par_file)
        
        assert len(x) == len(x_expected)
        np.testing.assert_allclose(x, x_expected, rtol=1e-6)
    
    def test_scaling_n(self):
        """Test scaling by number of scans."""
        dta_file, dsc_file, x_expected, y_expected = self.create_bruker_bes3t_files()
        
        x, y, params = eprload(dta_file, scaling='n')
        
        # Should be divided by 10 scans
        np.testing.assert_allclose(y, y_expected / 10, rtol=1e-5)
    
    def test_scaling_P(self):
        """Test scaling by sqrt(microwave power)."""
        dta_file, dsc_file, x_expected, y_expected = self.create_bruker_bes3t_files()
        
        x, y, params = eprload(dta_file, scaling='P')
        
        # Should be divided by sqrt(2.0 mW)
        np.testing.assert_allclose(y, y_expected / np.sqrt(2.0), rtol=1e-5)
    
    def test_scaling_G(self):
        """Test scaling by receiver gain."""
        dta_file, dsc_file, x_expected, y_expected = self.create_bruker_bes3t_files()
        
        x, y, params = eprload(dta_file, scaling='G')
        
        # Should be divided by 60.0 dB gain
        np.testing.assert_allclose(y, y_expected / 60.0, rtol=1e-5)
    
    def test_scaling_T(self):
        """Test scaling by temperature."""
        dta_file, dsc_file, x_expected, y_expected = self.create_bruker_bes3t_files()
        
        x, y, params = eprload(dta_file, scaling='T')
        
        # Should be multiplied by 298.0 K
        np.testing.assert_allclose(y, y_expected * 298.0, rtol=1e-5)
    
    def test_scaling_combined(self):
        """Test combined scaling options."""
        dta_file, dsc_file, x_expected, y_expected = self.create_bruker_bes3t_files()
        
        x, y, params = eprload(dta_file, scaling='nPG')
        
        # Should be divided by (10 scans * sqrt(2.0 mW) * 60 dB)
        expected = y_expected / 10 / np.sqrt(2.0) / 60.0
        np.testing.assert_allclose(y, expected, rtol=1e-5)
    
    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            eprload('nonexistent.DTA')
    
    def test_missing_dsc_file(self):
        """Test error when .DSC file is missing."""
        dta_file = self.test_dir / 'test.DTA'
        dta_file.write_bytes(b'\x00' * 100)
        
        with pytest.raises(FileNotFoundError, match='Parameter file not found'):
            eprload(dta_file)
    
    def test_unsupported_format(self):
        """Test error for unsupported file format."""
        txt_file = self.test_dir / 'test.txt'
        txt_file.write_text('data')
        
        with pytest.raises(ValueError, match='Unsupported file extension'):
            eprload(txt_file)


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_convert_value_int(self):
        """Test conversion of integer strings."""
        assert _convert_value('42') == 42
        assert _convert_value('  42  ') == 42
    
    def test_convert_value_float(self):
        """Test conversion of float strings."""
        assert _convert_value('3.14') == 3.14
        assert _convert_value('1.5e-3') == 1.5e-3
    
    def test_convert_value_string(self):
        """Test handling of string values."""
        assert _convert_value('text') == 'text'
        assert _convert_value("'quoted'") == 'quoted'
        assert _convert_value('"double"') == 'double'
    
    def test_apply_scaling_empty(self):
        """Test no scaling applied."""
        y = np.array([1.0, 2.0, 3.0])
        params = {'JSD': 10, 'MWPW': 2.0}
        
        y_scaled = _apply_scaling(y, params, '')
        np.testing.assert_array_equal(y_scaled, y)
    
    def test_apply_scaling_n(self):
        """Test scan number scaling."""
        y = np.array([10.0, 20.0, 30.0])
        params = {'JSD': 5}
        
        y_scaled = _apply_scaling(y, params, 'n')
        np.testing.assert_array_equal(y_scaled, [2.0, 4.0, 6.0])
    
    def test_apply_scaling_multiple(self):
        """Test multiple scaling factors."""
        y = np.array([100.0])
        params = {'JSD': 10, 'MWPW': 4.0, 'STMP': 200.0}
        
        # nPT: divide by 10, divide by sqrt(4)=2, multiply by 200
        # 100 / 10 / 2 * 200 = 1000
        y_scaled = _apply_scaling(y, params, 'nPT')
        np.testing.assert_allclose(y_scaled, [1000.0])


# ---------------------------------------------------------------------------
# Vendor format tests using real test data from tests/eprfiles/
# ---------------------------------------------------------------------------

EPR_DIR = Path(__file__).resolve().parents[2] / 'tests' / 'eprfiles'


@pytest.mark.skipif(not EPR_DIR.exists(), reason="tests/eprfiles/ not found")
class TestVendorFormats:
    """Tests for all vendor-format eprload parsers using real EPR data files."""

    # ---- Bruker BES3T (real files) ----

    def test_bruker_bes3t_dta(self):
        x, y, p = eprload(EPR_DIR / '00012107.dta')
        assert x.ndim == 1 and y.ndim == 1
        assert len(x) == len(y)
        assert len(y) > 0
        assert np.isfinite(y).all()

    def test_bruker_bes3t_xepr(self):
        x, y, p = eprload(EPR_DIR / 'E580_Xepr26b6_cwX_WillMyers.DTA')
        assert len(x) == len(y)
        assert len(y) > 0

    # ---- Bruker ESP (real files) ----

    def test_bruker_esp_real(self):
        x, y, p = eprload(EPR_DIR / '00011201.spc')
        assert len(x) == len(y)
        assert len(y) > 0
        assert np.isfinite(y).all()

    def test_bruker_esp_emx(self):
        x, y, p = eprload(EPR_DIR / 'EMX_field1d.spc')
        assert len(x) == len(y)
        assert len(y) > 0

    # ---- Active Spectrum (.ESR) ----

    def test_active_spectrum(self):
        x, y, p = eprload(EPR_DIR / '10uM_TEMPOL_CAP_SQ__140828_122013.ESR')
        assert len(x) == len(y)
        assert len(y) > 100
        assert np.isfinite(y).all()

    # ---- Adani DAT ----

    def test_adani_dat(self):
        x, y, p = eprload(EPR_DIR / 'adani' / 'pyrelen-2mT5mmAugstaak300sek.dat')
        assert len(x) == len(y)
        assert len(y) > 0
        assert np.isfinite(y).all()

    # ---- Adani JSON ----

    def test_adani_json_1d(self):
        x, y, p = eprload(EPR_DIR / 'adani' / '1d.json')
        assert x.ndim == 1
        assert y.ndim == 1
        assert len(x) == len(y)

    def test_adani_json_2d(self):
        x, y, p = eprload(EPR_DIR / 'adani' / '2d.json')
        assert isinstance(x, list) and len(x) == 2   # one abscissa per dimension (EasySpin)
        assert y.ndim == 2
        assert y.shape == (len(x[0]), len(x[1]))

    def test_adani_json_mn(self):
        x, y, p = eprload(EPR_DIR / 'adani' / 'Mn.json')
        assert len(x) > 0

    # ---- CIQTEK (.epr) ----

    def test_ciqtek_1d(self):
        x, y, p = eprload(EPR_DIR / 'ciqtek' / 'CIQTEK_1D.epr')
        assert x.ndim == 1
        assert len(x) > 0
        # CIQTEK returns complex data
        assert y.dtype == complex or np.isfinite(np.real(y)).all()

    def test_ciqtek_2d(self):
        x, y, p = eprload(EPR_DIR / 'ciqtek' / 'CIQTEK_2D.epr')
        assert isinstance(x, list) and len(x) == 2   # one abscissa per dimension (EasySpin)
        assert y.ndim == 2
        assert y.shape == (len(x[0]), len(x[1]))

    # ---- Magnettech binary (.spe) ----

    def test_magnettech_binary_old(self):
        x, y, p = eprload(EPR_DIR / 'magnettech' / 'oldformat.spe')
        assert len(x) == len(y)
        assert 'B0_Field' in p
        assert 'B0_Scan' in p
        assert np.isfinite(y).all()

    def test_magnettech_binary_new(self):
        x, y, p = eprload(EPR_DIR / 'magnettech' / 'newformat.spe')
        assert len(x) == len(y)
        assert np.isfinite(y).all()

    def test_magnettech_binary_1024(self):
        f = EPR_DIR / 'magnettech' / 'magnettech_mt500l_1024points.spe'
        if f.exists():
            x, y, p = eprload(f)
            assert len(x) == 1024
            assert len(y) == 1024

    # ---- Magnettech XML (.xml) ----

    def test_magnettech_xml_field_sweep(self):
        x, y, p = eprload(EPR_DIR / 'magnettech' / 'Mangan_Ausgangslage.xml')
        assert len(x) == len(y)
        assert len(y) > 0
        assert np.isfinite(y).all()

    def test_magnettech_xml_dip_sweep(self):
        x, y, p = eprload(EPR_DIR / 'magnettech' / 'DipSweep.xml')
        assert len(x) == len(y)
        assert len(y) > 0

    # ---- d00 Weizmann/ETH (.d00) ----

    def test_d00_wis_eth(self):
        x, y, p = eprload(EPR_DIR / '03060101.d00')
        assert y.size > 0

    # ---- SpecMan (.d01) ----

    def test_specman_cw(self):
        x, y, p = eprload(EPR_DIR / 'specman' / 'specman_cw.d01')
        assert y.size > 0

    def test_specman_2pfs(self):
        x, y, p = eprload(EPR_DIR / 'specman' / 'specman_2pfs.d01')
        assert y.size > 0

    def test_specman_t1(self):
        x, y, p = eprload(EPR_DIR / 'specman' / 'T1_BDPA_dtol_95K.d01')
        assert y.ndim == 1
        assert len(y) > 0

    def test_specman_spaces_in_name(self):
        x, y, p = eprload(EPR_DIR / 'specman' / 'steps -16pi to 16pi short.d01')
        assert y.size > 0

    # ---- JEOL ----

    def test_jeol_c60(self):
        x, y, p = eprload(EPR_DIR / 'jeol' / 'C60')
        assert len(x) == len(y)
        assert len(y) > 0
        assert 'Header' in p
        assert 'General' in p

    def test_jeol_data2(self):
        x, y, p = eprload(EPR_DIR / 'jeol' / 'data2')
        assert len(x) > 0

    def test_jeol_2d(self):
        x, y, p = eprload(EPR_DIR / 'jeol' / 'jeol-CuSO4_HItemp.0.2d')
        assert y.size > 0

    def test_jeol_endor(self):
        x, y, p = eprload(EPR_DIR / 'jeol' / 'jeol-endor_coal')
        assert len(x) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
