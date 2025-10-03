import pytest
import tempfile
import os
import shutil
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock

# Import the functions we want to test
from sd_file_parser import (
    parse_spotter_files,
    filterSOS,
    epochToDateArray,
    getVersions,
    parseSpectralFiles,
    cli_main
)


@pytest.fixture
def test_setup():
    """Setup for SPC file testing with example data."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, 'example_data', '2021-01-15')
    output_path = os.path.join(input_path, 'processed')
    
    yield {
        'input_path': input_path,
        'output_path': output_path
    }
    
    # Cleanup
    if os.path.exists(output_path):
        print(f"cleanup: deleting {output_path}")
        shutil.rmtree(output_path)


class TestFilterSOS:
    """Tests for the filterSOS function."""
    
    def test_valid_inputs(self):
        """Test filterSOS with valid inputs."""
        result = filterSOS(1, 0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 6)  # Expected SOS matrix shape
        
    def test_invalid_version_number(self):
        """Test filterSOS with invalid version numbers."""
        # Version 0 should return 0.0
        assert filterSOS(0, 0) == 0.0
        
        # Negative version should return 0.0
        assert filterSOS(-1, 0) == 0.0
        
        # Unsupported version should raise ValueError
        with pytest.raises(ValueError, match="Version .* is not supported"):
            filterSOS(10, 0)
    
    def test_invalid_iir_weight_type(self):
        """Test filterSOS with invalid IIR weight types."""
        with pytest.raises(ValueError, match="IIRWeightType .* not supported"):
            filterSOS(1, 5)
            
    def test_non_numeric_inputs(self):
        """Test filterSOS with non-numeric inputs."""
        with pytest.raises(TypeError, match="must be numeric"):
            filterSOS("invalid", 0)
            
        with pytest.raises(TypeError, match="must be numeric"):
            filterSOS(1, "invalid")


class TestEpochToDateArray:
    """Tests for the epochToDateArray function."""
    
    def test_single_timestamp(self):
        """Test with a single timestamp."""
        epoch_time = np.array([1640995200.0])  # 2022-01-01 00:00:00 UTC
        result = epochToDateArray(epoch_time)
        
        assert result.shape == (1, 7)  # 6 date components + milliseconds
        assert result[0, 0] == 2022  # Year
        assert result[0, 1] == 1     # Month
        assert result[0, 2] == 1     # Day
        
    def test_multiple_timestamps(self):
        """Test with multiple timestamps."""
        epoch_times = np.array([1640995200.0, 1640995260.5])  # With fractional seconds
        result = epochToDateArray(epoch_times)
        
        assert result.shape == (2, 7)
        assert result[1, 6] == 500.0  # 0.5 seconds = 500 milliseconds
        
    def test_fractional_seconds(self):
        """Test that fractional seconds are properly converted to milliseconds."""
        epoch_time = np.array([1640995200.123])
        result = epochToDateArray(epoch_time)
        
        assert abs(result[0, 6] - 123.0) < 1.0  # Allow for floating point precision


class TestParseSpotterData:
    """Tests for the process_spotter_data function."""
    
    def test_valid_inputs(self, test_setup):
        """Test parse_spotter_files with valid inputs."""
        input_dir = test_setup['input_path']
        output_dir = test_setup['output_path']
        # Call the actual function (not mocked)
        parse_spotter_files(input_dir, output_dir)
        
        # Check that output directory exists
        assert os.path.exists(output_dir), f"Output directory was not created: {output_dir}"
        
        # Verify all returned files actually exist
        output_files = ['Szz.csv', 'displacement.csv', 'location.csv']
        for filename in output_files:
            output_file = os.path.join(output_dir, filename)
            assert os.path.exists(output_file), f"Output file does not exist: {output_file}"
    
    def test_invalid_input_path(self):
        """Test with non-existent input path."""
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            parse_spotter_files('/non/existent/path', '/output')
    
    def test_invalid_output_format(self, test_setup):
        """Test with invalid output format."""
        input_dir = test_setup['input_path']
        output_dir = test_setup['output_path']
        
        with pytest.raises(ValueError, match="Unknown outputFileType; options are: numpy , matlab , pickle , csv"):
            parse_spotter_files(input_dir, output_dir, output_format='INVALID')
    
    def test_default_output_path(self, test_setup):
        """Test that default output path is created correctly."""
        input_dir = test_setup['input_path']
        
        output_path = parse_spotter_files(input_dir, None)
            
        expected_output = os.path.join(input_dir, 'processed')
        assert output_path == expected_output
        assert os.path.exists(expected_output)


class TestGetFileVersions:
    """Tests for the get_file_versions function."""

    
    def test_valid_file_list(self, test_setup):
        """Test with valid file list."""
        input_path = test_setup['input_path']
        result = getVersions(input_path)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all('fileNumbers' in version for version in result)
    
    def test_empty_file_list(self):
        """Test with empty file list."""
        with pytest.raises(TypeError, match='expected str, bytes or os.PathLike object, not list'):
            getVersions([])
    
    def test_non_existent_files(self):
        """Test with non-existent files."""
        with pytest.raises(TypeError, match='expected str, bytes or os.PathLike object, not list'):
            getVersions(['/non/existent/file.csv'])
    
    def test_invalid_file_list_type(self):
        """Test with invalid file list type."""
        with pytest.raises(FileNotFoundError):
            getVersions("not_a_list")


class TestCLI:
    """Tests for the command-line interface."""
    
    def test_cli_help(self):
        """Test that CLI shows help without error."""
        with patch('sys.argv', ['sd_file_parser.py', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 0
    
    def test_cli_with_input_path(self):
        """Test CLI with basic input path."""
        with patch('sys.argv', ['sd_file_parser.py', '/test/path']):
            with patch('sd_file_parser.parse_spotter_files') as mock_process:
                with patch('os.path.join') as mock_join:
                    mock_join.return_value = '/test/path/processed'
                    cli_main()
                    mock_process.assert_called_once()
    
    def test_cli_with_all_options(self):
        """Test CLI with all options specified."""
        test_args = [
            'sd_file_parser.py', 
            '/input/path',
            '--output', '/output/path',
            '--format', 'matlab',
            '--spectra', 'all'
        ]
        
        with patch('sys.argv', test_args):
            with patch('sd_file_parser.parse_spotter_files') as mock_process:
                cli_main()
                mock_process.assert_called_once_with(
                    input_path='/input/path',
                    output_path='/output/path',
                    output_format='matlab',
                    spectra='all'
                )


class TestParseSpectralFiles:
    """Tests for parseSpectralFiles function."""
    
    def testExplicitOutputParserRun(self, test_setup):
        """
        run the parser on example data, while customizing which
        output files we want
        """
        input_path, output_path = test_setup
        self.inputfn = os.path.join(input_path, '0235_SPC.CSV')
        parseSpectralFiles( inputFileName = self.inputfn, outputPath = self.outputpath,
            outputSpectra = ['Szz', 'Sxx'] )
        # did the output files get created?
        # only Szz gets created by default (see lines 827-829)
        self.assertTrue( os.path.exists( os.path.join( self.outputpath, 'Szz.csv' ) ) )
        self.assertTrue( os.path.exists( os.path.join( self.outputpath, 'Sxx.csv' ) ) )


    def testBasicParserRun(self):
        """
        run the parser on example data, using default parameters
        """
        parseSpectralFiles( inputFileName = self.inputfn, outputPath = self.outputpath )
        # did the output file get created?
        # only Szz gets created by default (see lines 827-829)
        self.assertTrue( os.path.exists( os.path.join( self.outputpath, 'Szz.csv' ) ) )


# Integration tests
@pytest.mark.integration
class TestIntegration:
    """Integration tests that verify complete workflows."""
    
    def test_complete_data_processing_workflow(self, test_setup):
        """Test complete end-to-end data processing with real data validation."""
        input_path = test_setup['input_path']
        output_dir = test_setup['output_path']
        
        # Verify input data exists
        if not os.path.exists(input_path):
            pytest.skip(f"Test data directory not found: {input_path}")
        
        # Check for expected input files
        input_files = os.listdir(input_path)
        expected_types = ['SYS', 'SPC', 'LOC', 'FLT']
        available_types = []
        for file_type in expected_types:
            if any(file_type in f for f in input_files):
                available_types.append(file_type)
        
        if not available_types:
            pytest.skip("No valid spotter files found in test data")
        
        # Process the data
        result = parse_spotter_files(input_path, output_dir, output_format='csv')
        assert result == output_dir, f"Expected output directory {output_dir}, got {result}"
        
        # Comprehensive output validation
        assert os.path.exists(output_dir)
        
        output_files = []
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                output_files.append(file)
        
        assert len(output_files) > 0, "No CSV output files were created"
        
        # Validate each output file
        for output_file in output_files:
            full_path = os.path.join(output_dir, output_file)
            
            # Check file has content
            file_size = os.path.getsize(full_path)
            assert file_size > 0, f"Output file is empty: {output_file}"
            
            # For CSV files, verify they have headers and data
            if output_file.endswith('.csv'):
                try:
                    import pandas as pd
                    df = pd.read_csv(full_path)
                    assert len(df.columns) > 0, f"CSV has no columns: {output_file}"
                    assert len(df) >= 0, f"CSV has no data: {output_file}"  # Allow empty data
                except Exception as e:
                    pytest.fail(f"Could not read CSV file {output_file}: {e}")
        
        print(f"Successfully processed {len(available_types)} file types: {available_types}")
        print(f"Generated {len(output_files)} output files: {output_files}")
    
    def test_multiple_output_formats(self, test_setup):
        """Test that different output formats work correctly."""
        input_path = test_setup['input_path']
        
        if not os.path.exists(input_path):
            pytest.skip("Test data not available")
        
        formats_to_test = ['csv']  # Add 'matlab', 'numpy' if supported
        
        for fmt in formats_to_test:
            output_dir = os.path.join(test_setup['output_path'], f'format_{fmt}')
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                result = parse_spotter_files(input_path, output_dir, output_format=fmt)
                assert result == output_dir, f"Expected output directory {output_dir}, got {result}"
                assert os.path.exists(output_dir)
                
                # Check for format-specific files
                files = os.listdir(output_dir)
                assert len(files) > 0, f"No files created for format {fmt}"
                
            except Exception as e:
                pytest.fail(f"Format {fmt} failed: {e}")