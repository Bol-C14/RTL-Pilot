"""
Simplified Unit Tests for Testbench Generation

Basic tests for the RTLTestbenchGenerator functionality.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from rtl_pilot.agents.testbench_gen import RTLTestbenchGenerator
from rtl_pilot.config.settings import Settings


class TestRTLTestbenchGenerator:
    """Basic test cases for RTLTestbenchGenerator class"""
    
    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing"""
        return Settings(
            llm_model="gpt-4",
            llm_api_key="test-key",
            llm_temperature=0.1,
            llm_max_tokens=2000,
            default_coverage_target=95.0
        )
    
    @pytest.fixture
    def temp_directory(self):
        """Temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def testbench_generator(self, mock_settings):
        """RTLTestbenchGenerator instance for testing"""
        return RTLTestbenchGenerator(mock_settings)
    
    def test_initialization(self, mock_settings):
        """Test RTLTestbenchGenerator initialization"""
        generator = RTLTestbenchGenerator(mock_settings)
        assert generator.settings == mock_settings
        assert hasattr(generator, 'logger')
        assert hasattr(generator, 'template_env')
    
    def test_analyze_rtl_design(self, testbench_generator, temp_directory):
        """Test RTL design analysis with a simple file"""
        # Create a test RTL file
        rtl_file = temp_directory / "test_design.v"
        rtl_file.write_text('''
module simple_counter (
    input wire clk,
    input wire rst_n,
    output reg [7:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 8'b0;
        end else begin
            count <= count + 1;
        end
    end
endmodule
''')
        
        analysis = testbench_generator.analyze_rtl_design(rtl_file)
        assert analysis is not None
        assert isinstance(analysis, dict)
    
    def test_generate_verification_scenarios(self, testbench_generator):
        """Test verification scenario generation"""
        # Create a mock analysis result
        analysis = {
            'module_name': 'simple_counter',
            'ports': [
                {'name': 'clk', 'direction': 'input', 'width': 1},
                {'name': 'rst_n', 'direction': 'input', 'width': 1},
                {'name': 'count', 'direction': 'output', 'width': 8}
            ]
        }
        
        scenarios = testbench_generator.generate_verification_scenarios(analysis)
        assert scenarios is not None
        assert isinstance(scenarios, list)
    
    def test_generate_testbench_code(self, testbench_generator):
        """Test testbench code generation"""
        # Create mock analysis and scenarios
        analysis = {'module_name': 'simple_counter'}
        scenarios = [{'name': 'basic_test', 'description': 'Basic functionality test'}]
        
        testbench_code = testbench_generator.generate_testbench_code(analysis, scenarios)
        assert testbench_code is not None
        assert isinstance(testbench_code, str)
