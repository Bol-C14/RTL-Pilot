"""
Unit Tests for Testbench Generation

Tests the TestbenchGenerator agent functionality including
RTL analysis, scenario generation, and testbench synthesis.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from rtl_pilot.agents.testbench_gen import RTLTestbenchGenerator
from rtl_pilot.config.settings import Settings
from rtl_pilot.config.schema import LLMConfig, VerificationConfig


class TestTestbenchGenerator:
    """Test cases for TestbenchGenerator class"""
    
    @pytest.fixture
    def sample_rtl_code(self):
        """Sample RTL code for testing"""
        return '''
module simple_counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [7:0] count
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 8'h00;
    else if (enable)
        count <= count + 1;
end

endmodule
'''
    
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
        """TestbenchGenerator instance for testing"""
        return RTLTestbenchGenerator(mock_settings)
    
    def test_initialization(self, mock_settings):
        """Test RTLTestbenchGenerator initialization"""
        generator = RTLTestbenchGenerator(mock_settings)
        assert generator.settings == mock_settings
        assert hasattr(generator, 'logger')
        assert hasattr(generator, 'template_env')
    
    def test_analyze_rtl_design(self, testbench_generator, temp_directory):
        """Test RTL design analysis"""
        # Create a test RTL file
        rtl_file = temp_directory / "test_design.v"
        rtl_file.write_text('''
module simple_counter (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [7:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 8'b0;
        end else if (enable) begin
            count <= count + 1;
        end
    end
endmodule
''')
        
        analysis = testbench_generator.analyze_rtl_design(rtl_file)
        assert analysis is not None
        assert isinstance(analysis, dict)
    
    def test_generate_verification_scenarios(self, testbench_generator, temp_directory):
        """Test verification scenario generation"""
        # Create a mock analysis result
        analysis = {
            'module_name': 'simple_counter',
            'ports': [
                {'name': 'clk', 'direction': 'input', 'width': 1},
                {'name': 'rst_n', 'direction': 'input', 'width': 1},
                {'name': 'enable', 'direction': 'input', 'width': 1},
                {'name': 'count', 'direction': 'output', 'width': 8}
            ]
        }
        
        scenarios = testbench_generator.generate_verification_scenarios(analysis)
        assert scenarios is not None
        assert isinstance(scenarios, list)
    
    def test_generate_testbench_code(self, testbench_generator, temp_directory):
        """Test testbench code generation"""
        # Create mock analysis and scenarios
        analysis = {'module_name': 'simple_counter'}
        scenarios = [{'name': 'basic_test', 'description': 'Basic functionality test'}]
        
        testbench_code = testbench_generator.generate_testbench_code(analysis, scenarios)
        assert testbench_code is not None
        assert isinstance(testbench_code, str)
    
    @pytest.mark.asyncio
    @patch('rtl_pilot.agents.testbench_gen.AsyncOpenAI')
    async def test_generate_testbench_success(self, mock_openai, testbench_generator, 
                                           temp_directory, sample_rtl_code):
        """Test successful testbench generation"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
`timescale 1ns / 1ps

module tb_simple_counter;
    reg clk, rst_n, enable;
    wire [7:0] count;
    
    simple_counter dut (
        .clk(clk),
        .rst_n(rst_n),
        .enable(enable),
        .count(count)
    );
    
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        rst_n = 0;
        enable = 0;
        #10 rst_n = 1;
        #10 enable = 1;
        #100 $finish;
    end
endmodule
'''
        
        # Configure mock
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create test file
        rtl_file = temp_directory / "simple_counter.v"
        rtl_file.write_text(sample_rtl_code)
        
        # Generate testbench
        result = await testbench_generator.generate_testbench(
            rtl_files=[str(rtl_file)],
            output_dir=str(temp_directory)
        )
        
        assert result['success'] is True
        assert result['testbench_file'] is not None
        assert Path(result['testbench_file']).exists()
    
    @pytest.mark.asyncio
    async def test_generate_testbench_invalid_rtl(self, testbench_generator, temp_directory):
        """Test testbench generation with invalid RTL"""
        invalid_rtl = "This is not valid Verilog code"
        rtl_file = temp_directory / "invalid.v"
        rtl_file.write_text(invalid_rtl)
        
        result = await testbench_generator.generate_testbench(
            rtl_files=[str(rtl_file)],
            output_dir=str(temp_directory)
        )
        
        assert result['success'] is False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_generate_testbench_missing_file(self, testbench_generator, temp_directory):
        """Test testbench generation with missing file"""
        result = await testbench_generator.generate_testbench(
            rtl_files=["nonexistent.v"],
            output_dir=str(temp_directory)
        )
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_extract_module_info_complex(self, testbench_generator):
        """Test module info extraction for complex module"""
        complex_rtl = '''
module complex_module #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 10
) (
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] data_in,
    input wire [ADDR_WIDTH-1:0] addr,
    input wire write_en,
    input wire read_en,
    output reg [DATA_WIDTH-1:0] data_out,
    output wire valid,
    output wire ready
);

// Internal registers
reg [DATA_WIDTH-1:0] memory [0:(1<<ADDR_WIDTH)-1];
reg valid_reg;

assign valid = valid_reg;
assign ready = !write_en || !read_en;

always @(posedge clk) begin
    if (!rst_n) begin
        valid_reg <= 1'b0;
        data_out <= {DATA_WIDTH{1'b0}};
    end else begin
        if (write_en) begin
            memory[addr] <= data_in;
            valid_reg <= 1'b1;
        end else if (read_en) begin
            data_out <= memory[addr];
            valid_reg <= 1'b1;
        end else begin
            valid_reg <= 1'b0;
        end
    end
end

endmodule
'''
        
        analysis = testbench_generator._parse_rtl(complex_rtl)
        
        assert analysis['module_name'] == 'complex_module'
        assert len(analysis['parameters']) == 2
        assert len(analysis['ports']) == 9
        
        # Check for parameterized ports
        data_in_port = next(p for p in analysis['ports'] if p['name'] == 'data_in')
        assert data_in_port['width'] == 'DATA_WIDTH-1:0'
    
    def test_generate_constraints_and_stimulus(self, testbench_generator, sample_rtl_code):
        """Test constraint and stimulus generation"""
        analysis = testbench_generator._parse_rtl(sample_rtl_code)
        constraints = testbench_generator._generate_constraints(analysis)
        
        assert 'clock_period' in constraints
        assert 'reset_duration' in constraints
        assert constraints['clock_period'] > 0
        assert constraints['reset_duration'] > 0
    
    @pytest.mark.parametrize("module_name,expected_tb_name", [
        ("simple_counter", "tb_simple_counter"),
        ("fifo_module", "tb_fifo_module"),
        ("complex_design", "tb_complex_design")
    ])
    def test_testbench_naming_convention(self, testbench_generator, module_name, expected_tb_name):
        """Test testbench naming convention"""
        tb_name = testbench_generator._generate_testbench_name(module_name)
        assert tb_name == expected_tb_name
    
    def test_coverage_points_generation(self, testbench_generator, sample_rtl_code):
        """Test coverage points generation"""
        analysis = testbench_generator._parse_rtl(sample_rtl_code)
        coverage_points = testbench_generator._identify_coverage_points(analysis)
        
        assert len(coverage_points) > 0
        
        # Should identify state coverage
        state_coverage = [cp for cp in coverage_points if cp['type'] == 'state']
        assert len(state_coverage) > 0
        
        # Should identify transition coverage
        transition_coverage = [cp for cp in coverage_points if cp['type'] == 'transition']
        assert len(transition_coverage) > 0


@pytest.mark.integration
class TestTestbenchGeneratorIntegration:
    """Integration tests for TestbenchGenerator"""
    
    @pytest.fixture
    def real_settings(self):
        """Real settings for integration testing"""
        return RTLPilotSettings()
    
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY'),
        reason="Requires OpenAI API key for integration testing"
    )
    @pytest.mark.asyncio
    async def test_end_to_end_generation(self, real_settings, temp_directory):
        """End-to-end testbench generation with real LLM"""
        generator = RTLTestbenchGenerator(real_settings)
        
        # Simple RTL design
        rtl_code = '''
module test_module (
    input wire clk,
    input wire rst_n,
    input wire [7:0] data_in,
    output reg [7:0] data_out
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        data_out <= 8'h00;
    else
        data_out <= data_in;
end

endmodule
'''
        
        rtl_file = temp_directory / "test_module.v"
        rtl_file.write_text(rtl_code)
        
        result = await generator.generate_testbench(
            rtl_files=[str(rtl_file)],
            output_dir=str(temp_directory),
            top_module="test_module"
        )
        
        assert result['success'] is True
        assert result['testbench_file'] is not None
        
        # Verify generated testbench
        tb_file = Path(result['testbench_file'])
        assert tb_file.exists()
        
        tb_content = tb_file.read_text()
        assert 'module tb_' in tb_content
        assert 'test_module' in tb_content
        assert 'initial begin' in tb_content
