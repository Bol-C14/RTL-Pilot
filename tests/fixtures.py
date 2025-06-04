"""
Test fixtures and utilities for RTL-Pilot tests.

This module provides common test fixtures, mock objects, and helper
functions used across different test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock
import asyncio

from rtl_pilot.config.settings import Settings
from rtl_pilot.config.schema import ProjectConfig


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for test projects."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_rtl_file(temp_project_dir):
    """Create a sample RTL file for testing."""
    rtl_content = """
module simple_adder (
    input wire [7:0] a,
    input wire [7:0] b,
    input wire clk,
    input wire rst_n,
    output reg [8:0] sum
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        sum <= 9'b0;
    else
        sum <= a + b;
end

endmodule
"""
    rtl_file = temp_project_dir / "simple_adder.v"
    rtl_file.write_text(rtl_content)
    return rtl_file


@pytest.fixture
def sample_testbench_file(temp_project_dir):
    """Create a sample testbench file for testing."""
    tb_content = """
module tb_simple_adder;
    reg [7:0] a, b;
    reg clk, rst_n;
    wire [8:0] sum;

    simple_adder dut (
        .a(a),
        .b(b),
        .clk(clk),
        .rst_n(rst_n),
        .sum(sum)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 0;
        a = 0;
        b = 0;
        #20 rst_n = 1;
        #10 a = 8'h0F; b = 8'h10;
        #10 a = 8'hFF; b = 8'h01;
        #10 $finish;
    end

endmodule
"""
    tb_file = temp_project_dir / "tb_simple_adder.v"
    tb_file.write_text(tb_content)
    return tb_file


@pytest.fixture
def test_settings():
    """Create test settings configuration."""
    return Settings(
        vivado_path="/opt/Xilinx/Vivado/2023.2/bin/vivado",
        llm_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-4",
        project_root="/tmp/test_project",
        simulation_timeout=300,
        log_level="DEBUG"
    )


@pytest.fixture
def test_project_config():
    """Create test project configuration."""
    return ProjectConfig(
        name="test_project",
        rtl_sources=["src/adder.v", "src/counter.v"],
        testbench_dir="tb",
        simulation_dir="sim",
        top_module="test_top",
        clock_signal="clk",
        reset_signal="rst_n",
        target_device="xc7a35t-cpg236-1"
    )


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock_client = Mock()
    
    # Mock OpenAI client
    mock_client.chat.completions.create = Mock()
    mock_client.chat.completions.create.return_value = Mock(
        choices=[Mock(
            message=Mock(
                content="Generated testbench code here..."
            )
        )]
    )
    
    return mock_client


@pytest.fixture
def mock_vivado_interface():
    """Mock Vivado interface for testing."""
    mock_vivado = Mock()
    mock_vivado.run_simulation = Mock(return_value={
        "success": True,
        "log_file": "/tmp/simulation.log",
        "waveform_file": "/tmp/dump.vcd",
        "execution_time": 10.5
    })
    mock_vivado.check_installation = Mock(return_value=True)
    mock_vivado.get_version = Mock(return_value="2023.2")
    return mock_vivado


@pytest.fixture
def sample_simulation_log():
    """Sample simulation log content."""
    return """
INFO: [Vivado 2023.2] Starting simulation
INFO: [Vivado] Loading design files
INFO: [Vivado] Compiling testbench
INFO: [Vivado] Running simulation
Time: 0 ns  Iteration: 0  Instance: /tb_simple_adder  Process: @initial_1
Time: 5 ns  Iteration: 1  Instance: /tb_simple_adder  Process: clk
Time: 10 ns  Iteration: 1  Instance: /tb_simple_adder  Process: clk
Time: 20 ns  Iteration: 2  Instance: /tb_simple_adder  Process: @initial_2
Time: 30 ns  Iteration: 3  Instance: /tb_simple_adder  Process: @initial_2
Time: 40 ns  Iteration: 4  Instance: /tb_simple_adder  Process: @initial_2
INFO: [Vivado] Simulation completed successfully
INFO: [Vivado] Coverage: 85.5%
"""


@pytest.fixture
def sample_coverage_report():
    """Sample coverage report data."""
    return {
        "line_coverage": 85.5,
        "branch_coverage": 78.2,
        "toggle_coverage": 92.1,
        "functional_coverage": 67.8,
        "overall_coverage": 80.9,
        "coverage_details": {
            "simple_adder": {
                "line_coverage": 100.0,
                "branch_coverage": 85.0,
                "uncovered_lines": []
            }
        }
    }


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""
    mock_ops = Mock()
    mock_ops.read_file = Mock(return_value="file content")
    mock_ops.write_file = Mock(return_value=True)
    mock_ops.create_directory = Mock(return_value=True)
    mock_ops.file_exists = Mock(return_value=True)
    mock_ops.list_files = Mock(return_value=["file1.v", "file2.v"])
    return mock_ops


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_async_context():
    """Mock async context manager."""
    class MockAsyncContext:
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        async def run_command(self, command):
            return {
                "returncode": 0,
                "stdout": "Command executed successfully",
                "stderr": ""
            }
    
    return MockAsyncContext()


class MockSettings:
    """Mock settings class for testing."""
    def __init__(self, **kwargs):
        self.vivado_path = kwargs.get("vivado_path", "/opt/Xilinx/Vivado/2023.2/bin/vivado")
        self.llm_provider = kwargs.get("llm_provider", "openai")
        self.openai_api_key = kwargs.get("openai_api_key", "test-key")
        self.openai_model = kwargs.get("openai_model", "gpt-4")
        self.project_root = kwargs.get("project_root", "/tmp/test")
        self.simulation_timeout = kwargs.get("simulation_timeout", 300)
        self.log_level = kwargs.get("log_level", "INFO")
        self.max_retries = kwargs.get("max_retries", 3)
        self.enable_coverage = kwargs.get("enable_coverage", True)
        self.parallel_jobs = kwargs.get("parallel_jobs", 4)


class MockProjectConfig:
    """Mock project configuration for testing."""
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "test_project")
        self.rtl_sources = kwargs.get("rtl_sources", ["test.v"])
        self.testbench_dir = kwargs.get("testbench_dir", "tb")
        self.simulation_dir = kwargs.get("simulation_dir", "sim")
        self.top_module = kwargs.get("top_module", "test_top")
        self.clock_signal = kwargs.get("clock_signal", "clk")
        self.reset_signal = kwargs.get("reset_signal", "rst_n")
        self.target_device = kwargs.get("target_device", "xc7a35t-cpg236-1")


# Helper functions for test data
def create_test_rtl_files(base_dir: Path) -> Dict[str, Path]:
    """Create a set of test RTL files."""
    files = {}
    
    # Simple adder
    adder_content = """
module adder #(parameter WIDTH = 8) (
    input [WIDTH-1:0] a, b,
    output [WIDTH:0] sum
);
    assign sum = a + b;
endmodule
"""
    files["adder"] = base_dir / "adder.v"
    files["adder"].write_text(adder_content)
    
    # Counter
    counter_content = """
module counter #(parameter WIDTH = 8) (
    input clk, rst_n, enable,
    output reg [WIDTH-1:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= 0;
        else if (enable)
            count <= count + 1;
    end
endmodule
"""
    files["counter"] = base_dir / "counter.v"
    files["counter"].write_text(counter_content)
    
    return files


def create_test_project_structure(base_dir: Path) -> Dict[str, Path]:
    """Create a complete test project structure."""
    structure = {}
    
    # Create directories
    dirs = ["src", "tb", "sim", "scripts", "docs"]
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        structure[dir_name] = dir_path
    
    # Create RTL files
    rtl_files = create_test_rtl_files(structure["src"])
    structure.update(rtl_files)
    
    # Create project file
    project_content = """
{
    "name": "test_project",
    "rtl_sources": ["src/adder.v", "src/counter.v"],
    "testbench_dir": "tb",
    "simulation_dir": "sim",
    "top_module": "test_top",
    "clock_signal": "clk",
    "reset_signal": "rst_n",
    "target_device": "xc7a35t-cpg236-1"
}
"""
    project_file = base_dir / "project.json"
    project_file.write_text(project_content)
    structure["project"] = project_file
    
    return structure


# Assertion helpers
def assert_file_exists(file_path: Path, message: str = ""):
    """Assert that a file exists."""
    assert file_path.exists(), f"File {file_path} does not exist. {message}"


def assert_file_contains(file_path: Path, content: str, message: str = ""):
    """Assert that a file contains specific content."""
    assert_file_exists(file_path)
    file_content = file_path.read_text()
    assert content in file_content, f"File {file_path} does not contain '{content}'. {message}"


def assert_valid_verilog_syntax(content: str):
    """Basic check for valid Verilog syntax."""
    # Basic checks for Verilog syntax
    assert "module" in content, "Content should contain module declaration"
    assert "endmodule" in content, "Content should contain endmodule"
    
    # Check balanced parentheses
    open_parens = content.count("(")
    close_parens = content.count(")")
    assert open_parens == close_parens, "Unbalanced parentheses in Verilog code"


def assert_simulation_success(result: Dict[str, Any]):
    """Assert that simulation completed successfully."""
    assert result.get("success", False), f"Simulation failed: {result.get('error', 'Unknown error')}"
    assert "log_file" in result, "Simulation result should include log file path"


# Performance testing helpers
class PerformanceTimer:
    """Simple performance timer for testing."""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.time()
    
    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
