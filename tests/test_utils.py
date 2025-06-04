"""
测试工具接口模块和测试工具函数
"""
import pytest
import tempfile
import subprocess
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from typing import Dict, Any, List

from rtl_pilot.utils.vivado_interface import VivadoInterface
from rtl_pilot.utils.file_ops import FileManager, TempFileManager
from rtl_pilot.config.settings import Settings


@pytest.fixture(scope="session")
def temp_workspace():
    """Session-scoped temporary workspace for tests"""
    temp_dir = tempfile.mkdtemp(prefix="rtl_pilot_test_")
    workspace_path = Path(temp_dir)
    
    yield workspace_path
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_project_dir(temp_workspace):
    """Temporary project directory for individual tests"""
    project_dir = temp_workspace / "test_project"
    project_dir.mkdir(exist_ok=True)
    return project_dir


@pytest.fixture
def sample_rtl_files(temp_project_dir):
    """Sample RTL files for testing"""
    rtl_dir = temp_project_dir / "src"
    rtl_dir.mkdir(exist_ok=True)
    
    # Simple counter module
    counter_rtl = '''
module counter #(
    parameter WIDTH = 8
) (
    input wire clk,
    input wire rst_n,
    input wire enable,
    output reg [WIDTH-1:0] count
);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        count <= 0;
    else if (enable)
        count <= count + 1;
end

endmodule
'''
    
    # Simple adder module
    adder_rtl = '''
module adder (
    input [3:0] a,
    input [3:0] b,
    input cin,
    output [3:0] sum,
    output cout
);

assign {cout, sum} = a + b + cin;

endmodule
'''
    
    counter_file = rtl_dir / "counter.v"
    adder_file = rtl_dir / "adder.v"
    
    counter_file.write_text(counter_rtl)
    adder_file.write_text(adder_rtl)
    
    return {
        'counter': str(counter_file),
        'adder': str(adder_file),
        'all_files': [str(counter_file), str(adder_file)]
    }


@pytest.fixture
def sample_testbench(temp_project_dir):
    """Sample testbench for testing"""
    tb_dir = temp_project_dir / "tb"
    tb_dir.mkdir(exist_ok=True)
    
    testbench_sv = '''
module counter_tb;

parameter WIDTH = 8;
parameter CLK_PERIOD = 10;

logic clk;
logic rst_n;
logic enable;
logic [WIDTH-1:0] count;

// DUT instantiation
counter #(.WIDTH(WIDTH)) dut (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .count(count)
);

// Clock generation
always #(CLK_PERIOD/2) clk = ~clk;

initial begin
    $dumpfile("counter_tb.vcd");
    $dumpvars(0, counter_tb);
    
    clk = 0;
    rst_n = 0;
    enable = 0;
    
    #100 rst_n = 1;
    #50 enable = 1;
    
    repeat(20) @(posedge clk);
    
    $display("Test completed");
    $finish;
end

endmodule
'''
    
    tb_file = tb_dir / "counter_tb.sv"
    tb_file.write_text(testbench_sv)
    
    return str(tb_file)


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing"""
    client = Mock()
    
    # Default response for chat completion
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = '''
{
    "testbench": "module test_tb(); endmodule",
    "explanation": "Generated test module"
}
'''
    
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def vivado_interface():
    """Vivado接口fixture"""
    return VivadoInterface(vivado_path="/opt/Xilinx/Vivado/2023.1/bin/vivado")


@pytest.fixture
def file_manager(temp_workspace):
    """文件管理器fixture"""
    return FileManager(base_path=temp_workspace)


# Test Utility Functions
def create_test_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a test configuration dictionary"""
    base_config = {
        'project_name': 'test_project',
        'rtl_files': ['test.v'],
        'top_module': 'test',
        'verification_goals': {
            'line_coverage_target': 80,
            'branch_coverage_target': 75
        }
    }
    
    if overrides:
        base_config.update(overrides)
    
    return base_config


def assert_file_exists(file_path: str, message: str = ""):
    """Assert that a file exists"""
    path = Path(file_path)
    assert path.exists(), f"File does not exist: {file_path}. {message}"
    assert path.is_file(), f"Path is not a file: {file_path}. {message}"


def assert_directory_exists(dir_path: str, message: str = ""):
    """Assert that a directory exists"""
    path = Path(dir_path)
    assert path.exists(), f"Directory does not exist: {dir_path}. {message}"
    assert path.is_dir(), f"Path is not a directory: {dir_path}. {message}"


class TestVivadoInterface:
    """Vivado接口测试"""
    
    def test_init(self):
        """测试初始化"""
        vivado = VivadoInterface("/custom/vivado/path")
        assert vivado.vivado_path == "/custom/vivado/path"
        assert vivado.default_timeout == 300
    
    def test_init_with_custom_timeout(self):
        """测试自定义超时初始化"""
        vivado = VivadoInterface("/vivado", timeout=600)
        assert vivado.default_timeout == 600
    
    @patch('subprocess.run')
    def test_run_tcl_command_success(self, mock_run, vivado_interface):
        """测试成功运行TCL命令"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Command executed successfully",
            stderr=""
        )
        
        result = vivado_interface.run_tcl_command("create_project test")
        
        assert result['success'] is True
        assert result['output'] == "Command executed successfully"
        assert 'error' not in result
    
    @patch('subprocess.run')
    def test_run_tcl_command_failure(self, mock_run, vivado_interface):
        """测试TCL命令执行失败"""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Error: Invalid command"
        )
        
        result = vivado_interface.run_tcl_command("invalid_command")
        
        assert result['success'] is False
        assert result['error'] == "Error: Invalid command"
    
    @patch('subprocess.run')
    def test_run_tcl_command_timeout(self, mock_run, vivado_interface):
        """测试TCL命令超时"""
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd=['vivado'], timeout=10
        )
        
        result = vivado_interface.run_tcl_command("long_running_command", timeout=10)
        
        assert result['success'] is False
        assert 'timeout' in result['error'].lower()
    
    @patch('subprocess.run')
    def test_run_tcl_commands_batch(self, mock_run, vivado_interface):
        """测试批量运行TCL命令"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="All commands executed",
            stderr=""
        )
        
        commands = [
            "create_project test",
            "add_files file1.v",
            "run_synthesis"
        ]
        
        result = vivado_interface.run_tcl_commands(commands)
        
        assert result['success'] is True
        mock_run.assert_called_once()
    
    def test_create_tcl_script(self, vivado_interface, temp_workspace):
        """测试创建TCL脚本"""
        commands = [
            "create_project test_project",
            "add_files {file1.v file2.v}",
            "run_synthesis"
        ]
        
        script_path = vivado_interface.create_tcl_script(commands, temp_workspace)
        
        assert Path(script_path).exists()
        assert script_path.endswith('.tcl')
        
        # 验证脚本内容
        with open(script_path, 'r') as f:
            content = f.read()
            for cmd in commands:
                assert cmd in content


class TestFileManager:
    """文件管理器测试"""
    
    def test_init(self, file_manager, temp_workspace):
        """测试初始化"""
        assert file_manager.base_path == temp_workspace
        assert Path(file_manager.base_path).exists()
    
    def test_create_directory(self, file_manager):
        """测试创建目录"""
        dir_path = file_manager.create_directory("test_dir")
        
        assert Path(dir_path).exists()
        assert Path(dir_path).is_dir()
    
    def test_create_nested_directory(self, file_manager):
        """测试创建嵌套目录"""
        dir_path = file_manager.create_directory("parent/child/grandchild")
        
        assert Path(dir_path).exists()
        assert Path(dir_path).is_dir()
    
    def test_copy_file(self, file_manager, temp_workspace):
        """测试复制文件"""
        # 创建源文件
        source_file = Path(temp_workspace) / "source.txt"
        source_file.write_text("Test content")
        
        # 复制文件
        dest_path = file_manager.copy_file(str(source_file), "dest.txt")
        
        assert Path(dest_path).exists()
        assert Path(dest_path).read_text() == "Test content"
    
    def test_create_file(self, file_manager):
        """测试创建文件"""
        content = "Hello, World!"
        file_path = file_manager.create_file("hello.txt", content)
        
        assert Path(file_path).exists()
        assert Path(file_path).read_text() == content
    
    def test_read_file(self, file_manager):
        """测试读取文件"""
        content = "Read test content"
        file_path = file_manager.create_file("read_test.txt", content)
        
        read_content = file_manager.read_file(file_path)
        assert read_content == content
    
    def test_list_files(self, file_manager):
        """测试列出文件"""
        # 创建一些文件
        for ext in ['v', 'sv', 'txt']:
            file_manager.create_file(f"test.{ext}", "content")
        
        # 列出所有文件
        all_files = file_manager.list_files()
        assert len(all_files) >= 3
        
        # 按扩展名过滤
        v_files = file_manager.list_files(pattern="*.v")
        assert len(v_files) == 1
        assert v_files[0].endswith('.v')


class TestTempFileManager:
    """临时文件管理器测试"""
    
    def test_init(self):
        """测试初始化"""
        temp_manager = TempFileManager()
        assert temp_manager.temp_dir is not None
        assert Path(temp_manager.temp_dir).exists()
    
    def test_create_temp_file(self):
        """测试创建临时文件"""
        with TempFileManager() as temp_manager:
            file_path = temp_manager.create_temp_file("test.txt", "content")
            
            assert Path(file_path).exists()
            assert Path(file_path).read_text() == "content"
    
    def test_create_temp_directory(self):
        """测试创建临时目录"""
        with TempFileManager() as temp_manager:
            dir_path = temp_manager.create_temp_directory("test_dir")
            
            assert Path(dir_path).exists()
            assert Path(dir_path).is_dir()
    
    def test_cleanup_on_exit(self):
        """测试退出时清理"""
        temp_paths = []
        
        with TempFileManager() as temp_manager:
            file_path = temp_manager.create_temp_file("cleanup_test.txt", "content")
            dir_path = temp_manager.create_temp_directory("cleanup_dir")
            temp_paths.extend([file_path, dir_path])
        
        # 验证清理完成
        for path in temp_paths:
            assert not Path(path).exists()


class TestFileOperations:
    """文件操作工具函数测试"""
    
    def test_ensure_directory_exists(self, temp_workspace):
        """测试确保目录存在"""
        from rtl_pilot.utils.file_ops import ensure_directory_exists
        
        test_dir = Path(temp_workspace) / "new_directory"
        ensure_directory_exists(str(test_dir))
        
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_safe_remove(self, temp_workspace):
        """测试安全删除"""
        from rtl_pilot.utils.file_ops import safe_remove
        
        # 创建测试文件
        test_file = Path(temp_workspace) / "remove_test.txt"
        test_file.write_text("test")
        
        # 安全删除
        success = safe_remove(str(test_file))
        assert success is True
        assert not test_file.exists()
        
        # 删除不存在的文件
        success = safe_remove(str(test_file))
        assert success is True  # 不应该抛出异常
    
    def test_get_file_extension(self):
        """测试获取文件扩展名"""
        from rtl_pilot.utils.file_ops import get_file_extension
        
        assert get_file_extension("test.v") == ".v"
        assert get_file_extension("design.sv") == ".sv"
        assert get_file_extension("no_extension") == ""
        assert get_file_extension("multiple.dots.txt") == ".txt"
    
    def test_is_rtl_file(self):
        """测试RTL文件识别"""
        from rtl_pilot.utils.file_ops import is_rtl_file
        
        assert is_rtl_file("design.v") is True
        assert is_rtl_file("testbench.sv") is True
        assert is_rtl_file("package.vhd") is True
        assert is_rtl_file("readme.txt") is False
        assert is_rtl_file("script.tcl") is False


if __name__ == '__main__':
    pytest.main([__file__])
