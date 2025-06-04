"""
测试配置模块
"""
import pytest
import tempfile
import os
import yaml
import json
from pathlib import Path
from pydantic import ValidationError

from rtl_pilot.config.settings import Settings
from rtl_pilot.config.schema import (
    ConfigSchema,
    VerificationGoals, 
    VerificationScenario,
    RTLDesignInfo
)


@pytest.fixture
def temp_config_dir():
    """临时配置目录fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_yaml_config():
    """示例YAML配置内容"""
    return """
llm:
  provider: openai
  model: gpt-4
  temperature: 0.1
  max_tokens: 2048
  api_key: test_key

tools:
  vivado_path: /opt/Xilinx/Vivado/2023.1/bin/vivado
  timeout: 300

workspace:
  base_path: ./workspace
  temp_dir: /tmp/rtl_pilot
  cleanup_temp: true

verification:
  default_timeout: 600
  max_iterations: 5
  coverage_threshold: 80.0
"""


@pytest.fixture
def sample_json_config():
    """示例JSON配置内容"""
    return {
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.2
        },
        "tools": {
            "vivado_path": "/usr/local/vivado/bin/vivado"
        },
        "workspace": {
            "base_path": "./test_workspace"
        }
    }


class TestSettings:
    """设置类测试"""
    
    def test_default_settings(self):
        """测试默认设置"""
        settings = Settings()
        
        assert settings.llm_provider == "openai"
        assert settings.llm_model == "gpt-4"
        assert settings.llm_temperature == 0.1
        assert settings.workspace_path is not None
        assert settings.cleanup_temp is True
    
    def test_settings_from_env(self):
        """测试从环境变量加载设置"""
        os.environ['OPENAI_API_KEY'] = 'test_env_key'
        os.environ['VIVADO_PATH'] = '/custom/vivado/path'
        os.environ['RTL_PILOT_WORKSPACE'] = '/custom/workspace'
        
        try:
            settings = Settings()
            assert settings.openai_api_key == 'test_env_key'
            assert settings.vivado_path == '/custom/vivado/path'
            assert settings.workspace_path == '/custom/workspace'
        finally:
            # 清理环境变量
            del os.environ['OPENAI_API_KEY']
            del os.environ['VIVADO_PATH']
            del os.environ['RTL_PILOT_WORKSPACE']
    
    def test_load_from_yaml(self, temp_config_dir, sample_yaml_config):
        """测试从YAML文件加载配置"""
        config_file = Path(temp_config_dir) / "config.yaml"
        config_file.write_text(sample_yaml_config)
        
        settings = Settings.load_from_file(str(config_file))
        
        assert settings.llm_provider == "openai"
        assert settings.llm_model == "gpt-4"
        assert settings.vivado_path == "/opt/Xilinx/Vivado/2023.1/bin/vivado"
        assert settings.cleanup_temp is True
    
    def test_load_from_json(self, temp_config_dir, sample_json_config):
        """测试从JSON文件加载配置"""
        config_file = Path(temp_config_dir) / "config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_json_config, f)
        
        settings = Settings.load_from_file(str(config_file))
        
        assert settings.llm_provider == "openai"
        assert settings.llm_model == "gpt-3.5-turbo"
        assert settings.vivado_path == "/usr/local/vivado/bin/vivado"
    
    def test_save_to_file(self, temp_config_dir):
        """测试保存配置到文件"""
        settings = Settings(
            llm_model="gpt-4",
            vivado_path="/custom/path",
            workspace_path="/custom/workspace"
        )
        
        config_file = Path(temp_config_dir) / "saved_config.yaml"
        settings.save_to_file(str(config_file))
        
        assert config_file.exists()
        
        # 验证保存的内容
        loaded_settings = Settings.load_from_file(str(config_file))
        assert loaded_settings.llm_model == "gpt-4"
        assert loaded_settings.vivado_path == "/custom/path"
    
    def test_invalid_file_format(self, temp_config_dir):
        """测试无效文件格式"""
        config_file = Path(temp_config_dir) / "invalid.txt"
        config_file.write_text("not yaml or json")
        
        with pytest.raises(ValueError):
            Settings.load_from_file(str(config_file))
    
    def test_file_not_found(self):
        """测试文件不存在"""
        with pytest.raises(FileNotFoundError):
            Settings.load_from_file("nonexistent_config.yaml")
    
    def test_validation(self):
        """测试配置验证"""
        # 有效配置
        settings = Settings(
            llm_temperature=0.5,
            verification_timeout=600
        )
        assert settings.llm_temperature == 0.5
        
        # 无效温度值
        with pytest.raises(ValidationError):
            Settings(llm_temperature=2.0)  # 超出范围
        
        # 无效超时值
        with pytest.raises(ValidationError):
            Settings(verification_timeout=-1)  # 负值
    
    def test_merge_configs(self):
        """测试配置合并"""
        base_settings = Settings(llm_model="gpt-3.5-turbo")
        override_config = {"llm_model": "gpt-4", "llm_temperature": 0.2}
        
        merged = base_settings.merge(override_config)
        
        assert merged.llm_model == "gpt-4"
        assert merged.llm_temperature == 0.2
    
    def test_to_dict(self):
        """测试转换为字典"""
        settings = Settings(
            llm_model="gpt-4",
            workspace_path="/test"
        )
        
        config_dict = settings.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['llm_model'] == "gpt-4"
        assert config_dict['workspace_path'] == "/test"


class TestProjectConfig:
    """项目配置测试"""
    
    def test_valid_project_config(self):
        """测试有效的项目配置"""
        config = ProjectConfig(
            project_name="test_project",
            rtl_files=["adder.v", "utils.v"],
            top_module="adder",
            verification_goals=VerificationGoals(
                line_coverage_target=90,
                branch_coverage_target=85
            )
        )
        
        assert config.project_name == "test_project"
        assert len(config.rtl_files) == 2
        assert config.verification_goals.line_coverage_target == 90
    
    def test_project_config_validation(self):
        """测试项目配置验证"""
        # 空项目名
        with pytest.raises(ValidationError):
            ProjectConfig(
                project_name="",
                rtl_files=["test.v"],
                top_module="test"
            )
        
        # 空RTL文件列表
        with pytest.raises(ValidationError):
            ProjectConfig(
                project_name="test",
                rtl_files=[],
                top_module="test"
            )
    
    def test_project_config_from_file(self, temp_config_dir):
        """测试从文件加载项目配置"""
        config_data = {
            "project_name": "fifo_test",
            "rtl_files": ["fifo.sv", "fifo_tb.sv"],
            "top_module": "fifo",
            "verification_goals": {
                "line_coverage_target": 95,
                "branch_coverage_target": 90,
                "functional_tests": ["empty", "full", "overflow"]
            },
            "test_phases": [
                {
                    "name": "basic",
                    "tests": ["read_write", "empty_full"],
                    "coverage_target": 70
                }
            ]
        }
        
        config_file = Path(temp_config_dir) / "project.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = ProjectConfig.load_from_file(str(config_file))
        
        assert config.project_name == "fifo_test"
        assert config.verification_goals.line_coverage_target == 95
        assert len(config.test_phases) == 1


class TestVerificationGoals:
    """验证目标测试"""
    
    def test_coverage_targets_validation(self):
        """测试覆盖率目标验证"""
        # 有效值
        targets = CoverageTargets(
            line_coverage=85.0,
            branch_coverage=80.0,
            toggle_coverage=75.0
        )
        assert targets.line_coverage == 85.0
        
        # 无效值 - 超出范围
        with pytest.raises(ValidationError):
            CoverageTargets(line_coverage=150.0)
        
        # 无效值 - 负数
        with pytest.raises(ValidationError):
            CoverageTargets(branch_coverage=-10.0)
    
    def test_verification_goals_defaults(self):
        """测试验证目标默认值"""
        goals = VerificationGoals()
        
        assert goals.line_coverage_target >= 0
        assert goals.branch_coverage_target >= 0
        assert isinstance(goals.functional_tests, list)
    
    def test_custom_verification_goals(self):
        """测试自定义验证目标"""
        goals = VerificationGoals(
            line_coverage_target=95,
            branch_coverage_target=90,
            functional_tests=["test1", "test2", "test3"],
            performance_requirements={"max_delay": "10ns"}
        )
        
        assert goals.line_coverage_target == 95
        assert len(goals.functional_tests) == 3
        assert goals.performance_requirements["max_delay"] == "10ns"


class TestTestPhase:
    """测试阶段配置测试"""
    
    def test_test_phase_creation(self):
        """测试测试阶段创建"""
        phase = TestPhase(
            name="basic_tests",
            tests=["test1", "test2"],
            coverage_target=75,
            timeout=300,
            parallel=True
        )
        
        assert phase.name == "basic_tests"
        assert len(phase.tests) == 2
        assert phase.coverage_target == 75
        assert phase.parallel is True
    
    def test_test_phase_validation(self):
        """测试测试阶段验证"""
        # 空名称
        with pytest.raises(ValidationError):
            TestPhase(name="", tests=["test1"])
        
        # 空测试列表
        with pytest.raises(ValidationError):
            TestPhase(name="test", tests=[])
        
        # 无效覆盖率目标
        with pytest.raises(ValidationError):
            TestPhase(name="test", tests=["test1"], coverage_target=150)


class TestConfigurationHelpers:
    """配置辅助函数测试"""
    
    def test_detect_config_format(self):
        """测试配置格式检测"""
        from rtl_pilot.config.settings import _detect_config_format
        
        assert _detect_config_format("config.yaml") == "yaml"
        assert _detect_config_format("config.yml") == "yaml"
        assert _detect_config_format("config.json") == "json"
        
        with pytest.raises(ValueError):
            _detect_config_format("config.txt")
    
    def test_validate_file_paths(self):
        """测试文件路径验证"""
        from rtl_pilot.config.schema import validate_file_paths
        
        # 创建临时文件进行测试
        with tempfile.NamedTemporaryFile(suffix='.v', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # 有效路径
            assert validate_file_paths([temp_path]) is True
            
            # 无效路径
            assert validate_file_paths(["nonexistent.v"]) is False
        finally:
            os.unlink(temp_path)
    
    def test_environment_variable_substitution(self):
        """测试环境变量替换"""
        os.environ['TEST_VAR'] = 'test_value'
        
        try:
            from rtl_pilot.config.settings import substitute_env_vars
            
            config = {
                'path': '${TEST_VAR}/subdir',
                'name': 'project_${TEST_VAR}'
            }
            
            result = substitute_env_vars(config)
            
            assert result['path'] == 'test_value/subdir'
            assert result['name'] == 'project_test_value'
        finally:
            del os.environ['TEST_VAR']


if __name__ == '__main__':
    pytest.main([__file__])
