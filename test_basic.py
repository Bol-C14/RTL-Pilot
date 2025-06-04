# 简单的配置测试
import pytest
from rtl_pilot.config.schema import ConfigSchema, VerificationGoals

def test_basic_import():
    """测试基本导入是否工作"""
    config = ConfigSchema()
    assert config is not None

def test_verification_goals():
    """测试验证目标配置"""
    goals = VerificationGoals(coverage_target=95.0)
    assert goals.coverage_target == 95.0

if __name__ == "__main__":
    test_basic_import()
    test_verification_goals()
    print("基本测试通过!")
