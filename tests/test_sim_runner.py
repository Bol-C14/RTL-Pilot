"""
测试仿真运行器模块
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os

from rtl_pilot.agents.sim_runner import SimulationRunner
from rtl_pilot.config.settings import Settings
from rtl_pilot.utils.vivado_interface import VivadoInterface


@pytest.fixture
def settings():
    """测试设置fixture"""
    return Settings(
        vivado_path="/opt/Xilinx/Vivado/2023.1/bin/vivado",
        workspace_path=tempfile.mkdtemp(),
        cleanup_temp=True
    )


@pytest.fixture
def sim_runner(settings):
    """仿真运行器fixture"""
    return SimulationRunner(settings)


@pytest.fixture
def sample_files():
    """示例文件fixture"""
    return {
        'rtl': ['adder.v', 'utils.v'],
        'testbench': 'adder_tb.sv',
        'constraints': 'timing.xdc'
    }


class TestSimulationRunner:
    """仿真运行器测试类"""
    
    def test_init(self, sim_runner, settings):
        """测试初始化"""
        assert sim_runner.settings == settings
        assert isinstance(sim_runner.vivado, VivadoInterface)
    
    @pytest.mark.asyncio
    async def test_setup_project_success(self, sim_runner, sample_files):
        """测试项目设置成功"""
        with patch.object(sim_runner.vivado, 'run_tcl_commands') as mock_run:
            mock_run.return_value = {'success': True, 'output': 'Project created'}
            
            result = await sim_runner._setup_project(
                'test_project',
                sample_files['rtl'],
                sample_files['testbench']
            )
            
            assert result['success'] is True
            mock_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_project_failure(self, sim_runner, sample_files):
        """测试项目设置失败"""
        with patch.object(sim_runner.vivado, 'run_tcl_commands') as mock_run:
            mock_run.return_value = {'success': False, 'error': 'File not found'}
            
            result = await sim_runner._setup_project(
                'test_project',
                sample_files['rtl'],
                sample_files['testbench']
            )
            
            assert result['success'] is False
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_run_simulation_success(self, sim_runner):
        """测试仿真运行成功"""
        with patch.object(sim_runner.vivado, 'run_tcl_commands') as mock_run:
            mock_run.return_value = {
                'success': True, 
                'output': 'Simulation completed successfully'
            }
            
            result = await sim_runner._run_simulation('test_project')
            
            assert result['success'] is True
            mock_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collect_results(self, sim_runner):
        """测试结果收集"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建模拟结果文件
            log_file = Path(temp_dir) / 'simulation.log'
            waveform_file = Path(temp_dir) / 'dump.vcd'
            coverage_file = Path(temp_dir) / 'coverage.xml'
            
            log_file.write_text('Test passed\nAll assertions succeeded')
            waveform_file.write_text('VCD file content')
            coverage_file.write_text('<coverage>90%</coverage>')
            
            with patch.object(sim_runner, '_get_project_dir', return_value=temp_dir):
                results = await sim_runner._collect_results('test_project')
                
                assert 'log_file' in results
                assert 'waveform_file' in results
                assert 'coverage_file' in results
                assert results['log_content'] == 'Test passed\nAll assertions succeeded'
    
    @pytest.mark.asyncio
    async def test_run_simulation_full_flow(self, sim_runner, sample_files):
        """测试完整仿真流程"""
        with patch.object(sim_runner, '_setup_project') as mock_setup, \
             patch.object(sim_runner, '_run_simulation') as mock_run, \
             patch.object(sim_runner, '_collect_results') as mock_collect:
            
            # 设置mock返回值
            mock_setup.return_value = {'success': True}
            mock_run.return_value = {'success': True}
            mock_collect.return_value = {
                'log_file': 'sim.log',
                'waveform_file': 'dump.vcd',
                'success': True
            }
            
            result = await sim_runner.run_simulation(
                rtl_files=sample_files['rtl'],
                testbench_file=sample_files['testbench'],
                project_name='test_sim'
            )
            
            assert result['success'] is True
            assert 'log_file' in result
            mock_setup.assert_called_once()
            mock_run.assert_called_once()
            mock_collect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_simulation_setup_failure(self, sim_runner, sample_files):
        """测试项目设置失败时的处理"""
        with patch.object(sim_runner, '_setup_project') as mock_setup:
            mock_setup.return_value = {'success': False, 'error': 'Setup failed'}
            
            result = await sim_runner.run_simulation(
                rtl_files=sample_files['rtl'],
                testbench_file=sample_files['testbench']
            )
            
            assert result['success'] is False
            assert 'error' in result
    
    def test_generate_tcl_commands(self, sim_runner):
        """测试TCL命令生成"""
        commands = sim_runner._generate_tcl_commands(
            project_name='test_proj',
            rtl_files=['file1.v', 'file2.v'],
            testbench_file='tb.sv'
        )
        
        assert 'create_project' in commands[0]
        assert 'test_proj' in commands[0]
        assert any('file1.v' in cmd for cmd in commands)
        assert any('file2.v' in cmd for cmd in commands)
        assert any('tb.sv' in cmd for cmd in commands)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, sim_runner):
        """测试超时处理"""
        async def slow_operation():
            await asyncio.sleep(10)  # 模拟慢操作
            return {'success': True}
        
        with patch.object(sim_runner, '_run_simulation', side_effect=slow_operation):
            result = await sim_runner.run_simulation(
                rtl_files=['test.v'],
                testbench_file='test_tb.sv',
                timeout=1  # 1秒超时
            )
            
            assert result['success'] is False
            assert 'timeout' in result.get('error', '').lower()
    
    def test_file_validation(self, sim_runner):
        """测试文件验证"""
        # 测试空文件列表
        with pytest.raises(ValueError, match="RTL files cannot be empty"):
            asyncio.run(sim_runner.run_simulation(
                rtl_files=[],
                testbench_file='test_tb.sv'
            ))
        
        # 测试空测试台文件
        with pytest.raises(ValueError, match="Testbench file cannot be empty"):
            asyncio.run(sim_runner.run_simulation(
                rtl_files=['test.v'],
                testbench_file=''
            ))
    
    @pytest.mark.asyncio
    async def test_cleanup_on_failure(self, sim_runner, sample_files):
        """测试失败时的清理"""
        with patch.object(sim_runner, '_setup_project') as mock_setup, \
             patch.object(sim_runner, '_cleanup_project') as mock_cleanup:
            
            mock_setup.side_effect = Exception("Setup error")
            
            result = await sim_runner.run_simulation(
                rtl_files=sample_files['rtl'],
                testbench_file=sample_files['testbench']
            )
            
            assert result['success'] is False
            mock_cleanup.assert_called_once()


class TestSimulationConfiguration:
    """仿真配置测试类"""
    
    def test_default_configuration(self, sim_runner):
        """测试默认配置"""
        config = sim_runner._get_default_config()
        
        assert 'simulation_time' in config
        assert 'time_unit' in config
        assert config['time_unit'] == 'ns'
    
    def test_custom_configuration(self, sim_runner):
        """测试自定义配置"""
        custom_config = {
            'simulation_time': '1000ns',
            'time_unit': 'ps',
            'enable_coverage': True
        }
        
        config = sim_runner._merge_config(custom_config)
        
        assert config['simulation_time'] == '1000ns'
        assert config['time_unit'] == 'ps'
        assert config['enable_coverage'] is True


if __name__ == '__main__':
    pytest.main([__file__])
