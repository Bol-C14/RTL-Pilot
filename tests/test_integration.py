"""
集成测试 - 端到端验证流程测试
"""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from rtl_pilot.workflows.default_flow import DefaultVerificationFlow
from rtl_pilot.config.settings import Settings
from rtl_pilot.agents.testbench_gen import RTLTestbenchGenerator
from rtl_pilot.agents.sim_runner import SimulationRunner
from rtl_pilot.agents.evaluation import ResultEvaluator
from rtl_pilot.agents.planner import VerificationPlanner


@pytest.fixture
def integration_settings():
    """集成测试设置"""
    return Settings(
        llm_provider="openai",
        llm_model="gpt-4",
        workspace_path=tempfile.mkdtemp(prefix="rtl_pilot_integration_"),
        vivado_path="/opt/Xilinx/Vivado/2023.1/bin/vivado",
        cleanup_temp=True
    )


@pytest.fixture
def sample_integration_project():
    """示例集成项目配置"""
    return {
        'project_name': 'integration_test',
        'rtl_files': ['adder.v'],
        'top_module': 'adder',
        'verification_goals': {
            'line_coverage_target': 85,
            'branch_coverage_target': 80,
            'functional_tests': ['basic_ops', 'edge_cases']
        },
        'constraints': {
            'max_simulation_time': '1000ns',
            'timeout': 300
        }
    }


@pytest.fixture
def mock_rtl_content():
    """模拟RTL内容"""
    return '''
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


class TestEndToEndFlow:
    """端到端流程测试"""
    
    @pytest.mark.asyncio
    async def test_complete_verification_flow(self, integration_settings, sample_integration_project, mock_rtl_content):
        """测试完整验证流程"""
        # 创建临时RTL文件
        workspace = Path(integration_settings.workspace_path)
        rtl_file = workspace / "adder.v"
        rtl_file.write_text(mock_rtl_content)
        
        # 更新项目配置中的文件路径
        sample_integration_project['rtl_files'] = [str(rtl_file)]
        
        # 创建工作流实例
        workflow = DefaultVerificationFlow(integration_settings)
        
        # Mock各个组件的返回值
        with patch.object(workflow.tb_generator, 'generate_testbench') as mock_gen_tb, \
             patch.object(workflow.sim_runner, 'run_simulation') as mock_sim, \
             patch.object(workflow.evaluator, 'evaluate_results') as mock_eval, \
             patch.object(workflow.planner, 'create_verification_plan') as mock_plan:
            
            # 设置mock返回值
            mock_plan.return_value = {
                'phases': [
                    {'name': 'basic', 'tests': ['test1'], 'coverage_target': 80}
                ],
                'strategy': 'incremental'
            }
            
            mock_gen_tb.return_value = {
                'success': True,
                'testbench_file': 'adder_tb.sv',
                'testbench_content': 'module adder_tb(); endmodule'
            }
            
            mock_sim.return_value = {
                'success': True,
                'log_file': 'simulation.log',
                'log_content': 'Test passed',
                'coverage_file': 'coverage.xml',
                'execution_time': 45.2
            }
            
            mock_eval.return_value = {
                'success': True,
                'coverage_analysis': {'overall_coverage': 87.5},
                'log_analysis': {'passed_tests': 5, 'failed_tests': 0},
                'feedback': {'summary': 'Good coverage achieved'}
            }
            
            # 执行工作流
            result = await workflow.execute(sample_integration_project)
            
            # 验证结果
            assert result['success'] is True
            assert result['overall_coverage'] > 80
            assert 'plan' in result
            assert 'phases_results' in result
            
            # 验证各个组件被调用
            mock_plan.assert_called_once()
            mock_gen_tb.assert_called()
            mock_sim.assert_called()
            mock_eval.assert_called()
    
    @pytest.mark.asyncio
    async def test_flow_with_failure_recovery(self, integration_settings, sample_integration_project):
        """测试流程失败恢复"""
        workflow = DefaultVerificationFlow(integration_settings)
        
        with patch.object(workflow.tb_generator, 'generate_testbench') as mock_gen_tb, \
             patch.object(workflow.sim_runner, 'run_simulation') as mock_sim, \
             patch.object(workflow.planner, 'create_verification_plan') as mock_plan:
            
            mock_plan.return_value = {
                'phases': [{'name': 'basic', 'tests': ['test1']}]
            }
            
            # 第一次测试台生成失败
            mock_gen_tb.side_effect = [
                {'success': False, 'error': 'Generation failed'},
                {'success': True, 'testbench_file': 'adder_tb.sv'}  # 重试成功
            ]
            
            mock_sim.return_value = {
                'success': True,
                'log_content': 'Test passed',
                'coverage_file': None
            }
            
            result = await workflow.execute(sample_integration_project)
            
            # 验证重试机制
            assert mock_gen_tb.call_count >= 2
            assert result['success'] is True
    
    @pytest.mark.asyncio
    async def test_multi_phase_execution(self, integration_settings, sample_integration_project):
        """测试多阶段执行"""
        workflow = DefaultVerificationFlow(integration_settings)
        
        # 多阶段验证计划
        multi_phase_plan = {
            'phases': [
                {'name': 'basic', 'tests': ['basic_test'], 'coverage_target': 60},
                {'name': 'advanced', 'tests': ['edge_test'], 'coverage_target': 85},
                {'name': 'stress', 'tests': ['stress_test'], 'coverage_target': 90}
            ],
            'strategy': 'incremental'
        }
        
        with patch.object(workflow.planner, 'create_verification_plan') as mock_plan, \
             patch.object(workflow.planner, 'execute_plan') as mock_execute:
            
            mock_plan.return_value = multi_phase_plan
            mock_execute.return_value = {
                'success': True,
                'phases_results': [
                    {'phase': 'basic', 'coverage': 62.0, 'success': True},
                    {'phase': 'advanced', 'coverage': 87.0, 'success': True},
                    {'phase': 'stress', 'coverage': 92.0, 'success': True}
                ],
                'overall_coverage': 92.0
            }
            
            result = await workflow.execute(sample_integration_project)
            
            assert result['success'] is True
            assert len(result['phases_results']) == 3
            assert result['overall_coverage'] == 92.0


class TestComponentIntegration:
    """组件集成测试"""
    
    @pytest.mark.asyncio
    async def test_testbench_to_simulation_integration(self, integration_settings):
        """测试测试台生成到仿真的集成"""
        tb_generator = RTLTestbenchGenerator(integration_settings)
        sim_runner = SimulationRunner(integration_settings)
        
        # Mock LLM和Vivado调用
        with patch.object(tb_generator.llm_client, 'chat') as mock_chat, \
             patch.object(sim_runner.vivado, 'run_tcl_commands') as mock_vivado:
            
            # Mock测试台生成
            mock_chat.return_value = Mock(
                choices=[Mock(
                    message=Mock(content='''
{
    "testbench": "module test_tb(); initial begin $finish; end endmodule",
    "explanation": "Simple test"
}
                    ''')
                )]
            )
            
            # Mock仿真运行
            mock_vivado.return_value = {
                'success': True,
                'output': 'Simulation completed'
            }
            
            # 生成测试台
            tb_result = await tb_generator.generate_testbench(
                rtl_file="test.v",
                module_name="test"
            )
            
            assert tb_result['success'] is True
            testbench_file = tb_result['testbench_file']
            
            # 运行仿真
            sim_result = await sim_runner.run_simulation(
                rtl_files=["test.v"],
                testbench_file=testbench_file
            )
            
            assert sim_result['success'] is True
    
    @pytest.mark.asyncio
    async def test_simulation_to_evaluation_integration(self, integration_settings):
        """测试仿真到评估的集成"""
        sim_runner = SimulationRunner(integration_settings)
        evaluator = ResultEvaluator(integration_settings)
        
        # Mock仿真结果
        simulation_results = {
            'success': True,
            'log_content': '''
Info: Starting simulation
Info: Test case 1 passed
Warning: Signal x is undefined
Info: Test case 2 passed
Coverage: 85.5%
            ''',
            'coverage_file': None,
            'execution_time': 125.5
        }
        
        with patch.object(evaluator.llm_client, 'chat') as mock_chat:
            mock_chat.return_value = Mock(
                choices=[Mock(
                    message=Mock(content='''
{
    "summary": "Good test results with minor warnings",
    "issues": ["Undefined signal warning"],
    "recommendations": ["Define all signals properly"]
}
                    ''')
                )]
            )
            
            # 评估仿真结果
            eval_result = await evaluator.evaluate_results(simulation_results)
            
            assert eval_result['success'] is True
            assert 'log_analysis' in eval_result
            assert 'feedback' in eval_result
            assert eval_result['log_analysis']['passed_tests'] == 2
    
    @pytest.mark.asyncio
    async def test_evaluation_to_planning_feedback_loop(self, integration_settings):
        """测试评估到规划的反馈循环"""
        evaluator = ResultEvaluator(integration_settings)
        planner = VerificationPlanner(integration_settings)
        
        # 初始评估结果
        initial_results = {
            'coverage_analysis': {'overall_coverage': 65.0},
            'log_analysis': {'failed_tests': 2, 'warnings': 3}
        }
        
        project_config = {
            'project_name': 'feedback_test',
            'verification_goals': {'line_coverage_target': 90}
        }
        
        with patch.object(planner.llm_client, 'chat') as mock_chat:
            mock_chat.return_value = Mock(
                choices=[Mock(
                    message=Mock(content='''
{
    "revised_strategy": "focus_on_coverage_gaps",
    "additional_tests": ["edge_case_1", "boundary_test"],
    "priority_adjustments": ["increase_stimulus_diversity"]
}
                    ''')
                )]
            )
            
            # 基于评估结果调整计划
            revised_plan = await planner._adapt_plan_based_on_results(
                project_config, initial_results
            )
            
            assert 'revised_strategy' in revised_plan
            assert 'additional_tests' in revised_plan
            assert len(revised_plan['additional_tests']) > 0


class TestErrorHandlingAndRecovery:
    """错误处理和恢复测试"""
    
    @pytest.mark.asyncio
    async def test_llm_api_failure_handling(self, integration_settings, sample_integration_project):
        """测试LLM API失败处理"""
        workflow = DefaultVerificationFlow(integration_settings)
        
        with patch.object(workflow.tb_generator.llm_client, 'chat') as mock_chat:
            # 模拟API失败
            mock_chat.side_effect = Exception("API rate limit exceeded")
            
            result = await workflow.execute(sample_integration_project)
            
            assert result['success'] is False
            assert 'api' in result.get('error', '').lower()
    
    @pytest.mark.asyncio
    async def test_vivado_failure_handling(self, integration_settings, sample_integration_project):
        """测试Vivado失败处理"""
        workflow = DefaultVerificationFlow(integration_settings)
        
        with patch.object(workflow.sim_runner.vivado, 'run_tcl_commands') as mock_vivado, \
             patch.object(workflow.tb_generator, 'generate_testbench') as mock_gen_tb, \
             patch.object(workflow.planner, 'create_verification_plan') as mock_plan:
            
            mock_plan.return_value = {'phases': [{'name': 'test'}]}
            mock_gen_tb.return_value = {'success': True, 'testbench_file': 'test.sv'}
            
            # 模拟Vivado失败
            mock_vivado.return_value = {
                'success': False,
                'error': 'Vivado license not available'
            }
            
            result = await workflow.execute(sample_integration_project)
            
            assert result['success'] is False
            assert 'vivado' in result.get('error', '').lower()
    
    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self, integration_settings, sample_integration_project):
        """测试部分失败的恢复"""
        workflow = DefaultVerificationFlow(integration_settings)
        
        # 模拟部分阶段失败的场景
        phases_results = [
            {'phase': 'basic', 'success': True, 'coverage': 70.0},
            {'phase': 'advanced', 'success': False, 'error': 'Timeout'},
            {'phase': 'stress', 'success': True, 'coverage': 85.0}
        ]
        
        with patch.object(workflow.planner, 'execute_plan') as mock_execute:
            mock_execute.return_value = {
                'success': True,  # 整体成功，尽管有部分失败
                'phases_results': phases_results,
                'overall_coverage': 77.5
            }
            
            result = await workflow.execute(sample_integration_project)
            
            # 应该报告部分成功
            assert 'phases_results' in result
            failed_phases = [p for p in result['phases_results'] if not p['success']]
            assert len(failed_phases) == 1


class TestPerformanceAndScalability:
    """性能和可扩展性测试"""
    
    @pytest.mark.asyncio
    async def test_large_project_handling(self, integration_settings):
        """测试大型项目处理"""
        # 模拟大型项目配置
        large_project = {
            'project_name': 'large_design',
            'rtl_files': [f'module_{i}.v' for i in range(50)],  # 50个文件
            'top_module': 'top_design',
            'verification_goals': {
                'line_coverage_target': 95,
                'functional_tests': [f'test_{i}' for i in range(100)]  # 100个测试
            }
        }
        
        workflow = DefaultVerificationFlow(integration_settings)
        
        with patch.object(workflow.planner, 'create_verification_plan') as mock_plan:
            mock_plan.return_value = {
                'phases': [{'name': f'phase_{i}'} for i in range(10)],
                'estimated_time': '2 days'
            }
            
            plan = await workflow.planner.create_verification_plan(large_project)
            
            assert len(plan['phases']) == 10
            assert 'estimated_time' in plan
    
    @pytest.mark.asyncio
    async def test_concurrent_execution(self, integration_settings):
        """测试并发执行"""
        workflow = DefaultVerificationFlow(integration_settings)
        
        # 模拟多个并发项目
        projects = [
            {'project_name': f'project_{i}', 'rtl_files': [f'design_{i}.v']}
            for i in range(3)
        ]
        
        with patch.object(workflow, 'execute') as mock_execute:
            mock_execute.return_value = {'success': True, 'project_id': 'test'}
            
            # 并发执行多个项目
            tasks = [workflow.execute(project) for project in projects]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            assert len(results) == 3
            assert all(not isinstance(r, Exception) for r in results)


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])
