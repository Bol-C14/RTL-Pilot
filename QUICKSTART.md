# RTL-Pilot 快速入门指南

欢迎使用RTL-Pilot！本指南将帮助您快速上手这个AI驱动的RTL验证自动化工具。

## 📋 前置要求

在开始之前，请确保您的系统满足以下要求：

- **Python 3.8+**
- **Xilinx Vivado** (用于仿真，可选)
- **OpenAI API密钥** (用于AI功能)

## 🚀 安装

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/RTL-Pilot.git
cd RTL-Pilot
```

### 2. 安装依赖

```bash
# 基础安装
pip install -r requirements.txt

# 或开发模式安装
pip install -e .

# 安装可选依赖
pip install -e ".[web,dev]"
```

### 3. 配置环境

```bash
# 复制示例配置
cp config.example.yaml config.yaml

# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"
export VIVADO_PATH="/opt/Xilinx/Vivado/2023.1/bin/vivado"
```

## ⚡ 5分钟快速体验

### 方法1: 使用示例项目

```bash
# 进入示例项目目录
cd examples/adder

# 运行完整验证流程
rtl-pilot run-verification --config config.yaml

# 查看结果
ls output/
```

### 方法2: 使用CLI命令

```bash
# 生成测试台
rtl-pilot generate-tb \
  --rtl examples/adder/adder.v \
  --module adder \
  --output testbench.sv

# 运行仿真
rtl-pilot run-simulation \
  --rtl examples/adder/adder.v \
  --testbench testbench.sv \
  --output simulation_results/
```

### 方法3: 启动Web界面

```bash
# 启动Web界面
rtl-pilot web-ui

# 在浏览器中打开 http://localhost:8501
```

## 📚 基础用法

### 1. 创建新项目

```bash
# 创建项目目录
mkdir my_rtl_project
cd my_rtl_project

# 初始化项目配置
rtl-pilot init --name my_project --top-module top_design
```

这将创建以下结构：
```
my_rtl_project/
├── config.yaml        # 项目配置
├── src/               # RTL源文件目录
├── tb/                # 测试台目录
└── output/            # 输出目录
```

### 2. 配置项目

编辑 `config.yaml`:

```yaml
project_name: my_project
rtl_files:
  - src/design.v
  - src/utils.v
top_module: design
verification_goals:
  line_coverage_target: 85
  branch_coverage_target: 80
  functional_tests:
    - basic_operations
    - edge_cases
    - error_conditions
```

### 3. 生成测试台

```bash
# 自动生成测试台
rtl-pilot generate-tb \
  --rtl src/design.v \
  --module design \
  --scenarios basic,edge_cases \
  --output tb/design_tb.sv
```

### 4. 运行验证

```bash
# 运行完整验证流程
rtl-pilot run-verification
```

## 🔧 高级功能

### 自定义验证策略

```python
# custom_strategy.py
from rtl_pilot.workflows import BaseWorkflow

class CustomVerificationFlow(BaseWorkflow):
    async def execute(self, project_config):
        # 实现自定义验证逻辑
        plan = await self.create_custom_plan(project_config)
        results = await self.execute_custom_phases(plan)
        return results
```

### 使用Python API

```python
import asyncio
from rtl_pilot.agents import TestbenchGenerator
from rtl_pilot.config import Settings

async def main():
    # 初始化配置
    settings = Settings.load_from_file("config.yaml")
    
    # 创建测试台生成器
    tb_gen = TestbenchGenerator(settings)
    
    # 生成测试台
    result = await tb_gen.generate_testbench(
        rtl_file="src/design.v",
        module_name="design",
        test_scenarios=["basic", "edge_cases"]
    )
    
    if result['success']:
        print(f"测试台已生成: {result['testbench_file']}")
    else:
        print(f"生成失败: {result['error']}")

# 运行
asyncio.run(main())
```

### 批量处理多个项目

```bash
# 批量验证脚本
for project in projects/*; do
    echo "验证项目: $project"
    cd "$project"
    rtl-pilot run-verification --config config.yaml
    cd ..
done
```

## 📊 结果分析

验证完成后，查看生成的报告：

```bash
# 查看覆盖率报告
rtl-pilot show-coverage --format html

# 查看详细日志
rtl-pilot show-logs --level INFO

# 生成综合报告
rtl-pilot generate-report --include-all
```

报告包含：
- 📈 覆盖率分析
- 🧪 测试结果汇总
- ⚠️ 问题和建议
- 📊 性能指标
- 🌊 波形文件链接

## 🎯 最佳实践

### 1. 项目组织

```
project/
├── config.yaml           # 主配置文件
├── src/                  # RTL源文件
│   ├── design.v
│   └── utils.v
├── constraints/          # 约束文件
│   └── timing.xdc
├── tb/                   # 测试台
│   ├── auto_generated/   # 自动生成的测试台
│   └── manual/          # 手写测试台
├── scripts/             # 脚本文件
├── docs/                # 文档
└── output/              # 输出结果
    ├── simulation/
    ├── coverage/
    └── reports/
```

### 2. 配置管理

```yaml
# 为不同环境使用不同配置
verification:
  # 开发环境 - 快速验证
  dev:
    coverage_threshold: 70
    timeout: 300
  
  # 生产环境 - 严格验证  
  prod:
    coverage_threshold: 90
    timeout: 1800
    enable_formal_verification: true
```

### 3. 版本控制

```bash
# .gitignore 示例
output/
*.log
*.vcd
*.wlf
__pycache__/
.rtl_pilot_cache/
temp_*
```

## 🔍 故障排除

### 常见问题

1. **OpenAI API错误**
   ```bash
   # 检查API密钥
   echo $OPENAI_API_KEY
   
   # 测试API连接
   rtl-pilot test-llm
   ```

2. **Vivado找不到**
   ```bash
   # 检查Vivado路径
   which vivado
   
   # 设置正确路径
   export VIVADO_PATH="/correct/path/to/vivado"
   ```

3. **权限问题**
   ```bash
   # 检查工作空间权限
   ls -la workspace/
   
   # 修复权限
   chmod -R 755 workspace/
   ```

### 调试模式

```bash
# 启用详细日志
rtl-pilot --log-level DEBUG run-verification

# 保留临时文件用于调试
rtl-pilot --keep-temp run-verification
```

## 📖 下一步

- 📚 阅读[完整文档](https://rtl-pilot.readthedocs.io/)
- 🎓 学习[高级教程](docs/tutorials/)
- 💡 查看[示例项目](examples/)
- 🤝 参与[社区讨论](https://github.com/your-org/RTL-Pilot/discussions)

## 🆘 获取帮助

- 📋 [问题追踪](https://github.com/your-org/RTL-Pilot/issues)
- 💬 [讨论区](https://github.com/your-org/RTL-Pilot/discussions)
- 📧 [邮件支持](mailto:support@rtl-pilot.com)

---

🎉 恭喜！您已经成功开始使用RTL-Pilot。现在开始自动化您的RTL验证之旅吧！
