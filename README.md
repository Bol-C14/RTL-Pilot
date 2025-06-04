# RTL-Pilot: AI-Powered RTL Verification Automation

RTL-Pilot是一个智能化的RTL验证自动化工具，采用多智能体大语言模型(LLM)架构，自动化测试台生成、Vivado仿真执行、结果评估和工作流编排等验证流程。

## 🚀 核心特性

- **多智能体架构**: 专门的LLM智能体负责不同验证任务
- **自动测试台生成**: 基于RTL分析和验证场景的智能测试台生成
- **Vivado集成**: 无缝集成Xilinx Vivado仿真工具
- **智能结果评估**: 自动化覆盖率分析和反馈生成
- **工作流编排**: 灵活的验证流程规划和执行
- **模板系统**: Jinja2模板支持自定义测试台和报告
- **多界面支持**: CLI命令行和Web界面两种交互方式

## 📁 项目结构

```
RTL-Pilot/
├── rtl_pilot/                  # 核心包
│   ├── agents/                 # LLM智能体
│   │   ├── testbench_gen.py   # 测试台生成器
│   │   ├── sim_runner.py      # 仿真运行器
│   │   ├── evaluation.py      # 结果评估器
│   │   └── planner.py         # 验证规划器
│   ├── config/                 # 配置管理
│   │   ├── settings.py        # 全局设置
│   │   └── schema.py          # 数据模式
│   ├── interface/              # 用户界面
│   │   ├── cli.py             # 命令行界面
│   │   └── web_ui.py          # Web界面
│   ├── prompts/                # 提示模板
│   │   ├── verilog_tb.jinja2  # Verilog测试台模板
│   │   └── feedback_loop.jinja2 # 反馈报告模板
│   ├── utils/                  # 工具模块
│   │   ├── file_ops.py        # 文件操作
│   │   └── vivado_interface.py # Vivado接口
│   ├── workflows/              # 工作流
│   │   └── default_flow.py    # 默认验证流程
│   └── scripts/                # 脚本文件
│       └── vivado_run.tcl     # Vivado自动化脚本
├── examples/                   # 示例项目
│   ├── adder/                 # 加法器示例
│   └── fifo/                  # FIFO示例
├── tests/                     # 测试代码
└── docs/                      # 文档
```

## 🛠️ 安装指南

### 前置要求

- Python 3.8+
- Xilinx Vivado (仅用于仿真功能)
- OpenAI API密钥 (用于LLM功能)

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/your-org/RTL-Pilot.git
cd RTL-Pilot

# 安装基础依赖
pip install -r requirements.txt

# 或使用开发模式安装
pip install -e .
```

### 可选依赖

```bash
# 安装Web界面依赖
pip install streamlit plotly

# 安装开发工具
pip install pytest black isort mypy
```

## ⚙️ 配置设置

### 环境变量

创建 `.env` 文件或设置环境变量：

```bash
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Vivado工具路径
VIVADO_PATH=/opt/Xilinx/Vivado/2023.1/bin/vivado

# 工作目录
RTL_PILOT_WORKSPACE=/path/to/workspace
```

### 配置文件

创建 `config.yaml`:

```yaml
llm:
  provider: openai
  model: gpt-4
  temperature: 0.1
  max_tokens: 2048

tools:
  vivado_path: /opt/Xilinx/Vivado/2023.1/bin/vivado
  
workspace:
  base_path: ./workspace
  cleanup_temp: true

verification:
  default_timeout: 300
  max_iterations: 5
```

## 🚀 快速开始

### 命令行界面

```bash
# 查看帮助
rtl-pilot --help

# 生成测试台
rtl-pilot generate-tb --rtl examples/adder/adder.v --output testbench.sv

# 运行完整验证流程
rtl-pilot run-verification --project examples/adder/

# 启动Web界面
rtl-pilot web-ui
```

### Python API

```python
from rtl_pilot.agents import TestbenchGenerator, SimulationRunner
from rtl_pilot.config import Settings

# 初始化配置
settings = Settings()

# 生成测试台
tb_gen = TestbenchGenerator(settings)
testbench = await tb_gen.generate_testbench(
    rtl_file="examples/adder/adder.v",
    module_name="adder"
)

# 运行仿真
sim_runner = SimulationRunner(settings)
results = await sim_runner.run_simulation(
    testbench_file="testbench.sv",
    rtl_files=["examples/adder/adder.v"]
)
```

### Web界面

启动Web界面后，访问 `http://localhost:8501`:

1. **项目管理**: 创建和管理验证项目
2. **测试台生成**: 上传RTL文件并生成测试台
3. **仿真运行**: 配置并执行仿真
4. **结果分析**: 查看覆盖率报告和波形

## 📋 示例项目

### 1. 简单加法器

```bash
cd examples/adder
rtl-pilot run-verification --config config.yaml
```

包含:
- 4位加法器RTL代码
- 基础测试台模板
- 验证配置文件

### 2. 同步FIFO

```bash
cd examples/fifo
rtl-pilot run-verification --config config.yaml
```

包含:
- 参数化FIFO设计
- 全面的测试场景
- 高级覆盖率配置

## 🧪 智能体详解

### TestbenchGenerator (测试台生成器)

- 分析RTL接口和时序要求
- 生成综合测试场景
- 输出SystemVerilog测试台代码

### SimulationRunner (仿真运行器)

- 设置Vivado仿真项目
- 执行仿真并收集结果
- 生成波形和日志文件

### ResultEvaluator (结果评估器)

- 分析覆盖率报告
- 检测功能问题
- 生成改进建议

### VerificationPlanner (验证规划器)

- 制定验证策略
- 协调多智能体协作
- 优化验证流程

## 🔧 高级配置

### 自定义提示模板

修改 `rtl_pilot/prompts/` 中的Jinja2模板来定制生成逻辑:

```jinja2
// 自定义测试台模板
module {{ module_name }}_tb;
    // 自定义初始化逻辑
    {% for signal in input_signals %}
    logic {{ signal.width }} {{ signal.name }};
    {% endfor %}
    
    // 您的自定义测试代码
endmodule
```

### 扩展工作流

创建自定义验证工作流:

```python
from rtl_pilot.workflows import BaseWorkflow

class CustomVerificationFlow(BaseWorkflow):
    async def execute(self, project_config):
        # 实现自定义验证逻辑
        pass
```

## 🧪 测试

运行测试套件:

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_tb_generation.py

# 生成覆盖率报告
pytest --cov=rtl_pilot --cov-report=html
```

## 📖 API文档

详细的API文档可在 `docs/` 目录中找到，或访问在线文档。

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤:

1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行代码格式化
black rtl_pilot/
isort rtl_pilot/

# 运行类型检查
mypy rtl_pilot/
```

## 📝 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🔗 相关链接

- [文档](https://rtl-pilot.readthedocs.io/)
- [问题追踪](https://github.com/your-org/RTL-Pilot/issues)
- [讨论区](https://github.com/your-org/RTL-Pilot/discussions)

## 📧 联系方式

- 项目维护者: [Your Name](mailto:your.email@example.com)
- 技术支持: [support@rtl-pilot.com](mailto:support@rtl-pilot.com)

## 🙏 致谢

感谢以下开源项目和贡献者:

- OpenAI GPT模型
- Xilinx Vivado工具链
- Python生态系统
- 所有贡献者和用户

---

**RTL-Pilot** - 让RTL验证更智能、更高效！