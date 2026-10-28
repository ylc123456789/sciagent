# sciagent

sciagent 是一个面向科研任务的 Agent 框架开发预览版。它将研究判断、代码与实验执行、任务管理分开组织，支持通过命令行运行、查看和继续研究任务。

## 能做什么

- 由 Scientific Agent 阅读材料、提出工作需求，并根据证据形成结论。
- 由 Executor 按 Coding、Experiment 等配置执行代码或实验，支持有深度限制的下级委派。
- 通过固定代码管理任务、依赖、交付、预算和暂停恢复。
- 使用文件工具、Shell、网页搜索和文献工具取得材料，记录正式产物及其来源。
- 使用人工审批，或选择 AI 辅助审核具体操作。

模型决定研究和执行步骤。工具成功、任务完成和科学结论正确是不同层面的检查，运行成功并不保证结论正确。

## 安装

执行环境为 Linux，Windows 用户请在 WSL2 中安装。需要 Git 和 Conda，Python 3.12 及其余依赖由环境文件安装。

```bash
git clone https://github.com/ylc123456789/sciagent.git
cd sciagent
conda env create -f environment.yml
conda activate sciagent
sciagent --help
```

核心框架不要求 GPU。训练任务需要的 CUDA、深度学习库、数据与算力由具体任务决定。

## 先做离线检查

在仓库根目录运行：

```bash
python -m e2e.mock_e2e
```

这个示例使用确定性替身，不发送真实模型请求，用于检查任务执行和产物交接链路。

## 运行真实任务

配置模型密钥和运行记录目录。默认模型服务是 DeepSeek，模型名为 `deepseek-v4-flash`，必须支持原生工具调用。

```bash
export DEEPSEEK_API_KEY="YOUR_API_KEY"
export SCIAGENT_DATA_ROOT="$HOME/sciagent-data"

sciagent run \
  --workspace /absolute/path/to/project \
  --goal "阅读项目文件，说明研究目标、已有证据和仍需验证的问题" \
  --read-only \
  --no-execute-commands
```

更多示例、问答与恢复方法见 [快速上手](QUICKSTART.md)。真实任务会调用外部服务并产生相应费用。

## L3 真实研究验收

仓库提供 CIFAR-10 / ResNet18 置信度校准案例，包含中性训练输入、数据校验与工作区准备代码、生产 CLI 启动入口及独立验收规程。它用于检查完整科研流程，允许系统自主选择一种后处理方法，负结果也可以通过。

准备和运行步骤见 [L3 测试说明](e2e/L3_CALIBRATION.md)。L3 是本项目的验收称呼，不是公开榜单等级；完整运行需要已有数据、训练资源和真实模型预算。

## 开发与协作

安装环境后，可运行本地测试：

```bash
python -m pytest tests apps/cli/tests -q
```

代码主要位于 `packages/`，命令行入口位于 `apps/cli/`，端到端辅助入口位于 `e2e/`。提交问题时，请提供复现步骤、运行平台和经过脱敏的错误信息，不要上传模型密钥或完整私有研究记录。

这是开发预览版，目前没有原生 Windows 支持，也没有完整公开基准成绩声明。Shell 在启动账号的宿主权限下运行，工作区检查与审批不构成操作系统沙箱，请在专用、可信的执行环境中使用。

## 许可

框架部分暂未指定开源许可证。L3 第三方基线的 MIT 许可单独保留在 [baseline/LICENSE](e2e/l3_calibration/baseline/LICENSE)。
