# 快速上手

本指南从安装和离线检查开始，再运行一个小任务，最后介绍如何查看、回答和继续同一次运行。

## 1. 准备环境

使用 Linux 或 Windows 下的 WSL2。下面的安装和使用命令都在 Linux Shell 中执行。

```bash
git clone https://github.com/ylc123456789/sciagent.git
cd sciagent
conda env create -f environment.yml
conda activate ResAgent2
resagent2 --help
```

请在仓库根目录创建环境，`environment.yml` 会安装仓库中的本地包。现阶段环境名为 `ResAgent2`，命令名为 `resagent2`。

已有环境可在检查依赖变更后执行：

```bash
conda env update -n ResAgent2 -f environment.yml
```

## 2. 不用模型，先检查安装

```bash
python -m e2e.mock_e2e
```

它使用替身执行一个小闭环，不需要 API 密钥、GPU 或真实模型请求。成功输出中应包含 `status=completed`。这只验证框架链路，不衡量模型的科研能力。

查看 CLI 子命令：

```bash
resagent2 run --help
resagent2 show --help
resagent2 answer --help
resagent2 resume --help
```

## 3. 配置模型与数据目录

最小配置：

```bash
export DEEPSEEK_API_KEY="YOUR_API_KEY"
export RESAGENT2_DATA_ROOT="$HOME/sciagent-data"
```

密钥只保存在执行环境中，不要写入代码或提交到 Git。默认服务地址为 `https://api.deepseek.com/v1`，默认模型是 `deepseek-v4-flash`。

如需显式设置服务和模型：

```bash
export RESAGENT2_API_BASE="https://api.deepseek.com/v1"
export RESAGENT2_MODEL="deepseek-v4-flash"
export RESAGENT2_API_KEY_ENV="DEEPSEEK_API_KEY"
```

`RESAGENT2_API_KEY_ENV` 填写保存密钥的环境变量名称，不填密钥本身。更换兼容服务时，需要自行确认其原生 function calling 协议和模型容量配置。

所有命令应使用相同的 `RESAGENT2_DATA_ROOT`，否则后续命令可能找不到之前的 Run。

网页搜索默认使用 DeepSeek 搜索服务。不需要网页搜索时，可在创建任务前关闭：

```bash
export RESAGENT2_WEB_SEARCH_PROVIDER="off"
```

## 4. 先运行只读任务

准备一个含有 README、论文笔记或其他文本材料的目录，替换下面的工作区绝对路径：

```bash
resagent2 run \
  --workspace /absolute/path/to/project \
  --goal "阅读现有材料，概括研究问题和已有证据，指出哪些结论仍缺少验证。不要执行命令或修改文件。" \
  --read-only \
  --no-execute-commands \
  --max-llm-calls 40 \
  --timeout-seconds 600
```

这是一个有请求和时间上限的小例子，达到上限时可能未完成。请记下输出中的 Run ID。

## 5. 再尝试代码与实验任务

在一个专用工作区准备简单数据：

```bash
mkdir -p "$HOME/sciagent-demo"
cd "$HOME/sciagent-demo"
git init
printf 'x,y\n1,2\n2,4\n3,6\n' > data.csv
```

然后发起任务：

```bash
resagent2 run \
  --workspace "$HOME/sciagent-demo" \
  --goal "编写并运行 Python 程序读取 data.csv，计算每列的均值和样本标准差（ddof=1），把结果保存为 results/summary.json。交付程序、计算结果和简短说明，不修改输入数据。" \
  --approval-mode manual \
  --max-llm-calls 80 \
  --timeout-seconds 1800
```

允许执行的任务可能创建文件、安装依赖或运行代码。默认使用人工审批；看到操作申请时，检查准确命令和目标，再决定是否允许。

具体实验所需的数据、软件和算力需要另行准备。环境创建与选择由执行 Agent 通过命令处理，框架不承诺自动解决任意依赖问题。

## 6. 查看、回答与恢复

将 `RUN_ID` 替换为实际运行标识：

```bash
resagent2 show RUN_ID
```

如果系统提出业务问题，按照展示的字段名填写：

```bash
resagent2 answer RUN_ID --field FIELD_NAME="你的回答"
```

如果是具体操作的权限确认：

```bash
resagent2 answer RUN_ID yes
resagent2 answer RUN_ID no --comment "拒绝这次操作的原因"
```

批准只绑定该次具体操作，不授予通用权限。`answer` 接收回答后会继续同一 Run，通常不需要再紧接着调用 `resume`。

需要继续保存的运行现场时：

```bash
resagent2 resume RUN_ID
```

本版本在创建 Run 时支持 `manual` 和 `auto_review`。后者让 AI 辅助审核具体操作，无法判断或审核故障时仍可能请求人工。没有运行中审批模式切换或 `full_access` 模式。

CLI 常见退出码：

| 退出码 | 含义 |
| --- | --- |
| 0 | 命令成功；run、answer、resume 返回此码时表示 Run 完成 |
| 1 | 失败 |
| 3 | 暂停，通常需要查看问题或待处理事项 |
| 4 | 返回的 Run 状态为 running，可用 show 查看 |

暂停不等于研究失败，应先用 `show` 查看原因。

## 7. 测试与使用边界

在仓库根运行本地测试：

```bash
python -m pytest tests apps/cli/tests -q
```

部分测试依赖额外的公开基准资源或显式开启的网络检查，未具备条件时会跳过。真实模型任务与离线测试分开执行。

- Shell 使用 Linux Bash，每次调用都是独立进程。上一次的 `cd`、`export` 或环境激活不会自动延续到下一次。
- 使用专用工作区和可信执行环境。当前宿主执行方式没有操作系统级沙箱，人工或 AI 审批都不能替代隔离。
- 任务可能暂停提问，可能失败，也可能生成需要检查的代码和结论。请保留实际输入与输出，独立核对重要结果。
- 修改或更新代码后，不要假定旧版保存的 Run 一定兼容恢复。
