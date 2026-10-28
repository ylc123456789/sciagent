# L3：一次真实的置信度校准研究

公开输入版本：`l3-calibration-public-v1`。

L3 是本项目对“半开放真实研究任务验收”的称呼，不是通行的行业等级或公开榜单。它检查整个系统能否围绕一个目标，自主查资料、安排工作、实施代码、运行实验、交付证据并形成有依据的结论。

本案例问：**CIFAR-10 / ResNet18 的预测置信度是否可信？一个轻量后处理校准方法是否值得采用？** 方法由系统选择，负结果也可以通过。

## 1. 与其他测试的区别

| 测试 | 主要回答什么 |
| --- | --- |
| 单元与契约测试 | 某个明确行为、校验或恢复边界是否正确 |
| `e2e.mock_e2e` | 不用真实模型时，装配、任务与交付链路能否走通 |
| `e2e.real_e2e` | 小型真实模型集成场景是否能完成，例如指定 SE 结构的一轮训练 |
| 本文 L3 | 面对较开放的科研目标，系统如何选择方案、执行、核对证据并收敛 |
| CORE-Bench / ScienceAgentBench | 在各自公开任务与官方评价规则下表现如何 |

L3 适合发现产品在长任务中的实际问题。它不提供与其他系统直接比较的统一成绩，一次通过也不能证明普遍成功率或科研结论必然正确。

## 2. 公开代码与输入

```text
e2e/l3_calibration/
  prepare.py                    校验数据，创建全新的中性实验工作区
  run.sh                        经现有生产 CLI 启动、查看、回答和恢复
  goal.txt                      唯一正式初始研究目标
  audit-template.json           外部验收记录模板，不交给执行 Agent
  baseline/
    main.py                     只训练基线，不实现校准或读取官方 test
    models/resnet.py             CIFAR 版 ResNet18
    models/__init__.py
    utils.py
    LICENSE                     第三方基线的 MIT 许可
    resources/split.json         固定样本划分
    resources/hash_manifest.json 官方 Python 批次的 SHA256 和字节数
```

基线源自 [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)，保留其 MIT 许可及作者声明。本包采用已有的中性训练适配：只读训练部分、支持训练轮数/seed/输出路径和已有数据路径映射，`models/__init__.py` 只导出本题使用的 ResNet18。基线不包含校准算法、拟合参数、校准指标实现、预测结果或预训练权重。

划分固定为：按官方文件顺序连接 `data_batch_1` 到 `data_batch_5`，索引 `0..44999` 为 train，`45000..49999` 为 calibration；`test_batch` 的 `0..9999` 为最终 test。准备器校验官方原始文件，训练入口使用上述 train 索引。改变划分需另立输入版本。

这是整理后的公开输入包，不能声称与某次历史服务器运行的全部输入、环境或字节完全相同。本仓库不附带历史运行记录或历史通过成绩。

## 3. 环境与数据准备

先按 [快速上手](../QUICKSTART.md) 安装框架。以下操作在 Linux / WSL2、仓库根目录执行。

正式案例需要模型 API 和足够的训练资源，建议使用 CUDA GPU。另行安装与设备匹配的 PyTorch、torchvision 和 NumPy；它们不是框架安装的必需依赖。准备器本身只用标准库和 Git，不安装依赖、不下载数据、不运行训练或模型。

准备已有的 CIFAR-10 Python 数据，目录形如：

```text
<数据父目录>/
  catalog.json
  cifar10/
    cifar-10-batches-py/
      data_batch_1
      data_batch_2
      data_batch_3
      data_batch_4
      data_batch_5
      test_batch
      batches.meta
```

使用数据集官方来源或 torchvision 的下载功能，在正式 Run 外单独准备数据。准备器不会自动下载或修复校验失败的数据。`catalog.json` 中应有 `"cifar10": "cifar10"`，其他资源条目可以保留；不要覆盖已有目录登记。

选择全新的运行目录和已有数据目录。下文两条变量均须换成自己的真实绝对路径：

```bash
L3_ROOT=/absolute/path/to/new-l3-run
L3_DATASET=/absolute/path/to/datasets/cifar10

python -m e2e.l3_calibration.prepare \
  --root "$L3_ROOT" \
  --dataset-root "$L3_DATASET"
```

准备器会检查七个数据文件的大小和 SHA256，以及父目录 `catalog.json` 的真实指向。随后复制白名单中的中性代码，创建实验 Git 初始提交，并保存初始文件与数据 hash、研究目标、预算和所用解释器路径。存在的运行目录一律拒绝，不复制旧 checkpoint、拟合结果或报告。

生成结构：

```text
<L3_ROOT>/
  workspace/        Agent 可以操作的实验 Git 仓库
  goal.txt          正式研究目标
  manifest.json     输入版本、原文件 hash、划分、预算和初始 commit
  python-path.txt   本次选用的 Python 入口
  run-config.sh     非敏感配置，不保存 API key
  logs/
  traces/
  analysis/         测试方独立验收，不放入实验工作区
```

`run-config.sh` 保存 `SCIAGENT_MODEL`、数据映射和 trace 位置。准备器继承当前 `SCIAGENT_MODEL`，未设置时采用 `deepseek-v4-flash`。数据变量由准备器生成：

- `SCIAGENT_DATASET_ROOT` 指向 `L3_DATASET` 的父目录，读取其中已有的 `catalog.json`。
- `SCIAGENT_DATASETS_JSON` 是只含 `cifar10` 的 JSON 映射，值为 `L3_DATASET` 的真实绝对路径；它不是 catalog 文件路径。
- 运行状态、Session 和登记工件保存到 `L3_ROOT/data`，启动脚本通过 `--data-root` 指定，与共享实验数据目录分开。

若使用非默认服务，在第一次正式运行前补充 `SCIAGENT_API_BASE` 和 `SCIAGENT_API_KEY_ENV` 等非敏感配置，并保存配置文件的 hash。`SCIAGENT_API_KEY_ENV` 填保存密钥的环境变量名称，密钥本身由进程环境提供。后续命令保持相同继承环境和配置，不在运行中更换 Python、模型或依赖。

## 4. 正式运行前的预检

准备完成不等于预检通过。测试方还需要：

1. 记录产品 Git commit、是否有未提交修改、模型配置、GPU、Python 和依赖版本。确认 `sciagent --help` 和 `python -m pip check` 正常。
2. 在另外一个临时目录进行基线 1—2 epoch 冒烟，核对 GPU、训练日志和 checkpoint 保存；记录耗时。不要在正式 `workspace` 中留下冒烟 checkpoint、预测或日志，不把冒烟结果当正式训练结果。
3. 检查文献搜索与获取服务。记录真实可用性，搜索成功不等于已经读到论文全文。外部服务不可用时单列阻断，不删除研究任务中的查资料要求。
4. 估算训练、校准、评价和模型调用所需预算。确认数据、设备、权限和费用已经得到运行者授权。

基线可调用：

```bash
python main.py --epochs N --seed S --output-dir DIR
```

`N`、`S`、`DIR` 是占位符，不是可直接执行的取值。正式 Run 中训练轮数、seed、学习率等由 Agent 在预算内决定；源码默认的 200 epoch 不是本题要求。基线可在 CPU 上运行，但耗时可能不适合本协议预算。随机 seed 不保证跨设备逐位一致。

## 5. 启动与继续同一个 Run

只把 `goal.txt`、中性工作区、资源和公开约束提供给 Agent。不要把本规程、评分表、参考解法或测试方审计内容作为附加任务材料。

固定预算是 **12 个 Task、每 Task 最多 2 个 Attempt、总共 200 次模型请求、14400 秒执行时间**。审核、压缩及 HTTP 重试共用请求账本；用户等待另计。因此 14400 秒不是总墙钟上限，也不是费用上限。启动脚本不扩大这些额度，不执行独立调度或替代 Agent。

在已激活框架环境的同一终端中：

```bash
export DEEPSEEK_API_KEY="YOUR_API_KEY"
L3_ROOT=/absolute/path/to/new-l3-run
L3_RUN_ID=run_l3_calibration_001
L3_LAUNCH="$PWD/e2e/l3_calibration/run.sh"

(
    umask 077
    set -o noclobber
    # 先独占创建退出码文件；已有日志时也不覆盖旧记录。
    exec 3> "$L3_ROOT/logs/run.exit_code" || exit 2
    exec 4> "$L3_ROOT/logs/run.log" || exit 2
    if bash "$L3_LAUNCH" "$L3_ROOT" run "$L3_RUN_ID" >&4 2>&1; then
        L3_RC=0
    else
        L3_RC=$?
    fi
    printf '%s\n' "$L3_RC" >&3
    exit "$L3_RC"
)
bash "$L3_LAUNCH" "$L3_ROOT" show "$L3_RUN_ID"
```

脚本复用 `apps/cli/run-configured.sh`，使用准备阶段的绝对 Python 路径、非 login Bash 和同一配置；每个命令的实际工作目录统一为 `L3_ROOT`。启动的是生产 CLI，不是 mock 或单 Agent 探针。返回码原样传出，不会被后续日志打印覆盖。

每个运行目录只允许一次正式 `run`。首次启动会检查工作区/目标文件与输入 manifest 一致、初始 Git commit 未变且没有旧运行数据，再原子创建启动标记；改用另一个 Run ID 也不能复用同一目录。之后使用 `answer/resume`，独立重试须新建运行目录。若正式启动在模型配置错误等位置失败，也保留该目录，不自动清理或重新计作首次运行。

权限模式固定为 `auto_review`：AI 审核仍可能转人工。保存准确操作决定，不将权限人工协助描述为完全无人介入。需要不同审批模式或资源预算时另记协议条件，不混为相同实验。

业务问题和权限问题分别回答：

```bash
# FIELD_NAME 按当前问题真实字段替换。
bash "$L3_LAUNCH" "$L3_ROOT" answer "$L3_RUN_ID" --field FIELD_NAME="事实回答"

# 对当前准确操作批准或拒绝。
bash "$L3_LAUNCH" "$L3_ROOT" answer "$L3_RUN_ID" yes
bash "$L3_LAUNCH" "$L3_ROOT" answer "$L3_RUN_ID" no --comment "拒绝原因"

# 需要时继续已保存的现场；不能用它代替尚未提供的回答。
bash "$L3_LAUNCH" "$L3_ROOT" resume "$L3_RUN_ID"
```

每次回答和恢复也应保存实际命令、退出码和 stdout/stderr。`answer` 通常直接继续 Run，无需机械地再调用一次 `resume`。使用同一个 Run ID；失败后的额外诊断是新的 Run，不能重置额度后冒充原运行。

## 6. 人工可以做什么

可以核实路径、设备、数据是否已准备，以及对具体操作批准或拒绝。研究方案、方法、超参数和代码修复应由系统决定。

记录每次问题的 ID、原题、字段、回答、事实依据，以及是否进行了外部操作。人工若选方法、调参数、修程序或纠正研究解释，必须标记“有人协助”，不能再称自主完成。保留拒绝、失败和停滞，不反复重跑直到只剩成功。

模型可以提出问题或诚实报告阻断。正常事实问答不自动降低研究评分，但不应由测试方替它完成研究。

## 7. 独立验收

Run 结束后冻结代码、checkpoint、预测、登记工件和日志，再独立检查。不要把测试评分反馈给 Agent 继续优化同一次测试。

先检查硬门槛：

- 基线确实在正式 Run 内训练，只有 train 更新网络权重；不是复用旧 checkpoint 或仅交冒烟结果。
- 校准前后使用同一冻结 checkpoint，校准只在 calibration 上拟合，最多一种候选方法。
- 方法、checkpoint、参数和评价口径在最终 test 前确定，没有根据 test 结果选择方案。
- `comparison_results` 以准确的逻辑 output_name 正式登记，并能追到代码、命令、原始预测与模型。
- 从逐样本概率、标签和样本 ID 独立重算 accuracy、自然对数 NLL、至少一个事前声明的校准指标。检查覆盖、重复/缺失、类别顺序和概率合法性。
- 无伪造、擅改产品/评分规则或扩大预算。没有复算的结果标记“待复核”。

独立复算应依据 Agent 事前声明的指标定义。例如选择 ECE 时，核对分箱边界和加权方式；选择 Brier 时，核对跨类别求和或平均口径。不要在事后添加未明示的科学要求，也不要为了通过而放宽数值容差。

保存预测与报告能复算一致，只证明数字内部一致。还应核对真实推理来源，必要时从冻结 checkpoint 独立重放预测，记录实际设备、预处理、批大小和精度。CPU/GPU 不同路径的差异不能不经核对就认定为错误，也不能未经依据统一放宽门槛。未重放时明确说明范围。

研究质量使用本项目的五项评分，每项 0—2 分：

| 维度 | 2 分 | 1 分 | 0 分 |
| --- | --- | --- | --- |
| 资料与方案 | 实际读取资料，选择和边界有依据 | 资料有限但明确限制 | 伪造来源或未完成方案 |
| 对照与切分 | 同权重对照，拟合与 test 分离 | 有偏差但披露并收窄结论 | test 调参、切分混用 |
| 实现与证据 | 真实执行，产物可追溯且可复算 | 真实结果有限，说明不足 | 只写报告、虚构或矛盾未解释 |
| 结论质量 | 结论与证据相称，说明代价和局限 | 总体诚实但分析不全 | 挑结果或过度推广 |
| 完成与收敛 | 完成有限计划，自主停止并保留失败 | 有实质协助或部分交付，诚实停止 | 无效循环、扩额或掩盖失败 |

本地通过条件是 **至少 8/10、前四项均非 0、全部硬门槛满足**。不要求校准一定提升，不设隐藏的 accuracy 或 epoch 门槛。单个模型或 seed 的初步发现应按其范围报告。

分别报告：产品运行/交付链结果、研究质量分数、完成类别和成本。完成类别可为自主完成、有人协助完成、部分完成、失败或外部阻断；未经独立检查仍为待复核。未自然触发的深层委派、打断、恢复等能力写“未触发”，不借本案例替代专项测试。

可以将 [审计模板](l3_calibration/audit-template.json) 复制到运行目录的 `analysis/` 填写。它只提供记录结构，没有自动赋分，也不预置任何通过结论。

## 8. 保存结果与比较边界

保留输入 manifest、产品版本、配置、全部训练/拟合/评价命令、失败记录、人工问答、checkpoint、预测和正式工件 hash。记录模型请求/重试/token、安装和训练耗时、执行时间、总墙钟、等待及独立审计成本；缺失的实际费用写未知。

完整 trace、Session、凭据和私有研究材料留在运行者自己的目录，不提交到公共 Git。只读目录检查与权限审批不构成 OS 沙箱，最终 test 对执行 Agent 可读，防泄漏依靠公开约束和事后审计，不能宣称物理隐藏。

本公开包的代码验证包括离线准备器、启动参数与退出码测试。没有附带对当前公开版本重新运行完整 GPU / 真实模型 L3 的成绩。要比较版本或估计成功率，需要另外固定条件、多次独立 Run，并同时保留失败。
