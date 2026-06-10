# 技术概览（中文）

## 项目定位

`earnings-event-vol` 是一个围绕美股财报事件的研究型量化平台。它的目标不是做通用波动率预测，而是回答一个更具体的问题：

> 模型能否在财报前后，改进对期权隐含事件方差错价的交易判断？

项目将数据获取、事件构造、特征工程、模型训练、策略评估和报告生成串成一条可复现的研究流水线，偏论文工程而不是单一策略脚本。

---

## 技术栈

### 语言与基础库

- Python 3.11 到 3.13
- `pandas`、`numpy`、`scipy`：表格计算、数值计算、统计函数
- `pydantic`：领域对象和输入约束
- `pyyaml`：配置管理
- `httpx`：访问 Massive、SEC 等外部数据源
- `matplotlib`：研究图表输出
- `polars`、`pyarrow`：列式数据与 Parquet 支持
- `torch`：深度学习与序列模型

### 可选机器学习依赖

- `scikit-learn`：线性模型、预处理与验证
- `lightgbm`：树模型
- `xgboost`：树模型
- `optuna`：超参数搜索
- `mamba_ssm`：Mamba 序列模型后端

### 工程工具

- `uv`：环境和依赖管理
- `just`：统一命令入口
- `pytest`：测试
- `mypy`：静态类型检查
- `ruff`：格式化和 lint
- `mkdocs` + `mkdocs-material`：文档站点

---

## 架构总览

项目采用“研究流水线 + 领域模块”结构。

1. 配置层：统一管理路径、数据根目录、API 密钥、运行参数。
2. 数据层：从 Massive、SEC、FRED 等来源构造可用样本。
3. 事件层：定义财报事件、交易日历、事件窗口与期权合约选择。
4. 特征层：从事件面板生成横截面和序列特征。
5. 模型层：运行 baseline、传统 ML、深度学习和序列模型。
6. 策略层：把预测结果转成 premium-space 交易信号和代理 PnL。
7. 评估层：输出 forecasting、ranking、strategy 等指标。
8. 报告层：生成研究报告、图表和摘要文档。

从命令面看，主要有两条主线：

- `just data`：构建研究所需的事件级数据
- `just research`：在现有数据上做特征、建模、评估和报告

---

## 目录与核心模块

### 命令入口

- `src/earnings_event_vol/cli.py`
  - 全项目的统一 CLI。
  - `just` 命令最终基本都会走这里。

- `justfile`
  - 项目的公共操作面板。
  - 常用命令包括 `just status`、`just check`、`just data`、`just research`、`just docs`。

### 配置与领域对象

- `src/earnings_event_vol/config.py`
  - 定义 `ProjectConfig`。
  - 负责解析 `DATA_DIR`、`REPORTS_DIR`、`ARTIFACTS_DIR`、Massive API 配置等。
  - 所有数据路径和密钥文件基本都从这里进入系统。

- `src/earnings_event_vol/schemas.py`
  - 定义核心枚举和领域对象。
  - 包括 `EarningsEvent`、`EventWindow`、`OptionQuote`、`TradeLeg`、`StrategyTrade` 等。
  - 相当于全项目的数据契约层。

### 事件与市场日历

- `src/earnings_event_vol/events.py`
  - 处理美股交易日、节假日、提前收盘和事件窗口对齐。
  - 财报盘前和盘后会对应不同的事件窗口定义。

- `src/earnings_event_vol/earnings_calendar.py`
  - 构建财报事件候选集。
  - 会从 SEC 提交文件、8-K 文本等来源提取和校验财报信息。
  - 是“事件来源”的核心模块。

### 股票池与事件面板

- `src/earnings_event_vol/universe.py`
  - 构造月度流动性股票池。
  - 会过滤非单名股票标的，如 ETF、指数、波动率产品等。

- `src/earnings_event_vol/event_panel.py`
  - 负责期权合约发现、前向价与 ATM 选择、事件面板拼接。
  - 把财报事件转换成模型可用的事件级样本。

### 代理价格与交易近似

- `src/earnings_event_vol/trade_proxy.py`
  - 这是项目最关键的工程模块之一。
  - 使用 Massive 秒级聚合数据估算事件前入场、开盘后出场、收盘前出场等代理价格。
  - 负责计算 `option_vwap`、局部 IV、事件代理面板和大量交易诊断字段。
  - 本项目当前不是基于 NBBO 的逐笔成交回测，而是基于受约束的“代理交易价格”研究。

### 方差定义与金融公式

- `src/earnings_event_vol/variance.py`
  - 定义项目中的 realized event variance 和 implied event variance 提取逻辑。
  - 会把财报事件拆成 `jump_c2o`、`day_c2c`、`reaction_o2c` 三类目标。
  - 同时负责 `IVAR_event` 抽取与 variance edge 计算。

### 特征工程与模型

- `src/earnings_event_vol/features.py`
  - 生成模型输入矩阵。
  - 包括历史滚动特征、训练集拟合归一化、特征 schema 过滤、序列矩阵构造等。

- `src/earnings_event_vol/models.py`
  - 定义模型注册表和训练逻辑。
  - 包括 baseline、ElasticNet、LightGBM、XGBoost、FT-Transformer、BiGRU、Mamba 等。
  - 支持 sequence 模型与普通表格模型并行存在。

### 策略评估与指标

- `src/earnings_event_vol/backtest.py`
  - 把预测结果映射成 premium-space 信号。
  - 关注的是交易价值是否覆盖入场成本和交易成本，而不是只比较预测误差。

- `src/earnings_event_vol/metrics.py`
  - 负责 forecasting、ranking、calibration、PnL、drawdown、cost sensitivity 等指标。
  - 是结果解释和报告生成的主要支持模块。

- `src/earnings_event_vol/research.py`
  - 组织 sequence audit、features、models、report 四个研究阶段。
  - 是论文级研究产线的总调度器。

---

## 数据流水线

### `just data` 的主 DAG

默认数据流水线会执行如下阶段：

```text
options-day-aggs-bulk
-> universe
-> dynamic-calendar
-> sec-companyfacts
-> event-window-panel
-> contract-reference-validation
-> trade-proxy-panel
```

### 各阶段作用

- `options-day-aggs-bulk`
  - 下载或整理期权日频聚合数据。
  - 为股票池选择、合约发现和部分特征提供基础数据。

- `universe`
  - 依据期权流动性构造月度 top-N 单名股票池。

- `dynamic-calendar`
  - 基于 SEC/8-K 等来源构造动态财报日历。

- `sec-companyfacts`
  - 拉取 SEC CompanyFacts 数据，用于财务基本面特征。

- `event-window-panel`
  - 把财报事件与标的价格、期权合约、事件窗口拼成事件级样本。

- `contract-reference-validation`
  - 校验期权合约元数据，如 `shares_per_contract`、deliverable 是否标准化。
  - 排除非标准合约。

- `trade-proxy-panel`
  - 基于秒级期权交易聚合数据构造代理交易价格面板。
  - 为建模和策略评估提供核心输入。

### 设计特点

- 使用“事件级”而不是单日级样本。
- 将交易可执行性限制纳入面板构造，而不只做学术预测。
- 对时间戳、盘前盘后、开盘后窗口和收盘前窗口做了明确约束。

---

## 目标变量与金融定义

项目关注的是财报事件方差，而不是常规日波动率。

### 三类 realized target

- `jump_c2o`
  - 财报跳空部分的方差
  - AMC 通常对应 `close_d -> open_{d+1}`
  - BMO 通常对应 `close_{d-1} -> open_d`

- `day_c2c`
  - 财报反应日完整 close-to-close 方差
  - 当前 V1 代理策略主要围绕它评估

- `reaction_o2c`
  - 开盘后到收盘的消化阶段方差
  - 主要用于诊断和拆分事件反应

### 市场基准

- `IVAR_event`
  - 用两条覆盖事件的邻近期权期限进行插值抽取事件隐含方差。
  - 它是模型需要去挑战的“市场共识”。

### 错价定义

- ex post mispricing
  - `RVAR_event_day_c2c - IVAR_event`

这意味着项目不是只在做“预测 realized variance”，而是在做“预测是否存在可交易的事件方差错价”。

---

## 特征工程技术

特征工程由 `features.py` 主导，核心技术包括：

- 点时信息约束
  - 特征必须遵守事件前可获得信息边界，防止泄漏。

- 滚动事件历史
  - 为同一股票累积过去数次财报事件统计量。

- 训练集拟合归一化
  - z-score、rank 等变换只在 train 拟合，再应用到 validation/test。

- 特征 schema 管理
  - 通过 schema report 控制哪些列可以进入模型。
  - 显式排除 ID、结果列、PnL 列、未来信息列。

- 序列特征
  - 支持将期权表面路径、日度或盘中路径编码成 sequence matrix。

### 特征来源

- 事件级价格与波动率字段
- 历史财报方差与隐含方差
- 财务报表和 SEC XBRL 字段
- 单名股票 run-up 与 surface proxy
- VIX 等市场协变量
- 期权序列通道

---

## 模型体系

`models.py` 通过 `MODEL_REGISTRY` 管理模型。模型分成几类：

### Baseline

- `market_implied_event_variance`
  - 直接使用市场的 `IVAR_event`

- `last_four_rvar`
  - 同股票最近四次财报的 realized variance 平均值

- `last_four_ivar`
  - 同股票最近四次财报的 implied variance 平均值

### 表格模型

- `linear_elastic_net_tuned`
- `lightgbm_tuned`
- `xgboost_tuned`
- `lightgbm_xgboost_mean_ensemble`

### 深度和序列模型

- `ft_transformer`
- `ridge_flat_aggregates_sequence`
- `attention_pooling_sequence`
- `bigru_sequence_5seed`
- Mamba 相关 sequence 模型

### 调参与评估原则

- 测试集不参与调参
- Optuna 和 `ElasticNetCV` 只读 train/validation
- 锁定测试集只在最终评估阶段使用

这说明项目的建模设计偏研究规范，而不是追求单次回测最好看的结果。

---

## 策略与回测技术

本项目的交易评估重点不是“预测准不准”，而是“预测能否转化成净收益为正的交易判断”。

### 核心思路

- 先预测事件 realized variance 或相关错价
- 再与市场 `IVAR_event` 比较，得到 edge
- 再映射到期权 premium-space 的期望交易价值
- 只有当期望价值覆盖入场成本和交易成本时，信号才有意义

### `backtest.py` 的职责

- Black-Scholes 价格近似
- 期权 payoff 计算
- 事件跳跃分布近似
- 市场入场成本估计
- premium-space 信号构造
- 策略框架与风险上限控制

### 当前回测边界

- 不是完整撮合回测
- 不是 NBBO 级别成交研究
- 主要是风险定义期权结构下的研究型代理回测
- 更适合论文分析和策略筛选，不适合直接当生产交易系统

---

## 指标体系

`metrics.py` 同时支持三类评估：

- Forecast metrics
  - 例如误差和类 QLIKE 损失

- Ranking metrics
  - 衡量模型是否能把高 edge 事件排在前面

- Strategy metrics
  - 包括 PnL、return on premium、drawdown、cost sensitivity 等

这套设计很符合项目目标：模型即使在误差上不最优，也可能在交易排序上更有价值。

---

## 工程与可复现性设计

项目明显重视研究可复现性和协议一致性。

### 具体体现

- `just` 提供统一命令面
- `uv` 管理依赖和环境
- `pytest` 覆盖大量协议细节
- `mypy` 与 `ruff` 提高静态质量
- `mkdocs` 输出文档站点
- `artifacts/`、`reports/`、`gold data` 明确区分中间产物和结果产物

### 配置约束

- `DATA_DIR` 默认要求在仓库外
- 项目显式避免把大数据根目录放在云盘仓库中
- 外部数据、密钥路径和产物目录通过环境变量配置

这说明仓库更像一个长期研究工程，而不是临时分析脚本集合。

---

## 优势与局限

### 优势

- 研究问题明确，不是泛化到一切波动率任务
- 从事件构造到回测结果有完整闭环
- 对数据泄漏、测试集污染、非标准合约排除等细节考虑较多
- 同时支持传统 ML 和序列模型
- 测试覆盖面广，协议约束清晰

### 局限

- 工程复杂度高，上手成本不低
- 依赖外部付费数据源，复现门槛较高
- 当前主要是 proxy execution，而不是真实订单簿级研究
- 部分深度模型依赖较重，对环境要求更高

---

## 推荐阅读顺序

如果是第一次接手，建议按下面顺序阅读：

1. `README.md`
2. `SPEC.md`
3. `justfile`
4. `src/earnings_event_vol/config.py`
5. `src/earnings_event_vol/cli.py`
6. `src/earnings_event_vol/data_pipeline.py`
7. `src/earnings_event_vol/trade_proxy.py`
8. `src/earnings_event_vol/features.py`
9. `src/earnings_event_vol/models.py`
10. `src/earnings_event_vol/backtest.py`
11. `src/earnings_event_vol/research.py`
12. `tests/test_protocol.py`

---

## 一句话总结

`earnings-event-vol` 的技术本质是：

> 一个围绕财报事件期权错价研究构建的、带有严格时间边界和代理交易建模的可复现量化研究平台。

它把“市场数据工程、事件定义、机器学习、序列建模、策略评估、研究报告”整合进了同一套协议里。
