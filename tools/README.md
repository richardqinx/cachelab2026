# Case3 矩阵乘法优化实验

针对 27×41 与 41×27 矩阵相乘，依次采用寄存器分块、蛇形遍历、K 循环
起点选择和同 cache 行提前读取，比较各阶段的访存开销。

| 阶段 | 本阶段增加的设计 | cache miss | register miss | latency | speedup |
|---|---|---:|---:|---:|---:|
| baseline | 仓库的朴素矩阵乘法 | 36869 | 60507 | 613542 | 1.0000 |
| blocked | 4×7 寄存器分块；边界使用 3 行或 6 列；按行块顺序访问，K 正扫 | 4462 | 12906 | 79836 | 7.6850 |
| schedule | 固定列块、蛇形遍历行块；为每块选择 K 正扫或反扫 | 4171 | 12906 | 75471 | 8.1295 |
| cyclic | 同一块顺序下，允许 K 从任意位置起步再循环 | 4131 | 12906 | 74871 | 8.1947 |
| lookahead | 同一分块与块顺序下，允许提前读取下一轮同 cache 行的 A 元素 | 4086 | 12906 | 74196 | 8.2692 |

分块阶段将当前输出块的累加值保存在寄存器中，每次保留 A 的当前行值，逐项
加载 B。后续阶段沿用相同的块形状，逐步调整块顺序、K 遍历参数和预读行。
各阶段分别优化其允许的参数组合。

## 实验复现

在项目根目录执行：

```bash
python3 -B tools/reproduce_case3.py
```

默认读取已保存的遍历参数。输出目录为
`/tmp/cachelab-case3-...`，包含生成的 C++、参数 JSON、trace、评测日志和
`results.json`。编译和评测在该临时目录中进行。

程序使用仓库原始 Makefile 进行合法性检查和编译，由 `printTrace` 检查计算结果、
输入矩阵是否保持不变以及 36 个寄存器限制，再由 `csim-ref` 计算 cache miss。
同时检查：

- 每个阶段的实际 miss 和访问次数都与地址模型一致；
- 每条访问的操作类型、地址和大小与模型一致；
- 最终阶段与 `gemm.cpp` 中 case3 的访存序列完全一致
  （寄存器编号不参与比较）；
- `gemm.cpp` 的 case0、case1、case2 通过框架的正确性与寄存器检查。

指定实验阶段：

```bash
python3 -B tools/reproduce_case3.py --stage cyclic --stage lookahead
```

指定输出位置时，该目录必须尚不存在且位于 `/tmp`：

```bash
python3 -B tools/reproduce_case3.py --work-dir /tmp/cachelab-report-run
```

## 参数搜索

`case3_search.py` 使用动态规划搜索遍历参数。重新搜索指定阶段：

```bash
python3 -B tools/reproduce_case3.py --stage schedule --search schedule
python3 -B tools/reproduce_case3.py --stage cyclic --search cyclic
python3 -B tools/reproduce_case3.py --stage lookahead --search lookahead
```

后三个阶段共用固定的 28 个输出块顺序；每块可以选择：

- schedule：K 从 0 正扫，或从 40 反扫；
- cyclic：41 个起点 × 2 个方向；
- lookahead：上述起点和方向，再选择寄存器预算允许的预读行集合，最多预读 3 行。
  空集合也在搜索范围内，故包含 cyclic 的候选。

每个块保留各 cache 组首次访问和最后留下的行。对于相同的块后 cache 状态，
动态规划只保留 miss 最少的前缀，求得固定块顺序和上述候选集合内的最优参数。

## 访存开销分析

分块将 C 的部分和一直留在寄存器中，直到完成整个 K 循环才写回。共有 4 个列块、
7 个行块，每个输入值在相应块中只读一次，因此：

```text
register miss = 41 × (4 × 27 + 7 × 27) + 27 × 27 = 12906
latency = 15 × cache miss + register miss
```

后续阶段保持访问总数不变，通过调整访问顺序减少 cache miss。提前读取将
下一轮必需的 load 移到当前轮，并把值保存在寄存器中。判断两个 A 元素是否
位于同一 cache 行，需比较完整字下标除以 4 的商，因为 A 的行跨度为 41。

4×7 块的 28 个累加值、4 个 A 值、一个 B 值、A 基址和循环变量共占 35 个寄存器，
最后一个寄存器可用于提前读取。较小的边界块有更多空闲寄存器。

## 文件说明

- `reproduce_case3.py`：唯一命令入口；暂存项目、正式评测、逐条 trace 核对。
- `case3_model.py`：访存顺序、cache 状态摘要、寄存器与 K 遍历约束。
- `case3_search.py`：五阶段定义、固定参数、动态规划和 C++ 生成。
- `test_case3_search.py`：独立逐条模拟、完整 K 覆盖、预读访问守恒等检查。

在项目根目录运行模型检查，或增加正式框架验证：

```bash
python3 -B tools/test_case3_search.py
python3 -B tools/test_case3_search.py --integration
```

正式编译与评测的工作目录位于 `/tmp`。

需要 Python 3、g++、make 和仓库提供的可执行 `csim-ref`，不需要第三方 Python 包。
