"""case3 的分块、固定参数、动态规划和 C++ 生成。

阅读顺序：make_tiles → fixed_plan / search → generate。
每个输出块有一组 Traversal 参数，Plan.path 按块顺序保存这些参数。
"""

from dataclasses import dataclass
from itertools import combinations

from case3_model import (
    EMPTY_CACHE, M, N, P, Score, Tile, Traversal,
    evaluate, tile_trace, transition, validate_option,
)

STAGES = ("baseline", "blocked", "schedule", "cyclic", "lookahead")


def make_tiles(snake: bool = False) -> tuple[Tile, ...]:
    """生成 4×7 输出块；边界缩为 3 行或 6 列，蛇形时相邻列条带反向。"""
    row_starts = list(range(0, M, 4))
    col_starts = list(range(0, P, 7))
    output_tiles = []

    if not snake:
        for row in row_starts:
            for col in col_starts:
                output_tiles.append(Tile(row, col, min(4, M - row), min(7, P - col)))
    else:
        for stripe_index, col in enumerate(col_starts):
            row_order = row_starts if stripe_index % 2 == 0 else row_starts[::-1]
            for row in row_order:
                output_tiles.append(Tile(row, col, min(4, M - row), min(7, P - col)))

    return tuple(output_tiles)


@dataclass(frozen=True)
class Plan:
    """一个完整方案：块顺序和逐块对应的遍历参数。"""

    name: str
    tiles: tuple[Tile, ...]
    path: tuple[Traversal, ...]

    @property
    def score(self) -> Score:
        return evaluate(self.tiles, self.path)


# 以下是搜索得到的固定参数，不是缓存的性能数字。
# 每行对应一个输出块：(K 起点, 方向, 允许预读的 A 块内行号)。
# 空元组表示不预读；块的对应顺序由 make_tiles(snake=True) 定义。
SCHEDULE = (
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
    (40, -1, ()),
    (0, 1, ()),
)

CYCLIC = (
    (39, -1, ()),
    (40, 1, ()),
    (30, 1, ()),
    (33, -1, ()),
    (36, 1, ()),
    (22, 1, ()),
    (21, -1, ()),
    (22, 1, ()),
    (22, -1, ()),
    (23, 1, ()),
    (19, -1, ()),
    (19, 1, ()),
    (18, -1, ()),
    (19, 1, ()),
    (18, -1, ()),
    (18, 1, ()),
    (17, -1, ()),
    (18, 1, ()),
    (17, -1, ()),
    (19, 1, ()),
    (18, -1, ()),
    (22, 1, ()),
    (21, -1, ()),
    (22, 1, ()),
    (17, -1, ()),
    (14, 1, ()),
    (10, -1, ()),
    (11, 1, ()),
)

LOOKAHEAD = (
    (38, -1, (1,)),
    (40, 1, (1,)),
    (30, 1, (0,)),
    (33, -1, (0,)),
    (36, 1, (3,)),
    (22, 1, (0,)),
    (21, -1, (0, 1)),
    (22, 1, ()),
    (22, -1, (0,)),
    (23, 1, (2,)),
    (19, -1, (3,)),
    (18, 1, (2,)),
    (17, -1, (1,)),
    (18, 1, (0,)),
    (17, -1, (0,)),
    (18, 1, (0,)),
    (17, -1, (1,)),
    (18, 1, (3,)),
    (17, -1, (1,)),
    (19, 1, (2,)),
    (18, -1, (0, 1)),
    (22, 1, (0, 1, 2)),
    (21, -1, (0, 1, 2)),
    (22, 1, (0, 1, 3)),
    (17, -1, (0, 1)),
    (14, 1, (0, 1)),
    (10, -1, (0, 2, 3)),
    (11, 1, (1, 2, 3)),
)


def fixed_plan(name: str) -> Plan:
    """读取各阶段保存的遍历参数。"""
    if name not in STAGES[1:]:
        raise ValueError("baseline 直接使用仓库的朴素实现")

    block_order = make_tiles(snake=name != "blocked")
    if name == "blocked":
        path = (Traversal(0, 1),) * len(block_order)
    else:
        saved_parameters = {"schedule": SCHEDULE, "cyclic": CYCLIC, "lookahead": LOOKAHEAD}
        path = tuple(Traversal(*parameters) for parameters in saved_parameters[name])

    return Plan(name, block_order, path)


def traversal_choices(tile: Tile, stage: str):
    """逐阶段增加搜索自由度，后一个阶段包含前一阶段的参数候选。"""
    if stage == "schedule":
        k_orders = ((0, 1), (N - 1, -1))
    else:
        k_orders = ((start, direction) for start in range(N) for direction in (1, -1))

    rows, cols = tile.rows, tile.cols
    preload_limit = 0
    if stage == "lookahead":
        # C 累加值、当前 A 和 3 个辅助寄存器以外的空闲位置可用于预读。
        free_registers = 36 - (rows * cols + rows + 3)
        preload_limit = min(rows, 3, free_registers)

    for start, direction in k_orders:
        for count in range(preload_limit + 1):
            for preload_rows in combinations(range(rows), count):
                yield Traversal(start, direction, preload_rows)


def search(name: str) -> Plan:
    """求出固定块顺序和候选集合内 miss 最少的遍历参数。"""
    if name == "blocked":
        return fixed_plan(name)
    if name not in STAGES[2:]:
        raise ValueError("可搜索的阶段为 schedule、cyclic、lookahead")

    block_order = make_tiles(snake=True)
    # 状态表：cache 内容 -> (累计 miss, 到达该状态的逐块参数)。
    states = {EMPTY_CACHE: (0, ())}

    for tile in block_order:
        next_states = {}
        for traversal in traversal_choices(tile, name):
            summary = tile_trace(tile, traversal)
            for cache_before, (misses_so_far, path) in states.items():
                cache_after, added_misses = transition(cache_before, summary)
                candidate = misses_so_far + added_misses, path + (traversal,)

                # 后续成本只依赖 cache 内容。入口路径不同、出口状态相同时，
                # 只需保留累计 miss 更少的路径；平局时按参数顺序确定结果。
                previous = next_states.get(cache_after)
                if previous is None or candidate < previous:
                    next_states[cache_after] = candidate
        states = next_states

    # 各候选的访存总数相同，所以最小化 miss 就等价于最小化 latency。
    _, best_path = min(states.values())
    return Plan(name, block_order, best_path)


def make_macro(signature: str, body: list[str]) -> str:
    """给 C++ 宏的各行补上续行符。"""
    continuation = " \\"  # C++ 宏行末的反斜杠。
    return (continuation + "\n").join([f"#define {signature}", *body]) + "\n"


def make_products_macro(rows: int, cols: int) -> tuple[str, str]:
    """一个 K 的外积：A 行值已在寄存器中，依次读 B 并更新整个 C 块。"""
    name = f"CASE3_PRODUCTS_{rows}_{cols}"
    body = ["  do", "    {", "      reg b;"]
    for local_col in range(cols):
        body.append(
            f"      b = A[CASE3_B_OFFSET + (K) * case3_p + (COL) + {local_col}];"
        )
        for local_row in range(rows):
            accumulator = local_row * cols + local_col
            body.append(f"      sum[{accumulator}] += a[{local_row}] * b;")
    body.extend(["    }", "  while (false)"])
    return name, make_macro(f"{name}(COL, K)", body)


def preload_slot(local_row: int) -> str:
    """SAVE0… 是常量开关；前面开启的数量就是该行在 ahead 中的位置。"""
    if local_row == 0:
        return "0"
    return " + ".join(f"(SAVE{row})" for row in range(local_row))


def make_tile_macro(rows: int, cols: int, paired: bool) -> tuple[str, str]:
    """生成一种块形状的内核；paired 表示两轮 K 合并以支持预读。"""
    kind = "PAIR" if paired else "PLAIN"
    name = f"CASE3_{kind}_{rows}_{cols}"
    products = f"CASE3_PRODUCTS_{rows}_{cols}"
    iteration_step = 2 if paired else 1
    current_k = "CASE3_K (START, STEP, t)"
    next_k = "CASE3_K (START, STEP, t + 1)"

    body = [
        "  do", "    {", f"      reg sum[{rows * cols}];",
        f"      for (reg t = 0; t < case3_n; t += {iteration_step})",
        "        {", f"          reg a[{rows}];",
    ]
    if paired:
        body.append("          reg ahead[COUNT];")

    # 读当前 A；允许的行在同一 cache 行内顺便读出下一轮的值。
    for local_row in range(rows):
        body.append(f"          a[{local_row}] = A[CASE3_A (ROW, {local_row}, {current_k})];")
        if paired:
            condition = (
                f"(SAVE{local_row}) && "
                f"CASE3_SAME_LINE (ROW, {local_row}, {current_k}, {next_k})"
            )
            body.extend([
                f"          if (t + 1 < case3_n && {condition})",
                f"            ahead[{preload_slot(local_row)}]"
                f" = A[CASE3_A (ROW, {local_row}, {next_k})];",
            ])
    body.append(f"          {products} (COL, {current_k});")

    if paired:
        # N=41，最后一轮没有第二个 K；这里的判断不能省略。
        body.extend(["          if (t + 1 < case3_n)", "            {"])
        for local_row in range(rows):
            body.extend([
                f"              if ((SAVE{local_row}) && "
                f"CASE3_SAME_LINE (ROW, {local_row}, {current_k}, {next_k}))",
                f"                a[{local_row}] = ahead[{preload_slot(local_row)}];",
                "              else",
                f"                a[{local_row}] = A[CASE3_A (ROW, {local_row}, {next_k})];",
            ])
        body.extend([f"              {products} (COL, {next_k});", "            }"])
    body.append("        }")

    for local_row in range(rows):
        for local_col in range(cols):
            accumulator = local_row * cols + local_col
            body.append(
                f"      A[CASE3_C_OFFSET + ((ROW) + {local_row}) * case3_p"
                f" + (COL) + {local_col}] = sum[{accumulator}];"
            )
    body.extend(["    }", "  while (false)"])

    parameters = ["ROW", "COL", "START", "STEP"]
    if paired:
        parameters += ["COUNT"] + [f"SAVE{row}" for row in range(rows)]
    return name, make_macro(f"{name}({', '.join(parameters)})", body)


def generate(plan: Plan) -> str:
    """把方案翻译成只使用 reg/ptr_reg 的 gemm_case3；坐标和开关均为常量。"""
    if len(plan.tiles) != len(plan.path):
        raise ValueError("每个输出块都需要一组遍历参数")
    for tile, traversal in zip(plan.tiles, plan.path):
        validate_option(tile, traversal)

    definitions = [
        "#define CASE3_B_OFFSET (case3_m * case3_n)",
        "#define CASE3_C_OFFSET (CASE3_B_OFFSET + case3_n * case3_p)",
        "#define CASE3_K(START, STEP, T) (((START) + case3_n + (STEP) * (T)) % case3_n)",
        "#define CASE3_A(ROW, U, K) (((ROW) + (U)) * case3_n + (K))",
        "#define CASE3_SAME_LINE(ROW, U, K, NEXT) "
        "(CASE3_A (ROW, U, K) / 4 == CASE3_A (ROW, U, NEXT) / 4)",
    ]
    macro_names = [
        "CASE3_B_OFFSET", "CASE3_C_OFFSET", "CASE3_K", "CASE3_A", "CASE3_SAME_LINE",
    ]
    for rows, cols in sorted({(tile.rows, tile.cols) for tile in plan.tiles}):
        name, source = make_products_macro(rows, cols)
        definitions.append(source)
        macro_names.append(name)

    calls = []
    emitted_kernels = set()
    for tile, traversal in zip(plan.tiles, plan.path):
        row, col, rows, cols = tile
        start, direction, preload_rows = traversal
        paired = bool(preload_rows)
        kernel_shape = rows, cols, paired
        name = f"CASE3_{'PAIR' if paired else 'PLAIN'}_{rows}_{cols}"
        if kernel_shape not in emitted_kernels:
            _, source = make_tile_macro(rows, cols, paired)
            definitions.append(source)
            macro_names.append(name)
            emitted_kernels.add(kernel_shape)

        arguments = [row, col, start, direction]
        if paired:
            arguments.append(len(preload_rows))
            arguments.extend(int(local_row in preload_rows) for local_row in range(rows))
        calls.append(f"  {name} ({', '.join(map(str, arguments))});")

    # 统一用 A 作为基址，释放另三个指针占用的寄存器；块坐标展开成常量。
    function = [
        "void gemm_case3 (ptr_reg A, ptr_reg B, ptr_reg C, ptr_reg buffer)",
        "{", "  {",
        "    ptr_reg unused_buffer = std::move (buffer);",
        "    ptr_reg unused_B = std::move (B);",
        "    ptr_reg unused_C = std::move (C);",
        "  }",
        *calls, "}",
    ]
    cleanup = [f"#undef {name}" for name in macro_names]
    return "\n".join([*definitions, *function, *cleanup, ""])
