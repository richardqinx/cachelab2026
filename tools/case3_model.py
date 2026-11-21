"""描述 case3 的访存顺序，并模拟实验规定的直接映射 cache。

地址统一用「相对 A 基址的 int 下标」表示。每条 cache 行有 4 个 int，
共 32 组；实际字节地址由复现脚本转换。
"""

from functools import lru_cache
from typing import NamedTuple

M, N, P = 27, 41, 27
B_OFFSET = M * N
C_OFFSET = B_OFFSET + N * P
CACHE_SETS = 32
WORDS_PER_LINE = 4
EMPTY_CACHE = (-1,) * CACHE_SETS


class Tile(NamedTuple):
    """C 中一个输出块的位置、行数和列数。"""

    row: int
    col: int
    rows: int
    cols: int


class Traversal(NamedTuple):
    """一个输出块的 K 起点、方向，以及允许提前读取的 A 块内行号。"""

    start: int
    direction: int
    preload_rows: tuple[int, ...] = ()


class Score(NamedTuple):
    latency: int
    misses: int
    accesses: int


class CacheSummary(NamedTuple):
    """一段访存对 cache 的影响；-1 表示该组没有被访问。"""

    first_lines: tuple[tuple[int, int], ...]  # (组号，该组首次访问的行号)
    final_lines: tuple[int, ...]             # 访存结束后，各组留下的行号
    cold_misses: int                        # 从空 cache 开始执行的 miss 数
    accesses: int


def validate_option(tile: Tile, traversal: Traversal) -> None:
    """只接受覆盖完整 K 维度、且不超过寄存器预算的方案。"""
    row, col, rows, cols = tile
    start, direction, preload_rows = traversal

    if not (0 <= row < row + rows <= M and 0 <= col < col + cols <= P):
        raise ValueError("输出块超出 C 的范围")
    # 此工具只搜索正扫与反扫，两者都恰好访问每个 K 一次。
    if not (0 <= start < N and direction in (-1, 1)):
        raise ValueError("K 起点必须在范围内，方向必须为 1 或 -1")
    if tuple(sorted(set(preload_rows))) != preload_rows:
        raise ValueError("预读行号必须按升序排列，且不能重复")
    if any(local_row not in range(rows) for local_row in preload_rows):
        raise ValueError("预读行号超出输出块的范围")

    # 同时存活：C 累加值、当前 A、预读 A，以及 A 基址、循环变量 t、一个 B 值。
    register_count = rows * cols + rows + len(preload_rows) + 3
    if register_count > 36:
        raise ValueError("方案超过 36 个寄存器的限制")


def baseline_words():
    """朴素 i-j-k 实现：每个 C 元素都单独累加。"""
    for row in range(M):
        for col in range(P):
            for k in range(N):
                yield row * N + k
                yield B_OFFSET + k * P + col
            yield C_OFFSET + row * P + col


def tile_words(tile: Tile, traversal: Traversal):
    """产生单个输出块的访存序列；预读只提前一次 load，不增加 load 总数。"""
    validate_option(tile, traversal)
    row, col, rows, cols = tile
    start, direction, preload_rows = traversal

    # 每轮处理两个 K；N=41，最后一轮只能处理第一个。
    for iteration in range(0, N, 2):
        k = (start + direction * iteration) % N
        next_k = (k + direction) % N
        has_next = iteration + 1 < N
        saved_rows = set()

        for local_row in range(rows):
            address = (row + local_row) * N + k
            next_address = (row + local_row) * N + next_k
            yield address

            same_cache_line = address // WORDS_PER_LINE == next_address // WORDS_PER_LINE
            if has_next and local_row in preload_rows and same_cache_line:
                yield next_address
                saved_rows.add(local_row)

        yield from range(B_OFFSET + k * P + col, B_OFFSET + k * P + col + cols)

        if has_next:
            for local_row in range(rows):
                # 已预读的值保留在寄存器中，下一轮不再访问内存。
                if local_row not in saved_rows:
                    yield (row + local_row) * N + next_k
            yield from range(B_OFFSET + next_k * P + col, B_OFFSET + next_k * P + col + cols)

    # 整个 K 循环结束，才把寄存器中的 C 块写回。
    for local_row in range(rows):
        yield from range(C_OFFSET + (row + local_row) * P + col,
                         C_OFFSET + (row + local_row) * P + col + cols)


def summarize(words) -> CacheSummary:
    """把一段访存压缩为首行、末行和计数，避免搜索时反复重放整段 trace。"""
    first_lines = [-1] * CACHE_SETS
    final_lines = [-1] * CACHE_SETS
    misses = 0
    accesses = 0

    for word_index in words:
        cache_line = word_index // WORDS_PER_LINE
        set_index = cache_line % CACHE_SETS
        accesses += 1
        if final_lines[set_index] != cache_line:
            misses += 1
            if first_lines[set_index] == -1:
                first_lines[set_index] = cache_line
            final_lines[set_index] = cache_line

    touched_sets = tuple(
        (set_index, cache_line)
        for set_index, cache_line in enumerate(first_lines)
        if cache_line != -1
    )
    return CacheSummary(touched_sets, tuple(final_lines), misses, accesses)


def transition(cache_before: tuple[int, ...], summary: CacheSummary):
    """把摘要应用到已有 cache，返回新状态与本段实际 miss 数。"""
    # 对直接映射 cache，每组只有第一次访问受入口状态影响。
    # 若入口已经有首次要读的行，就能比「从空 cache 开始」少一次 miss。
    initial_hits = sum(
        cache_before[set_index] == cache_line
        for set_index, cache_line in summary.first_lines
    )
    misses = summary.cold_misses - initial_hits

    # 未被访问的组必须保留原内容，不能被摘要中的 -1 清空。
    if len(summary.first_lines) == CACHE_SETS:
        cache_after = summary.final_lines
    else:
        cache_after = tuple(
            old_line if new_line == -1 else new_line
            for old_line, new_line in zip(cache_before, summary.final_lines)
        )
    return cache_after, misses


@lru_cache(maxsize=12000)
def tile_trace(tile: Tile, traversal: Traversal) -> CacheSummary:
    """相同块和参数会被多次比较，缓存其摘要即可。"""
    return summarize(tile_words(tile, traversal))


def evaluate(tiles: tuple[Tile, ...], traversals: tuple[Traversal, ...]) -> Score:
    """按块顺序计算总成本；不考虑算术和分支指令的耗时。"""
    if len(tiles) != len(traversals):
        raise ValueError("每个输出块都需要一组遍历参数")

    cache_state = EMPTY_CACHE
    total_misses = 0
    total_accesses = 0
    for tile, traversal in zip(tiles, traversals):
        summary = tile_trace(tile, traversal)
        cache_state, misses = transition(cache_state, summary)
        total_misses += misses
        total_accesses += summary.accesses

    return Score(15 * total_misses + total_accesses, total_misses, total_accesses)
