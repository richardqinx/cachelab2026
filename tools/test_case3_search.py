"""验证模型与搜索的正确性；加 --integration 后，在 /tmp 正式复现全部阶段。"""

from collections import Counter
import itertools
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from case3_model import (
    B_OFFSET, C_OFFSET, EMPTY_CACHE, M, N, P,
    Score, Tile, Traversal, baseline_words, summarize,
    tile_words, transition, validate_option,
)
from case3_search import (
    STAGES, Plan, fixed_plan, generate, make_tiles, search, traversal_choices,
)
from reproduce_case3 import reproduce, replace_case3

RUN_INTEGRATION = "--integration" in sys.argv
if RUN_INTEGRATION:
    sys.argv.remove("--integration")


def replay(words, cache_before=EMPTY_CACHE):
    """独立逐条模拟访存，不使用待测的 cache 摘要算法。"""
    cache_lines = list(cache_before)
    misses = 0
    accesses = 0
    for word_index in words:
        cache_line = word_index // 4
        set_index = cache_line % 32
        if cache_lines[set_index] != cache_line:
            misses += 1
            cache_lines[set_index] = cache_line
        accesses += 1
    return tuple(cache_lines), misses, accesses


class ModelTests(unittest.TestCase):
    def test_summary_preserves_partial_cache_states(self):
        """摘要必须保留未访问组的内容，并给出与逐条模拟相同的 miss。"""
        random_source = random.Random(2026)
        for _ in range(100):
            prefix = [random_source.randrange(3000)
                      for _ in range(random_source.randrange(100))]
            words = [random_source.randrange(3000)
                     for _ in range(random_source.randrange(100))]
            cache_before, _, _ = replay(prefix)
            cache_after, misses, accesses = replay(words, cache_before)

            summary = summarize(words)
            self.assertEqual(transition(cache_before, summary), (cache_after, misses))
            self.assertEqual(summary.accesses, accesses)

    def test_k_coverage_and_preload_preserve_all_accesses(self):
        """检查四种边界块：每个输入元素的读取次数和 C 的写回次数都不能改变。"""
        edge_tiles = (
            Tile(0, 0, 4, 7), Tile(24, 0, 3, 7),
            Tile(0, 21, 4, 6), Tile(24, 21, 3, 6),
        )
        for tile in edge_tiles:
            row, col, rows, cols = tile
            expected_accesses = Counter()
            for k in range(N):
                expected_accesses.update((row + local_row) * N + k
                                         for local_row in range(rows))
                expected_accesses.update(B_OFFSET + k * P + col + local_col
                                         for local_col in range(cols))
            expected_accesses.update(
                C_OFFSET + (row + local_row) * P + col + local_col
                for local_row in range(rows) for local_col in range(cols)
            )

            for traversal in traversal_choices(tile, "lookahead"):
                self.assertEqual(Counter(tile_words(tile, traversal)), expected_accesses)

    def test_invalid_options_are_rejected(self):
        tile = Tile(0, 0, 4, 7)
        invalid_parameters = (
            Traversal(0, 0), Traversal(0, 41), Traversal(41, 1),
            Traversal(0, 1, (4,)),       # 块内没有第 4 行。
            Traversal(0, 1, (0, 1)),    # 4×7 块最多只能预读一行。
        )
        for traversal in invalid_parameters:
            with self.subTest(traversal=traversal), self.assertRaises(ValueError):
                validate_option(tile, traversal)

    def test_fixed_stages_cover_c_once_and_improve(self):
        """独立检查固定参数，不能只验证搜索代码自己计算的 score。"""
        baseline = summarize(baseline_words())
        previous_latency = 15 * baseline.cold_misses + baseline.accesses

        for stage_name in STAGES[1:]:
            plan = fixed_plan(stage_name)
            covered_elements = Counter(
                (row + local_row, col + local_col)
                for row, col, rows, cols in plan.tiles
                for local_row in range(rows) for local_col in range(cols)
            )
            self.assertEqual(covered_elements, Counter(itertools.product(range(M), range(P))))

            words = (
                word for tile, traversal in zip(plan.tiles, plan.path)
                for word in tile_words(tile, traversal)
            )
            _, misses, accesses = replay(words)
            self.assertEqual(plan.score, Score(15 * misses + accesses, misses, accesses))
            self.assertEqual(accesses, 12906)
            self.assertLess(plan.score.latency, previous_latency)
            previous_latency = plan.score.latency

    def test_direction_search_against_exhaustive_choices(self):
        """取三个真实输出块，穷举所有方向组合，核对动态规划的最优值。"""
        block_order = make_tiles(snake=True)[:3]
        candidates = [
            tuple(traversal_choices(tile, "schedule")) for tile in block_order
        ]
        expected = min(
            Plan("schedule", block_order, path).score
            for path in itertools.product(*candidates)
        )

        with patch("case3_search.make_tiles", return_value=block_order):
            self.assertEqual(search("schedule").score, expected)

    def test_replacement_preserves_surrounding_code(self):
        """替换整个 case3 区段，不能误删其他函数或遗留旧宏。"""
        source = (
            "void gemm_case2() {}\n#define CASE3_OLD 1\n"
            "void\ngemm_case3() { { } }\n#undef CASE3_OLD\n// suffix\n"
        )
        result = replace_case3(source, "void gemm_case3() {}\n")
        self.assertEqual(result, "void gemm_case2() {}\nvoid gemm_case3() {}\n// suffix\n")
        self.assertEqual(result.count("gemm_case3"), 1)
        self.assertNotIn("CASE3_OLD", result)
        self.assertIn("CASE3_PRODUCTS", generate(fixed_plan("lookahead")))


@unittest.skipUnless(RUN_INTEGRATION, "加 --integration 才运行正式编译和评测")
class OfficialTests(unittest.TestCase):
    def test_all_stages_match_reference_and_submission(self):
        with tempfile.TemporaryDirectory(prefix="case3-tests-", dir="/tmp") as directory:
            results = reproduce(Path(directory), STAGES)
            self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
