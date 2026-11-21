#!/usr/bin/env python3
"""在 /tmp 中评测 case3 的五个优化阶段，输出访存开销对比。"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from case3_model import C_OFFSET, Score, baseline_words, summarize, tile_words
from case3_search import STAGES, Plan, fixed_plan, generate, search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATED_BASE_ADDRESS = 0x30000000
INT_BYTES = 4


@dataclass(frozen=True)
class Result:
    """正式框架的测量值；trace_hash 用于比较完整访存顺序。"""

    stage: str
    misses: int
    accesses: int
    latency: int
    registers: int
    trace_hash: str

    @property
    def score(self) -> Score:
        return Score(self.latency, self.misses, self.accesses)


def run(project_dir: Path, *command: str) -> str:
    """在临时项目目录运行命令，并把输出留在 commands.log 中。"""
    completed = subprocess.run(
        command, cwd=project_dir, text=True, capture_output=True, timeout=120,
    )
    output = completed.stdout + completed.stderr
    with (project_dir / "commands.log").open("a") as log:
        log.write(f"$ {' '.join(command)}\n{output}\n")
    if completed.returncode:
        raise RuntimeError(f"命令失败：{' '.join(command)}\n{output}")
    return output


def stage_project(project_dir: Path, source: str) -> None:
    """复制未修改的评测框架；之后所有编译都只影响这个临时副本。"""
    project_dir.mkdir()
    framework_files = (
        "Makefile", "common.h", "cachelab.h", "gemm.h", "matrix.h",
        "simulator.h", "matrix.cpp", "simulator.cpp", "gemm_baseline.cpp",
        "printTrace.cpp", "csim-ref",
    )
    for name in framework_files:
        shutil.copy2(PROJECT_ROOT / name, project_dir / name)
    shutil.copytree(
        PROJECT_ROOT / "test", project_dir / "test",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (project_dir / "gemm.cpp").write_text(source)


def replace_case3(source: str, replacement: str) -> str:
    """在源码字符串中替换 case3 及其宏，保留 case0、case1、case2。"""
    matches = list(re.finditer(r"\bvoid\s+gemm_case3\s*\(", source))
    if len(matches) != 1:
        raise ValueError("源码中应恰好有一个 gemm_case3 函数")

    function_start = matches[0].start()
    macro_definitions = list(re.finditer(r"(?m)^#define CASE3_\w+", source[:function_start]))
    section_start = macro_definitions[0].start() if macro_definitions else function_start

    # 按大括号层数找到函数结束位置，不能在第一个内层 } 处截断。
    opening_brace = source.index("{", function_start)
    brace_depth = 1
    section_end = opening_brace + 1
    while brace_depth:
        if section_end == len(source):
            raise ValueError("gemm_case3 的大括号未闭合")
        if source[section_end] == "{":
            brace_depth += 1
        elif source[section_end] == "}":
            brace_depth -= 1
        section_end += 1

    macro_cleanup = list(re.finditer(
        r"(?m)^#undef CASE3_\w+[^\n]*(?:\n|$)", source[section_end:],
    ))
    if macro_cleanup:
        section_end += macro_cleanup[-1].end()
    return source[:section_start] + replacement + source[section_end:]


def build(project_dir: Path, source: str) -> None:
    """调用原始 Makefile，执行代码合法性检查并编译当前阶段。"""
    (project_dir / "gemm.cpp").write_text(source)
    # 即使两次写文件的时间很接近，也不能复用前一个阶段的可执行文件。
    (project_dir / "printTrace").unlink(missing_ok=True)
    run(project_dir, "make", "printTrace")


def trace_digest(trace_path: Path) -> tuple[int, str]:
    """统计访问次数，同时对规范化的操作、地址、大小求哈希。"""
    digest = hashlib.sha256()
    accesses = 0
    with trace_path.open() as trace:
        for line in trace:
            fields = line.split()
            if len(fields) != 3 or fields[0] not in ("L", "S"):
                raise ValueError(f"无法识别的 trace 行：{line!r}")
            operation = fields[0]
            address, size = fields[1].split(",")
            # 寄存器编号可以不同；内存操作和访问顺序必须一致。
            normalized = f"{operation} {int(address, 16):x},{int(size)}\n"
            digest.update(normalized.encode())
            accesses += 1
    return accesses, digest.hexdigest()


def expected_digest(words) -> str:
    """把模型的 int 下标转换为与正式 trace 相同的字节地址格式。"""
    digest = hashlib.sha256()
    for word_index in words:
        # 分块算法只读 A/B、只写 C，因此可用 C 的起始下标区分 L/S。
        operation = "L" if word_index < C_OFFSET else "S"
        address = SIMULATED_BASE_ADDRESS + word_index * INT_BYTES
        digest.update(f"{operation} {address:x},{INT_BYTES}\n".encode())
    return digest.hexdigest()


def measure(project_dir: Path, stage_name: str, case: str = "case3") -> Result:
    """先运行正确性和寄存器检查，再用 csim-ref 统计 cache 行为。"""
    trace_path = project_dir.parent / f"{stage_name}.trace"
    with trace_path.open("w") as trace_file:
        completed = subprocess.run(
            [str(project_dir / "printTrace"), case], cwd=project_dir,
            text=True, stdout=trace_file, stderr=subprocess.PIPE, timeout=120,
        )
    (project_dir.parent / f"{stage_name}.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"{case} 运行失败：{completed.stderr}")

    register_match = re.search(r"Pass using (\d+) regs", completed.stderr)
    if not register_match or int(register_match[1]) > 36:
        raise RuntimeError(f"{case} 没有通过寄存器检查")
    registers = int(register_match[1])

    # 清除旧结果，避免误读上一次 csim-ref 生成的统计。
    results_path = project_dir / ".csim_results"
    results_path.unlink(missing_ok=True)
    run(project_dir, "./csim-ref", "-s", "5", "-E", "1", "-b", "4", "-t", str(trace_path))
    _, misses, _ = map(int, results_path.read_text().split())
    accesses, digest = trace_digest(trace_path)
    latency = 15 * misses + accesses
    return Result(stage_name, misses, accesses, latency, registers, digest)


def check_model(result: Result, words, expected_score: Score) -> None:
    """总数相同还不够：也要检查整条访存序列是否与模型一致。"""
    if result.score != expected_score:
        raise RuntimeError(
            f"{result.stage} 的模型与实测不符：模型 {expected_score}，实测 {result.score}"
        )
    if result.trace_hash != expected_digest(words):
        raise RuntimeError(f"{result.stage} 的实际访存顺序与模型不符")


def verify_plan(project_dir: Path, source: str, plan: Plan) -> Result:
    """生成一个阶段，保存复现材料，再正式编译和测量。"""
    output_dir = project_dir.parent
    candidate_source = replace_case3(source, generate(plan))
    (output_dir / f"{plan.name}.cpp").write_text(candidate_source)
    parameters = {"tiles": plan.tiles, "path": plan.path}
    (output_dir / f"{plan.name}.json").write_text(json.dumps(parameters, indent=2) + "\n")

    build(project_dir, candidate_source)
    result = measure(project_dir, plan.name)
    words = (
        word
        for tile, traversal in zip(plan.tiles, plan.path)
        for word in tile_words(tile, traversal)
    )
    check_model(result, words, plan.score)
    return result


def print_results(results: list[Result], submitted: Result, baseline: Result) -> None:
    print("\n| Stage | Cache miss | Register miss | Latency | Speedup | Registers |")
    print("|---|---:|---:|---:|---:|---:|")
    for result in [*results, submitted]:
        speedup = baseline.latency / result.latency
        print(
            f"| {result.stage} | {result.misses} | {result.accesses} | {result.latency} | "
            f"{speedup:.4f} | {result.registers} |"
        )


def reproduce(work_dir: Path, selected: tuple[str, ...], search_stage: str | None = None):
    """先核对提交版本，再逐阶段复现；不写回工作区源码。"""
    source = (PROJECT_ROOT / "gemm.cpp").read_text()
    project_dir = work_dir / "project"
    stage_project(project_dir, source)
    build(project_dir, source)

    submitted = measure(project_dir, "submitted")
    for case_index in range(3):
        measure(project_dir, f"submitted_case{case_index}", f"case{case_index}")

    baseline = measure(project_dir, "baseline", "case3_baseline")
    baseline_summary = summarize(baseline_words())
    baseline_score = Score(
        15 * baseline_summary.cold_misses + baseline_summary.accesses,
        baseline_summary.cold_misses,
        baseline_summary.accesses,
    )
    check_model(baseline, baseline_words(), baseline_score)

    results = []
    for stage_name in selected:
        if stage_name == "baseline":
            results.append(baseline)
            continue

        plan = search(stage_name) if stage_name == search_stage else fixed_plan(stage_name)
        print(f"核对 {stage_name}：模型预测 latency = {plan.score.latency}", flush=True)
        result = verify_plan(project_dir, source, plan)
        if stage_name == "lookahead" and result.trace_hash != submitted.trace_hash:
            raise RuntimeError("最终阶段与当前提交的 case3 访存序列不一致")
        results.append(result)

    print_results(results, submitted, baseline)
    if "lookahead" in selected:
        print("\n最终阶段与当前提交的 case3 操作、地址和大小序列完全一致。")
    print("当前提交的 case0–case3 均通过正确性和寄存器检查。")
    result_records = [asdict(result) for result in [*results, submitted]]
    (work_dir / "results.json").write_text(json.dumps(result_records, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", choices=STAGES,
                        help="只复现指定阶段，可多次指定；默认复现全部五个阶段")
    parser.add_argument("--search", choices=STAGES[2:],
                        help="重新搜索该阶段的参数，而不是读取固定参数")
    parser.add_argument("--work-dir", type=Path,
                        help="/tmp 下尚不存在的输出目录；默认自动创建")
    args = parser.parse_args()

    selected = tuple(name for name in STAGES if not args.stage or name in args.stage)
    if args.search and args.search not in selected:
        parser.error("--search 指定的阶段必须包含在 --stage 中")

    if args.work_dir:
        work_dir = args.work_dir.resolve()
        temporary_root = Path("/tmp").resolve()
        if not work_dir.is_relative_to(temporary_root) or work_dir == temporary_root:
            parser.error("--work-dir 必须是 /tmp 下的新目录")
        if work_dir.exists():
            parser.error("--work-dir 已存在，请选择新目录")
        work_dir.mkdir(parents=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="cachelab-case3-", dir="/tmp"))

    print(f"编译、生成代码、trace 和结果均保存在：{work_dir}", flush=True)
    reproduce(work_dir, selected, args.search)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        raise SystemExit(str(error)) from error
