"""Compile+simulate candidate Verilog with Icarus Verilog and check the testbench passed."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PASS_TOKEN = "ALL_TESTS_PASSED"
COMPILE_TIMEOUT_S = 20
SIM_TIMEOUT_S = 30


@dataclass
class Problem:
    pid: str
    prompt: str
    module_header: str
    testbench: str
    reference: str = ""

    @classmethod
    def load(cls, path: Path) -> "Problem":
        raw = json.loads(Path(path).read_text())
        return cls(**raw)


@dataclass
class CandidateResult:
    compiled: bool
    passed: bool
    log: str = ""


@dataclass
class EvalReport:
    model_name: str
    n_problems: int
    n_samples: int
    compile_rate: float
    pass_at: dict = field(default_factory=dict)
    per_problem: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


def extract_verilog(text: str) -> str:
    """Strip markdown fences / chatter, return raw text if no module found."""
    fence = re.search(r"```(?:verilog|systemverilog|v)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    mod = re.search(r"(module\b.*?endmodule)", text, re.S)
    return (mod.group(1) if mod else text).strip()


def check_candidate(code: str, testbench: str, workdir: Path | None = None) -> CandidateResult:
    """Compile candidate + testbench with iverilog, then simulate with vvp."""
    if shutil.which("iverilog") is None:
        raise RuntimeError(
            "iverilog not found. Install it first (apt-get install iverilog)."
        )
    tmp = Path(tempfile.mkdtemp(dir=workdir))
    try:
        (tmp / "dut.v").write_text(code)
        (tmp / "tb.v").write_text(testbench)
        out = tmp / "sim.out"
        try:
            comp = subprocess.run(
                ["iverilog", "-g2012", "-o", str(out), str(tmp / "dut.v"), str(tmp / "tb.v")],
                capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return CandidateResult(False, False, "compile timeout")
        if comp.returncode != 0:
            return CandidateResult(False, False, comp.stderr[-2000:])
        try:
            sim = subprocess.run(
                ["vvp", str(out)], capture_output=True, text=True, timeout=SIM_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return CandidateResult(True, False, "simulation timeout")
        passed = PASS_TOKEN in sim.stdout
        return CandidateResult(True, passed, (sim.stdout + sim.stderr)[-2000:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def evaluate(
    model_name: str,
    generate_fn,
    problems: list[Problem],
    n_samples: int = 5,
    ks: tuple[int, ...] = (1, 5),
) -> EvalReport:
    """generate_fn(prompt, n) -> list[str] of n completions for that prompt."""
    per_problem: dict = {}
    total, compiled_total = 0, 0
    for prob in problems:
        raw = generate_fn(prob.prompt, n_samples)
        results = [check_candidate(extract_verilog(r), prob.testbench) for r in raw]
        c = sum(r.passed for r in results)
        compiled = sum(r.compiled for r in results)
        total += len(results)
        compiled_total += compiled
        per_problem[prob.pid] = {
            "n": len(results),
            "compiled": compiled,
            "correct": c,
            "pass_at": {f"pass@{k}": round(pass_at_k(len(results), c, k), 4)
                        for k in ks if k <= len(results)},
        }
    agg = {
        f"pass@{k}": round(
            sum(p["pass_at"].get(f"pass@{k}", 0.0) for p in per_problem.values())
            / max(len(per_problem), 1), 4)
        for k in ks if k <= n_samples
    }
    return EvalReport(
        model_name=model_name,
        n_problems=len(problems),
        n_samples=n_samples,
        compile_rate=round(compiled_total / max(total, 1), 4),
        pass_at=agg,
        per_problem=per_problem,
    )


def load_problems(problems_dir: Path) -> list[Problem]:
    return [Problem.load(p) for p in sorted(Path(problems_dir).glob("*.json"))]
