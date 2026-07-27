"""Gradio demo for SiliconLM. Runs fine on a free CPU Space.

Tabs: generate Verilog (real model if MODEL_PATH is set, otherwise a
labelled demo mode), verify code with Icarus Verilog, view the ablation
table from silicon_eval results, and ask questions over an indexed RAG
corpus if one exists.
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from silicon_eval.harness import check_candidate, extract_verilog  # noqa: E402

RESULTS_DIR = ROOT / "results"
PROBLEMS_DIR = ROOT / "silicon_eval" / "problems"
MODEL_PATH = os.environ.get("MODEL_PATH", "")
RAG_INDEX = ROOT / "rag" / "index"

_llm = None


def get_llm():
    global _llm
    if _llm is None and MODEL_PATH and Path(MODEL_PATH).exists():
        from llama_cpp import Llama
        _llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=os.cpu_count(),
                     verbose=False)
    return _llm


def demo_bank() -> dict[str, tuple[str, str]]:
    bank = {}
    for p in sorted(PROBLEMS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        bank[d["prompt"].splitlines()[0]] = (d["prompt"], d["reference"])
    return bank


DEMOS = demo_bank()


def generate(prompt: str) -> tuple[str, str]:
    llm = get_llm()
    if llm is not None:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=500)
        code = extract_verilog(out["choices"][0]["message"]["content"])
        return code, badge("SiliconLM · quantized GGUF · CPU", "gold")
    for title, (full, ref) in DEMOS.items():
        if prompt.strip() and prompt.strip().lower() in full.lower():
            return ref, badge("demo mode — verified reference (set MODEL_PATH "
                              "to serve the trained model)", "violet")
    first = next(iter(DEMOS.values()))
    return first[1], badge("demo mode — verified reference (set MODEL_PATH "
                           "to serve the trained model)", "violet")


def verify(code: str, testbench: str) -> str:
    if not code.strip() or not testbench.strip():
        return card("Paste a module and a testbench, or load the example.", "dim")
    res = check_candidate(extract_verilog(code), testbench)
    log = html.escape(res.log[-1500:])
    if res.passed:
        head = "<span class='pill pass'>PASS</span> compiled &amp; all tests passed"
    elif res.compiled:
        head = "<span class='pill fail'>FAIL</span> compiled, testbench failed"
    else:
        head = "<span class='pill fail'>COMPILE ERROR</span>"
    return f"<div class='verdict'>{head}</div><pre class='simlog'>{log}</pre>"


def load_example() -> tuple[str, str]:
    d = json.loads((PROBLEMS_DIR / "counter3.json").read_text())
    return d["reference"], d["testbench"]


STAGE_ORDER = ["scratch", "base", "dapt", "sft", "dpo", "reference", "buggy-baseline"]


def ablation_html() -> str:
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json"),
                    key=lambda p: next((i for i, s in enumerate(STAGE_ORDER)
                                        if s in p.stem), 99)):
        r = json.loads(f.read_text())
        p1 = r.get("pass_at", {}).get("pass@1", 0.0)
        rows.append((r["model_name"], r.get("compile_rate", 0.0), p1,
                     r.get("pass_at", {}).get("pass@5")))
    if not rows:
        return card("No results yet — run silicon_eval/run_eval.py on each "
                    "checkpoint; every JSON it writes appears here.", "dim")
    body = ""
    for name, cr, p1, p5 in rows:
        p5s = f"{p5:.2f}" if p5 is not None else "—"
        body += (
            f"<tr><td class='mname'>{html.escape(name)}</td>"
            f"<td>{cr:.2f}</td><td>{p1:.2f}</td><td>{p5s}</td>"
            f"<td class='barcell'><div class='bar' style='width:{max(p1*100,2):.0f}%'>"
            f"</div></td></tr>")
    return (
        "<table class='abl'><thead><tr><th>checkpoint</th><th>compile</th>"
        "<th>pass@1</th><th>pass@5</th><th>pass@1 (simulator-verified)</th>"
        f"</tr></thead><tbody>{body}</tbody></table>")


def ask_docs(q: str) -> str:
    if not q.strip():
        return card("Ask something about the indexed documentation.", "dim")
    if not (RAG_INDEX / "faiss.idx").exists():
        return card("No index yet — build one with rag/datasheet_rag.py build "
                    "--docs your_docs/. The engine refuses to answer without "
                    "grounded sources.", "dim")
    from rag.datasheet_rag import answer
    res = answer(RAG_INDEX, q)
    cites = "".join(
        f"<div class='cite'>[{html.escape(c['source'])} p.{c['chunk']}] "
        f"score {c['score']:.2f}</div>" for c in res["citations"])
    return (f"<div class='ragans'>{html.escape(res['answer'])}</div>{cites}")


def badge(text: str, tone: str) -> str:
    return f"<span class='badge {tone}'>{html.escape(text)}</span>"


def card(text: str, tone: str = "") -> str:
    return f"<div class='card {tone}'>{html.escape(text)}</div>"


CSS = """
@keyframes fadeUp { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
#hero, .tabitem { animation: fadeUp 0.35s ease-out; }

#hero { padding:20px 4px 8px; }
#hero h1 { margin:0; font-size:2rem; letter-spacing:-0.01em; }
#hero p { opacity:0.75; max-width:62ch; margin:8px 0 18px; line-height:1.55; }

.steps { display:flex; flex-wrap:wrap; gap:10px; }
.step { display:flex; align-items:center; gap:8px; padding:8px 14px;
  border-radius:999px; background:rgba(124,58,237,.08);
  border:1px solid rgba(124,58,237,.25); font-size:0.85rem; }
.step .num { display:inline-flex; align-items:center; justify-content:center;
  width:20px; height:20px; border-radius:50%; background:#7c3aed; color:#fff;
  font-size:0.72rem; font-weight:700; flex:none; }

.pill { font-weight:600; padding:3px 12px; border-radius:4px;
  margin-right:10px; font-size:0.85rem; }
.pill.pass { background:rgba(34,197,94,.12); color:#22c55e; border:1px solid #22c55e; }
.pill.fail { background:rgba(239,68,68,.12); color:#ef4444; border:1px solid #ef4444; }
.verdict { padding:14px 4px 8px; font-weight:500; }
.simlog { background:rgba(127,127,127,.08); border-radius:8px; padding:14px;
  max-height:260px; overflow:auto; white-space:pre-wrap; font-family:monospace; }
.badge { font-size:0.75rem; padding:4px 10px; border-radius:4px; }
.badge.gold { color:#b8860b; border:1px solid #b8860b; }
.badge.violet { color:#7c3aed; border:1px solid #7c3aed; }
.card { border:1px solid rgba(124,58,237,.2); background:rgba(124,58,237,.05);
  border-radius:10px; padding:16px 18px; transition:border-color .15s ease; }
.card.dim { opacity:0.85; }
table.abl { width:100%; border-collapse:collapse; font-size:0.9rem; }
table.abl th { text-align:left; font-weight:600; padding:10px 12px;
  border-bottom:1px solid rgba(127,127,127,.3); }
table.abl tr { transition:background-color .12s ease; }
table.abl tbody tr:hover { background:rgba(124,58,237,.06); }
table.abl td { padding:10px 12px; border-bottom:1px solid rgba(127,127,127,.2); }
td.barcell { width:38%; }
.bar { height:10px; border-radius:2px; background:linear-gradient(90deg,#7c3aed,#a78bfa);
  transition:width .4s ease; }
.ragans { padding:14px; border:1px solid rgba(127,127,127,.3); border-radius:8px;
  line-height:1.6; white-space:pre-wrap; }
.cite { font-size:0.8rem; opacity:0.75; margin-top:8px; }

button.primary, .gr-button-primary { transition:transform .12s ease, box-shadow .12s ease !important; }
button.primary:hover, .gr-button-primary:hover {
  transform:translateY(-1px); box-shadow:0 4px 14px rgba(124,58,237,.3); }
"""

HERO = """
<div id="hero">
  <h1>SiliconLM</h1>
  <p>Chip-design language model: from-scratch pretraining, domain adaptation,
  SFT, and DPO where the preference signal comes from running candidate
  Verilog through Icarus Verilog instead of a human or an LLM judge.</p>
  <div class="steps">
    <div class="step"><span class="num">1</span>Generate a spec into Verilog</div>
    <div class="step"><span class="num">2</span>Verify it compiles &amp; simulates</div>
    <div class="step"><span class="num">3</span>Compare checkpoints in Ablation</div>
    <div class="step"><span class="num">4</span>Ask docs, grounded &amp; cited</div>
  </div>
</div>
"""

with gr.Blocks(title="SiliconLM") as demo:
    gr.HTML(HERO)

    with gr.Tab("Generate"):
        gr.HTML("<div class='card dim'>No trained checkpoint is loaded yet, so "
                "this runs in demo mode: it matches your spec against 5 known "
                "problems and returns their verified reference solution. Try "
                "one of the examples below, or type your own spec once a "
                "model is available.</div>")
        prompt = gr.Textbox(label="Design spec", lines=4,
                            placeholder="Implement a 3-bit up-counter with "
                                        "synchronous reset and enable…")
        gr.Examples(
            examples=[
                ["Implement a 4-bit unsigned adder with carry out."],
                ["Implement a 2-to-1 multiplexer for 8-bit data."],
                ["Implement a D flip-flop with active-high asynchronous reset."],
                ["Implement a 3-bit up-counter with synchronous reset and enable."],
                ["Implement a 4-to-2 priority encoder with valid flag."],
            ],
            inputs=[prompt], label="Try one:",
        )
        gen_btn = gr.Button("Generate Verilog", variant="primary")
        gen_status = gr.HTML()
        gen_out = gr.Code(label="SiliconLM output", language=None)
        gen_btn.click(generate, prompt, [gen_out, gen_status])

    with gr.Tab("Verify"):
        gr.HTML("<div class='card dim'>Compiles and simulates a Verilog module "
                "against a testbench with Icarus Verilog — real simulation, not "
                "an LLM guessing whether the code is correct. New here? Click "
                "\"Load example\" to fill both boxes with a working pair, then "
                "hit \"Compile & simulate\".</div>")
        with gr.Row():
            code_in = gr.Code(label="Verilog module", language=None, lines=12)
            tb_in = gr.Code(label="Testbench (must print ALL_TESTS_PASSED)",
                            language=None, lines=12)
        with gr.Row():
            ex_btn = gr.Button("Load example", variant="primary")
            ver_btn = gr.Button("Compile & simulate", variant="primary")
        verdict = gr.HTML()
        ex_btn.click(load_example, None, [code_in, tb_in])
        ver_btn.click(verify, [code_in, tb_in], verdict)

    with gr.Tab("Ablation"):
        gr.HTML("<div class='card dim'>Nothing to type here — this table shows "
                "pass@1 / pass@5 for every model checkpoint that's been scored "
                "so far (silicon_eval/run_eval.py writes one JSON file per "
                "checkpoint into results/, and every row below is one of "
                "those files). Click Refresh after scoring a new checkpoint.</div>")
        abl = gr.HTML(ablation_html())
        gr.Button("Refresh").click(lambda: ablation_html(), None, abl)

    with gr.Tab("Ask docs"):
        gr.HTML("<div class='card dim'>Retrieval-augmented Q&amp;A over a small "
                "indexed set of sample documents (a fictional datasheet + a "
                "RISC-V basics doc). It refuses to answer anything not covered "
                "by the index instead of guessing — try an off-topic question "
                "to see that guardrail.</div>")
        q_in = gr.Textbox(label="Question about the indexed documentation",
                          placeholder="What is the absolute maximum VDD?")
        gr.Examples(
            examples=[
                ["What is the absolute maximum VDD?"],
                ["What is the recommended operating voltage?"],
                ["What privilege levels does RISC-V define?"],
                ["What are the standard RISC-V extensions?"],
                ["What is the capital of France?"],
            ],
            inputs=[q_in], label="Try one (the last one should be refused):",
        )
        rag_btn = gr.Button("Search", variant="primary")
        rag_out = gr.HTML()
        rag_btn.click(ask_docs, q_in, rag_out)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft(primary_hue="violet"),
                server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT", 7860)))
