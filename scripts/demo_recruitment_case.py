"""CPU recruitment demo: saved search tradeoffs and optional real public BM25.

No training, model inference, benchmark labels, network, or cloud dependency.
Historical profiles are replayed, never represented as live LTR/dense search.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CORPUS_SHA = "c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71"
MANUAL_QUERIES = (
    "Arctic sea ice decline",
    "carbon dioxide greenhouse effect",
    "zxqv_nonexistent_term_20260927",
)


def load(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def historical_summary(root: Path = ROOT) -> dict[str, Any]:
    """Project only known aggregate fields; no generic dump of source records."""
    search = load(root / "docs/verified-runs/search-tradeoffs-20260927.json")
    training = load(
        root / "docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json"
    )
    behavior = load(root / "docs/verified-runs/fair-three-arm-behavior-20261003.json")
    profiles = []
    for name in ("ltr", "rerank20", "rerank100"):
        item = search["profiles"][name]
        profiles.append(
            {
                "profile": name,
                "recall_at_5": item["metrics"]["recall@5"],
                "evidence_f1_at_5": item["metrics"]["evidence_f1@5"],
                "p95_ms": item["timings"]["end_to_end_ms"]["p95"],
                "torch_allocated_gib": item["peak_torch_allocated_bytes"] / 1024**3,
            }
        )
    comparison = search["paired_vs_ltr"]["rerank100"]["paired_bootstrap"]
    evaluation = training["evaluation"]
    autonomous = behavior["routes"]["autonomous"]
    return {
        "mode": "saved_aggregate_replay_not_live_dense_ltr_or_verdict",
        "training": {
            "scope": "restricted offline dev; not test or pretraining",
            "documents": evaluation["document_count"],
            "queries": evaluation["claim_count"],
            "steps": training["adapter"]["training_steps"],
            "base_recall_at_5": evaluation["base"]["recall_at_5"],
            "adapted_recall_at_5": evaluation["adapted"]["recall_at_5"],
        },
        "search": {
            "scope": "126 public validation queries; serial warmed offline timing",
            "live_ltr_assets_available": False,
            "selected_offline_candidate": search["decision"]["default_candidate"],
            "profiles": profiles,
            "rerank100_minus_ltr_recall_ci": [
                comparison["recall@5"]["ci_lower"],
                comparison["recall@5"]["ci_upper"],
            ],
            "rerank100_minus_ltr_f1_ci": [
                comparison["evidence_f1@5"]["ci_lower"],
                comparison["evidence_f1@5"]["ci_upper"],
            ],
            "decision": "LTR is the offline low-latency candidate; no significant Top5 superiority, equivalence, online SLA or money saving claimed",
        },
        "agent": {
            "scope": "32-task fair development comparison, not independent test",
            "stop": autonomous["validated_actions"].get("stop", 0),
            "acquire": autonomous["validated_actions"].get("acquire", 0),
            "decision": "do not promote autonomous acquisition; semantic support unmeasured by the original experiment",
        },
    }


def public_packets(
    evidence: Path | None,
    root: Path = ROOT,
    *,
    claim: str | None = None,
    candidate_k: int = 10,
    top_k: int = 3,
) -> dict[str, Any]:
    if not 1 <= top_k <= candidate_k <= 100:
        raise ValueError("require 1 <= top_k <= candidate_k <= 100")
    if claim is not None:
        claim = claim.strip()
        if not claim or len(claim) > 2000:
            raise ValueError("claim must contain 1 to 2000 characters")
    if evidence is None:
        if claim is not None or candidate_k != 10 or top_k != 3:
            raise ValueError(
                "new claims or ranking settings require --evidence, not saved replay"
            )
        return load(root / "docs/verified-runs/recruitment-public-demo-20261003.json")
    started = perf_counter()
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if digest != PUBLIC_CORPUS_SHA:
        raise ValueError("only the registered public evidence corpus is accepted")
    validated = perf_counter()
    # Import no dense encoder, reranker, generator, LangSmith or model provider.
    from climate_rag.bm25 import BM25Index
    from climate_rag.io import iter_evidence

    load_started = perf_counter()
    documents = list(iter_evidence(evidence))
    if len(documents) != 5240:
        raise ValueError("unexpected public corpus size")
    loaded = perf_counter()
    index = BM25Index().fit(documents)
    indexed = perf_counter()
    provenance = {doc.evidence_id: dict(doc.metadata) for doc in documents}
    setup_finished = perf_counter()
    cases: list[dict[str, Any]] = []
    queries = (claim,) if claim is not None else MANUAL_QUERIES
    for query in queries:
        query_started = perf_counter()
        candidates = index.search(query, top_k=candidate_k)
        searched = perf_counter()
        ranked = candidates[:top_k]
        items = [
            {
                "evidence_id": item.evidence_id,
                "text": item.text,
                "retrieval": {
                    "rank": item.rank,
                    "score": item.score,
                    "route": "bm25",
                },
                "provenance": provenance[item.evidence_id],
                "text_sha256": hashlib.sha256(item.text.encode()).hexdigest(),
            }
            for item in ranked
        ]
        candidates_summary = [
            {"evidence_id": item.evidence_id, "rank": item.rank, "score": item.score}
            for item in candidates
        ]
        packaged = perf_counter()
        cases.append(
            {
                "claim_text": query,
                "status": "evidence_found" if ranked else "empty_result",
                "answer": None,
                "candidates": candidates_summary,
                "selection": "BM25 score descending, evidence_id tie break; no second-stage reranker",
                "items": items,
                "timings_ms": {
                    "search_and_rank": (searched - query_started) * 1000,
                    "evidence_packet": (packaged - searched) * 1000,
                    "request_total": (packaged - query_started) * 1000,
                },
            }
        )
    return {
        "mode": "live_public_BM25_CPU_manual_queries_no_model_or_quality_evaluation",
        "document_count": len(documents),
        "corpus_sha256": digest,
        "input_mode": "custom_claim"
        if claim is not None
        else "three_authored_demo_queries",
        "configuration": {
            "retriever": "BM25Index",
            "k1": index.k1,
            "b": index.b,
            "candidate_k": candidate_k,
            "evidence_top_k": top_k,
            "dense": False,
            "ltr": False,
            "reranker": False,
            "verdict_model": False,
        },
        "setup_timings_ms": {
            "asset_validation": (validated - started) * 1000,
            "document_load": (loaded - load_started) * 1000,
            "index_build": (indexed - loaded) * 1000,
            "setup_total": (setup_finished - started) * 1000,
        },
        "timing_scope": "single local invocation; request=search/rank/packet, excludes output serialization and CLI overhead; setup includes imports/hash/load/index/provenance; not P50/P95 or online SLA",
        "cases": cases,
    }


def build_demo(
    evidence: Path | None = None,
    root: Path = ROOT,
    *,
    claim: str | None = None,
    candidate_k: int = 10,
    top_k: int = 3,
) -> dict[str, Any]:
    result = historical_summary(root)
    result["public_retrieval"] = public_packets(
        evidence, root, claim=claim, candidate_k=candidate_k, top_k=top_k
    )
    return result


def render_text(result: dict[str, Any]) -> str:
    train, search, agent = result["training"], result["search"], result["agent"]
    lines = [
        "Climate Evidence Retrieval — training and search decision case",
        "HISTORICAL RESULT REPLAY, NOT live dense/LTR/LLM execution",
        f"Offline-dev training: {train['documents']:,} docs / {train['queries']} queries / {train['steps']} adaptation steps",
        f"Recall@5: {train['base_recall_at_5']:.2%} -> {train['adapted_recall_at_5']:.2%}",
        "Separate public validation (126 queries):",
        "profile       Recall@5   F1@5      offline P95(ms)   Torch GiB",
    ]
    for item in search["profiles"]:
        lines.append(
            f"{item['profile']:<13} {item['recall_at_5']:.4f}     "
            f"{item['evidence_f1_at_5']:.4f}     {item['p95_ms']:>10.1f}       "
            f"{item['torch_allocated_gib']:.2f}"
        )
    lines.extend(
        [
            "Decision: " + search["decision"],
            f"Autonomous behavior: stop={agent['stop']}, acquire={agent['acquire']}; no Agent gain claimed.",
            "Public retrieval mode: " + result["public_retrieval"]["mode"],
        ]
    )
    live = result["public_retrieval"]
    if "configuration" in live:
        lines.append(
            "Actual current configuration: " + json.dumps(live["configuration"])
        )
        lines.append("Setup timings(ms): " + json.dumps(live["setup_timings_ms"]))
        lines.append("Timing scope: " + live["timing_scope"])
    for case in result["public_retrieval"]["cases"]:
        lines.append("\nQuery: " + case["claim_text"])
        if "candidates" in case:
            lines.append("  Candidate pool (BM25 order, not LTR/rerank):")
            for item in case["candidates"]:
                lines.append(
                    f"    #{item['rank']} {item['evidence_id']} score={item['score']:.6f}"
                )
            lines.append("  Selected evidence:")
        for item in case["items"]:
            text = item.get(
                "text", "[full text omitted; --evidence enables live public BM25]"
            )
            lines.append(
                f"  #{item['retrieval']['rank']} {item['evidence_id']} score={item['retrieval']['score']:.6f}: {text}"
            )
            if "provenance" in item:
                lines.append(
                    "    Source: " + json.dumps(item["provenance"], ensure_ascii=False)
                )
        if "timings_ms" in case:
            lines.append(
                "  Current request timings(ms): " + json.dumps(case["timings_ms"])
            )
        if not case["items"]:
            lines.append("  Empty result: return no evidence, not an invented answer.")
    lines.append(
        "No verdict generated. Manual retrieval/empty-result demos are not benchmark results or autonomous recovery cases."
    )
    return "\n".join(lines) + "\n"


def render_html(result: dict[str, Any]) -> str:
    search = result["search"]
    rows = "".join(
        f"<tr><td>{escape(item['profile'])}</td><td>{item['recall_at_5']:.2%}</td>"
        f"<td>{item['evidence_f1_at_5']:.4f}</td><td>{item['p95_ms']:.1f} ms</td>"
        f"<td>{item['torch_allocated_gib']:.2f} GiB</td></tr>"
        for item in search["profiles"]
    )
    packets = []
    for case in result["public_retrieval"]["cases"]:
        items = (
            "".join(
                f"<li><small>{escape(item['evidence_id'])}</small><p>{escape(item.get('text', '[完整文本未发布；用 --evidence 在本地展示公开检索]'))}</p></li>"
                for item in case["items"]
            )
            or "<li>空结果：不生成没有证据的回答。</li>"
        )
        packets.append(
            f"<section><h3>{escape(case['claim_text'])}</h3>"
            f"<pre>{escape(json.dumps({'candidates': case.get('candidates', 'saved metadata only'), 'timings_ms': case.get('timings_ms', 'not recorded in saved replay')}, ensure_ascii=False, indent=2))}</pre>"
            f"<ol>{items}</ol></section>"
        )
    train, agent = result["training"], result["agent"]
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Climate · 检索训练与证据搜索</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#263445;background:#f4f6fa;margin:0}}
main{{max-width:1000px;margin:36px auto;padding:0 24px}}h1,h2{{color:#173961}}
section,.card{{background:white;border:1px solid #dce3ed;border-radius:12px;padding:20px;margin:18px 0}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:12px 8px;text-align:left;border-bottom:1px solid #e2e8f0}}
.boundary{{border-left:4px solid #b87912;padding:12px 18px;background:#fff9ed}}
small{{color:#65758b}}li p{{margin-top:4px}}code{{overflow-wrap:anywhere}}
@media(max-width:600px){{main{{padding:0 12px}}table{{font-size:12px}}th,td{{padding:8px 3px}}}}
</style><main><h1>Climate：检索模型训练与可信证据搜索</h1>
<p>业务问题：为声明找到相关且充分的证据；根据质量、时延选择检索与排序方案。</p>
<p class="boundary">历史选型结果回放 ≠ 当前实时 dense/LTR/生成式查证。不同语料与评测分母分开展示。</p>
<div class="card"><h2>1 · 模型任务适配</h2><p>claim 分组 → BM25/dense hard negatives → InfoNCE/LoRA</p>
<p>{train["documents"]:,} 条受限语料 · {train["queries"]} 条 offline dev · {train["steps"]} steps</p>
<p>Recall@5：<strong>{train["base_recall_at_5"]:.2%} → {train["adapted_recall_at_5"]:.2%}</strong></p>
<small>任务适配实验，不是预训练、独立 test 或线上 A/B；本页面不加载权重。</small></div>
<div class="card"><h2>2 · 质量—时延选型</h2><p>同一语料、126 条公开 validation、相同候选池。</p>
<table><tr><th>方案</th><th>Recall@5</th><th>F1@5</th><th>离线 P95</th><th>Torch 峰值</th></tr>{rows}</table>
<p><strong>选择 LTR 为低时延候选。</strong>4B Top100 的 R@5/F1 点估计更高，但差值区间跨零。</p>
<small>暖机、串行、进程内请求；排除模型加载/HTTP/并发。不是线上 SLA、统计等效或美元节省。
历史完整 LTR 路线的 query encoder 使用 GPU，不是 CPU-only。</small></div>
<div class="card"><h2>3 · 当前公共证据展示</h2><code>{escape(result["public_retrieval"]["mode"])}</code>
<p>仅 BM25；默认三条预先编写的查询，或 --claim 输入声明；无标签、无模型调用、无 verdict。</p>
<pre>{escape(json.dumps({key: result["public_retrieval"][key] for key in ("configuration", "setup_timings_ms", "timing_scope") if key in result["public_retrieval"]}, ensure_ascii=False, indent=2))}</pre></div>
{"".join(packets)}
<div class="card"><h2>4 · 自主方案未推广</h2><p>实际开发对照：stop {agent["stop"]} / acquire {agent["acquire"]}。</p>
<p>保留负结果；引用合法不等于语义支持。当前展示不模拟自主补证据成功。</p></div>
<p><small>此页面由已有汇总与公开检索包生成；不读取 gold、sealed test 或私密模型运行轨迹。</small></p>
</main></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        help="optional SHA-pinned public corpus for live CPU BM25",
    )
    parser.add_argument("--format", choices=("text", "html", "json"), default="text")
    parser.add_argument(
        "--claim",
        help="a user-authored claim; requires --evidence, never answered from saved replay",
    )
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = build_demo(
            args.evidence,
            claim=args.claim,
            candidate_k=args.candidate_k,
            top_k=args.top_k,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.format == "html":
        rendered = render_html(result)
    elif args.format == "json":
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    else:
        rendered = render_text(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
