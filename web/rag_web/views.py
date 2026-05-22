import json
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST, require_GET

from .models import QueryRun
from .tasks import run_rag_query


def index(request):
    recent_runs = QueryRun.objects.all()[:5]
    context = {
        "recent_runs":    recent_runs,
        "dev_mode":       getattr(settings, "RAG_DEV_MODE", True),
        "hybrid_top_k":   getattr(settings, "RAG_HYBRID_TOP_K", 100),
        "reranker_top_k": getattr(settings, "RAG_RERANKER_TOP_K", 10),
        "use_stage2":     getattr(settings, "RAG_USE_STAGE2_RERANKER", False),
    }
    return render(request, "rag_web/index.html", context)


@require_POST
def run_query(request):
    query_text = request.POST.get("query_text", "").strip()
    if not query_text:
        return redirect("/")

    query_run = QueryRun.objects.create(
        query_text     = query_text,
        hybrid_top_k   = getattr(settings, "RAG_HYBRID_TOP_K", 100),
        reranker_top_k = getattr(settings, "RAG_RERANKER_TOP_K", 10),
        use_stage2     = getattr(settings, "RAG_USE_STAGE2_RERANKER", False),
        #task_id        = "pending",  # temporary, updated below
    )

    task = run_rag_query.delay(
        query_run_id = query_run.pk,
        query_text   = query_text,
    )

    query_run.task_id = task.id
    query_run.save(update_fields=["task_id"])

    return redirect(f"/status/{task.id}/")


@require_GET
def status(request, task_id):
    query_run = get_object_or_404(QueryRun, task_id=task_id)

    if request.headers.get("Accept") == "application/json":
        data = {
            "status":   query_run.status,
            "step":     _get_step_label(query_run.status),
            "progress": _get_progress(query_run.status),
            "elapsed":  query_run.elapsed_secs,
            "error":    query_run.error_message[:200] if query_run.error_message else "",
        }
        if query_run.status == QueryRun.Status.SUCCESS:
            data["redirect"] = f"/results/{task_id}/"
        return JsonResponse(data)

    return render(request, "rag_web/status.html", {
        "query_run": query_run,
        "task_id":   task_id,
    })


def _get_step_label(status):
    return {"PENDING": "Waiting for worker...", "STARTED": "Pipeline running...",
            "SUCCESS": "Complete!", "FAILURE": "Failed"}.get(status, status)


def _get_progress(status):
    return {"PENDING": 5, "STARTED": 50, "SUCCESS": 100, "FAILURE": 100}.get(status, 0)


@require_GET
def results(request, task_id):
    query_run = get_object_or_404(QueryRun, task_id=task_id)
    citations = []
    results_data = []
    if query_run.is_complete:
        citations    = json.loads(query_run.citations_json or "[]")
        results_data = json.loads(query_run.results_json or "[]")
    return render(request, "rag_web/results.html", {
        "query_run":    query_run,
        "citations":    citations,
        "results_data": results_data,
        "n_law":   sum(1 for r in results_data if r.get("source") == "law"),
        "n_court": sum(1 for r in results_data if r.get("source") == "court"),
        "n_regex": sum(1 for r in results_data if r.get("regex_match")),
    })


@require_GET
def evaluation(request):
    eval_dir = getattr(settings, "ARTIFACTS_DIR", None)
    summary = {}
    chart_labels = chart_f1 = chart_precision = chart_recall = []
    worst_queries = missed_citations = per_query_data = []

    if eval_dir:
        from pathlib import Path
        eval_dir = Path(eval_dir) / "evaluation"
        summary_path = eval_dir / "evaluation_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        per_query_path = eval_dir / "per_query_results.csv"
        if per_query_path.exists():
            import pandas as pd
            df = pd.read_csv(per_query_path)
            per_query_data  = df.to_dict("records")
            chart_labels    = df["query_id"].astype(str).tolist()
            chart_f1        = df["f1"].round(3).tolist()
            chart_precision = df["precision"].round(3).tolist()
            chart_recall    = df["recall"].round(3).tolist()
        worst_path = eval_dir / "worst_queries.csv"
        if worst_path.exists():
            import pandas as pd
            worst_queries = pd.read_csv(worst_path).to_dict("records")
        missed_path = eval_dir / "most_missed_citations.csv"
        if missed_path.exists():
            import pandas as pd
            missed_citations = pd.read_csv(missed_path).head(10).to_dict("records")

    return render(request, "rag_web/evaluation.html", {
        "summary":          summary,
        "per_query_data":   per_query_data,
        "worst_queries":    worst_queries,
        "missed_citations": missed_citations,
        "chart_labels":     json.dumps(chart_labels),
        "chart_f1":         json.dumps(chart_f1),
        "chart_precision":  json.dumps(chart_precision),
        "chart_recall":     json.dumps(chart_recall),
        "has_eval_data":    bool(summary),
    })


@require_GET
def history(request):
    return render(request, "rag_web/history.html", {
        "runs": QueryRun.objects.all()
    })
