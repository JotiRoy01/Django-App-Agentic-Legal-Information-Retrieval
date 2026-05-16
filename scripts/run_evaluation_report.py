from agentic.evaluation.evaluation_report import run_evaluation



if __name__ == "__main__":
    report = run_evaluation(
        submission_path = "artifacts/submission.csv",
        gold_path       = "data/val.csv",
        k               = 10,
    )
    print(report.summary())