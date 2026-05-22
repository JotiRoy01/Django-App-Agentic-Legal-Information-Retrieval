from agentic.evaluation.evaluation_report import run_evaluation



if __name__ == "__main__":
    report = run_evaluation()
    print(report.summary())