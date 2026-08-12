import os
import shutil


def main():
    print("====================================================")
    print("AIR-NET V1 — BACKUP HISTORICAL RESULTS & METADATA")
    print("====================================================")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outputs_dir = os.path.join(project_root, "outputs")
    backup_dir = os.path.join(outputs_dir, "v1_resume_backup")

    os.makedirs(backup_dir, exist_ok=True)

    files_to_backup = [
        "benchmark_report.txt",
        "bicubic_baseline.txt",
        "evaluation_audit.txt",
        "experiment_info.json",
        "metric_sample_audit.csv",
        "model_summary.txt",
        "results.csv",
        "train_stats.txt",
    ]

    backed_up_count = 0
    for fname in files_to_backup:
        src = os.path.join(outputs_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, fname)
            shutil.copy2(src, dst)
            backed_up_count += 1
            print(f"  [OK] Backed up: {fname} -> {dst}")

    print(
        f"\nSuccessfully backed up {backed_up_count} historical metadata/log files to {backup_dir}"
    )
    print("====================================================")


if __name__ == "__main__":
    main()
