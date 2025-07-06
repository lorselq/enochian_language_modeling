import datetime
from typing import Optional
from pathlib import Path

def save_log(log_entries: str, label: Optional[str] = None, cluster_number: Optional[str] = None, cluster_total: Optional[str] = None, accepted: Optional[bool] = False, style: str="StyleUnclear"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{label.upper()}" if label else "ngram-missing"
    cluster_info = ""
    if cluster_number and cluster_total:
        cluster_info = f"clstr{cluster_number}-of-{cluster_total}"
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_output = log_entries.strip() if log_entries else ""
    log_path = ""
    if len(file_output) > 0:
        log_path = log_dir / f"{name}_{style}_{timestamp}_{cluster_info}_{'accepted' if accepted else 'rejected'}_log.txt"
    else:
        log_path = log_dir / f"{name}_empty_{timestamp}_{cluster_info}log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_entries.strip() + "\n\n")

    print(f"\n\n📁 Log saved to `{log_path}`\n\n")

    return str(log_path)
