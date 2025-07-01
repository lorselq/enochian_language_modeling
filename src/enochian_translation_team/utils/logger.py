import datetime
import re
from typing import Optional
from pathlib import Path

def save_log(log_entries, label: Optional[str] = None, cluster_number: Optional[str] = None, cluster_total: Optional[str] = None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{label.upper()}" if label else "ngram-missing"
    cluster_info = ""
    if cluster_number and cluster_total:
        cluster_info = f"clstr{cluster_number}-of-{cluster_total}_"
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{timestamp}_{cluster_info}log.txt"

    header_re = re.compile(r"^\*\*(.+?):\*\*", re.MULTILINE)

    with open(log_path, "w", encoding="utf-8") as f:
        for role, message in log_entries:
            md = header_re.sub(lambda m: f"## {m.group(1)}", message, count=1)
            f.write(md.strip() + "\n\n")

    print(f"\n\n📁 Log saved to `{log_path}`\n\n")

    return str(log_path)
