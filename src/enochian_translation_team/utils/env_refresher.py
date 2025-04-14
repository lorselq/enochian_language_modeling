import os
import re
import subprocess

def get_windows_host_ip():
    try:
        result = subprocess.run(["ip", "route"], stdout=subprocess.PIPE, check=True)
        output = result.stdout.decode()
        match = re.search(r"default via (\d+\.\d+\.\d+\.\d+)", output)
        if match:
            return match.group(1)
        else:
            print("[WARN] No default gateway IP found.")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to parse routing table: {e}")
        return None

def refresh_env():
    ip = get_windows_host_ip()
    if not ip:
        print("[FAIL] Could not detect Windows host IP.")
        return False

    base_url = f"http://{ip}:1234/v1"
    # write .env in the root project folder
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))

    try:
        with open(env_path, "w") as f:
            f.write(f"OPENAI_API_KEY=sk-local-testing-lol\n")
            f.write(f"OPENAI_API_BASE={base_url}\n")
            f.write(f"MODEL_NAME=openai/local-model\n")
            f.write(f"PYTHONPATH=src\n")
        print(f"[SUCCESS] .env updated with IP: {ip}")
        return True
    except Exception as e:
        print(f"[ERROR] Writing .env failed: {e}")
        return False
