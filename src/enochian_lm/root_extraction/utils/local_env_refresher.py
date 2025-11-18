import os
import re
import subprocess
from dotenv import load_dotenv, find_dotenv


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


def refresh_local_env(local=False) -> bool:
    """
    If local=True, regenerate .local_env with current Windows host IP.
    Then load .local_env into os.environ. Return True on success.
    """
    # location of the env file at your project root
    explicit_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../.env_local")
    )
    
    if os.path.exists(explicit_path):
        env_path = explicit_path
    else:
        env_path = find_dotenv(".env_local")

    if local:
        ip = get_windows_host_ip()
        if not ip:
            print("[FAIL] Could not detect Windows host IP.")
            return False

        base_url = f"http://{ip}:1234/v1"
        try:
            with open(env_path, "w") as f:
                f.write("LOCAL_OPENAI_API_KEY=sk-local-testing-lol\n")
                f.write(f"LOCAL_OPENAI_API_BASE={base_url}\n")
                f.write("LOCAL_MODEL_NAME=deepseek/deepseek-r1-0528-qwen3-8b\n")
                f.write("PYTHONPATH=src\n")
        except Exception as e:
            print(f"[ERROR] Writing .local_env failed: {e}")
            return False

    # Now load it
    if not os.path.exists(env_path):
        print(f"[ERROR] Could not find local env file at {env_path}.")
        return False

    # load_dotenv will read KEY=VALUE lines into os.environ
    loaded = load_dotenv(env_path, override=True)
    if not loaded:
        print(f"[ERROR] Failed to load environment from {env_path}.")
        return False

    return True
