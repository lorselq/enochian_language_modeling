from enochian_translation_team.utils.env_refresher import refresh_env
import subprocess

def main():
    if refresh_env():
        print("[✓] Launching Streamlit app...")
        subprocess.run(["streamlit", "run", "src/enochian_translation_team/app.py"])
    else:
        print("[ERROR] Could not initialize environment. GUI launch aborted.")

if __name__ == "__main__":
    main()
