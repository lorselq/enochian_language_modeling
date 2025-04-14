from enochian_translation_team.utils.env_refresher import refresh_env

if __name__ == "__main__":
    if refresh_env():
        from enochian_translation_team.crew import run_crew    
        run_crew()
    else:
        print("[ERROR] Could not initialize environment. Crew launch aborted.")
