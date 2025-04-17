from typing import Optional
from enochian_translation_team.crew.root_extraction_crew import RootExtractionCrew

def run_root_extraction_gui(max_words: Optional[int] = None, stream_callback=None):
    crew = RootExtractionCrew()
    crew.run_with_streaming(max_words=max_words, stream_callback=stream_callback)
