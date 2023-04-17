import os

PROJECT_ROOT: str = os.getcwd()
"""
We assume the application always starts at the project root.
"""

class DatasetSetting:
    """
    Please change the path variables so that it matches your environment.
    As a convention to avoid confusion, please alway use absolute path.
    """
    ASAP_PATH: str = None

    CONCATENATED_MIDI_OUTPUT_PATH: str = PROJECT_ROOT + "/.cache"
    """
    Intermediate MIDI file cache output directory.
    """
    MODEL_OUTPUT_PATH: str = PROJECT_ROOT + "/output"
    """
    Hold binary of the trained model.
    """