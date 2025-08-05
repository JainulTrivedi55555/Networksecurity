from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data class for data ingestion artifact.
    """
    trained_file_path: str
    test_file_path: str
    