from datetime import datetime
import os 

from networksecurity.constant import training_pipeline

print(training_pipeline.PIPELINE_NAME)
print(training_pipeline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self,timestamp = datetime.now()):
        timestamp = timestamp.strftime('%m_%d_%Y_%H_%M_%S')
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        self.timestamp: str = timestamp
        

class DataIngestionConfig:
    """
    Configuration for the Data Ingestion stage.
    Defines all paths and parameters needed for ingesting data.
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Base directory for all data ingestion artifacts
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )

        # Path for the raw feature store FILE.
        # FIXED: Changed to 'Data_...' to match the definition in your constants file.
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.Data_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )

        # Path for the final training FILE.
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
        )

        # Path for the final testing FILE.
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
        )

        # Ratio for splitting data
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        
        # MongoDB parameters
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME