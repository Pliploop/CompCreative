import logging
import os
from dataclasses import asdict, dataclass
from typing import Optional

from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

logger = logging.getLogger(__name__)


@dataclass
class ProcessingInputConfig:
    source: str
    input_name: Optional[str] = None  # autogenerated if not provided
    destination: str = "/opt/ml/processing/input/"  # must start with /opt/ml/processing
    s3_data_type: str = "S3Prefix"
    s3_input_mode: str = "File"
    s3_data_distribution_type: str = "ShardedByS3Key"


@dataclass
class ProcessingOutputConfig:
    destination: str
    output_name: Optional[str] = None  # autogenerated if not provided
    source: str = "/opt/ml/processing/output/"  # must start with /opt/ml/processing
    s3_upload_mode: str = "Continuous"


@dataclass
class ProcessorConfig:
    base_job_name: str
    image_uri: str
    instance_count: int
    instance_type: str
    volume_size_in_gb: int
    tags: list[dict]
    entrypoint: list[str]
    max_runtime_in_seconds: int = 5 * 24 * 60 * 60  # corresponds to 5 days
    role: str = ""


def launch_sagemaker_processing(cfg: dict):
    # INFER LAST CONFIGURATION DETAILS FROM THE INPUT
    base_job_name = "-".join([cfg["base_name"], "processing"])

    # CONFIGURING THE INPUTS
    inputs = [ProcessingInput(**asdict(ProcessingInputConfig(**cfg_i))) for cfg_i in cfg["processing_inputs"]]

    # CONFIGURING THE OUTPUTS
    outputs = [ProcessingOutput(**asdict(ProcessingOutputConfig(**cfg_i))) for cfg_i in cfg["processing_outputs"]]

    # CONFIGURING THE PROCESSOR
    processor_config = ProcessorConfig(base_job_name=base_job_name, **cfg["processor"])
    processor = Processor(**asdict(processor_config))

    # RUN!
    logging.info("Start the processing!")
    # fmt: off
    # arguments = [
    #     "--input_dir", os.path.commonpath([i.destination for i in inputs]),
    #     "--output_dir", os.path.commonpath([o.source for o in outputs]),
    #     *cfg.get("additional_run_arguments", []),
    # ]  # fmt:on
    
    
    #time this
    
    from time import time
    
    start = time()
    
    processor.run(inputs=inputs, outputs=outputs, logs='All')
    
    formatted_time = time() - start
    
    print(f"Time taken to run the processor: {formatted_time}")