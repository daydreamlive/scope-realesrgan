"""Configuration schema for Real-ESRGAN super resolution pipeline."""

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults


class RealESRGANConfig(BasePipelineConfig):
    """Configuration for Real-ESRGAN 2x super resolution pipeline.

    Uses RRDBNet architecture with pre-trained Real-ESRGAN x2plus weights.
    Upscales input video by 2x (e.g., 512x320 → 1024x640).

    Model: https://github.com/xinntao/Real-ESRGAN
    Architecture: RRDBNet (23 RRDB blocks, 64 features, 32 grow channels)
    Weights: RealESRGAN_x2plus.pth (~67 MB, auto-downloaded on first use)
    """

    pipeline_id = "realesrgan"
    pipeline_name = "Real-ESRGAN 2x"
    pipeline_description = (
        "2x super resolution using Real-ESRGAN (RRDBNet). "
        "Upscales video frames, e.g. 512x320 to 1024x640."
    )
    docs_url = "https://github.com/xinntao/Real-ESRGAN"
    artifacts = []
    supports_prompts = False
    modified = False
    estimated_vram_gb = 0.3

    usage = []

    modes = {"video": ModeDefaults(default=True)}
