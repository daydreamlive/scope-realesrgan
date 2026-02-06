"""Real-ESRGAN 2x super resolution pipeline for Daydream Scope."""

import logging
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope.core.pipelines.process import normalize_frame_sizes

from .arch import RRDBNet
from .schema import RealESRGANConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)

WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"


class RealESRGANPipeline(Pipeline):
    """Real-ESRGAN 2x super resolution postprocessor.

    Upscales video frames by 2x using the RRDBNet architecture with
    pre-trained Real-ESRGAN x2plus weights. Stateless, frame-by-frame
    processing.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return RealESRGANConfig

    def __init__(
        self,
        config=None,
        device=None,
        dtype=torch.float16,
        **kwargs,
    ):
        if config is not None:
            self.config = config
        else:
            config_fields = {
                k: v
                for k, v in kwargs.items()
                if k in RealESRGANConfig.model_fields
            }
            self.config = RealESRGANConfig(**config_fields)

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        logger.info("Loading Real-ESRGAN x2plus (RRDBNet)...")
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=2,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )

        state_dict = torch.hub.load_state_dict_from_url(
            WEIGHTS_URL, map_location="cpu"
        )
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        model.load_state_dict(state_dict, strict=True)
        model.eval().to(self.device, self.dtype)
        self.model = model
        logger.info("Real-ESRGAN x2plus loaded successfully")

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    @torch.inference_mode()
    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")

        if video is None:
            raise ValueError("Input 'video' cannot be None for RealESRGANPipeline")

        if isinstance(video, list):
            video = normalize_frame_sizes(video)

        output_frames = []
        for frame in video:
            # (1, H, W, C) uint8 [0,255] → (1, C, H, W) float [0,1]
            x = frame.squeeze(0).float().div(255.0)
            x = x.permute(2, 0, 1).unsqueeze(0)
            x = x.to(self.device, self.dtype)

            _, _, h, w = x.shape
            pad_h = (2 - h % 2) % 2
            pad_w = (2 - w % 2) % 2
            if pad_h or pad_w:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

            out = self.model(x)

            if pad_h or pad_w:
                out = out[:, :, : h * 2, : w * 2]

            # (1, C, 2H, 2W) → (2H, 2W, C) float [0,1]
            out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).float()
            output_frames.append(out)

        return {"video": torch.stack(output_frames, dim=0)}
