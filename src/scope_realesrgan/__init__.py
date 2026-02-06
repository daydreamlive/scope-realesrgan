"""Real-ESRGAN 2x super resolution postprocessor plugin for Daydream Scope."""

from scope.core.plugins import hookimpl

from .pipeline import RealESRGANPipeline


@hookimpl
def register_pipelines(register):
    register(RealESRGANPipeline)


__all__ = ["RealESRGANPipeline"]
