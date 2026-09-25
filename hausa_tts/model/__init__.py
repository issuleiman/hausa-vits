def __getattr__(name: str):
    if name == "HausaVITS":
        from hausa_tts.model.vits import HausaVITS
        return HausaVITS
    raise AttributeError(f"module 'hausa_tts.model' has no attribute {name!r}")

__all__ = ["HausaVITS"]

