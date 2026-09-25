def __getattr__(name: str):
    if name == "HausaSynthesizer":
        from hausa_tts.inference.synthesize import HausaSynthesizer
        return HausaSynthesizer
    raise AttributeError(f"module 'hausa_tts.inference' has no attribute {name!r}")

__all__ = ["HausaSynthesizer"]

