def __getattr__(name: str):
    if name == "Trainer":
        from hausa_tts.training.trainer import Trainer
        return Trainer
    raise AttributeError(f"module 'hausa_tts.training' has no attribute {name!r}")

__all__ = ["Trainer"]

