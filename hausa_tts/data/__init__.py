def __getattr__(name: str):
    _imports = {
        "HausaTTSDataset": "hausa_tts.data.dataset",
        "AudioProcessor": "hausa_tts.data.audio",
        "HausaCollate": "hausa_tts.data.collate",
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'hausa_tts.data' has no attribute {name!r}")

__all__ = ["HausaTTSDataset", "AudioProcessor", "HausaCollate"]

