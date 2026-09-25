"""HausaVITS - Hausa Text-to-Speech Framework

A comprehensive training framework for single-speaker and multi-speaker
Hausa text-to-speech models, extending the VITS architecture with
Hausa-specific linguistic processing (tone, prosody, vowel length,
implosive/ejective consonants).
"""
__version__ = '0.1.0'


def __getattr__(name: str):
    """Lazy imports to avoid pulling in heavy dependencies at package level."""
    if name == "HausaVITS":
        from hausa_tts.model.vits import HausaVITS
        return HausaVITS
    elif name == "HausaSynthesizer":
        from hausa_tts.inference.synthesize import HausaSynthesizer
        return HausaSynthesizer
    raise AttributeError(f"module 'hausa_tts' has no attribute {name!r}")


__all__ = ["HausaVITS", "HausaSynthesizer"]
