"""Public CountAnything package."""


def build_count_anything_model(*args, **kwargs):
    from count_anything.model_builder import build_count_anything_model as _build

    return _build(*args, **kwargs)

__all__ = ["build_count_anything_model"]
