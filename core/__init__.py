"""Shared infrastructure for the decoupled Download/Craft data pipeline.

``core`` is intentionally dependency-free (standard library only, plus a
lazy ``torch`` import inside :mod:`core.checkpointing`) so both the Colab
Download Mode and the Kaggle Craft Mode can import it without pulling in
chemistry or tensor libraries.
"""
