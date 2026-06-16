# Moved to src/features/context/state_context.py (Layer 3).
# obv_signal is now computed alongside volume_zscore in detect_volume_state().
# This file is kept only as a redirect so any direct script call still works.

from features.context.state_context import detect_volume_state as run  # noqa: F401
