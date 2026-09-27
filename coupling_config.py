"""Shared WRF-LBM coupling configuration.

Single source of truth for coupling parameters.
"""

# Forecast duration (hours)
HOURS = 1

# Coupling interval (seconds)
INTERVAL_SECONDS = 10

# LBM steps = HOURS * 3600 / INTERVAL_SECONDS = 1 * 3600 / 10 = 360
