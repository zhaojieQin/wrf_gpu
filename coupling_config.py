"""Shared WRF-LBM coupling configuration.

Single source of truth for coupling parameters.
"""

# Forecast duration (hours)
HOURS = 1

# d03 timestep (seconds): time_step=18 / (ratio_d02=3 * ratio_d03=3) = 2s
# Coupling fires every d03 timestep, so interval = target_dt
D03_DT_SECONDS = 2

# Coupling interval (seconds) — kept for reference, not used for alarm scheduling
INTERVAL_SECONDS = D03_DT_SECONDS

# LBM steps = HOURS * 3600 / D03_DT_SECONDS = 1 * 3600 / 2 = 1800
