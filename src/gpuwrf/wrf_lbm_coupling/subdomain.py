"""Subdomain extraction for WRF-LBM coupling."""

from typing import Any, Dict
import jax.numpy as jnp


class CouplingConfig:
    """Configuration for WRF-LBM coupling.

    Attributes:
        domain: Target domain name (e.g., 'd03')
        fields: List of field specifications, each a dict with:
            - name: field name in OperationalCarry.state
            - i_range: (start, end) or (start, -1) for i-dimension
            - j_range: (start, end) or (start, -1) for j-dimension
            - k_range: (start, end) or (start, -1) for k-dimension (optional, for 3D fields)
        interval_seconds: Coupling interval in seconds
    """

    def __init__(self, domain: str, fields: list, interval_seconds: float):
        self.domain = domain
        self.fields = fields
        self.interval_seconds = interval_seconds


def normalize_range(range_spec: tuple, actual_size: int) -> tuple:
    """Convert range with -1 to actual indices.

    Args:
        range_spec: (start, end) where end can be -1 meaning "to end"
        actual_size: Actual dimension size

    Returns:
        (start, end) with -1 replaced by actual_size
    """
    start, end = range_spec
    if end == -1:
        end = actual_size
    return (start, end)


def extract_subdomain(carry: Any, config: CouplingConfig) -> Dict[str, Any]:
    """Extract subdomain data for coupling.

    Args:
        carry: OperationalCarry with .state attribute
        config: CouplingConfig specifying fields and ranges

    Returns:
        Dictionary with extracted field data as JAX arrays
    """
    state = carry.state
    subdomain = {}

    for field_spec in config.fields:
        field_name = field_spec["name"]
        field_data = getattr(state, field_name)

        # Get actual shape
        shape = field_data.shape

        # Normalize ranges
        i_range = normalize_range(field_spec["i_range"], shape[-2])
        j_range = normalize_range(field_spec["j_range"], shape[-1])

        # Extract based on dimensionality
        if "k_range" in field_spec:
            # 3D field
            k_range = normalize_range(field_spec["k_range"], shape[-3])
            extracted = field_data[..., k_range[0]:k_range[1], i_range[0]:i_range[1], j_range[0]:j_range[1]]
        else:
            # 2D field
            extracted = field_data[..., i_range[0]:i_range[1], j_range[0]:j_range[1]]

        subdomain[field_name] = extracted

    return subdomain
