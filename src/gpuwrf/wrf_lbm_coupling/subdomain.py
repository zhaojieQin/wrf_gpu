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


def extract_subdomain(carry: Any, config) -> Dict[str, Any]:
    """Extract subdomain data for coupling.

    Args:
        carry: OperationalCarry with .state attribute
        config: CouplingConfig object or dict with coupling configuration

    Returns:
        Dictionary with extracted field data as JAX arrays
    """
    state = carry.state
    subdomain = {}

    # Support both CouplingConfig object and dict
    if isinstance(config, dict):
        # Dict format: extract fields list from subdomain_config
        subdomain_config = config.get('subdomain_config', {})
        field_names = subdomain_config.get('fields', ['u', 'v', 'theta', 'qv'])

        # Build field_specs from simple field names and ranges
        z_range = subdomain_config.get('z_range', (0, -1))
        y_range = subdomain_config.get('y_range', (0, -1))
        x_range = subdomain_config.get('x_range', (0, -1))

        field_specs = []
        for field_name in field_names:
            spec = {
                "name": field_name,
                "i_range": y_range,  # WRF convention: i is y-direction
                "j_range": x_range,  # WRF convention: j is x-direction
            }
            # Add k_range for 3D fields (check if field exists and has 3+ dims)
            if hasattr(state, field_name):
                field_data = getattr(state, field_name)
                if len(field_data.shape) >= 3:
                    spec["k_range"] = z_range
            field_specs.append(spec)
    else:
        # CouplingConfig object
        field_specs = config.fields

    for field_spec in field_specs:
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
