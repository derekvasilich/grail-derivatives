import ctypes

from app.api import DerivType, OptionConfig, OptionGreeks, deriv_labels


def print_greeks(i: DerivType, fdm_greeks: OptionGreeks, bin_greeks: OptionGreeks):
    print(f"\n{deriv_labels[i]} ========")
    print(f"Vₛ: {fdm_greeks.price:.4f}, Δ: {fdm_greeks.delta:.4f}, Γ: {fdm_greeks.gamma:.4f}, Θ: {fdm_greeks.theta:.4f}")
    print(f"Vₑ: {bin_greeks.price:.4f}, Δ: {bin_greeks.delta:.4f}, Γ: {bin_greeks.gamma:.4f}, Θ: {bin_greeks.theta:.4f}")


# ------------------------------------------------------------------
# Binary struct helpers shared by the /pricing/binary and /pricing/grid tests.
# ------------------------------------------------------------------
CONFIG_STRUCT_SIZE = ctypes.sizeof(OptionConfig)
GREEKS_STRUCT_SIZE = ctypes.sizeof(OptionGreeks)


def build_config_struct(**overrides) -> OptionConfig:
    """Build a single OptionConfig ctypes struct with sensible defaults."""
    defaults = dict(deriv=0, Tn=300, time=1.0, h=1.0, r=0.1, sigma=0.5, s=100.0, k=110.0, q=0.0)
    defaults.update(overrides)
    return OptionConfig(**defaults)


def pack_configs(configs) -> bytes:
    """Pack an iterable of OptionConfig structs into a contiguous byte buffer."""
    configs = list(configs)
    array_type = OptionConfig * len(configs)
    return bytes(array_type(*configs))


def unpack_greeks(raw: bytes):
    """Reconstruct the OptionGreeks struct array from a raw octet-stream response body."""
    count = len(raw) // GREEKS_STRUCT_SIZE
    array_type = OptionGreeks * count
    return array_type.from_buffer_copy(raw)
