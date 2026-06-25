import platform
import os
import ctypes

# Load the shared library
dir_path = os.path.dirname(os.path.realpath(__file__))
dylib_path = os.path.join(dir_path, "lib", "libblackfdmcore.dylib")
so_path = os.path.join(dir_path, "lib", "libblackfdmcore.so")

if platform.system() == "Darwin" and platform.machine() == "arm64":
    assert os.path.exists(dylib_path), f"Library not found at: {dylib_path}"
    fdm_lib = ctypes.CDLL(dylib_path)
else:
    assert os.path.exists(so_path), f"Library not found at: {so_path}"
    fdm_lib = ctypes.CDLL(so_path)
    
class DerivType:
    VanillaCall = 0
    VanillaPut = 1
    AmericanCall = 2
    AmericanPut = 3
    BermudanCall = 4
    BermudanPut = 5
    BarrierOutCall = 6
    BarrierOutPut = 7
    BarrierInCall = 8
    BarrierInPut = 9
    DblBarrierOutCall = 10
    DblBarrierOutPut = 11
    DblBarrierInCall = 12
    DblBarrierInPut = 13
    
deriv_labels = {
    0: "Vanilla Call", 1: "Vanilla Put", 
    2: "American Call", 3: "American Put", 
    4: "Bermudan Call", 5: "Bermudan Put"
}

class BarrierType:
    UpAndOut = 0
    DownAndOut = 1

class FrequencyType:
    SemiAnnual = 2
    Quarterly = 4

class OptionConfig(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("deriv", ctypes.c_int),
        ("barrier", ctypes.c_int),
        ("frequency", ctypes.c_int),
        ("Tn", ctypes.c_int),
        ("top", ctypes.c_int),
        ("bottom", ctypes.c_int),
        ("left", ctypes.c_int),
        ("right", ctypes.c_int),
        ("time", ctypes.c_double),
        ("h", ctypes.c_double),
        ("r", ctypes.c_double),
        ("sigma", ctypes.c_double),
        ("s", ctypes.c_double),
        ("k", ctypes.c_double),
        ("q", ctypes.c_double),
    ]

class OptionGreeks(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("price", ctypes.c_double),
        ("delta", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("vega", ctypes.c_double),
        ("x_min", ctypes.c_double),
        ("dx", ctypes.c_double),
        ("Tn", ctypes.c_int),
        ("Xm", ctypes.c_int),
    ]

# ==============================================================================
# EXPLICIT FUNCTION SIGNATURE DECLARATIONS (LP64 SAFE MAPPINGS)
# ==============================================================================

# 1. Map Binomial Engine Functions cleanly
fdm_lib.price_binomial.argtypes = [
    ctypes.POINTER(OptionConfig),
    ctypes.c_int,
    ctypes.POINTER(OptionGreeks),
]
fdm_lib.price_binomial.restype = ctypes.c_int

# 2. Map Single FDM Engine Functions
fdm_lib.price_single.argtypes = [
    ctypes.POINTER(OptionConfig),
    ctypes.POINTER(OptionGreeks),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_bool,
]
fdm_lib.price_single.restype = ctypes.c_int

# 3. Map OpenMP Parallel Batch Pricing Functions
fdm_lib.price_batch.argtypes = [
    ctypes.POINTER(OptionConfig),
    ctypes.POINTER(OptionGreeks),
    ctypes.c_int,
    ctypes.c_bool,
]
fdm_lib.price_batch.restype = ctypes.c_int

GreeksArray6Type = OptionGreeks * 6

# ==============================================================================
# REFACTOR RUNTIME INTERFACE EXECUTION CALLS
# ==============================================================================

def fdm_price_binomial_all(config: OptionConfig, n: int):
    greeks_buffer = GreeksArray6Type()

    status = fdm_lib.price_binomial(
        ctypes.byref(config),
        ctypes.c_int(n),
        greeks_buffer
    )
    return status, greeks_buffer

def fdm_price_binomial(config: OptionConfig, n: int):
    greeks_buffer = GreeksArray6Type()

    status = fdm_lib.price_binomial(
        ctypes.byref(config),
        ctypes.c_int(n),
        greeks_buffer
    )
    return status, greeks_buffer[config.deriv]

def fdm_price_single(config: OptionConfig, calc_vega: bool = False):
    greeks = OptionGreeks()
    prices_ptr = ctypes.c_void_p()

    status = fdm_lib.price_single(
        ctypes.byref(config),
        ctypes.byref(greeks),
        ctypes.byref(prices_ptr),
        ctypes.c_bool(calc_vega),
    )
    return status, greeks, prices_ptr

def fdm_price_batch(c_configs_buffer, batch_size, calc_vega: bool = False):
    if batch_size == 0:
        return 0, []
    
    greeks_array_type = OptionGreeks * batch_size
    c_greeks_buffer = greeks_array_type()
    
    status = fdm_lib.price_batch(
        ctypes.cast(c_configs_buffer, ctypes.POINTER(OptionConfig)),
        ctypes.cast(c_greeks_buffer, ctypes.POINTER(OptionGreeks)),
        ctypes.c_int(batch_size),
        ctypes.c_bool(calc_vega),
    )
    return status, c_greeks_buffer
