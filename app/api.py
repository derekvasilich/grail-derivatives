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
    
# 1. Define Enum Constants matching your C++ definitions
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

# 2. Mirror the OptionConfig Struct
class OptionConfig(ctypes.Structure):
    _fields_ = [
        ("deriv", ctypes.c_int),       # Maps to enum DerivType (4 bytes)
        ("barrier", ctypes.c_int),     # Maps to enum BarrierType (4 bytes)
        ("frequency", ctypes.c_int),   # Maps to enum FrequencyType (4 bytes)
        ("Tn", ctypes.c_int),          # int
        ("top", ctypes.c_int),         # int
        ("bottom", ctypes.c_int),      # int
        ("left", ctypes.c_int),        # int
        ("right", ctypes.c_int),       # int
        ("time", ctypes.c_double),     # double (8 bytes)
        ("h", ctypes.c_double),        # double
        ("r", ctypes.c_double),        # double
        ("sigma", ctypes.c_double),    # double
        ("s", ctypes.c_double),        # double
        ("k", ctypes.c_double),        # double
        ("q", ctypes.c_double),        # double
    ]

# 3. Mirror the OptionGreeks Struct
class OptionGreeks(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("delta", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("vega", ctypes.c_double),
        ("Tn", ctypes.c_int),
        ("Xm", ctypes.c_int),
    ]

def fdm_price_binomial_all(config: OptionConfig, n: int):
    greeks_array_type = OptionGreeks * 6
    greeks_buffer = greeks_array_type()

    # int price_single(const OptionConfig* config, OptionGreeks* out_greeks, double **prices);
    fdm_lib.price_single.argtypes = [
        ctypes.POINTER(OptionConfig),
        ctypes.c_int,
        ctypes.POINTER(OptionGreeks),
    ]
    fdm_lib.price_binomial.restype = ctypes.c_int

    status = fdm_lib.price_binomial(
        ctypes.byref(config),
        n,
        greeks_buffer  # Array objects pass automatically by reference pointer
    )

    # 4. Return the execution status and your populated Greeks array safely
    return status, greeks_buffer

def fdm_price_binomial(config: OptionConfig, n: int):
    greeks_array_type = OptionGreeks * 6
    greeks_buffer = greeks_array_type()

    # int price_single(const OptionConfig* config, OptionGreeks* out_greeks, double **prices);
    fdm_lib.price_single.argtypes = [
        ctypes.POINTER(OptionConfig),
        ctypes.c_int,
        ctypes.POINTER(OptionGreeks),
    ]
    fdm_lib.price_binomial.restype = ctypes.c_int

    status = fdm_lib.price_binomial(
        ctypes.byref(config),
        n,
        greeks_buffer  # Array objects pass automatically by reference pointer
    )

    # 4. Return the execution status and your populated Greeks array safely
    return status, greeks_buffer[config.deriv]

def fdm_price_single(config: OptionConfig, calc_vega: bool = False):
    greeks = OptionGreeks()

    # 3. Create an empty pointer to pointer for the matrix out-param
    prices_ptr = ctypes.c_void_p(0)
    
    # int price_single(const OptionConfig* config, OptionGreeks* out_greeks, double **prices);
    fdm_lib.price_single.argtypes = [
        ctypes.POINTER(OptionConfig),
        ctypes.POINTER(OptionGreeks),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_bool,
    ]
    fdm_lib.price_single.restype = ctypes.c_int

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
    
    fdm_lib.price_batch.argtypes = [
        ctypes.POINTER(OptionConfig),
        ctypes.POINTER(OptionGreeks),
        ctypes.c_int,
        ctypes.c_bool,
    ]
    fdm_lib.price_batch.restype = ctypes.c_int
    status = fdm_lib.price_batch(
        ctypes.cast(c_configs_buffer, ctypes.POINTER(OptionConfig)),
        ctypes.cast(c_greeks_buffer, ctypes.POINTER(OptionGreeks)),
        ctypes.c_int(batch_size),
        ctypes.c_bool(calc_vega),
    )
    return status, c_greeks_buffer