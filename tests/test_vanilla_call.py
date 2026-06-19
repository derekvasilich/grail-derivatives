import pytest
from app.black_api import DerivType, OptionConfig, fdm_price_binomial, fdm_price_single

def test_vanilla_call_fine():
    # 1. Initialize the input configuration values
    config = OptionConfig(
        deriv=DerivType.VanillaCall,
        Tn=300,
        time=1.0000,
        h=0.05,
        r=0.1,
        sigma=1.0000,
        s=1.0000,
        k=1.0000,
    )

    status, fdm_greeks, prices =  fdm_price_single(config)

    assert status == 0

    if status == 0:  # Assuming 0 means success
        print("Vanilla Call ========")
        print(f"Price: {fdm_greeks.price}")
        print(f"Delta: {fdm_greeks.delta}")
        print(f"Gamma: {fdm_greeks.gamma}")
        print(f"Theta: {fdm_greeks.theta}")
        # Note: If C++ allocated memory here, you must free it later to avoid leaks!
    else:
        print(f"FDM Solver failed with status code: {status}")
    
    status, bin_greeks = fdm_price_binomial(config, 2048)
    
    assert status == 0
        
    print("\n================ Binomial Tree Greeks ================")
    print(f"Binomial Call --------")
    print(f"  S: {bin_greeks.price:.8f}")
    print(f"  Δ: {bin_greeks.delta:.8f}")
    print(f"  Γ: {bin_greeks.gamma:.8f}")
    print(f"  Θ: {bin_greeks.theta:.8f}")
    print("======================================================")     
    
    assert fdm_greeks.price > 0
    assert fdm_greeks.delta >= 0 and fdm_greeks.delta <= 1
    assert fdm_greeks.gamma > 0
    assert fdm_greeks.theta < 0
    
    # Prices should match exceptionally closely (within 1-2 cents)
    assert fdm_greeks.price == pytest.approx(bin_greeks.price, abs=0.02)

    # Delta represents the first derivative (should match within a tiny margin)
    assert fdm_greeks.delta == pytest.approx(bin_greeks.delta, abs=0.01)

    # Gamma is the second derivative (highly sensitive to grid layout, use a wider abs gate)
    assert fdm_greeks.gamma == pytest.approx(bin_greeks.gamma, abs=0.005)

    # Theta is time-decay (highly sensitive to boundary shapes and smoothing, use an absolute buffer)
    assert fdm_greeks.theta == pytest.approx(bin_greeks.theta, abs=0.005)    

# test_vanilla_call_fine()