import pytest
from app.api import DerivType, OptionConfig, fdm_price_binomial, fdm_price_single
from tests import print_greeks

def test_american_put_fine():
    # 1. Initialize the input configuration values
    config = OptionConfig(
        deriv=DerivType.AmericanPut,
        Tn=300,
        right=0,
        left=0,
        time=1.0000,
        h=0.05,
        r=0.1,
        sigma=1.0000,
        s=0.1000,
        k=1.0000,
    )

    status, fdm_greeks, prices = fdm_price_single(config)

    assert status == 0

    if status != 0:  # Assuming 0 means success
        print(f"FDM Solver failed with status code: {status}")
    
    status, bin_greeks = fdm_price_binomial(config, 2048)
    
    assert status == 0
        
    print_greeks(config.deriv, fdm_greeks, bin_greeks)
    
    assert fdm_greeks.price > 0
    assert fdm_greeks.delta <= 0 and fdm_greeks.delta >= -1
    assert fdm_greeks.gamma < 0
    assert fdm_greeks.theta >= -0.05
    
    # Prices should match exceptionally closely (within 1-2 cents)
    assert fdm_greeks.price == pytest.approx(bin_greeks.price, abs=0.02)

    # Delta represents the first derivative (should match within a tiny margin)
    assert fdm_greeks.delta == pytest.approx(bin_greeks.delta, abs=0.01)

    # Gamma is the second derivative (highly sensitive to grid layout, use a wider abs gate)
    assert fdm_greeks.gamma == pytest.approx(bin_greeks.gamma, abs=0.005)

    # Theta is time-decay (highly sensitive to boundary shapes and smoothing, use an absolute buffer)
    assert fdm_greeks.theta == pytest.approx(bin_greeks.theta, abs=0.005)    
    
if __name__ == "__main__":
    test_american_put_fine()