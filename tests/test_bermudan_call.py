import pytest
from app.api import DerivType, FrequencyType, OptionConfig, fdm_price_binomial, fdm_price_single
from tests import print_greeks

def test_bermudan_call_fine():
    # 1. Initialize the input configuration values
    config = OptionConfig(
        deriv=DerivType.BermudanCall,
        frequency=FrequencyType.Quarterly,
        Tn=1000,
        time=1.0000,
        h=0.1,
        r=0.1,
        sigma=0.5000,
        s=100,
        k=110,
        q=0.0
    )

    status, fdm_greeks, prices =  fdm_price_single(config)

    assert status == 0

    if status != 0:  # Assuming 0 means success
        print(f"FDM Solver failed with status code: {status}")
    
    status, bin_greeks = fdm_price_binomial(config, 2048)
    
    assert status == 0
    
    print_greeks(config.deriv, fdm_greeks, bin_greeks)
            
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


@pytest.mark.parametrize("maturity", [0.25, 0.5, 0.6, 1.5, 2.0])
def test_bermudan_call_matches_binomial_across_maturities(maturity):
    """Regression for the Bermudan exercise-schedule bug.

    The FDM's `isBermudanStepMatch` previously derived steps-per-interval as `Tn / frequency`,
    which only equals the right count when time == 1.0. So the FDM agreed with the binomial at
    1y but diverged at other maturities. This sweeps maturities — including the previously-broken
    short (0.25, 0.5) and long (2.0) cases plus a non-quarter-aligned one (0.6) — and requires
    the FDM to track the binomial's calendar-time exercise schedule.
    """
    config = OptionConfig(
        deriv=DerivType.BermudanCall,
        frequency=FrequencyType.Quarterly,
        Tn=1000,
        time=maturity,
        h=0.25,
        r=0.1,
        sigma=0.5000,
        s=100,
        k=110,
        q=0.0,
    )

    status, fdm_greeks, _ = fdm_price_single(config)
    assert status == 0

    status, bin_greeks = fdm_price_binomial(config, 2048)
    assert status == 0

    print_greeks(config.deriv, fdm_greeks, bin_greeks)

    assert fdm_greeks.price > 0
    assert 0 <= fdm_greeks.delta <= 1

    # Key regression guard: the schedule bug surfaced as a dollars-wide price gap off 1y.
    assert fdm_greeks.price == pytest.approx(bin_greeks.price, abs=0.05)
    assert fdm_greeks.delta == pytest.approx(bin_greeks.delta, abs=0.02)


if __name__ == "__main__":
    test_bermudan_call_fine()