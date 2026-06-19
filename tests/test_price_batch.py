import pytest
from app.black_api import DerivType, FrequencyType, OptionConfig, fdm_price_binomial_all, fdm_price_batch
from app.black_api import deriv_labels
from app.tests import print_greeks

def test_price_batch_small():
    # 1. Generate the contiguous array class type layout (OptionConfig * 6)
    configs_array_type = OptionConfig * 6
    
    # 2. Instantiate the actual memory block array
    configs = configs_array_type()
    
    config = OptionConfig(
        deriv=DerivType.BermudanCall,
        frequency=FrequencyType.Quarterly,
        Tn=1000,
        time=1.0000,
        h=1,
        r=0.1,
        sigma=0.5000,
        s=100,
        k=110,
        q=0.0
    )
    
    # 3. Populate all 6 slots with your configuration data properties safely
    for i in range(6):
        # Copy fields directly from your base config object
        for field_name, _ in config._fields_:
            setattr(configs[i], field_name, getattr(config, field_name))
        configs[i].deriv = i
    
    status, all_fdm_greeks =  fdm_price_batch(configs, 6)

    assert status == 0

    status, all_bin_greeks = fdm_price_binomial_all(config, 2048)

    for i in range(6):
        fdm_greeks = all_fdm_greeks[i]
        bin_greeks = all_bin_greeks[i]
                
        assert status == 0

        print_greeks(i, fdm_greeks, bin_greeks)
                
        assert fdm_greeks.price > 0
        assert fdm_greeks.delta >= -1 and fdm_greeks.delta <= 1
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

# price_batch_small()