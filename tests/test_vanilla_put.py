from app.api import DerivType, OptionConfig, fdm_price_binomial, fdm_price_single

def test_vanilla_put_fine():
    # 1. Initialize the input configuration values
    config = OptionConfig(
        deriv=DerivType.VanillaPut,
        Tn=300,
        time=1.0000,
        h=0.05,
        r=0.1,
        sigma=1.0000,
        s=0.1000,
        k=1.0000,
    )

    status, greeks, prices = fdm_price_single(config)

    assert status == 0

    if status == 0:  # Assuming 0 means success
        print("Vanilla Put ========")
        print(f"Price: {greeks.price}")
        print(f"Delta: {greeks.delta}")
        print(f"Gamma: {greeks.gamma}")
        print(f"Theta: {greeks.theta}")
        # Note: If C++ allocated memory here, you must free it later to avoid leaks!
    else:
        print(f"FDM Solver failed with status code: {status}")
    
    status, bin_greeks = fdm_price_binomial(config, 2048)
    
    assert status == 0
        
    print("\n================ Binomial Tree Greeks ================")
    print(f"Binomial Put --------")
    print(f"  S: {greeks.price:.8f}")
    print(f"  Δ: {bin_greeks.delta:.8f}")
    print(f"  Γ: {bin_greeks.gamma:.8f}")
    print(f"  Θ: {bin_greeks.theta:.8f}")
    print("======================================================")    
    
    assert greeks.price > 0
    assert greeks.delta <= 0 and greeks.delta >= -1
    assert greeks.gamma > 0
    assert greeks.theta > 0

if __name__ == "__main__":
    test_vanilla_put_fine()