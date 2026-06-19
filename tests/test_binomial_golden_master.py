import pytest
from app.api import DerivType, FrequencyType, OptionConfig, fdm_price_binomial_all

def test_binomial_golden_master_n2048():
    """
    Validates your high-resolution Binomial Tree layout against your 
    audited numerical baseline values for N = 2048.
    
    Parameters: S=100, K=110, σ=0.5, r=0.1, T=1.0, q=0.0
    """
    N = 2048
    # 1. Setup the exact configuration parameters from your audited run
    config = OptionConfig(
        deriv=DerivType.VanillaPut,
        frequency=FrequencyType.Quarterly,
        Tn=2048, top=0, bottom=1, left=0, right=0,
        time=1.0, h=0.05, r=0.1, sigma=0.5, s=100.0, k=110.0, q=0.0
    )

    # 2. Invoke your cache-aligned, thread-safe C++ binomial entry point
    status, bin_greeks_array = fdm_price_binomial_all(config, N)
    
    # Assert successful completion from the C++ layer
    assert status == 0

    print("====================== Binomial Tree Greeks =====================")
    print("      |       European    |      American     |      Bermudan    ")
    print("      |    Call     Put   |    Call     Put   |    Call     Put  ")
    print("----- | -------- -------- | -------- -------- | -------- --------")
    print("  S   |", *(f"{bin_greeks_array[i].price:8.4f}" + (" |" if (i % 2 != 0 and i < 5) else "") for i in range(6)))
    print("  Δ   |", *(f"{bin_greeks_array[i].delta:8.4f}" + (" |" if (i % 2 != 0 and i < 5) else "") for i in range(6)))
    print("  Γ   |", *(f"{bin_greeks_array[i].gamma:8.4f}" + (" |" if (i % 2 != 0 and i < 5) else "") for i in range(6)))
    print("  Θ   |", *(f"{bin_greeks_array[i].theta:8.4f}" + (" |" if (i % 2 != 0 and i < 5) else "") for i in range(6)))
    print("=================================================================")

    # 3. Unpack your 6 contiguous option array rows cleanly
    euro_call  = bin_greeks_array[0]
    euro_put   = bin_greeks_array[1]
    amer_call  = bin_greeks_array[2]
    amer_put   = bin_greeks_array[3]
    berm_call  = bin_greeks_array[4]
    berm_put   = bin_greeks_array[5]

    # ============================================================
    # 📈 GOLDEN MASTER MATRIX PRICE ASSERTIONS (Within 0.0005)
    # ============================================================
    assert euro_call.price == pytest.approx(19.9313, abs=5e-4)
    assert euro_put.price  == pytest.approx(19.4635, abs=5e-4)
    
    assert amer_call.price == pytest.approx(19.9313, abs=5e-4) # Proven identical to Euro Call (q=0)
    assert amer_put.price  == pytest.approx(21.2662, abs=5e-4)
    
    assert berm_call.price == pytest.approx(19.9313, abs=5e-4)
    assert berm_put.price  == pytest.approx(20.8684, abs=5e-4)

    # ============================================================
    # 🎯 GOLDEN MASTER GREEKS ASSERTIONS (Within 0.0005)
    # ============================================================
    
    # European Greeks Verification
    assert euro_call.delta == pytest.approx(0.6023,  abs=5e-4)
    assert euro_call.gamma == pytest.approx(0.0077,  abs=5e-4)
    assert euro_call.theta == pytest.approx(-0.0375, abs=5e-4)
    
    assert euro_put.delta  == pytest.approx(-0.3977, abs=5e-4)
    assert euro_put.gamma  == pytest.approx(0.0077,  abs=5e-4)
    assert euro_put.theta  == pytest.approx(-0.0102, abs=5e-4)

    # American Greeks Verification
    assert amer_put.delta  == pytest.approx(-0.4561, abs=5e-4)
    assert amer_put.gamma  == pytest.approx(0.0098,  abs=5e-4)
    assert amer_put.theta  == pytest.approx(-0.0154, abs=5e-4)

    # Bermudan Greeks Verification
    assert berm_put.delta  == pytest.approx(-0.4492, abs=5e-4)
    assert berm_put.gamma  == pytest.approx(0.0097,  abs=5e-4)
    assert berm_put.theta  == pytest.approx(-0.0150, abs=5e-4)

    # ============================================================
    # ⚖️ FINANCIAL REGULATORY RULES SUB-CHECKS
    # ============================================================
    # Verify Put-Call Parity: Call - Put = S - K * exp(-r*T)
    parity_gap = euro_call.price - euro_put.price
    assert parity_gap == pytest.approx(0.4679, abs=5e-4)

# test_binomial_golden_master_n2048()