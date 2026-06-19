from app.api import DerivType, OptionGreeks, deriv_labels


def print_greeks(i: DerivType, fdm_greeks: OptionGreeks, bin_greeks: OptionGreeks):
    print(f"\n{deriv_labels[i]} ========")
    print(f"Vₛ: {fdm_greeks.price:.4f}, Δ: {fdm_greeks.delta:.4f}, Γ: {fdm_greeks.gamma:.4f}, Θ: {fdm_greeks.theta:.4f}")
    print(f"Vₑ: {bin_greeks.price:.4f}, Δ: {bin_greeks.delta:.4f}, Γ: {bin_greeks.gamma:.4f}, Θ: {bin_greeks.theta:.4f}")
