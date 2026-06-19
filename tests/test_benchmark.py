import time
import random
from app.api import OptionConfig, OptionGreeks, DerivType, fdm_price_batch

def test_production_benchmark(batch_size=10000):
    print("==================================================")
    print("      🚀 STARTING HIGH-THROUGHPUT BATCH SWEEP     ")
    print("==================================================")
    print(f"Allocating contiguous vector array arrays for {batch_size:,} options...")

    # 1. Allocate a truly contiguous block of C-style memory for inputs and outputs
    configs_array_type = OptionConfig * batch_size
    
    configs_vector = configs_array_type()

    # 2. Populate the array fields with varying randomized market scenarios
    # (Forces the CPU to perform active calculations, preventing cache shortcuts)
    random.seed(42) # Anchor seed for reproducible runs
    for i in range(batch_size):
        configs_vector[i].deriv = random.choice([DerivType.AmericanPut, DerivType.BermudanPut, DerivType.VanillaPut,DerivType.AmericanCall, DerivType.BermudanCall, DerivType.VanillaCall])
        configs_vector[i].Tn = 100      # Keep your optimized 1000 time steps layout
        configs_vector[i].time = random.uniform(0.5, 1.0)
        configs_vector[i].h = 1         # Keep your fast 0.1 spatial mesh layout
        configs_vector[i].r = random.uniform(0.02, 0.12)
        configs_vector[i].sigma = random.uniform(0.15, 0.60)
        configs_vector[i].s = random.uniform(80.0, 120.0)
        configs_vector[i].k = 110.0
        configs_vector[i].q = 0.0

    print("Warming up CPU hardware cache registers...")
    # Warm up call to spin up OpenMP core threads to maximum frequency
    fdm_price_batch(configs_vector, 10)

    print(f"Executing batch run over {batch_size:,} options across all cores...")
    
    # 💥 THE CORE BOUNDARY TIMER: Measures strict hardware compute duration
    start_time = time.perf_counter()
    status, greeks_output_vector = fdm_price_batch(configs_vector, batch_size)
    end_time = time.perf_counter()

    duration_ms = (end_time - start_time) * 1000.0
    throughput = batch_size / (end_time - start_time)

    print("\n==================================================")
    print("             CPU BENCHMARK RESULTS                ")
    print("==================================================")
    print(f"  Batch Processing Status:    {'SUCCESS (0)' if status == 0 else 'FAILED'}")
    print(f"  Total Hardware Compute Time: {duration_ms:.3f} ms")
    print(f"  Average Latency Per Option:  {(duration_ms / batch_size):.4f} ms")
    print(f"  🚀 Calculated Throughput:    {throughput:,.2f} options/second")
    print("==================================================\n")

if __name__ == "__main__":
    test_production_benchmark()