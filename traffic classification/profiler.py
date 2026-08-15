import time
import cProfile
import pstats
import joblib
import numpy as np

# 1. Load the trained pipeline
# Ensure the .pkl file is in the same directory
print("Loading model...")
model = joblib.load('encrypted_traffic_classifier_reused.pkl')

# Your pipeline selects 15 features via SelectKBest
# We simulate a normalized input array with 15 features
num_features = 15

# 2. Measure Latency (Single Prediction)
dummy_single = np.random.rand(1, num_features)
start_latency = time.perf_counter()
model.predict(dummy_single)
latency_ms = (time.perf_counter() - start_latency) * 1000

print("-" * 30)
print(f"Single Prediction Latency: {latency_ms:.4f} ms")

# 3. Measure Throughput (Batch of 1,000 Predictions)
dummy_batch = np.random.rand(1000, num_features)
start_throughput = time.perf_counter()
model.predict(dummy_batch)
throughput_sec = time.perf_counter() - start_throughput

print(f"Batch (1,000) Processing Time: {throughput_sec:.4f} seconds")
print(f"Estimated Throughput: {1000 / throughput_sec:.0f} predictions / second")
print("-" * 30)

# 4. cProfile the Batch Prediction
print("Running cProfile on Batch Prediction...\n")
profiler = cProfile.Profile()
profiler.enable()
model.predict(dummy_batch)
profiler.disable()

# Print the top 10 most time-consuming function calls
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(10)