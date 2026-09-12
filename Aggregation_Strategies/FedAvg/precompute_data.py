"""Run this once, by itself, before `flwr run`.

Downloads CIFAR10, builds the stratified 60/10/30 split, and precomputes
the Dirichlet client partitioning for the given num_partitions/alpha --
all sequentially, in this one process. This means no simulated client
process has to do any of this expensive work (or duplicate ~1GB+ of
tensors in memory) when the actual federated run starts; they'll just
read their small cached shard off disk.

Usage (from inside your app's installed environment):
    python precompute_data.py
"""
from Aggregation_Strategies.FedAvg.fedavg.dataset import DIRICHLET_ALPHA, _ensure_partitions, _ensure_raw_split

# Match these to what you'll actually pass to `flwr run` (num-supernodes /
# your run-config's alpha), so the cache this builds is the one your run
# will actually use.
NUM_PARTITIONS = 10
ALPHA = DIRICHLET_ALPHA

if __name__ == "__main__":
    print("Downloading CIFAR10 and building the stratified split...")
    _ensure_raw_split()
    print(f"Precomputing Dirichlet partitions (num_partitions={NUM_PARTITIONS}, alpha={ALPHA})...")
    _ensure_partitions(NUM_PARTITIONS, ALPHA)
    print("Done. Cache is ready -- `flwr run` should now start cleanly.")
