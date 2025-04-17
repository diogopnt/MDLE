from pyspark import SparkContext
import csv
from itertools import combinations

sc = SparkContext(appName="AprioriAlgorithm")

data = sc.textFile("conditions.csv/conditions.csv")

# Extract and remove header
header = data.first()
rows = data.filter(lambda line: line != header)

# Parse each CSV line using Python's csv module 
def parse_csv_line(line):
    return next(csv.reader([line]))

# Extract relevant fields: (patient_id, condition_code)
patient_code_pairs = rows.map(parse_csv_line).map(lambda x: (x[2], x[4]))

# Group condition codes per patient, creating transactions
# Each transaction is a set of condition codes associated with a single patient
patient_transactions = patient_code_pairs.groupByKey().mapValues(lambda codes: set(codes)).cache()

# Minimum support threshold (itemsets must appear in at least 1000 patients)
min_support = 1000

# PHASE 1: Frequent 1-itemsets
# Count how many patients have each individual condition code (1-itemsets)
frequent_1_itemsets = (
    patient_transactions.flatMap(lambda x: [(item, 1) for item in x[1]])
    .reduceByKey(lambda a, b: a + b) # Sum the counts per condition code
    .filter(lambda x: x[1] >= min_support)  # Keep only those with support ≥ threshold
    .map(lambda x: x[0])  # Keep only item IDs
    .collect()
)

# Store frequent 1-itemsets in a set and broadcast to all workers
frequent_1_set = set(frequent_1_itemsets)  
broadcast_L1 = sc.broadcast(frequent_1_set)

# PHASE 2: Frequent 2-itemsets
# Generate candidate 2-itemsets from each transaction using only frequent items from L1
def gen_candidates_2(transaction):
    items = sorted([item for item in transaction if item in broadcast_L1.value])
    return combinations(items, 2)

# Count and filter frequent 2-itemsets
frequent_2_itemsets = (
    patient_transactions.flatMap(lambda x: [(c, 1) for c in gen_candidates_2(x[1])])
    .reduceByKey(lambda a, b: a + b) # Sum counts of each pair
    .filter(lambda x: x[1] >= min_support) # keep only those with sufficient support
    .cache()
)

# Collect L2 (frequent pairs) and broadcast
L2 = frequent_2_itemsets.map(lambda x: x[0]).collect()
broadcast_L2 = sc.broadcast(set(L2))

# PHASE 3: Frequent 3-itemsets
# Generate candidate 3-itemsets from L1, and prune if any 2-subset is not in L2
def gen_candidates_3(transaction):
    items = sorted([item for item in transaction if item in broadcast_L1.value])
    candidates = combinations(items, 3)
    
    # Helper function to check if all 2-subsets are frequent (i.e., in L2)
    def all_2subsets_frequent(triple):
        return all(tuple(sorted(pair)) in broadcast_L2.value for pair in combinations(triple, 2))
    return [c for c in candidates if all_2subsets_frequent(c)] # Prune candidates that contain infrequent 2-subsets

# Count and filter frequent 3-itemsets
frequent_3_itemsets = (
    patient_transactions.flatMap(lambda x: [(c, 1) for c in gen_candidates_3(x[1])])
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= min_support)
)

# Show top 10 for k=2
top_2 = frequent_2_itemsets.takeOrdered(10, key=lambda x: -x[1])
print("Top 10 frequent itemsets of size 2:")
for itemset, support in top_2:
    print(f"{itemset} -> support: {support}")

# Show top 10 for k=3
top_3 = frequent_3_itemsets.takeOrdered(10, key=lambda x: -x[1])
print("\nTop 10 frequent itemsets of size 3:")
for itemset, support in top_3:
    print(f"{itemset} -> support: {support}")

sc.stop()
