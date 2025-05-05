from pyspark import SparkContext
import csv
from itertools import combinations

sc = SparkContext(appName="AprioriAlgorithm-Final")

data = sc.textFile("conditions.csv/conditions.csv")

# Extract and remove header
header = data.first()
rows = data.filter(lambda line: line != header)

# Parse each CSV line using Python's csv module 
def parse_csv_line(line):
    return next(csv.reader([line]))

# Extract relevant fields: (CODE, DESCRIPTION)
code_to_desc = (
    rows.map(parse_csv_line)
        .map(lambda x: (x[4], x[5]))  # (CODE, DESCRIPTION)
        .distinct()
        .collectAsMap()
)

# Extract relevant fields: (patient_id, condition_code)
patient_code_pairs = rows.map(parse_csv_line).map(lambda x: (x[2], x[4]))

# Group condition codes per patient, creating transactions
# Each transaction is a set of condition codes associated with a single patient
patient_transactions = patient_code_pairs.groupByKey().mapValues(lambda codes: set(codes)).cache()

# Minimum support threshold (itemsets must appear in at least 1000 patients)
min_support = 1000

# PHASE 1: Frequent 1-itemsets
# Count how many patients have each individual condition code (1-itemsets)
frequent_1_itemsets_with_counts = (
    patient_transactions.flatMap(lambda x: [(item, 1) for item in x[1]])
    .reduceByKey(lambda a, b: a + b) # Sum the counts per condition code
    .filter(lambda x: x[1] >= min_support)  # Keep only those with support ≥ threshold
    .cache()
)

# Extract just the item IDs from the frequent 1-itemsets for broadcasting
frequent_1_itemsets = frequent_1_itemsets_with_counts.map(lambda x: x[0]).collect()

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
    .filter(lambda x: x[1] >= min_support) # Keep only those with sufficient support
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
    
    # Helper function to check if all 2-subsets are frequent
    def all_2subsets_frequent(triple):
        return all(tuple(sorted(pair)) in broadcast_L2.value for pair in combinations(triple, 2))
    return [c for c in candidates if all_2subsets_frequent(c)] # Prune candidates that contain infrequent 2-subsets

# Count and filter frequent 3-itemsets
frequent_3_itemsets = (
    patient_transactions.flatMap(lambda x: [(c, 1) for c in gen_candidates_3(x[1])])
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= min_support)
    .cache()
)

# Take Top 10 most frequent itemsets for k = 1, 2, 3
top_1 = frequent_1_itemsets_with_counts.takeOrdered(10, key=lambda x: -x[1])
top_2 = frequent_2_itemsets.takeOrdered(10, key=lambda x: -x[1])
top_3 = frequent_3_itemsets.takeOrdered(10, key=lambda x: -x[1])

# Count of frequent itemsets
count_1 = frequent_1_itemsets_with_counts.count()
count_2 = frequent_2_itemsets.count()
count_3 = frequent_3_itemsets.count()

def conditions(itemset):
    if isinstance(itemset, str):
        return f"{itemset} ({code_to_desc.get(itemset, 'Unknown')})"
    return ", ".join(f"{item} ({code_to_desc.get(item, 'Unknown')})" for item in itemset)


with open("B1_output.txt", "w") as f:
    f.write("=== Frequent Itemsets Analysis ===\n\n")
    
    f.write(f"Count of frequent itemsets (size 1): {count_1}\n")
    f.write(f"Count of frequent itemsets (size 2): {count_2}\n")
    f.write(f"Count of frequent itemsets (size 3): {count_3}\n\n")
    
    f.write("Top 10 frequent itemsets of size 1:\n")
    for item, support in top_1:
        f.write(f"{conditions(item)} -> support: {support}\n")
    
    f.write("\nTop 10 frequent itemsets of size 2:\n")
    for itemset, support in top_2:
        f.write(f"{conditions(itemset)} -> support: {support}\n")
    
    f.write("\nTop 10 frequent itemsets of size 3:\n")
    for itemset, support in top_3:
        f.write(f"{conditions(itemset)} -> support: {support}\n")

print("Results saved to B1_output.txt")

sc.stop()