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
    .collectAsMap()
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
    .collectAsMap()
)

# Collect L2 (frequent pairs) and broadcast
L2 = list(frequent_2_itemsets.keys())
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
    .collectAsMap()
)

# Show top 10 for k=2
top_2 = sorted(frequent_2_itemsets.items(), key=lambda x: -x[1])[:10]
print("Top 10 frequent itemsets of size 2:")
for itemset, support in top_2:
    print(f"{itemset} -> support: {support}")

# Show top 10 for k=3
top_3 = sorted(frequent_3_itemsets.items(), key=lambda x: -x[1])[:10]
print("\nTop 10 frequent itemsets of size 3:")
for itemset, support in top_3:
    print(f"{itemset} -> support: {support}")

# -- B2 -- 

# List to collect all rules with their metrics
association_rules = []

def calculate_rule_metrics(antecedent, consequent, full_itemset_support):
    antecedent = tuple(sorted(antecedent))
    consequent = tuple(sorted(consequent))

    # Obter os suportes individuais
    if len(antecedent) == 1:
        support_A = frequent_1_itemsets.get(antecedent[0], 0)
    else:
        support_A = frequent_2_itemsets.get(antecedent, 0)

    if len(consequent) == 1:
        support_B = frequent_1_itemsets.get(consequent[0], 0)
    else:
        support_B = frequent_2_itemsets.get(consequent, 0)

    if support_A == 0 or support_B == 0:
        return None

    # Calcular métricas principais
    confidence = full_itemset_support / support_A
    prob_A = support_A / num_transactions
    prob_B = support_B / num_transactions

    lift = confidence / prob_B
    interest = confidence - prob_B

    # Calcular lift máximo e mínimo
    try:
        lift_max = min(1 / prob_B, 1 / prob_A)
    except ZeroDivisionError:
        lift_max = float('inf')

    try:
        lift_min = max((prob_A + prob_B - 1) / (prob_A * prob_B), 0)
    except ZeroDivisionError:
        lift_min = 0

    # Calcular standardized lift
    if lift_max == lift_min:
        standardized_lift = 0  # evitar divisão por zero
    else:
        standardized_lift = (lift - lift_min) / (lift_max - lift_min)

    return {
        "antecedent": antecedent,
        "consequent": consequent,
        "confidence": confidence,
        "interest": interest,
        "lift": lift,
        "standardized_lift": standardized_lift
    }

# Count total number of transactions (patients)
num_transactions = patient_transactions.count()

# Generate rules from frequent 2-itemsets
for itemset, supp in frequent_2_itemsets.items():
    items = list(itemset)
    for i in range(1, len(items)):
        for A in combinations(items, i):
            B = tuple(sorted(set(items) - set(A)))
            metrics = calculate_rule_metrics(A, B, supp)
            if metrics:
                association_rules.append(metrics)

# Generate rules from frequent 3-itemsets
for itemset, supp in frequent_3_itemsets.items():
    items = list(itemset)
    for i in range(1, len(items)):
        for A in combinations(items, i):
            B = tuple(sorted(set(items) - set(A)))
            metrics = calculate_rule_metrics(A, B, supp)
            if metrics:
                association_rules.append(metrics)

# Sort all rules by interest (descending)
top_rules_by_sl = sorted(association_rules, key=lambda x: -x["standardized_lift"])[:10]

print("\nTop 10 Association Rules by Interest:")
for rule in top_rules_by_sl:
    antecedent = ", ".join(rule["antecedent"])
    consequent = ", ".join(rule["consequent"])
    print(f"{antecedent} -> {consequent} | SL: {rule['standardized_lift']:.4f}, lift: {rule['lift']:.4f}, conf: {rule['confidence']:.4f}, interest: {rule['interest']:.4f}")
    
'''
A = '126906006'
B = '92691004'
AB = tuple(sorted([A, B]))

support_A = frequent_1_itemsets.get(A, 0)
support_B = frequent_1_itemsets.get(B, 0)
support_AB = frequent_2_itemsets.get(AB, 0)

print(f"\n--- Verificação manual para {A} -> {B} ---")
print(f"Suporte A (126906006): {support_A}")
print(f"Suporte B (92691004): {support_B}")
print(f"Suporte A ∪ B: {support_AB}")
print(f"Total de transações: {num_transactions}")


prob_B = support_B / num_transactions
conf = support_AB / support_A
interest = conf - prob_B
lift = conf / prob_B

# Lift máximo e mínimo
prob_A = support_A / num_transactions
lift_max = min(1 / prob_B, 1 / prob_A)
lift_min = max((prob_A + prob_B - 1) / (prob_A * prob_B), 0)

if lift_max == lift_min:
    standardized_lift = 0
else:
    standardized_lift = (lift - lift_min) / (lift_max - lift_min)

# Mostrar os resultados calculados
print(f"\nConfiança calculada: {conf:.4f}")
print(f"Interesse calculado: {interest:.4f}")
print(f"Lift calculado: {lift:.4f}")
print(f"Standardized Lift calculado: {standardized_lift:.4f}")
print(f"Lift max: {lift_max:.4f}, Lift min: {lift_min:.4f}") 
'''
    
sc.stop()
    
# Diferença entre RDD´s e Dicionários