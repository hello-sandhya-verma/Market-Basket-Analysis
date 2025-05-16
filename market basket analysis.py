import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------------------------
# Step 1: Simulate Retail Transactions
# ---------------------------------------------
transactions = [
    ['milk', 'bread', 'nuts', 'apple'],
    ['milk', 'bread', 'nuts'],
    ['milk', 'bread'],
    ['milk', 'apple'],
    ['bread', 'apple']
]

# ---------------------------------------------
# Step 2: One-Hot Encoding
# ---------------------------------------------
encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)
df = pd.DataFrame(encoded_array, columns=encoder.columns_)

print("\n=== One-Hot Encoded Transactions ===\n")
print(df.astype(int))

# ---------------------------------------------
# Step 3: Find Frequent Itemsets
# ---------------------------------------------
min_support = 0.6
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

print(f"\n=== Frequent Itemsets (Support ≥ {int(min_support * 100)}%) ===\n")
for _, row in frequent_itemsets.iterrows():
    items = ', '.join(row['itemsets'])
    support = round(row['support'] * 100, 2)
    print(f" - [{items}] appears in {support}% of transactions.")

# ---------------------------------------------
# Step 4: Generate Association Rules
# ---------------------------------------------
# Use confidence as metric instead of lift to capture more rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# Human-readable explanation and labels for chart
explained_rules = []
simple_labels = []

print("\n=== Association Rule Insights ===\n")

for idx, rule in rules.iterrows():
    lhs = ', '.join(rule['antecedents'])
    rhs = ', '.join(rule['consequents'])
    support = round(rule['support'] * 100, 2)
    confidence = round(rule['confidence'] * 100, 2)
    lift = round(rule['lift'], 2)

    print(f"{idx + 1}. If a customer buys [{lhs}], they may also buy [{rhs}] "
          f"(Support: {support}%, Confidence: {confidence}%, Lift: {lift})")

    simple_labels.append(f"{lhs} → {rhs}")
    explained_rules.append((confidence, lift))

# ---------------------------------------------
# Step 5: Visualize Top Rules
# ---------------------------------------------
top_n = 5

if len(explained_rules) == 0:
    print("\nNo association rules found that meet the confidence threshold.")
else:
    sorted_idx = sorted(
        range(len(explained_rules)),
        key=lambda i: explained_rules[i][1],  # sort by lift
        reverse=True
    )[:top_n]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        range(len(sorted_idx)),
        [explained_rules[i][1] for i in sorted_idx],  # lift values
        color='lightcoral'
    )

    plt.xticks(
        ticks=range(len(sorted_idx)),
        labels=[simple_labels[i] for i in sorted_idx],
        rotation=45,
        ha='right',
        fontsize=12
    )

    plt.ylabel("Lift Score", fontsize=12)
    plt.title("Top Association Rules (by Lift)", fontsize=14)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add confidence % above each bar
    for i, idx in enumerate(sorted_idx):
        conf, lift = explained_rules[idx]
        plt.text(i, lift + 0.02, f"Conf: {conf:.0f}%", ha='center', fontsize=10)

    plt.show()
