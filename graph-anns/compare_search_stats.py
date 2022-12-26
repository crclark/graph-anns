import pandas as pd

old = pd.read_csv('old_search_stats.csv')
new = pd.read_csv('new_search_stats.csv')

print("=== Old ===")
print(old.describe())

print("\n\n=== New ===")
print(new.describe())

print("\n\n")
