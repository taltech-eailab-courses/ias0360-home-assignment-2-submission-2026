import csv

with open("results/results.csv") as f:
    rows = list(csv.DictReader(f))

print("| Model | epochs | lr | params | file KB | ms/digit | MNIST test acc | own digits |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['name']} | {r['epochs']} | {r['lr']} | {r['params']} | {r['file_kb']} | "
          f"{r['avg_ms']} | {r['val_acc']} | {r['own_acc']} |")