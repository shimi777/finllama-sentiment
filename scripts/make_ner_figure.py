"""Two-panel NER comparison figure (FiNER-ORD strict-F1, FIN/Alvarado micro-F1),
highlighting the financial-tuned plutus-8B. Reads the two committed NER tables."""
import csv, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "key_figures" / "ner_comparison.png"

def short(m):
    return {"gliner-large":"GLiNER-L","gliner-small":"GLiNER-S","mistral7b":"Mistral-7B",
            "plutus8b":"plutus-8B","qwen25_7b":"Qwen2.5-7B","qwen3_4b":"Qwen3-4B",
            "qwen3_8b":"Qwen3-8B"}.get(m, m)

# FiNER-ORD (strict_f1)
fin_ord = []
for r in csv.DictReader((ROOT/"results/summary_ner/final_table_ner.csv").open()):
    fin_ord.append((short(r["model"]), float(r["strict_f1"])))
fin_ord.sort(key=lambda x: x[1], reverse=True)

# FIN/Alvarado (micro_f1)
fin = []
for r in csv.DictReader((ROOT/"results/summary/ner_table.csv").open()):
    fin.append((short(r["model"]), float(r["micro_f1"])))
fin.sort(key=lambda x: x[1], reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, data, title, ylab in [
    (axes[0], fin_ord, "FiNER-ORD (entity strict-F1)", "strict micro-F1"),
    (axes[1], fin, "FIN / Alvarado-2015 (entity micro-F1)", "micro-F1"),
]:
    names = [d[0] for d in data]; vals = [d[1] for d in data]
    colors = ["#c0392b" if "plutus" in n else "#5b8db8" for n in names]
    bars = ax.bar(names, vals, color=colors)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_ylabel(ylab); ax.set_ylim(0, max(vals)*1.25)
    ax.tick_params(axis="x", rotation=35)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+max(vals)*0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    for lbl in ax.get_xticklabels():
        if "plutus" in lbl.get_text(): lbl.set_color("#c0392b"); lbl.set_weight("bold")

fig.suptitle("NER: the financial-tuned model (plutus-8B, red) is worst on both datasets",
             fontsize=12, weight="bold")
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
