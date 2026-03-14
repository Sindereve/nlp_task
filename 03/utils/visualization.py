from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

def plot_top_n_plus_other(counts, top_n=20, figsize=(9, 6)):
    top = counts.nlargest(top_n)
    other = counts.drop(top.index).sum()
    
    plot_data = pd.concat([top, pd.Series({'Остальные': other})])
    
    plt.figure(figsize=figsize)
    sns.barplot(x=plot_data.values, y=plot_data.index, palette="mako")
    
    plt.title(f"Соотношение классов (топ-{top_n} + остальные)")
    plt.xlabel("Количество")
    plt.ylabel("Класс")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(plot_data.values):
        plt.text(v + max(plot_data)*0.005, i, f"{int(v):,}", va='center')
    
    plt.tight_layout()
    plt.show()

