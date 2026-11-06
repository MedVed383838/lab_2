import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def load_brain_data(path: str = "brainsize.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", na_values=["NA"])


def plot_scatter(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(20, 4), sharey=True)
    for ax, feature in zip(axes, FEATURES):
        sns.scatterplot(data=df, x=feature, y="MRI_Count", hue="Gender", ax=ax)
        ax.set_title(f"{feature} и MRI_Count")
    plt.tight_layout()
    plt.savefig("scatter_plots.png", dpi=300, bbox_inches="tight")


def plot_heatmaps(df: pd.DataFrame) -> None:
    columns = FEATURES + ["MRI_Count"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, gender, title in zip(axes, ["Female", "Male"], ["Женщины", "Мужчины"]):
        corr = df[df["Gender"] == gender][columns].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("heatmaps.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    data = load_brain_data()
    plot_scatter(data)
    plot_heatmaps(data)
    plt.show()
