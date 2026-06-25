from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "credit_card.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "plots"


def make_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def save_class_distribution(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Non-Fraud", "Fraud"], counts.values, color=["#2a9d8f", "#e76f51"])
    plt.title("Class Distribution")
    plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close()


def save_amount_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df,
        x="Amount",
        hue="Class",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        palette=["#2a9d8f", "#e76f51"],
    )
    plt.yscale("log")
    plt.title("Transaction Amount Distribution by Class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amount_distribution_by_class.png", dpi=200)
    plt.close()


def save_top_correlations(df: pd.DataFrame) -> None:
    correlations = df.corr(numeric_only=True)["Class"].drop("Class").abs().sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=correlations.values, y=correlations.index, color="#264653")
    plt.title("Top Absolute Correlations with Fraud Class")
    plt.xlabel("Absolute correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_correlations_to_class.png", dpi=200)
    plt.close()


def save_fraud_rate_by_time_decile(df: pd.DataFrame) -> None:
    time_bins = pd.qcut(df["Time"], q=10, duplicates="drop")
    fraud_rate = df.groupby(time_bins)["Class"].mean() * 100
    plt.figure(figsize=(10, 5))
    fraud_rate.plot(kind="bar", color="#457b9d")
    plt.title("Fraud Rate by Time Decile")
    plt.ylabel("Fraud rate (%)")
    plt.xlabel("Time decile")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fraud_rate_by_time_decile.png", dpi=200)
    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    class_counts = df["Class"].value_counts()
    class_pct = (class_counts / len(df) * 100).round(4)
    amount_summary = df.groupby("Class")["Amount"].describe().round(3)
    time_summary = df.groupby("Class")["Time"].describe().round(3)
    top_correlations = df.corr(numeric_only=True)["Class"].drop("Class").abs().sort_values(ascending=False).head(10)

    print("Dataset shape:", df.shape)
    print("\nClass counts:\n", class_counts)
    print("\nClass percentage:\n", class_pct)
    print("\nMissing values:", int(df.isna().sum().sum()))
    print("Duplicated rows:", int(df.duplicated().sum()))
    print("\nAmount summary by class:\n", amount_summary)
    print("\nTime summary by class:\n", time_summary)
    print("\nTop absolute correlations with Class:\n", top_correlations.round(4))


def main() -> None:
    make_output_dir()
    df = load_data()

    print_summary(df)
    save_class_distribution(df)
    save_amount_distribution(df)
    save_top_correlations(df)
    save_fraud_rate_by_time_decile(df)

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
