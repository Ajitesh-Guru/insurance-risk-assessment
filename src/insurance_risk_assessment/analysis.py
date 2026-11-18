import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_eda_report(data: pd.DataFrame, output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    data["sex"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax, labels=["Male", "Female"]
    )
    ax.set(title="Sex composition", ylabel="")
    fig.savefig(output_dir / "sex_composition.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(data["age"], bins=20)
    ax.set(title="Age composition", xlabel="Age", ylabel="Count")
    fig.savefig(output_dir / "age_composition.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.stripplot(data=data, x="smoker", y="charges", jitter=True, ax=ax)
    ax.set(title="Charges by smoking status", xlabel="Smoker", ylabel="Charges")
    fig.savefig(output_dir / "charges_by_smoker.png")
    plt.close(fig)

    region_charges = data.groupby("region")["charges"].mean().sort_values()
    fig, ax = plt.subplots()
    region_charges.plot.bar(ax=ax)
    ax.set(title="Average charges by region", xlabel="Region", ylabel="Average charge")
    fig.tight_layout()
    fig.savefig(output_dir / "charges_by_region.png")
    plt.close(fig)
