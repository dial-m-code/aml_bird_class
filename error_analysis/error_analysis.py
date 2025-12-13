import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys

from collections import Counter

def main():
    df_predicted = pd.read_csv("val_predictions.csv")
    df_predicted.columns = ["image_path","label"]
    df_gt = pd.read_csv("train_images.csv")
    df_merged = df_predicted.merge(df_gt, on="image_path", how="left", suffixes = ["_predicted", "_gt"])
    df_merged["correct"] = df_merged["label_predicted"] == df_merged["label_gt"]

    df_label_dist = pd.read_csv("label_dist.csv")
    df_merged = df_merged.merge(df_label_dist, left_on = "label_gt", right_on = "label")

    df_baseline = pd.read_csv("baseline_predictions.csv")
    df_own_model = pd.read_csv("predictions_own_model.csv")

    df_models_merged = df_own_model.merge(df_baseline, on="id", suffixes=["_own","_base"])
    df_models_merged["agreement"] = df_models_merged["label_own"] == df_models_merged["label_base"]
    print(df_models_merged.head())

    accuracy_per_class = (
    df_models_merged
    .groupby("label_base")["agreement"]
    .mean()
    .mul(100)
    .reset_index(name="agreement_pct")
    )
    accuracy_per_class = accuracy_per_class.merge(df_label_dist, left_on="label_base", right_on="label")
    df_models_merged_error = df_models_merged[df_models_merged["agreement"] == False]
    c=Counter(df_models_merged_error[["label_own", "label_base"]])
    print(c.most_common())
    sys.exit()





    plt.figure(figsize=(8,4.5), dpi=300)
    sns.regplot(accuracy_per_class, x="num_images", y="agreement_pct", scatter=False,
    lowess=True,
    color="#0797df",
    line_kws={'linewidth':10})
    #plt.title("Agreement vs. image count in train set", fontweight="bold")
    plt.xlabel("Number of images in train set", fontsize=15, fontweight="bold")
    plt.ylabel("Agreement baseline & own model", fontsize=15, fontweight="bold")
    #plt.show()
    plt.savefig("agreement_class_size.png")
if __name__ == '__main__':
    main()
