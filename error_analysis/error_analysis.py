import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys

def main():
    df_predicted = pd.read_csv("val_predictions.csv")
    df_predicted.columns = ["image_path","label"]
    df_gt = pd.read_csv("train_images.csv")
    df_merged = df_predicted.merge(df_gt, on="image_path", how="left", suffixes = ["_predicted", "_gt"])
    df_merged["correct"] = df_merged["label_predicted"] == df_merged["label_gt"]

    df_label_dist = pd.read_csv("label_dist.csv")
    df_merged = df_merged.merge(df_label_dist, left_on = "label_gt", right_on = "label")

    accuracy_per_class = (
    df_merged
    .groupby("label_gt")["correct"]
    .mean()
    .mul(100)
    .reset_index(name="accuracy_pct")
    )
    print(accuracy_per_class)
if __name__ == '__main__':
    main()
