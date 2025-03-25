import pandas as pd
import os
import matplotlib.pyplot as plt

def compare_labels_and_save():

    classification_file = "/home/rtliu/catkin_ws/src/online_klassifikation_pkg/classification_results/classification_results.csv"
    df_classification = pd.read_csv(classification_file)

    Interval = 0.01
    window_size = int(50/(100*Interval))
    print(window_size)
    kontakt = 'K'
    Index = 15
    
    real_label = df_classification.iloc[:, 1].reset_index(drop=True)

    df_combined = pd.DataFrame({
        "real_label": real_label
    })

    output_dir = f"/home/rtliu/catkin_ws/src/online_klassifikation_pkg/classification_results"
    os.makedirs(output_dir, exist_ok=True)  

    output_file = os.path.join(output_dir, f"label_comparison_{kontakt}{Index:02d}.csv")

    df_combined.to_csv(output_file, index=False)

    print(f"neue Dokument: {output_file}")
    plt.figure(figsize=(12, 6))
    plt.plot(df_combined["real_label"].values, label="Real Label", linestyle="solid", color='red',linewidth=0.6)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Label")
    plt.title("Real Labels Over Time")
    plt.legend()
    plt.grid(True)
    
    # **保存图像**
    plot_file = os.path.join(output_dir, f"label_1.png")
    plt.savefig(plot_file, dpi=300)
    plt.show()

if __name__ == "__main__":
    compare_labels_and_save()
