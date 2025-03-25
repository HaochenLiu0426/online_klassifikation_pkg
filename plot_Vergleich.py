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
    sliding_window_file = f"/home/rtliu/catkin_ws/src/Test_Datensatz/Trainingsdatensatz_data_nach_schneiden_test.csv" # Trainingsdatensatz_data_FTS_02
    # sliding_window_file = f"/home/rtliu/catkin_ws/src/MA_Liu_1ms/Trainingsdatensatz_data_FTS_02.csv" # Trainingsdatensatz_data_FTS_02
    # sliding_window_file = f"/home/rtliu/catkin_ws/src/MA_Liu/Trainingsdatensatz_data_FTS_02.csv" # Trainingsdatensatz_data_FTS_02
    # sliding_window_file = f"/home/rtliu/catkin_ws/src/04. Sliding Window_nach Entfernung/Window_{window_size}_op/SW_{kontakt}{Index:02d}.csv"
    df_sliding_window = pd.read_csv(sliding_window_file)
    
    # time_column = df_sliding_window.iloc[:, 0]
    # ideal_label = df_sliding_window.iloc[:, -1]
    # real_label = df_classification.iloc[:, 1].reset_index(drop=True)
    time_column = df_sliding_window.iloc[window_size:, 0].reset_index(drop=True)
    ideal_label = df_sliding_window.iloc[window_size:, -1].reset_index(drop=True)
    real_label = df_classification.iloc[:, 1].reset_index(drop=True)
    print(time_column.shape)  # 显示数据的形状
    print(ideal_label.shape)
    print(real_label.shape)


    df_combined = pd.DataFrame({
        "time": time_column,
        "ideal_label": ideal_label,
        "real_label": real_label
    })
    print(df_combined.shape)

    output_dir = f"/home/rtliu/catkin_ws/src/online_klassifikation_pkg/Vergleich/Window_{window_size}"
    os.makedirs(output_dir, exist_ok=True)  

    output_file = os.path.join(output_dir, f"label_comparison_{kontakt}{Index:02d}.csv")

    df_combined.to_csv(output_file, index=False)

    print(f"neue Dokument: {output_file}")
    plt.figure(figsize=(12, 6))
    plt.plot(df_combined["time"].values, df_combined["ideal_label"].values, label="Ideal Label", linestyle="dashed", color='blue',linewidth=1)


    plt.plot(df_combined["time"].values, df_combined["real_label"].values, label="Real Label", linestyle="solid", color='red',linewidth=0.6)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Label")
    plt.title("Comparison of Ideal and Real Labels Over Time")
    plt.legend()
    plt.grid(True)
    
    # **保存图像**
    plot_file = os.path.join(output_dir, f"label_comparison_{kontakt}{Index:02d}.png")
    plt.savefig(plot_file, dpi=300)
    plt.show()

if __name__ == "__main__":
    compare_labels_and_save()
