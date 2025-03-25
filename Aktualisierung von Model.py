import yaml
import os

parameters = {
    "Entscheidungsbaum": {
        "minS2_minL1": [30, 40, 50],
        "minS10_minL5": [30, 40, 50],
        "minS100_minL40": [30, 40, 50]
    },
    "Random_Forest": {
        "minS10_minL5": [30, 40, 50],
        "minS50_minL20": [30, 40, 50],
        "minS100_minL40": [30, 40, 50]
    },
    "SVM": {
        "": [30, 40, 50]  
    }
}

split_methods = {
    "Entscheidungsbaum": ["Split", "Split_SMOTE", "SMOTE_Split"],
    "Random_Forest": ["Split", "Split_SMOTE", "SMOTE_Split"],
    "SVM": ["Split"]  # SVM enthält Split und Split_SMOTE
}


def choose_option(prompt, options):
    print(prompt)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    while True:
        choice = input("Bitte geben Sie die Nummer ein: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Ungültige Eingabe. Bitte erneut versuchen.")

# Modelltyp
model_type = choose_option("Bitte wählen Sie den Modelltyp:", list(parameters.keys()))

# minSminL
if model_type == "SVM":
    minSminL = "" 
else:
    minSminL = choose_option("Bitte wählen Sie minSminL:", list(parameters[model_type].keys()))

# window_size
window_size = choose_option("Bitte wählen Sie die Fenstergröße (window_size):", parameters[model_type][minSminL])

# split_method
split_method = choose_option("Bitte wählen Sie die Split-Methode:", split_methods[model_type])

# ROS code
config = {
    "model_type": model_type,
    "minSminL": minSminL,
    "window_size": int(window_size),
    "split_method": split_method
}

script_dir = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(script_dir, "scripts", "Aktualisierung.yaml")
os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

with open(yaml_path, "w") as file:
    yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
print("Parameter wurden in Aktualisierung.yaml gespeichert!")