import os 
import json 

# JUST EXPERIMENTING

def setup_folders(base_dir="experiment_results", datasets=[], hp_configs={}, num_repeats=3):
    # Iterate over each dataset
    for dataset in datasets:
        # Define the directory path for the current dataset
        dataset_dir = os.path.join(base_dir, dataset)

        # Create the directory for the current dataset
        os.makedirs(dataset_dir, exist_ok=True)

        # Iterate over each hyperparameter configuration for the current dataset
        for hp_config in hp_configs.get(dataset, []):
            # Define the directory path for the current hyperparameter configuration
            hp_config_dir = os.path.join(dataset_dir, hp_config)

            # Create the directory for the current hyperparameter configuration
            os.makedirs(hp_config_dir, exist_ok=True)

            # Iterate over the number of repeats for the current hyperparameter configuration
            for repeat in range(num_repeats):
                # Define the filename for the current repeat
                filename = f"repeat_{repeat}.json"

                # Define the path for the current JSON file
                json_path = os.path.join(hp_config_dir, filename)

                # Print the path for the current JSON file (optional)
                print(f"Created directory: {json_path}")

