from pathlib import Path
from datasets import load_dataset

# Download/load the dataset from Hugging Face.
dataset = load_dataset("karpathy/tiny_shakespeare")

# Make sure the data folder exists.
data_folder = Path("data")
data_folder.mkdir(parents=True, exist_ok=True)

# Save each split (train/validation/test) as its own CSV.
for split_name, split_data in dataset.items():
    file_path = data_folder / f"{split_name}.csv"
    split_data.to_pandas().to_csv(file_path, index=False)
    print(f"Saved: {file_path}")

    # Print a tiny preview so you can quickly check content.
    if len(split_data) > 0:
        print(split_data[0]["text"][:200])
        print("-" * 40)