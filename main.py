from datasets import load_dataset


DATASET_NAME = "karpathy/tiny_shakespeare"
TEXT_COLUMN = "text"
PREVIEW_CHARS = 600


def preview_split(split_name: str, split_dataset) -> None:
    print("=" * 80)
    print(f"SPLIT: {split_name}")
    print(f"Rows: {len(split_dataset)}")
    print(f"Columns: {split_dataset.column_names}")
    print()

    if len(split_dataset) == 0:
        print("No rows in this split.")
        return

    sample_text = str(split_dataset[0].get(TEXT_COLUMN, ""))
    print(f"Sample {TEXT_COLUMN} preview:")
    print(sample_text[:PREVIEW_CHARS])
    if len(sample_text) > PREVIEW_CHARS:
        print("...")


def main() -> None:
    print(f"Loading Hugging Face dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    print(f"Dataset splits: {list(dataset.keys())}")
    for split_name, split_dataset in dataset.items():
        preview_split(split_name, split_dataset)


if __name__ == "__main__":
    main()
