import pandas as pd
import os

# Configure pandas display settings for better output visibility
pd.set_option("display.width", None)

# URL to the dataset from HuggingFace Hub
url = "https://huggingface.co/datasets/FareedKhan/1k_stories_100_genre/resolve/main/1k_stories_100_genre.csv"

# Download and load the CSV file into a pandas DataFrame
df = pd.read_csv(url)

# Display the first few rows and column names for data verification
print(df.head())
print(df.columns)

# Construct the output path dynamically using the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "..", "data", "1k_stories_100_genre.csv")

# Save the DataFrame to a CSV file
df.to_csv(output_path, index=False)