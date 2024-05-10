import sys
import pandas as pd
import matplotlib.pyplot as plt

# Check if CSV file paths are provided as command-line arguments
if len(sys.argv) < 2:
    print("Usage: python script.py <csv_file1> <csv_file2> ...")
    sys.exit(1)

# Process each CSV file provided as an argument
for csv_file in sys.argv[1:]:
    try:
        # Read CSV file into DataFrame
        df = pd.read_csv(csv_file)

        # Process DataFrame (Example: Print the first few rows)
        print(f"Processing {csv_file}:")
        print(df)  # Example processing, replace with your processing logic

        # Plot the DataFrame

        # List of column labels
        column_labels = ['Column 1', 'Column 2']

        # Set column labels
        df.columns = column_labels
        plt.plot(df['Column 1'], df['Column 2'])

        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('DataFrame Plot')

        # Show the plot
        plt.grid(True)
        plt.show()

        # If you need to write processed data back to a file, you can do it here
        # Example: df.to_csv('processed_' + csv_file, index=False)

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
