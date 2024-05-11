import sys
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

test_name = "trials/network_tweaks_"
# test_name = "trials/baseline_"

num_tests = 10

# Initialize lists to store data from each test
all_data = []

# Process each test
for test_number in range(1, num_tests + 1):
    try:
        # Construct CSV file name based on prefix and test number
        csv_file = f"{test_name}{test_number}.csv"

        # Check if CSV file already exists
        if not os.path.exists(csv_file):
            # Run the simulation code with the CSV name as an argument
            subprocess.run(['python', 'plane_simulator.py', csv_file])

        # Read CSV file into DataFrame
        df = pd.read_csv(csv_file)

        # Process DataFrame (Example: Print the first few rows)
        print(f"Processing {csv_file}:")
        print(df)  # Example processing, replace with your processing logic

        # Store DataFrame data for this test
        all_data.append(df)

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")

# Plot data from all tests on the same graph
plt.figure(figsize=(10, 6))  # Set the figure size
ticks_per_sec = 30  #assumed time step based on standard from plane_simulator
for i, df in enumerate(all_data):
    # Calculate displacement as the cumulative sum of speed
    displacement = (df['Speed'] * (1/ticks_per_sec)).cumsum()

    # Flip the altitude values (higher altitudes become negative)
    altitude_flipped = -df['Altitude']
    altitude_flipped = altitude_flipped + 800

    # Plot the DataFrame with flipped altitude values
    plt.plot(displacement, altitude_flipped, label=f"Test {i+1}")

# Add labels and title
plt.xlabel('Distance')
plt.ylabel('Altitude')
plt.title('Altitude vs Distance')

# Add legend
plt.legend()

# Save the plot as an image
plt.grid(True)
plt.savefig(test_name+'_altitude_vs_distance.png')  # Save as PNG format
# plt.show()

# Initialize a list to store the overall run time for each test
overall_run_time_per_test = []

# Process each test
for df in all_data:
    # Get the overall run time (last value in the "Time" column) for the current test
    overall_run_time = df['Time'].iloc[-1]
    
    # Append the overall run time to the list
    overall_run_time_per_test.append(overall_run_time)

# Plot overall run time with respect to the test number
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(all_data) + 1), overall_run_time_per_test, marker='o', linestyle='-')
plt.xlabel('Test Number')
plt.ylabel('Overall Run Time')
plt.title('Overall Run Time vs Test Number')
plt.grid(True)
# plt.show()
plt.savefig(test_name+'_process_time.png')  # Save as PNG format

# Calculate the total runtime and number of tests
total_runtime = sum(overall_run_time_per_test)
num_tests = len(overall_run_time_per_test)

# Calculate the average runtime
average_runtime = total_runtime / num_tests

# Print the average runtime
print("Average " + test_name + "Runtime:", average_runtime)
