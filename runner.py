import argparse
import subprocess

# Define the command to run script1.py with options
command = 'python script1.py'

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--arg1', type=str, help='The first argument')
parser.add_argument('--arg2', type=str, help='The second argument')
args = parser.parse_args()

# Add the arguments to the command
command += ' --arg1 {} --arg2 {}'.format(args.arg1, args.arg2)

# Run the command using subprocess and get the output
output = subprocess.check_output(command, shell=True)

# Decode the output (which is in bytes) to a string
result = output.decode('utf-8').strip()

# Print the result
print(int(result)+1)
