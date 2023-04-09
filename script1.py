import argparse

def main(args):
    # Do some calculations
    result = int(args.arg1) + int(args.arg2)
    return result

if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1', type=str, help='The first argument')
    parser.add_argument('--arg2', type=str, help='The second argument')
    args = parser.parse_args()
    
    # Call the main function and print the result
    result = main(args)
    print(result)
