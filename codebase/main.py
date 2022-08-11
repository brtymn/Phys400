# Library imports.
import time
import tracemalloc
import argparse
import sys
import params
from qse_hybrid import main

if __name__ == "__main__":
    # Start of the maximum memory allocation calculation process.
    tic = time.perf_counter()
    tracemalloc.start()

    # This part is for the command line arguements.
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help = "Toggle to use the Split Training Method.", action = 'store')


    # Parse the arguements from the command line.
    args = parser.parse_args()
    # Save the parsed arguements into global variables.
    params.method_select = args.method
    print(params.method_select)
    print('Code execution started.')

    # Run the code.
    main()

    peak = tracemalloc.get_traced_memory()[1]
    print(f"Peak memory usage was {peak / 10**6}MB.")

    # End of the maximum memory allocation calculation process.
    tracemalloc.stop()
    toc = time.perf_counter()

    print('Protocols completed.')
