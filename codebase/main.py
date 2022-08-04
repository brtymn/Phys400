# Library imports.
import time
import tracemalloc
import argparse
from results import fold_results, zip_results, delete_files
import sys
import globals

if __name__ == "__main__":
    # Start of the maximum memory allocation calculation process.
    tic = time.perf_counter()
    tracemalloc.start()

    # This part is for the command line arguements.
    parser = argparse.ArgumentParser()
    parser.add_argument("--STM", help = "", action = 'store_true')
    parser.add_argument("--SCTM", help = "", action = 'store_true')


    # Parse the arguements from the command line.
    args = parser.parse_args()
    # Save the parsed arguements into global variables.
    globals.STM = args.STM
    globals.SCTM = args.SCTM

    print('Code execution started.')

    # Run the code.
    main(xyz_dir, log_dir, eig_dir)

    peak = tracemalloc.get_traced_memory()[1]
    print(f"Peak memory usage was {peak / 10**6}MB.")

    # End of the maximum memory allocation calculation process.
    tracemalloc.stop()
    toc = time.perf_counter()

    print('Protocols completed.')
