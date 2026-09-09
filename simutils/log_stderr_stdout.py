import os
import sys
import shutil
from datetime import datetime


class Tee:
    def __init__(self, original, file):
        self.original = original
        self.file = file

    def write(self, data):
        self.original.write(data)
        self.file.write(data)
        self.original.flush()
        self.file.flush()

    def flush(self):
        self.original.flush()
        self.file.flush()


# ----------------------------------------------------------------------
# Initialize logging
# ----------------------------------------------------------------------

outfile = "output.txt"

# If an old output.txt exists, rename it with a timestamp.
if os.path.exists(outfile):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"output_{timestamp}.txt"
    os.rename(outfile, backup)

log = open(outfile, "a", buffering=1)

tee_stdout = Tee(sys.stdout, log)
tee_stderr = Tee(sys.stderr, log)

sys.stdout = tee_stdout
sys.stderr = tee_stderr


# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------

def move_log(folder1):
    """
    Move the current log to folder1 and continue logging there.

    Everything already written to "filename" is preserved.
    Future stdout/stderr output continues to be written to the new file.
    """
    global log

    new_path = os.path.join(folder1, outfile)

    # Create destination directory if necessary.
    os.makedirs(folder1, exist_ok=True)

#    if os.path.exists(new_path):
#        raise FileExistsError(f"Log destination already exists: {new_path}")

    # Finish writing to the current file.
    log.flush()
    log.close()

    # Move the existing log.
    shutil.move(outfile, new_path)

    # Reopen at the new location.
    log = open(new_path, "a", buffering=1)

    # Make both Tee objects use the new file.
    tee_stdout.file = log
    tee_stderr.file = log

# use with
#import simutils.log_stderr_stdout as log_stderr_stdout
#Then to move the file in your main program can simply do:
#log_stderr_stdout.move_log(folder1)
