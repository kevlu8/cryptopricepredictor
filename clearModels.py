import shutil
import os

shutil.rmtree("models", ignore_errors=True)
os.mkdir("models")