import os
import glob
import sys

INPUT_DIR = "input"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def find_model():
    files = os.listdir(INPUT_DIR)

    for f in files:
        path = os.path.join(INPUT_DIR, f)

        # CASE 1: Python script
        if f.endswith(".py"):
            print("Running Python script to generate model...")
            os.system(f"python {path}")

        # CASE 2: direct model
        elif f.endswith(".pt") or f.endswith(".h5"):
            return path

    # search generated models
    pt = glob.glob("*.pt") + glob.glob("models/*.pt")
    h5 = glob.glob("*.h5")

    if pt:
        return pt[0]
    if h5:
        return h5[0]

    raise Exception("No model found!")

def main():
    model_path = find_model()
    print("Model found:", model_path)

    # Stage 1
    os.system(f"python convert/export_onnx.py {model_path}")

    # Stage 2
    os.system("python optimize/quantize.py")

    # Stage 3
    os.system("python tests/check_accuracy.py")

if __name__ == "__main__":
    main()
