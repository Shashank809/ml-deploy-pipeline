import os
import glob
import subprocess
import sys

INPUT_DIR = "input"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


def find_model():
    files = os.listdir(INPUT_DIR)

    for f in files:
        path = os.path.join(INPUT_DIR, f)

        # -------- CASE 1: Python script --------
        if f.endswith(".py"):
            print("Running Python script to generate model...")

            try:
                subprocess.run(["python", path], check=True)
            except subprocess.CalledProcessError:
                raise Exception("Error while executing input Python script!")

        # -------- CASE 2: Direct model --------
        elif f.endswith(".pt") or f.endswith(".h5"):
            print("Direct model found:", path)
            return path

    # -------- Search for generated models --------
    print("Searching for generated model...")

    pt_files = glob.glob("*.pt") + glob.glob("models/*.pt")
    h5_files = glob.glob("*.h5") + glob.glob("models/*.h5")

    if pt_files:
        print("Found PyTorch model:", pt_files[0])
        return pt_files[0]

    if h5_files:
        print("Found TensorFlow model:", h5_files[0])
        return h5_files[0]

    # -------- If nothing found --------
    raise Exception("❌ No model found! Make sure your script saves model using torch.save() or model.save().")


def main():
    print("===== PIPELINE STARTED =====")

    # -------- Stage 0 --------
    model_path = find_model()
    print("Using model:", model_path)

    # -------- Stage 1 --------
    print("\n===== STAGE 1: ONNX CONVERSION =====")
    subprocess.run(["python", "convert/export_onnx.py", model_path], check=True)

    # -------- Stage 2 --------
    print("\n===== STAGE 2: QUANTIZATION =====")
    subprocess.run(["python", "optimize/quantize.py"], check=True)

    # -------- Stage 3 --------
    print("\n===== STAGE 3: VALIDATION =====")
    subprocess.run(["python", "tests/check_accuracy.py"], check=True)

    print("\n===== PIPELINE COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
