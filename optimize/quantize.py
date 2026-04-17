import os
import numpy as np
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantType
)

print("===== QUANTIZATION STEP =====")

INPUT_MODEL = "models/model.onnx"
OUTPUT_MODEL = "models/model_int8.onnx"

if not os.path.exists(INPUT_MODEL):
    raise Exception("❌ ONNX model not found!")

# -------------------------------
# 🔹 Static Quantization (PRIMARY)
# -------------------------------
class DummyDataReader(CalibrationDataReader):
    def __init__(self):
        self.data = [
            {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
            for _ in range(5)
        ]
        self.iterator = iter(self.data)

    def get_next(self):
        return next(self.iterator, None)


try:
    print("\n🔵 Trying STATIC quantization (best for CNN)...")

    quantize_static(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        calibration_data_reader=DummyDataReader(),
        weight_type=QuantType.QInt8
    )

    print("✅ Static quantization SUCCESS")

except Exception as e:
    print("⚠️ Static quantization failed!")
    print("Reason:", e)

    # -------------------------------
    # 🔹 Dynamic Quantization (fallback)
    # -------------------------------
    try:
        print("\n🟡 Trying DYNAMIC quantization...")

        quantize_dynamic(
            model_input=INPUT_MODEL,
            model_output=OUTPUT_MODEL,
            weight_type=QuantType.QInt8
        )

        print("✅ Dynamic quantization SUCCESS")

    except Exception as e2:
        print("❌ Dynamic quantization also failed!")
        print("Reason:", e2)

        # -------------------------------
        # 🔹 Final fallback
        # -------------------------------
        print("\n⚠️ Using original ONNX model (no quantization)")
        import shutil
        shutil.copy(INPUT_MODEL, OUTPUT_MODEL)

        print("⚠️ Fallback model saved as model_int8.onnx")
