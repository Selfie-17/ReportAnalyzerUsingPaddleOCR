import paddle


def _safe_bfloat16_supported(device=None):
    try:
        target = paddle.device.get_device() if device is None else device
        if target is None:
            return False
        target_text = str(target).lower()
        if not any(token in target_text for token in ("gpu", "xpu", "npu", "mlu")):
            return False
        try:
            return bool(paddle.amp.is_bfloat16_supported(target))
        except TypeError:
            try:
                if "gpu" in target_text:
                    return bool(paddle.base.core.is_bfloat16_supported(paddle.CUDAPlace(0)))
                return False
            except Exception:
                return False
    except Exception:
        return False


paddle.amp.is_bfloat16_supported = _safe_bfloat16_supported

print("=" * 50)
print("PADDLE GPU TEST")
print("=" * 50)

print("Paddle version:", paddle.__version__)
print("CUDA compiled:", paddle.is_compiled_with_cuda())
try:
    paddle.set_device("gpu:0")
except Exception as e:
    raise RuntimeError(f"Failed to set GPU device: {e}")

print("Device:", paddle.device.get_device())

if not paddle.is_compiled_with_cuda():
    raise RuntimeError("Paddle is NOT compiled with CUDA")

if "gpu" not in paddle.device.get_device().lower():
    raise RuntimeError("Paddle is NOT using GPU")

from paddleocr import PaddleOCRVL

print()
print("Loading PaddleOCR-VL 1.6...")

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    device="gpu:0",
)

print("Model loaded successfully on GPU.")

file_path = input(
    "Enter image or PDF path: "
).strip().strip('"')

print()
print("Running OCR...")

results = pipeline.predict(file_path)

print()
print("=" * 50)
print("RESULT")
print("=" * 50)

for result in results:

    print()
    print("Result type:", type(result))

    print()
    print("Object:")
    print(result)

    if hasattr(result, "markdown"):
        print()
        print("MARKDOWN:")
        print(result.markdown)

print()
print("DONE")