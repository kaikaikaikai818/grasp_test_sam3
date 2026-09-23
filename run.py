"""轻量文字分割主程序：修改下方配置后，点击运行即可。"""

import ctypes
import os
import sys
import warnings
from pathlib import Path

# ========================
# 当前阶段固定跑通螺丝刀：D455粗定位，D435i精定位，纯视觉不控制机器人
# ========================
RUN_MODE = "realsense"
IMAGE_PATH = "test.jpg"
TEXT_PROMPT = "a screwdriver"
HO_SERIAL = "215122257404"  # 手外固定 D455：粗定位
HI_SERIAL = "215222074676"  # 手内 D435i：精定位


os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

PROJECT_ROOT = Path(__file__).resolve().parent


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(0, message, "处理失败", 0x10)


def main() -> None:
    prompt = TEXT_PROMPT.strip()
    if not prompt:
        raise ValueError("TEXT_PROMPT 不能为空。")

    mode = RUN_MODE.strip().lower()
    checkpoint = PROJECT_ROOT / "weights" / "mobile_sam.pt"
    if mode == "image":
        from light_sam.pipeline import run_pipeline

        image_path = Path(IMAGE_PATH)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"找不到图片：{image_path}")
        output_dir = PROJECT_ROOT / "outputs" / image_path.stem
        run_pipeline(
            image_path=str(image_path),
            prompt=prompt,
            output_dir=str(output_dir),
            checkpoint=str(checkpoint),
            local_files_only=True,
        )
        os.startfile(output_dir / "sam_result.jpg")
    elif mode == "realsense":
        from light_sam.realsense_runner import run_realsense

        run_realsense(
            checkpoint=str(checkpoint),
            prompt=prompt,
            output_root=str(PROJECT_ROOT / "outputs" / "realsense"),
            d455_serial=HO_SERIAL,
            d435i_serial=HI_SERIAL,
        )
    else:
        raise ValueError('RUN_MODE 只能填写 "image" 或 "realsense"。')


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        show_error(str(exc))
