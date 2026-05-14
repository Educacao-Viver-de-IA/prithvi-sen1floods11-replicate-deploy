import base64
import io
import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/src/weights"

print(f"[module] predict.py loading at t={time.time()}", flush=True)
sys.stdout.flush()

import numpy as np
print(f"[module] numpy loaded", flush=True)
import torch
print(f"[module] torch {torch.__version__} loaded, cuda={torch.cuda.is_available()}", flush=True)
sys.stdout.flush()
from cog import BasePredictor, Input, Path
print(f"[module] cog loaded", flush=True)
from PIL import Image
print(f"[module] PIL loaded", flush=True)
sys.stdout.flush()

WEIGHTS_DIR = "/src/weights/prithvi-sen1floods11"
CHECKPOINT = f"{WEIGHTS_DIR}/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === t={t0}", flush=True)
        sys.stdout.flush()
        self.task = None
        self.setup_error = None
        print(f"[setup] WEIGHTS_DIR={WEIGHTS_DIR}", flush=True)
        try:
            print(f"[setup] dir: {sorted(os.listdir(WEIGHTS_DIR))[:15]}", flush=True)
        except Exception as e:
            print(f"[setup] err: {e}", flush=True)
        print(f"[setup] cuda: {torch.cuda.is_available()}", flush=True)
        sys.stdout.flush()

        print(f"[setup] importing terratorch.tasks... (t={time.time()-t0:.1f}s)", flush=True)
        sys.stdout.flush()
        try:
            t_imp = time.time()
            from terratorch.tasks import SemanticSegmentationTask
            print(f"[setup] terratorch imported in {time.time()-t_imp:.1f}s", flush=True)
            self.SemanticSegmentationTask = SemanticSegmentationTask
        except Exception as e:
            import traceback
            print(f"[setup] FATAL terratorch import: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"terratorch import failed: {e}"
            return

        print(f"[setup] load checkpoint... (t={time.time()-t0:.1f}s)", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.task = None
        try:
            self.task = SemanticSegmentationTask.load_from_checkpoint(CHECKPOINT, map_location=device)
            print(f"[setup] load_from_checkpoint OK", flush=True)
        except Exception as e:
            print(f"[setup] load_from_checkpoint falhou ({type(e).__name__}: {e})", flush=True)

        if self.task is None:
            print(f"[setup] state_dict fallback...", flush=True)
            sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
            print(f"[setup] checkpoint keys: {list(sd.keys())[:10] if isinstance(sd, dict) else type(sd)}", flush=True)
            try:
                self.task = SemanticSegmentationTask(
                    model_factory="EncoderDecoderFactory",
                    model_args={
                        "backbone": "prithvi_eo_v2_300_tl",
                        "decoder": "UperNetDecoder",
                        "decoder_channels": 256,
                        "num_classes": 2,
                        "necks": [{"name": "SelectIndices", "indices": [5, 11, 17, 23]}, {"name": "ReshapeTokensToImage"}],
                    },
                    loss="ce",
                    ignore_index=-1,
                    lr=1e-4,
                    optimizer="AdamW",
                    plot_on_val=False,
                )
                state = sd.get("state_dict", sd)
                missing, unexpected = self.task.load_state_dict(state, strict=False)
                print(f"[setup] state_dict load: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
            except Exception as e2:
                import traceback
                print(f"[setup] state_dict fallback FAILED: {type(e2).__name__}: {e2}", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                self.setup_error = f"state_dict load failed: {e2}"
                return

        self.task.eval()
        for m in self.task.modules():
            m.eval()
        if torch.cuda.is_available():
            self.task = self.task.cuda()
        print(f"[setup] DONE (t={time.time()-t0:.1f}s) | all modules eval()", flush=True)
        sys.stdout.flush()

    def _load_image(self, image_path: Path, image_size: int) -> torch.Tensor:
        path_str = str(image_path)
        arr = None
        if path_str.lower().endswith((".tif", ".tiff")):
            try:
                import rasterio
                with rasterio.open(path_str) as src:
                    arr = src.read().astype(np.float32)
            except Exception as e:
                print(f"[predict] rasterio fail: {e}", flush=True)

        if arr is None:
            pil = Image.open(image_path)
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
            arr = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)

        if arr.shape[0] < 6:
            pad = np.repeat(arr[-1:], 6 - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > 6:
            arr = arr[:6]

        if arr.max() > 255:
            arr = arr / 10000.0
        elif arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        import torch.nn.functional as F
        tensor = torch.from_numpy(arr).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        tensor = tensor.unsqueeze(2)
        return tensor

    def predict(
        self,
        image: Path = Input(description="GeoTIFF Sentinel-2 6-band ou imagem RGB (auto-pad)."),
        image_size: int = Input(default=512, ge=128, le=1024),
        return_format: str = Input(default="summary", choices=["summary", "mask_png"]),
    ) -> str:
        if self.task is None:
            return json.dumps({"error": f"Modelo não carregou: {getattr(self, 'setup_error', 'unknown')}"})
        device = next(self.task.parameters()).device
        tensor = self._load_image(image, image_size).to(device, dtype=torch.float32)

        with torch.no_grad():
            output = self.task(tensor)

        if isinstance(output, dict):
            logits = output.get("output", output.get("logits", next(iter(output.values()))))
        elif isinstance(output, (list, tuple)):
            logits = output[0]
        else:
            logits = output

        if logits.dim() == 4:
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        else:
            pred = logits.squeeze().cpu().numpy()

        total_pixels = pred.size
        flood_pixels = int(np.sum(pred == 1))
        dry_pixels = int(np.sum(pred == 0))
        flood_pct = float(flood_pixels / total_pixels * 100)

        result = {
            "model": "prithvi-eo-2-300m-tl-sen1floods11",
            "image_size": image_size,
            "total_pixels": int(total_pixels),
            "flood_pixels": flood_pixels,
            "dry_pixels": dry_pixels,
            "flood_pct": round(flood_pct, 2),
            "interpretation": "major_flooding" if flood_pct > 30 else ("moderate_flooding" if flood_pct > 5 else "minimal_or_no_flooding"),
        }

        if return_format == "mask_png":
            mask_img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            mask_img[pred == 1] = [0, 100, 255]  # azul = água/inundação
            mask_img[pred == 0] = [50, 50, 50]
            pil_mask = Image.fromarray(mask_img)
            buf = io.BytesIO()
            pil_mask.save(buf, format="PNG")
            result["mask_png_base64"] = base64.b64encode(buf.getvalue()).decode()

        return json.dumps(result, ensure_ascii=False)
