
import os
import gc
import math
import time
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class PureAdvancedNeuralGPU:

    VERSION = "X1.0"

    def __init__(
        self,
        memory_limit_mb=180,
        tile_size=128
    ):

        self.name = "PureAdvanced Neural GPU X1"

        self.memory_limit_mb = memory_limit_mb
        self.tile_size = tile_size

        self.lock = threading.RLock()

        self.models = {
            "text": None,
            "vision": None,
            "text_to_image": None,
            "image_to_image": None
        }

        self.stats = {
            "operations": 0,
            "pixels_processed": 0,
            "neural_operations": 0,
            "started_at": time.time()
        }

        self._initialize_neural_core()

        print("=" * 45)
        print(self.name)
        print("Version:", self.VERSION)
        print("Memory budget:", memory_limit_mb, "MB")
        print("Tile size:", tile_size)
        print("Device: Software CPU")
        print("=" * 45)

    # ==========================================
    # 1. MEMORY MANAGEMENT CORE
    # ==========================================

    def memory_status(self):

        try:
            import psutil

            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss / (1024 * 1024)

            return {
                "process_ram_mb": round(rss, 2),
                "budget_mb": self.memory_limit_mb,
                "within_budget": rss <= self.memory_limit_mb
            }

        except ImportError:

            return {
                "process_ram_mb": None,
                "budget_mb": self.memory_limit_mb,
                "within_budget": None,
                "message": "Install psutil for RAM monitoring"
            }

    def check_memory(self):

        status = self.memory_status()

        if status["process_ram_mb"] is not None:
            if not status["within_budget"]:
                raise MemoryError(
                    "Configured RAM budget exceeded: "
                    + str(status["process_ram_mb"]) + " MB"
                )

        return status

    def cleanup(self):

        gc.collect()

        return True

    # ==========================================
    # 2. NEURAL MATRIX CORE
    # ==========================================

    def _initialize_neural_core(self):

        self.neural_weights = None

        self.neural_config = {
            "input_size": 256,
            "hidden_size": 128,
            "output_size": 64
        }

    def load_neural_weights(self, filename):

        data = np.load(filename)

        required = {"w1", "w2"}

        if not required.issubset(data.files):
            raise ValueError("Missing neural weight arrays")

        w1 = data["w1"]
        w2 = data["w2"]

        if w1.shape != (256, 128):
            raise ValueError("Invalid first layer shape")

        if w2.shape != (128, 64):
            raise ValueError("Invalid second layer shape")

        self.neural_weights = {
            "w1": w1,
            "w2": w2
        }

        return "Trained neural weights loaded"

    def neuron_layer(self, inputs, weights):

        x = np.asarray(inputs, dtype=np.float32)

        w = np.asarray(weights)

        if x.shape[-1] != w.shape[0]:
            raise ValueError("Neural dimensions do not match")

        # Matrix multiplication
        output = np.matmul(x, w.astype(np.float32))

        # GELU-like activation
        output = 0.5 * output * (
            1.0 + np.tanh(
                0.79788456 *
                (output + 0.044715 * output ** 3)
            )
        )

        self.stats["neural_operations"] += output.size

        return output

    def neural_compute(self, inputs):

        if self.neural_weights is None:
            raise RuntimeError(
                "Trained weights are required for neural inference"
            )

        x = np.asarray(inputs, dtype=np.float32)

        if x.shape[-1] != 256:
            raise ValueError("Expected 256 input features")

        x = self.neuron_layer(
            x,
            self.neural_weights["w1"]
        )

        x = self.neuron_layer(
            x,
            self.neural_weights["w2"]
        )

        return x

    # ==========================================
    # 3. MODEL ADAPTER SYSTEM
    # ==========================================

    def register_model(self, task, model):

        if task not in self.models:
            raise ValueError("Unknown model task: " + task)

        if not callable(model):
            raise TypeError("Model must be callable")

        self.models[task] = model

        return task + " model registered"

    def run_model(self, task, *args, **kwargs):

        model = self.models.get(task)

        if model is None:
            raise RuntimeError(
                "No trained model connected for: " + task
            )

        with self.lock:

            self.check_memory()

            result = model(*args, **kwargs)

            self.stats["operations"] += 1

            return result

    # ==========================================
    # 4. TEXT-TO-TEXT CORE
    # ==========================================

    def text_to_text(self, prompt, **kwargs):

        if not isinstance(prompt, str):
            raise TypeError("Prompt must be text")

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        return self.run_model(
            "text",
            prompt,
            **kwargs
        )

    # ==========================================
    # 5. IMAGE-TO-TEXT CORE
    # ==========================================

    def image_to_text(self, image_path, **kwargs):

        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as source:

            image = source.convert("RGB")
            image.thumbnail((384, 384))

            return self.run_model(
                "vision",
                image.copy(),
                **kwargs
            )

    # ==========================================
    # 6. TEXT-TO-IMAGE CORE
    # ==========================================

    def text_to_image(
        self,
        prompt,
        output="generated.png",
        **kwargs
    ):

        if not prompt.strip():
            raise ValueError("Image prompt is empty")

        result = self.run_model(
            "text_to_image",
            prompt,
            **kwargs
        )

        if isinstance(result, Image.Image):
            result.save(output)
            return output

        if isinstance(result, (str, Path)):
            return str(result)

        raise TypeError(
            "Image model must return PIL Image or file path"
        )

    # ==========================================
    # 7. IMAGE-TO-IMAGE CORE
    # ==========================================

    def image_to_image(
        self,
        image_path,
        prompt,
        output="edited.png",
        **kwargs
    ):

        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as source:
            image = source.convert("RGB")

            result = self.run_model(
                "image_to_image",
                image.copy(),
                prompt,
                **kwargs
            )

        if isinstance(result, Image.Image):
            result.save(output)
            return output

        if isinstance(result, (str, Path)):
            return str(result)

        raise TypeError(
            "Image editing model must return an image"
        )

    # ==========================================
    # 8. PIXEL COMPUTE CORE
    # ==========================================

    def pixel_transform(
        self,
        image_path,
        output="pixel_output.png",
        brightness=1.0,
        contrast=1.0
    ):

        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as source:

            image = source.convert("RGB")

            # Pixel calculations in small tiles
            result = Image.new("RGB", image.size)

            width, height = image.size

            for y in range(0, height, self.tile_size):
                for x in range(0, width, self.tile_size):

                    box = (
                        x,
                        y,
                        min(x + self.tile_size, width),
                        min(y + self.tile_size, height)
                    )

                    tile = image.crop(box)

                    pixels = np.asarray(
                        tile,
                        dtype=np.float32
                    )

                    pixels *= brightness

                    pixels = np.clip(
                        pixels, 0, 255
                    )

                    pixels = np.asarray(
                        pixels,
                        dtype=np.uint8
                    )

                    processed = Image.fromarray(
                        pixels, "RGB"
                    )

                    processed = ImageEnhance.Contrast(
                        processed
                    ).enhance(contrast)

                    result.paste(processed, (x, y))

                    self.stats["pixels_processed"] += (
                        pixels.shape[0] * pixels.shape[1]
                    )

                    del pixels, tile, processed

            result.save(output)

        self.stats["operations"] += 1

        return output

    # ==========================================
    # 9. PROCEDURAL IMAGE CORE
    # ==========================================

    def procedural_image(
        self,
        width=1024,
        height=1024,
        seed=42,
        output="procedural.png"
    ):

        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")

        rng = np.random.default_rng(seed)

        result = Image.new(
            "RGB",
            (width, height)
        )

        for y in range(0, height, self.tile_size):
            for x in range(0, width, self.tile_size):

                tw = min(self.tile_size, width - x)
                th = min(self.tile_size, height - y)

                yy, xx = np.mgrid[0:th, 0:tw]

                nx = (xx + x) / width
                ny = (yy + y) / height

                wave = (
                    np.sin(nx * 18) *
                    np.cos(ny * 22)
                )

                noise = rng.random(
                    (th, tw)
                ) * 0.10

                intensity = np.clip(
                    0.5 + wave * 0.25 + noise,
                    0, 1
                )

                pixels = np.stack(
                    [
                        intensity * 35,
                        intensity * 100,
                        intensity * 65
                    ],
                    axis=-1
                ).astype(np.uint8)

                tile = Image.fromarray(
                    pixels,
                    "RGB"
                )

                result.paste(tile, (x, y))

                self.stats["pixels_processed"] += tw * th

                del pixels, tile, intensity, wave, noise

        result.save(output)

        self.stats["operations"] += 1

        return output

    # ==========================================
    # 10. ENGINE STATUS
    # ==========================================

    def status(self):

        return {
            "name": self.name,
            "version": self.VERSION,
            "device": "CPU Software Compute",
            "models": {
                key: value is not None
                for key, value in self.models.items()
            },
            "neural_weights_loaded": (
                self.neural_weights is not None
            ),
            "memory": self.memory_status(),
            "statistics": self.stats.copy()
        }


# ==============================================
# ENGINE STARTUP
# ==============================================

if __name__ == "__main__":

    gpu = PureAdvancedNeuralGPU(
        memory_limit_mb=180,
        tile_size=128
    )

    # Procedural graphics test
    image = gpu.procedural_image(
        width=1024,
        height=1024,
        seed=42
    )

    print("Generated:", image)

    # Pixel processing test
    edited = gpu.pixel_transform(
        image,
        output="pixel_test.png",
        brightness=1.2,
        contrast=1.1
    )

    print("Pixel output:", edited)

    # Engine report
    print("\nENGINE STATUS")

    print(gpu.status())

    gpu.cleanup()
    
# =====================================================================
# X2 ADD-ON LAYER
# The original X1 source above is intentionally left unchanged.
# =====================================================================

class LowMemoryTextCore:
    """Lightweight prompt validation and normalization; AI reasoning is delegated to a model."""

    @staticmethod
    def prepare(prompt, max_chars=12000):
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        if len(prompt) > max_chars:
            raise ValueError("Prompt exceeds configured character limit")
        return prompt


class TiledMatrixEngine:
    """Bounded temporary-memory matrix multiplication using NumPy tiles."""

    def __init__(self, tile_size=64):
        if not isinstance(tile_size, int) or tile_size < 1:
            raise ValueError("tile_size must be a positive integer")
        self.tile_size = tile_size

    def matmul(self, left, right):
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("Expected compatible 2-D matrices")
        out = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)
        step = self.tile_size
        for i in range(0, a.shape[0], step):
            for j in range(0, b.shape[1], step):
                acc = np.zeros((min(step, a.shape[0] - i),
                                min(step, b.shape[1] - j)), dtype=np.float32)
                for k in range(0, a.shape[1], step):
                    acc += a[i:i+step, k:k+step] @ b[k:k+step, j:j+step]
                out[i:i+step, j:j+step] = acc
        return out


class LowMemoryOCRCore:
    """Optional OCR adapter. Requires Tesseract binary and pytesseract package."""

    def __init__(self, max_side=1600):
        self.max_side = max(128, int(max_side))

    def read(self, image_or_path, language="eng"):
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError("OCR dependency missing: install pytesseract and Tesseract OCR") from exc
        if isinstance(image_or_path, Image.Image):
            image = image_or_path.copy()
        else:
            with Image.open(image_or_path) as src:
                image = src.convert("RGB")
        image.thumbnail((self.max_side, self.max_side))
        try:
            return pytesseract.image_to_string(image, lang=language)
        finally:
            image.close()


class JanusProAdapter:
    """Lazy CPU-first adapter for DeepSeek Janus-Pro-1B.

    Loading is explicit because the model is much larger than the X1 180 MB
    monitoring budget. This adapter does not pretend the model fits in 180 MB.
    """

    def __init__(self, model_id="deepseek-ai/Janus-Pro-1B", max_new_tokens=160,
                 image_size=384, trust_remote_code=True):
        self.model_id = model_id
        self.max_new_tokens = max(8, int(max_new_tokens))
        self.image_size = max(128, min(384, int(image_size)))
        self.trust_remote_code = bool(trust_remote_code)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.torch = None
        self.device = "cpu"

    @property
    def loaded(self):
        return self.model is not None and self.processor is not None

    def load(self):
        if self.loaded:
            return "Janus model already loaded"
        try:
            import torch
            from transformers import AutoModelForCausalLM
            from janus.models import VLChatProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Janus dependencies missing. Install the Janus repository requirements, "
                "PyTorch and Transformers before loading the model."
            ) from exc

        # CPU float32 is selected for broad CPU compatibility. It is not a
        # low-RAM quantized load; real model memory is substantially >180 MB.
        processor = VLChatProcessor.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        model.to("cpu")
        model.eval()
        self.torch = torch
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model = model
        return "Janus-Pro loaded on CPU"

    def unload(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        if self.torch is not None:
            self.torch = None
        gc.collect()
        return True

    def text(self, prompt, max_new_tokens=None, **_kwargs):
        if not self.loaded:
            raise RuntimeError("Call load_janus() before using Janus")
        torch = self.torch
        prompt = LowMemoryTextCore.prepare(prompt)
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        formatted = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        inputs = self.tokenizer(formatted, return_tensors="pt")
        with torch.inference_mode():
            output = self.model.language_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def vision(self, image, question="Describe this image", max_new_tokens=None, **_kwargs):
        if not self.loaded:
            raise RuntimeError("Call load_janus() before using Janus")
        if not isinstance(image, Image.Image):
            raise TypeError("vision expects a PIL image")
        image = image.convert("RGB")
        image.thumbnail((self.image_size, self.image_size))
        conversation = [
            {"role": "<|User|>", "content": "<image_placeholder>\\n" +
             LowMemoryTextCore.prepare(question), "images": [image]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepared = self.processor(conversations=conversation, images=[image], force_batchify=True)
        prepared = prepared.to("cpu")
        with self.torch.inference_mode():
            embeds = self.model.prepare_inputs_embeds(**prepared)
            output = self.model.language_model.generate(
                inputs_embeds=embeds,
                attention_mask=prepared.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        return self.tokenizer.decode(output[0].cpu().tolist(), skip_special_tokens=True).strip()

    def text_to_image(self, prompt, output="janus_generated.png", seed=42,
                      guidance=5.0, temperature=1.0, **_kwargs):
        if not self.loaded:
            raise RuntimeError("Call load_janus() before using Janus")
        torch = self.torch
        prompt = LowMemoryTextCore.prepare(prompt)
        torch.manual_seed(int(seed))
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        formatted = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        formatted += self.processor.image_start_tag
        ids = torch.tensor(self.tokenizer.encode(formatted), dtype=torch.long)
        # CFG requires a conditional + unconditional row. Keep batch at one image.
        tokens = torch.stack((ids, ids), dim=0)
        tokens[1, 1:-1] = self.processor.pad_id
        generated = torch.zeros((1, 576), dtype=torch.int)
        with torch.inference_mode():
            embeds = self.model.language_model.get_input_embeddings()(tokens)
            past = None
            for idx in range(576):
                out = self.model.language_model.model(
                    inputs_embeds=embeds, use_cache=True, past_key_values=past
                )
                past = out.past_key_values
                logits = self.model.gen_head(out.last_hidden_state[:, -1, :])
                cond, uncond = logits[0:1], logits[1:2]
                logits = uncond + float(guidance) * (cond - uncond)
                probs = torch.softmax(logits / max(float(temperature), 1e-4), dim=-1)
                token = torch.multinomial(probs, num_samples=1)
                generated[:, idx] = token.squeeze(-1)
                doubled = torch.cat((token, token), dim=0).reshape(-1)
                embeds = self.model.prepare_gen_img_embeds(doubled).unsqueeze(1)
            decoded = self.model.gen_vision_model.decode_code(
                generated, shape=[1, 8, 24, 24]
            )
        array = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)[0]
        array = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
        result = Image.fromarray(array, "RGB")
        result.save(output)
        result.close()
        return str(output)


class PureAdvancedNeuralGPUX2(PureAdvancedNeuralGPU):
    """X2 extends X1 without editing or removing any X1 methods."""

    VERSION = "X2.0"

    def __init__(self, memory_limit_mb=180, tile_size=64, janus_model_id="deepseek-ai/Janus-Pro-1B"):
        super().__init__(memory_limit_mb=memory_limit_mb, tile_size=tile_size)
        self.name = "PureAdvanced Neural GPU X2"
        self.text_core = LowMemoryTextCore()
        self.ocr_core = LowMemoryOCRCore()
        self.matrix_engine = TiledMatrixEngine(tile_size=tile_size)
        self.janus = JanusProAdapter(model_id=janus_model_id)
        self.core_status = {
            "text_processing": True,
            "ocr": "optional dependency; not loaded",
            "vision_processing": True,
            "optimized_matrix": True,
            "model_integration": True,
            "janus_loaded": False,
        }

    def load_janus(self):
        result = self.janus.load()
        self.register_model("text", self.janus.text)
        self.register_model("vision", self.janus.vision)
        self.register_model("text_to_image", self.janus.text_to_image)
        self.core_status["janus_loaded"] = True
        return result

    def unload_janus(self):
        self.janus.unload()
        for task in ("text", "vision", "text_to_image"):
            self.models[task] = None
        self.core_status["janus_loaded"] = False
        return "Janus unloaded; memory cleanup requested"

    def process_text(self, text, **kwargs):
        return self.text_to_text(self.text_core.prepare(text), **kwargs)

    def run_ocr(self, image_or_path, language="eng"):
        return self.ocr_core.read(image_or_path, language=language)

    def optimized_matmul(self, left, right):
        return self.matrix_engine.matmul(left, right)

    def status(self):
        report = super().status()
        report["name"] = self.name
        report["version"] = self.VERSION
        report["cores"] = self.core_status.copy()
        report["janus_model_id"] = self.janus.model_id
        report["janus_device"] = self.janus.device
        return report


# Convenient explicit factory; does not download or load Janus automatically.
def create_gpu_x2(memory_limit_mb=180, tile_size=64,
                  janus_model_id="deepseek-ai/Janus-Pro-1B"):
    return PureAdvancedNeuralGPUX2(
        memory_limit_mb=memory_limit_mb,
        tile_size=tile_size,
        janus_model_id=janus_model_id,
    )
