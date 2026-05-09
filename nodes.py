import copy
import math
import os
import re
import textwrap
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths


AXIS_TYPES = [
    "None",
    "Checkpoint",
    "Diffusion Model",
    "VAE",
    "LoRA",
    "CLIP Skip",
    "Prompt S/R",
    "Positive Prompt",
    "Negative Prompt",
    "Seed",
    "Steps",
    "CFG",
    "Sampler",
    "Scheduler",
    "Denoise",
]

UNET_DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
SEED_MODES = ["Fixed", "Increment per cell", "Increment per row", "Increment per column"]
class AnyType(str):
    def __ne__(self, other):
        return False


ANY_TYPE = AnyType("*")


class FlexibleOptionalInputType(dict):
    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return (ANY_TYPE,)


def _sampler_names():
    if hasattr(comfy.samplers, "KSampler") and hasattr(comfy.samplers.KSampler, "SAMPLERS"):
        return comfy.samplers.KSampler.SAMPLERS
    return comfy.samplers.SAMPLER_NAMES


def _scheduler_names():
    if hasattr(comfy.samplers, "KSampler") and hasattr(comfy.samplers.KSampler, "SCHEDULERS"):
        return comfy.samplers.KSampler.SCHEDULERS
    return comfy.samplers.SCHEDULER_NAMES


@dataclass
class PlotState:
    model: object
    clip: object
    vae: object
    positive_prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float


def _sorted_kwargs(kwargs, prefix):
    found = []
    for key, value in kwargs.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        try:
            index = int(suffix)
        except ValueError:
            index = 999999
        found.append((index, value))
    return [value for _, value in sorted(found, key=lambda item: item[0])]


class OGN_XYCheckpointAxis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_1": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": FlexibleOptionalInputType(),
        }

    RETURN_TYPES = ("OGN_XY_AXIS",)
    RETURN_NAMES = ("axis",)
    FUNCTION = "build_axis"
    CATEGORY = "OGN/XY Plot"

    def build_axis(self, checkpoint_1, **kwargs):
        values = [checkpoint_1] + _sorted_kwargs(kwargs, "checkpoint_")
        values = [value for value in values if value]
        return ({"type": "Checkpoint", "values": values},)


class OGN_XYDiffusionModelAxis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model_1": (folder_paths.get_filename_list("diffusion_models"),),
            },
            "optional": FlexibleOptionalInputType(),
        }

    RETURN_TYPES = ("OGN_XY_AXIS",)
    RETURN_NAMES = ("axis",)
    FUNCTION = "build_axis"
    CATEGORY = "OGN/XY Plot"

    def build_axis(self, diffusion_model_1, **kwargs):
        values = [diffusion_model_1] + _sorted_kwargs(kwargs, "diffusion_model_")
        values = [value for value in values if value]
        return ({"type": "Diffusion Model", "values": values},)


class OGN_XYLoraAxis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_1": (["None"] + folder_paths.get_filename_list("loras"),),
                "strength_model_1": (
                    "FLOAT",
                    {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05},
                ),
            },
            "optional": FlexibleOptionalInputType(),
        }

    RETURN_TYPES = ("OGN_XY_AXIS",)
    RETURN_NAMES = ("axis",)
    FUNCTION = "build_axis"
    CATEGORY = "OGN/XY Plot"

    def build_axis(self, lora_1, strength_model_1, **kwargs):
        sets = {1: {1: {"lora": lora_1, "strength_model": strength_model_1}}}
        for key, value in kwargs.items():
            lora_match = re.fullmatch(r"lora_(\d+)", key)
            strength_match = re.fullmatch(r"strength_model_(\d+)", key)
            nested_lora_match = re.fullmatch(r"lora(\d+)_(\d+)", key)
            nested_strength_match = re.fullmatch(r"strength_model_(\d+)_(\d+)", key)
            if lora_match:
                set_index = int(lora_match.group(1))
                sets.setdefault(set_index, {}).setdefault(1, {})["lora"] = value
            elif strength_match:
                set_index = int(strength_match.group(1))
                sets.setdefault(set_index, {}).setdefault(1, {})["strength_model"] = value
            elif nested_lora_match:
                set_index = int(nested_lora_match.group(1))
                item_index = int(nested_lora_match.group(2))
                sets.setdefault(set_index, {}).setdefault(item_index, {})["lora"] = value
            elif nested_strength_match:
                set_index = int(nested_strength_match.group(1))
                item_index = int(nested_strength_match.group(2))
                sets.setdefault(set_index, {}).setdefault(item_index, {})["strength_model"] = value

        values = []
        for set_index, items in sorted(sets.items()):
            loras = []
            for _, row in sorted(items.items()):
                lora = row.get("lora")
                if not lora or str(lora).lower() == "none":
                    continue
                strength = float(row.get("strength_model", row.get("strength", 1.0)))
                if not math.isfinite(strength):
                    strength = 1.0
                loras.append({"lora": lora, "strength_model": strength, "strength_clip": strength})
            values.append({"set": set_index, "loras": loras})
        return ({"type": "LoRA", "values": values},)


class OGN_XYPromptSRAxis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search": ("STRING", {"multiline": False, "default": ""}),
                "replace_1": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": FlexibleOptionalInputType(),
        }

    RETURN_TYPES = ("OGN_XY_AXIS",)
    RETURN_NAMES = ("axis",)
    FUNCTION = "build_axis"
    CATEGORY = "OGN/XY Plot"

    def build_axis(self, search, replace_1, **kwargs):
        replacements = [replace_1] + _sorted_kwargs(kwargs, "replace_")
        replacements = [value for value in replacements if value is not None]
        return (
            {
                "type": "Prompt S/R",
                "values": [{"search": search, "replacement": replacement} for replacement in replacements],
            },
        )


class OGN_XYPrimitiveAxis:
    @classmethod
    def INPUT_TYPES(cls):
        primitive_types = [
            "Positive Prompt",
            "Negative Prompt",
            "Seed",
            "Steps",
            "CFG",
            "Sampler",
            "Scheduler",
            "Denoise",
            "VAE",
            "CLIP Skip",
        ]
        return {
            "required": {
                "axis_type": (primitive_types,),
                "value_1": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": FlexibleOptionalInputType(),
        }

    RETURN_TYPES = ("OGN_XY_AXIS",)
    RETURN_NAMES = ("axis",)
    FUNCTION = "build_axis"
    CATEGORY = "OGN/XY Plot"

    def build_axis(self, axis_type, value_1, **kwargs):
        values = [value_1] + _sorted_kwargs(kwargs, "value_")
        values = [value for value in values if value is not None and str(value) != ""]
        return ({"type": axis_type, "values": values},)


class OGN_XYPlot:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "latent_image": ("LATENT",),
                "positive_prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True, "default": ""},
                ),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "dynamicPrompts": True, "default": ""},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed_mode": (SEED_MODES,),
                "diffusion_weight_dtype": (UNET_DTYPES,),
                "include_labels": ("BOOLEAN", {"default": True}),
                "label_font_size": ("INT", {"default": 16, "min": 8, "max": 64}),
                "label_padding": ("INT", {"default": 10, "min": 0, "max": 80}),
            },
            "optional": {
                "x_axis": ("OGN_XY_AXIS",),
                "y_axis": ("OGN_XY_AXIS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("plot_image", "cell_images")
    FUNCTION = "plot"
    CATEGORY = "OGN/XY Plot"

    def plot(
        self,
        model,
        clip,
        vae,
        latent_image,
        positive_prompt,
        negative_prompt,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        seed_mode,
        diffusion_weight_dtype,
        include_labels,
        label_font_size,
        label_padding,
        x_axis=None,
        y_axis=None,
    ):
        x_type, x_values = self._normalize_axis(x_axis)
        y_type, y_values = self._normalize_axis(y_axis)
        base = PlotState(
            model=model,
            clip=clip,
            vae=vae,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        )

        images = []
        grid_cells = []
        for y_index, y_value in enumerate(y_values):
            row = []
            for x_index, x_value in enumerate(x_values):
                state = copy.copy(base)
                state.seed = self._cell_seed(seed, seed_mode, x_index, y_index, len(x_values))
                self._apply_axis(state, x_type, x_value, x_values, diffusion_weight_dtype)
                self._apply_axis(state, y_type, y_value, y_values, diffusion_weight_dtype)

                positive = self._encode_text(state.clip, state.positive_prompt)
                negative = self._encode_text(state.clip, state.negative_prompt)
                sampled = self._run_ksampler(
                    state.model,
                    state.seed,
                    state.steps,
                    state.cfg,
                    state.sampler_name,
                    state.scheduler,
                    positive,
                    negative,
                    latent_image,
                    state.denoise,
                )
                decoded = self._decode(state.vae, sampled)
                images.append(decoded)
                row.append(decoded[0])
            grid_cells.append(row)

        cell_batch = torch.cat(images, dim=0)
        grid = self._make_grid_image(
            grid_cells,
            x_labels=[self._label(x_type, value) for value in x_values],
            y_labels=[self._label(y_type, value) for value in y_values],
            include_labels=include_labels,
            font_size=label_font_size,
            padding=label_padding,
        )
        return (grid, cell_batch)

    def _normalize_axis(self, axis):
        if not axis:
            return "None", [""]
        axis_type = axis.get("type", "None")
        values = axis.get("values", [""])
        if not values:
            raise ValueError(f"{axis_type} axis needs at least one value.")
        return axis_type, values

    def _apply_axis(self, state, axis_type, value, axis_values, diffusion_weight_dtype):
        if axis_type == "None":
            return
        if axis_type == "Checkpoint":
            state.model, state.clip, state.vae = self._load_checkpoint(value)
        elif axis_type == "Diffusion Model":
            state.model = self._load_diffusion_model(value, diffusion_weight_dtype)
        elif axis_type == "VAE":
            state.vae = self._load_vae(value)
        elif axis_type == "LoRA":
            state.model, state.clip = self._apply_lora(state.model, state.clip, value)
        elif axis_type == "CLIP Skip":
            state.clip = self._apply_clip_skip(state.clip, int(value))
        elif axis_type == "Prompt S/R":
            search, replacement = self._prompt_sr_pair(value, axis_values)
            state.positive_prompt = state.positive_prompt.replace(search, replacement)
            state.negative_prompt = state.negative_prompt.replace(search, replacement)
        elif axis_type == "Positive Prompt":
            state.positive_prompt = value
        elif axis_type == "Negative Prompt":
            state.negative_prompt = value
        elif axis_type == "Seed":
            state.seed = int(value)
        elif axis_type == "Steps":
            state.steps = int(value)
        elif axis_type == "CFG":
            state.cfg = float(value)
        elif axis_type == "Sampler":
            state.sampler_name = value
        elif axis_type == "Scheduler":
            state.scheduler = value
        elif axis_type == "Denoise":
            state.denoise = float(value)

    def _prompt_sr_pair(self, value, axis_values):
        if isinstance(value, dict):
            return value.get("search", ""), value.get("replacement", "")
        if "=>" in value:
            search, replacement = value.split("=>", 1)
            return search, replacement
        if "->" in value:
            search, replacement = value.split("->", 1)
            return search, replacement
        return axis_values[0], value

    def _load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]

    def _load_diffusion_model(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn" and hasattr(torch, "float8_e4m3fn"):
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast" and hasattr(torch, "float8_e4m3fn"):
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2" and hasattr(torch, "float8_e5m2"):
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        return comfy.sd.load_diffusion_model(unet_path, model_options=model_options)

    def _load_vae(self, vae_name):
        try:
            import nodes

            if hasattr(nodes, "VAELoader"):
                return nodes.VAELoader().load_vae(vae_name)[0]
        except Exception:
            pass
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        return comfy.sd.VAE(sd=sd)

    def _apply_lora(self, model, clip, value):
        if isinstance(value, dict):
            if "loras" in value:
                for lora_value in value.get("loras", []):
                    model, clip = self._apply_lora(model, clip, lora_value)
                return model, clip
            lora_name = value.get("lora", "None")
            strength_model = float(value.get("strength_model", 1.0))
            strength_clip = float(value.get("strength_clip", strength_model))
            if str(lora_name).strip().lower() in {"none", "no lora", "disabled", "off"}:
                return model, clip
            if strength_model == 0 and strength_clip == 0:
                return model, clip
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        if value.strip().lower() in {"none", "no lora", "disabled", "off"}:
            return model, clip
        parts = [part.strip() for part in value.split(":")]
        lora_name = parts[0]
        strength_model = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
        strength_clip = float(parts[2]) if len(parts) > 2 and parts[2] else strength_model
        if strength_model == 0 and strength_clip == 0:
            return model, clip
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        return comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

    def _apply_clip_skip(self, clip, stop_at_clip_layer):
        cloned = clip.clone()
        cloned.clip_layer(stop_at_clip_layer)
        return cloned

    def _encode_text(self, clip, text):
        if clip is None:
            raise RuntimeError("CLIP is required for prompt encoding.")
        tokens = clip.tokenize(text)
        if hasattr(clip, "encode_from_tokens_scheduled"):
            return clip.encode_from_tokens_scheduled(tokens)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def _run_ksampler(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise,
    ):
        latent_copy = self._clone_latent(latent)
        try:
            import nodes

            if hasattr(nodes, "common_ksampler"):
                return nodes.common_ksampler(
                    model,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent_copy,
                    denoise=denoise,
                )[0]
        except Exception:
            pass

        latent_samples = latent_copy["samples"]
        noise_mask = latent_copy.get("noise_mask")
        batch_inds = latent_copy.get("batch_index")
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            noise_mask=noise_mask,
        )
        out = latent_copy.copy()
        out["samples"] = samples
        return out

    def _clone_latent(self, latent):
        cloned = {}
        for key, value in latent.items():
            cloned[key] = value.clone() if hasattr(value, "clone") else copy.deepcopy(value)
        return cloned

    def _decode(self, vae, samples):
        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return images

    def _cell_seed(self, seed, mode, x_index, y_index, x_count):
        if mode == "Increment per cell":
            return seed + y_index * x_count + x_index
        if mode == "Increment per row":
            return seed + y_index
        if mode == "Increment per column":
            return seed + x_index
        return seed

    def _label(self, axis_type, value):
        if axis_type == "None":
            return ""
        if axis_type == "LoRA":
            if isinstance(value, dict):
                if "loras" in value:
                    labels = [
                        f"{self._display_name(item.get('lora'))} ({item.get('strength_model')})"
                        for item in value.get("loras", [])
                    ]
                    return " + ".join(labels) if labels else "None"
                lora_name = self._display_name(value.get("lora"))
                return f"{lora_name} ({value.get('strength_model')})"
            return self._display_name(value)
        if axis_type == "Prompt S/R" and isinstance(value, dict):
            return str(value.get("replacement", ""))
        if axis_type in {"Checkpoint", "Diffusion Model", "VAE"}:
            return self._display_name(value)
        return str(value)

    def _display_name(self, value):
        if value is None:
            return ""
        text = str(value).replace("\\", "/")
        name = text.rsplit("/", 1)[-1]
        return os.path.splitext(name)[0] or name

    def _make_grid_image(self, rows, x_labels, y_labels, include_labels, font_size, padding):
        first = rows[0][0]
        cell_h, cell_w = int(first.shape[0]), int(first.shape[1])
        cols = len(rows[0])
        row_count = len(rows)

        font = self._font(font_size)
        left = self._y_label_area(y_labels, font, padding) if include_labels and any(y_labels) else 0
        top = self._x_label_area(x_labels, font, padding) if include_labels and any(x_labels) else 0
        width = left + cols * cell_w
        height = top + row_count * cell_h
        canvas = Image.new("RGB", (width, height), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)

        if include_labels:
            for index, label in enumerate(x_labels):
                self._draw_label(draw, label, (left + index * cell_w, 0, cell_w, top), font, padding)
            for index, label in enumerate(y_labels):
                self._draw_label(draw, label, (0, top + index * cell_h, left, cell_h), font, padding)

        for y, row in enumerate(rows):
            for x, tensor in enumerate(row):
                canvas.paste(self._tensor_to_pil(tensor), (left + x * cell_w, top + y * cell_h))
        return self._pil_to_tensor(canvas)

    def _y_label_area(self, labels, font, padding):
        if not labels:
            return 0
        widths = []
        for label in labels:
            lines = self._wrapped_lines(label, font, 260)
            widths.extend([self._text_size(font, line)[0] for line in lines])
        return min(max(widths, default=0) + padding * 2, 320)

    def _x_label_area(self, labels, font, padding):
        if not labels:
            return 0
        line_h = self._text_size(font, "Ag")[1] + 2
        max_lines = max(len(self._wrapped_lines(label, font, 260)) for label in labels)
        return min(max_lines * line_h + padding * 2, 180)

    def _draw_label(self, draw, label, box, font, padding):
        x, y, width, height = box
        if width <= 0 or height <= 0:
            return
        draw.rectangle((x, y, x + width, y + height), fill=(35, 35, 35))
        max_text_width = max(1, width - padding * 2)
        lines = self._wrapped_lines(label, font, max_text_width)
        line_h = self._text_size(font, "Ag")[1] + 2
        total_h = line_h * len(lines)
        cursor_y = y + max(padding, (height - total_h) // 2)
        for line in lines:
            text_w = self._text_size(font, line)[0]
            cursor_x = x + max(padding, (width - text_w) // 2)
            draw.text((cursor_x, cursor_y), line, fill=(235, 235, 235), font=font)
            cursor_y += line_h

    def _wrapped_lines(self, text, font, max_width):
        if not text:
            return [""]
        chunks = []
        for raw_line in str(text).splitlines():
            if not raw_line:
                chunks.append("")
                continue
            estimate = max(8, int(max_width / max(1, self._text_size(font, "M")[0])))
            chunks.extend(textwrap.wrap(raw_line, width=estimate, break_long_words=True) or [""])
        return chunks[: max(1, math.floor(320 / max(1, self._text_size(font, "Ag")[1])))]

    def _font(self, size):
        font_candidates = {
            "Meiryo": ["meiryo.ttc", "meiryob.ttc"],
            "Yu Gothic": ["YuGothM.ttc", "YuGothR.ttc", "YuGothB.ttc"],
            "Arial": ["arial.ttf", "arialbd.ttf"],
            "DejaVu Sans": ["DejaVuSans.ttf"],
            "Auto": ["meiryo.ttc", "YuGothM.ttc", "arial.ttf", "DejaVuSans.ttf"],
        }
        candidates = font_candidates["Auto"]
        font_dirs = [
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/opentype/noto",
        ]
        for candidate in candidates:
            for font_dir in font_dirs:
                path = os.path.join(font_dir, candidate)
                if os.path.exists(path):
                    try:
                        return ImageFont.truetype(path, size=size)
                    except Exception:
                        pass
        try:
            return ImageFont.truetype(candidates[-1], size=size)
        except Exception:
            return ImageFont.load_default()

    def _text_size(self, font, text):
        if hasattr(font, "getbbox"):
            left, top, right, bottom = font.getbbox(text)
            return right - left, bottom - top
        return font.getsize(text)

    def _tensor_to_pil(self, tensor):
        array = tensor.detach().cpu().numpy()
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    def _pil_to_tensor(self, image):
        array = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(array)[None,]


NODE_CLASS_MAPPINGS = {
    "OGN_XYPlot": OGN_XYPlot,
    "OGN_XYCheckpointAxis": OGN_XYCheckpointAxis,
    "OGN_XYDiffusionModelAxis": OGN_XYDiffusionModelAxis,
    "OGN_XYLoraAxis": OGN_XYLoraAxis,
    "OGN_XYPromptSRAxis": OGN_XYPromptSRAxis,
    "OGN_XYPrimitiveAxis": OGN_XYPrimitiveAxis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OGN_XYPlot": "OGN_XY Plot",
    "OGN_XYCheckpointAxis": "OGN_XY Checkpoint Axis",
    "OGN_XYDiffusionModelAxis": "OGN_XY Diffusion Model Axis",
    "OGN_XYLoraAxis": "OGN_XY LoRA Axis",
    "OGN_XYPromptSRAxis": "OGN_XY Prompt S/R Axis",
    "OGN_XYPrimitiveAxis": "OGN_XY Primitive Axis",
}
