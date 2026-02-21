#!/usr/bin/env python3
"""
Perch v2 / BirdNET helper for auto naming selected REAPER items.

Input request JSON:
{
  "paths": [...],
  "top_k": 30,
  "perch_model": "/Users/apple/Documents/Perch v2/perch_v2.onnx",
  "perch_labels": "/Users/apple/Documents/Perch v2/labels.txt"
}

Output TSV:
path, name, action, subcat, material, confidence,
 top1, top2, top3, top1_desc, top2_desc, top3_desc,
 content_type, vehicle_model, scene_markers, list_mode_used, location_used,
 threshold_mode, name_raw_cutoff, marker_threshold_mode, marker_raw_cutoff,
 raw_top1, name_threshold_rel, marker_threshold_rel
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except Exception:
    print("ERROR: numpy missing", file=sys.stderr)
    sys.exit(2)

try:
    import soundfile as sf
except Exception:
    print("ERROR: soundfile missing", file=sys.stderr)
    sys.exit(2)

try:
    import soxr
except Exception:
    print("ERROR: soxr missing", file=sys.stderr)
    sys.exit(2)

SAMPLE_RATE = 32000
DEFAULT_WINDOW_S = 5.0
DEFAULT_HOP_RATIO = 0.5
DEFAULT_TOP_K = 30
MIN_SEGMENT_S = 1.0
MARKER_THRESHOLD = 0.02
MARKER_GAP_S = 8.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.join(SCRIPT_DIR, "SFX_BirdDetect_Pack")
BUNDLE_PERCH_DIR = os.path.join(BUNDLE_DIR, "models", "perch_v2")
DEFAULT_LOCAL_BIRDS_FILE = os.path.join(BUNDLE_DIR, "species_list.txt")
FALLBACK_LOCAL_BIRDS_FILES = [
    os.path.join(SCRIPT_DIR, "species_list.txt"),
    "/Users/apple/Documents/Perch v2/local_birds.txt",
]
DEFAULT_PERCH_MODEL_FILE = os.path.join(BUNDLE_PERCH_DIR, "perch_v2.onnx")
DEFAULT_PERCH_LABELS_FILE = os.path.join(BUNDLE_PERCH_DIR, "labels.txt")
FALLBACK_PERCH_MODEL_FILES = ["/Users/apple/Documents/Perch v2/perch_v2.onnx"]
FALLBACK_PERCH_LABELS_FILES = ["/Users/apple/Documents/Perch v2/labels.txt"]


@dataclass
class LabelRow:
    scientific: str
    common: str
    group: str


def sanitize(s: str) -> str:
    s = str(s or "")
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(s.split())


def normalize_key(s: str) -> str:
    s = sanitize(s).lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def as_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = sanitize(v).lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def first_existing_file(paths: Sequence[str]) -> str:
    for p in paths:
        pp = sanitize(p)
        if pp and os.path.isfile(pp):
            return pp
    if paths:
        return sanitize(paths[0])
    return ""


def load_audio(path: str, target_sr: int = SAMPLE_RATE, max_duration_s: float = 600.0) -> Optional[np.ndarray]:
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        return None

    if audio is None or len(audio) == 0:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if max_duration_s and max_duration_s > 0:
        max_samples = int(max_duration_s * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    if sr != target_sr:
        audio = soxr.resample(audio, sr, target_sr)

    if len(audio) < int(MIN_SEGMENT_S * target_sr):
        return None

    return np.asarray(audio, dtype=np.float32)


def load_labels(path: str) -> List[LabelRow]:
    out: List[LabelRow] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("~")
            if len(parts) >= 3:
                sci = sanitize(parts[0])
                common = sanitize(parts[1])
                group = sanitize(parts[2])
            elif len(parts) == 2:
                sci = sanitize(parts[0])
                common = sanitize(parts[1])
                group = ""
            else:
                sci = sanitize(parts[0])
                common = sanitize(parts[0])
                group = ""
            out.append(LabelRow(scientific=sci, common=common, group=group))
    return out


def load_local_birds(path: str) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    raw_names: Set[str] = set()
    norm_names: Set[str] = set()
    preferred_common: Dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return raw_names, norm_names, preferred_common
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = sanitize(line)
            if not t or t.startswith("#"):
                continue
            sci = ""
            common = ""
            if "_" in t:
                left, right = t.split("_", 1)
                sci = sanitize(left)
                common = sanitize(right)
            else:
                common = t

            candidates: List[str] = []
            if sci:
                candidates.append(sci)
            if common:
                candidates.append(common)
            if not candidates:
                continue

            for nm in candidates:
                low = nm.lower()
                raw_names.add(low)
                nrm = normalize_key(nm)
                if nrm:
                    norm_names.add(nrm)

            if common:
                pref = common
                if sci:
                    ns = normalize_key(sci)
                    if ns:
                        preferred_common[ns] = pref
                nc = normalize_key(common)
                if nc:
                    preferred_common[nc] = pref

    return raw_names, norm_names, preferred_common


def load_local_species_for_birdnet(path: str) -> Set[str]:
    out: Set[str] = set()
    if not path or not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = sanitize(line)
            if not t or t.startswith("#"):
                continue
            if "_" in t:
                left, _ = t.split("_", 1)
                sci = sanitize(left)
                if sci:
                    out.add(sci)
            else:
                # BirdNET filters are most reliable with scientific names.
                # Keep 2-token names as likely scientific names.
                parts = t.split()
                if len(parts) >= 2:
                    out.add(" ".join(parts[:2]))
    return out


def map_local_species_to_birdnet(local_species: Set[str], model: Any) -> Set[str]:
    mapped: Set[str] = set()
    if not local_species:
        return mapped

    model_species = None
    model_candidates: List[Any] = [model]
    if isinstance(model, dict):
        for key in ("audio_model", "model", "acoustic_model"):
            candidate = model.get(key)
            if candidate is not None:
                model_candidates.append(candidate)

    for candidate in model_candidates:
        model_species = getattr(candidate, "species_list", None)
        if model_species is None:
            model_species = getattr(candidate, "species", None)
        if model_species is not None:
            break

    if model_species is None:
        return mapped

    available: List[str] = [sanitize(x) for x in model_species if sanitize(x)]
    if not available:
        return mapped

    available_set = set(available)
    by_sci: Dict[str, str] = {}
    by_sci_norm: Dict[str, str] = {}

    for row in available:
        sci = row
        if "_" in row:
            sci, _ = row.split("_", 1)
        sci = sanitize(sci)
        if not sci:
            continue
        if sci not in by_sci:
            by_sci[sci] = row
        ns = normalize_key(sci)
        if ns and ns not in by_sci_norm:
            by_sci_norm[ns] = row

    for requested in local_species:
        req = sanitize(requested)
        if not req:
            continue
        if req in available_set:
            mapped.add(req)
            continue
        if req in by_sci:
            mapped.add(by_sci[req])
            continue
        nreq = normalize_key(req)
        if nreq and nreq in by_sci_norm:
            mapped.add(by_sci_norm[nreq])

    return mapped


def birdnet_species_scientific(species: str) -> str:
    s = sanitize(species)
    if "_" in s:
        left, _ = s.split("_", 1)
        return sanitize(left)
    return s


def birdnet_species_common(species: str) -> str:
    s = sanitize(species)
    if "_" in s:
        _, right = s.split("_", 1)
        return sanitize(right)
    return ""


def birdnet_display_name(species: str, preferred_common: Dict[str, str]) -> str:
    s = sanitize(species)
    sci = birdnet_species_scientific(s)
    com = birdnet_species_common(s)

    ns = normalize_key(sci)
    if ns and ns in preferred_common:
        return sanitize(preferred_common[ns])

    nc = normalize_key(com)
    if nc and nc in preferred_common:
        return sanitize(preferred_common[nc])

    if com:
        return com
    return sci or s


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def has_avx_support() -> bool:
    arch = platform.machine().lower()
    if arch.startswith("arm") or arch.startswith("aarch"):
        return True
    if arch not in {"x86_64", "amd64", "i386", "i686"}:
        return True
    if sys.platform != "darwin":
        return True
    keys = ["machdep.cpu.features", "machdep.cpu.leaf7_features"]
    for key in keys:
        try:
            out = subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL)
        except Exception:
            continue
        txt = sanitize(out.decode("utf-8", errors="ignore")).upper()
        if " AVX " in f" {txt} ":
            return True
    return False


def _parse_interval_key(interval_key: Any) -> Tuple[float, float]:
    if isinstance(interval_key, (list, tuple)) and len(interval_key) >= 2:
        s = _as_float(interval_key[0])
        e = _as_float(interval_key[1])
        if s is not None and e is not None:
            return max(0.0, s), max(0.0, e)
    txt = sanitize(interval_key)
    nums = re.findall(r"[-+]?\d*\.?\d+", txt)
    if len(nums) >= 2:
        s = _as_float(nums[0])
        e = _as_float(nums[1])
        if s is not None and e is not None:
            return max(0.0, s), max(0.0, e)
    return 0.0, 0.0


def _coerce_birdnet_event(pred: Any, default_start: float, default_end: float) -> Optional[Tuple[float, float, str, float]]:
    species: Any = ""
    conf: Any = None
    st: Any = default_start
    et: Any = default_end

    if isinstance(pred, dict):
        species = (
            pred.get("species_name")
            or pred.get("species")
            or pred.get("prediction")
            or pred.get("name")
            or pred.get("label")
            or ""
        )
        conf = pred.get("confidence")
        if conf is None:
            conf = pred.get("score")
        if conf is None:
            conf = pred.get("probability")
        st = pred.get("start_time", pred.get("start", default_start))
        et = pred.get("end_time", pred.get("end", default_end))
    elif isinstance(pred, (list, tuple)):
        if len(pred) < 2:
            return None
        a = pred[0]
        b = pred[1]
        if isinstance(a, (int, float)) and not isinstance(b, (int, float)):
            conf = a
            species = b
        else:
            species = a
            conf = b
        if len(pred) >= 3:
            st = pred[2]
        if len(pred) >= 4:
            et = pred[3]
    else:
        species = (
            getattr(pred, "species_name", None)
            or getattr(pred, "species", None)
            or getattr(pred, "prediction", None)
            or getattr(pred, "name", None)
            or getattr(pred, "label", None)
            or ""
        )
        conf = getattr(pred, "confidence", None)
        if conf is None:
            conf = getattr(pred, "score", None)
        if conf is None:
            conf = getattr(pred, "probability", None)
        st = getattr(pred, "start_time", default_start)
        et = getattr(pred, "end_time", default_end)

    species_s = sanitize(species)
    conf_f = _as_float(conf)
    st_f = _as_float(st)
    et_f = _as_float(et)
    if not species_s or conf_f is None:
        return None
    if st_f is None:
        st_f = default_start
    if et_f is None:
        et_f = default_end
    st_f = max(0.0, st_f)
    et_f = max(st_f, et_f)
    conf_f = max(0.0, min(1.0, conf_f))
    return st_f, et_f, species_s, conf_f


def extract_birdnet_events(raw_pred: Any) -> List[Tuple[float, float, str, float]]:
    events: List[Tuple[float, float, str, float]] = []

    # New API: pandas DataFrame-like output.
    if hasattr(raw_pred, "to_dict") and hasattr(raw_pred, "columns"):
        try:
            rows = raw_pred.to_dict(orient="records")
            for row in rows:
                ev = _coerce_birdnet_event(row, 0.0, 0.0)
                if ev is not None:
                    events.append(ev)
            if events:
                return events
        except Exception:
            pass

    # Legacy API: dict[interval] -> iterable[prediction].
    if isinstance(raw_pred, dict):
        for interval_key, preds in raw_pred.items():
            st, et = _parse_interval_key(interval_key)
            if isinstance(preds, (list, tuple)):
                for pred in preds:
                    ev = _coerce_birdnet_event(pred, st, et)
                    if ev is not None:
                        events.append(ev)
            else:
                ev = _coerce_birdnet_event(preds, st, et)
                if ev is not None:
                    events.append(ev)
        return events

    if isinstance(raw_pred, (list, tuple)):
        for pred in raw_pred:
            ev = _coerce_birdnet_event(pred, 0.0, 0.0)
            if ev is not None:
                events.append(ev)
        return events

    ev = _coerce_birdnet_event(raw_pred, 0.0, 0.0)
    if ev is not None:
        events.append(ev)
    return events


def load_birdnet_model() -> Tuple[str, Any]:
    errors: List[str] = []
    try:
        import birdnet  # type: ignore

        if hasattr(birdnet, "load"):
            try:
                model = birdnet.load("acoustic", "2.4", "tf")
                return "modern", model
            except Exception as ex:
                errors.append(f"birdnet.load failed: {ex}")

        # Older BirdNET Python API branch (e.g., 0.1.x).
        try:
            from birdnet.models.v2m4 import AudioModelV2M4TFLite  # type: ignore

            model = AudioModelV2M4TFLite(tflite_num_threads=1, language="en_us")
            return "legacy_v2m4_tflite", {"birdnet_module": birdnet, "audio_model": model}
        except Exception as ex:
            errors.append(f"AudioModelV2M4TFLite load failed: {ex}")
    except Exception as ex:
        errors.append(f"import birdnet failed: {ex}")

    try:
        from birdnet.models import ModelV2M4  # type: ignore

        return "legacy", ModelV2M4()
    except Exception as ex:
        errors.append(f"ModelV2M4 load failed: {ex}")

    raise RuntimeError(" ; ".join(errors))


def prepare_birdnet_legacy_input(src: str, max_audio_duration_min: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    max_duration_s: Optional[float] = None
    if max_audio_duration_min is not None and max_audio_duration_min > 0:
        max_duration_s = float(max_audio_duration_min) * 60.0

    audio = load_audio(src, target_sr=SAMPLE_RATE, max_duration_s=max_duration_s)
    if audio is None:
        return None, None

    tmp_file = tempfile.NamedTemporaryFile(prefix="sfx_birdnet_legacy_", suffix=".wav", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()
    sf.write(tmp_path, audio, SAMPLE_RATE)
    return tmp_path, tmp_path


def run_birdnet_predict(
    api_kind: str,
    model: Any,
    src: str,
    custom_species: Optional[Set[str]],
    top_k: int,
    max_audio_duration_min: Optional[float],
) -> Any:
    if api_kind == "modern":
        kwargs: Dict[str, Any] = {}
        kwargs["top_k"] = max(3, min(100, int(top_k or 5)))
        kwargs["n_feeders"] = 1
        kwargs["n_workers"] = 1
        kwargs["batch_size"] = 1
        kwargs["prefetch_ratio"] = 1
        kwargs["device"] = "CPU"
        if max_audio_duration_min is not None and max_audio_duration_min > 0:
            kwargs["max_audio_duration_min"] = float(max_audio_duration_min)
        if custom_species:
            kwargs["custom_species_list"] = sorted(custom_species)
        try:
            return model.predict(src, **kwargs)
        except TypeError:
            # Some builds may not support all keyword options.
            kwargs2: Dict[str, Any] = {}
            if custom_species:
                kwargs2["custom_species_list"] = sorted(custom_species)
            try:
                return model.predict(src, **kwargs2)
            except TypeError:
                pass

            # Some builds may use filter_species naming.
            if custom_species:
                return model.predict(src, filter_species=set(custom_species))
            return model.predict(src)

    if api_kind == "legacy_v2m4_tflite":
        birdnet_module = model.get("birdnet_module")
        audio_model = model.get("audio_model")
        if birdnet_module is None or audio_model is None:
            raise RuntimeError("legacy_v2m4_tflite model payload is incomplete")

        src_for_predict, cleanup_path = prepare_birdnet_legacy_input(src, max_audio_duration_min)
        if not src_for_predict:
            return []

        try:
            kwargs3: Dict[str, Any] = {
                "min_confidence": 0.0,
                "batch_size": 1,
                "chunk_overlap_s": 0.0,
                "use_bandpass": True,
                "apply_sigmoid": True,
                "custom_model": audio_model,
                "silent": True,
            }
            if custom_species:
                kwargs3["species_filter"] = set(custom_species)

            gen = birdnet_module.predict_species_within_audio_file(Path(src_for_predict), **kwargs3)
            out_events: List[Dict[str, Any]] = []
            per_chunk_top = max(3, min(100, int(top_k or 5)))
            for interval, preds in gen:
                if isinstance(interval, (list, tuple)) and len(interval) >= 2:
                    st = float(interval[0])
                    et = float(interval[1])
                else:
                    st = 0.0
                    et = 0.0

                if hasattr(preds, "items"):
                    iterable = list(preds.items())
                else:
                    iterable = list(preds)

                count = 0
                for entry in iterable:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        continue
                    species = sanitize(entry[0])
                    conf = _as_float(entry[1])
                    if not species or conf is None:
                        continue
                    out_events.append(
                        {
                            "start_time": st,
                            "end_time": et,
                            "species_name": species,
                            "confidence": max(0.0, min(1.0, float(conf))),
                        }
                    )
                    count += 1
                    if count >= per_chunk_top:
                        break
            return out_events
        finally:
            if cleanup_path:
                try:
                    os.remove(cleanup_path)
                except Exception:
                    pass

    kwargs2: Dict[str, Any] = {}
    if custom_species:
        kwargs2["filter_species"] = set(custom_species)
    return model.predict_species_within_audio_file(Path(src), **kwargs2)


def choose_label_indices(
    labels: List[LabelRow],
    list_mode: str,
    local_names_raw: Set[str],
    local_names_norm: Set[str],
    strict_local: bool = False,
) -> Tuple[List[int], str]:
    mode = (list_mode or "aves").strip().lower()
    if mode == "all":
        return list(range(len(labels))), "all"

    def idx_for_group(group_name: str) -> List[int]:
        g = sanitize(group_name).lower()
        return [i for i, l in enumerate(labels) if sanitize(l.group).lower() == g]

    aves = idx_for_group("aves")
    if mode in {"aves", "mammalia", "insecta", "amphibia"}:
        idx = idx_for_group(mode)
        if idx:
            return idx, mode
        if aves:
            return aves, f"{mode}_fallback_aves"
        return list(range(len(labels))), f"{mode}_fallback_all"

    if mode == "local":
        if not local_names_raw and not local_names_norm:
            if strict_local:
                return [], "local_missing"
            if aves:
                return aves, "aves_fallback"
            return list(range(len(labels))), "all_fallback"

        idx: List[int] = []
        for i, l in enumerate(labels):
            common = sanitize(l.common).lower()
            scientific = sanitize(l.scientific).lower()
            common_n = normalize_key(common)
            scientific_n = normalize_key(scientific)
            if (
                common in local_names_raw
                or scientific in local_names_raw
                or (common_n and common_n in local_names_norm)
                or (scientific_n and scientific_n in local_names_norm)
            ):
                idx.append(i)
        if idx:
            return idx, "local"
        if strict_local:
            return [], "local_no_match"
        if aves:
            return aves, "aves_fallback"
        return list(range(len(labels))), "all_fallback"

    if aves:
        return aves, "aves"
    return list(range(len(labels))), "all_fallback"


def pick_window_samples(input_shape: Sequence[object]) -> int:
    # Heuristic: find static temporal dimension in ONNX input.
    static_dims = []
    for d in input_shape:
        try:
            iv = int(d)
        except Exception:
            continue
        if iv and iv > 4096:
            static_dims.append(iv)
    if static_dims:
        return int(max(static_dims))
    return int(DEFAULT_WINDOW_S * SAMPLE_RATE)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    den = float(np.sum(ex))
    if den <= 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return ex / den


def moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size == 0:
        return x
    kernel = np.ones((n,), dtype=np.float32) / float(n)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def refine_onset_sample(
    audio: np.ndarray,
    center_sample: int,
    sr: int,
    lookback_s: float = 0.35,
    lookahead_s: float = 0.90,
) -> int:
    n = int(audio.size)
    if n <= 0:
        return max(0, center_sample)
    c = max(0, min(n - 1, int(center_sample)))
    s0 = max(0, int(c - max(0.0, lookback_s) * sr))
    s1 = min(n, int(c + max(0.0, lookahead_s) * sr))
    if s1 - s0 < int(0.05 * sr):
        return c

    seg = np.abs(audio[s0:s1]).astype(np.float32)
    if seg.size == 0:
        return c

    smooth_n = max(1, int(0.010 * sr))  # ~10ms smoothing
    env = moving_average(seg, smooth_n)
    if env.size == 0:
        return c

    pre_n = max(1, int(0.20 * env.size))
    noise = float(np.percentile(env[:pre_n], 25))
    peak = float(np.max(env))
    if peak <= 1e-8:
        return c

    thr = max(noise * 2.5, noise + (peak - noise) * 0.18, 1e-6)
    idx = np.where(env >= thr)[0]
    if idx.size == 0:
        return c

    i = int(idx[0])
    back = max(0, i - int(0.05 * sr))
    if i >= back:
        local = env[back : i + 1]
        if local.size > 0:
            i = back + int(np.argmin(local))

    return max(0, min(n - 1, s0 + i))


def run_session_scores(
    sess: ort.InferenceSession,
    input_name: str,
    output_name: str,
    wav: np.ndarray,
) -> Optional[np.ndarray]:
    # Try common shapes for audio models.
    candidates = [
        wav.reshape(1, -1).astype(np.float32),
        wav.reshape(1, -1, 1).astype(np.float32),
        wav.reshape(1, 1, -1).astype(np.float32),
        wav.reshape(1, 1, -1, 1).astype(np.float32),
    ]

    for inp in candidates:
        try:
            outputs = sess.run([output_name], {input_name: inp})
            if not outputs:
                continue
            arr = np.asarray(outputs[0])
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                logits = arr
            elif arr.ndim == 2:
                logits = arr[0]
            else:
                # Average all axes except last class axis.
                axes = tuple(range(arr.ndim - 1))
                logits = arr.mean(axis=axes)
            scores = np.asarray(logits, dtype=np.float32).reshape(-1)
            if scores.size > 0:
                # Keep model probabilities if already in [0,1].
                # Otherwise convert logits to ranked class probabilities.
                smin = float(np.min(scores))
                smax = float(np.max(scores))
                looks_prob = (smin >= -0.001) and (smax <= 1.001)
                if not looks_prob:
                    scores = softmax(scores)
                return scores
        except Exception:
            continue

    return None


def make_chunks(audio: np.ndarray, win_samples: int, hop_samples: int) -> List[Tuple[int, np.ndarray]]:
    out: List[Tuple[int, np.ndarray]] = []
    n = len(audio)
    if n <= win_samples:
        seg = np.zeros((win_samples,), dtype=np.float32)
        seg[:n] = audio
        out.append((0, seg))
        return out

    s = 0
    while s < n:
        e = s + win_samples
        if e > n:
            # Include one tail window aligned to end.
            s = max(0, n - win_samples)
            e = n
            seg = audio[s:e]
            out.append((s, seg.astype(np.float32)))
            break
        seg = audio[s:e]
        out.append((s, seg.astype(np.float32)))
        s += hop_samples

    return out


def label_name(lbl: LabelRow) -> str:
    c = sanitize(lbl.common)
    s = sanitize(lbl.scientific)
    if c and c.lower() != "none":
        return c
    return s or "Unknown"


def label_name_with_local_preference(lbl: LabelRow, preferred_common: Dict[str, str]) -> str:
    sci = sanitize(lbl.scientific)
    com = sanitize(lbl.common)
    ns = normalize_key(sci)
    nc = normalize_key(com)
    if ns and ns in preferred_common:
        return sanitize(preferred_common[ns])
    if nc and nc in preferred_common:
        return sanitize(preferred_common[nc])
    return label_name(lbl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--response", required=True)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    with open(args.request, "r", encoding="utf-8") as f:
        req = json.load(f)

    paths = req.get("paths", [])
    top_k = int(req.get("top_k", args.top_k) or args.top_k)
    top_k = max(3, min(100, top_k))
    engine = sanitize(req.get("engine", "perch_v2")).lower()
    if engine == "birdnet":
        engine = "birdnet"
    else:
        engine = "perch_v2"
    list_mode = sanitize(req.get("list_mode", "aves")).lower()
    strict_local = as_bool(req.get("strict_local", True), True)
    detections_per_segment = int(req.get("detections_per_segment", 1) or 1)
    detections_per_segment = max(1, min(10, detections_per_segment))
    name_top_n = int(req.get("name_top_n", 3) or 3)
    name_top_n = max(1, min(10, name_top_n))
    name_as_list = as_bool(req.get("name_as_list", False), False)
    marker_gap_s = float(req.get("marker_gap_s", MARKER_GAP_S) or MARKER_GAP_S)
    if marker_gap_s < 0.0:
        marker_gap_s = 0.0
    onset_snap = as_bool(req.get("onset_snap", True), True)
    onset_lookback_s = float(req.get("onset_lookback_s", 0.35) or 0.35)
    onset_lookahead_s = float(req.get("onset_lookahead_s", 0.90) or 0.90)
    onset_lookback_s = max(0.0, min(2.0, onset_lookback_s))
    onset_lookahead_s = max(0.0, min(3.0, onset_lookahead_s))
    # Backward compatible thresholds:
    # - threshold (legacy raw): used for both name and marker if specific values are absent.
    # - name_threshold / marker_threshold (raw 0..1): explicit raw thresholds.
    # - name_threshold_rel / marker_threshold_rel (0..100): relative thresholds.
    legacy_threshold = float(req.get("threshold", 0.45) or 0.45)
    if legacy_threshold > 1.0:
        legacy_threshold = legacy_threshold / 100.0
    legacy_threshold = max(0.0, min(1.0, legacy_threshold))

    name_threshold_raw = float(req.get("name_threshold", legacy_threshold) or legacy_threshold)
    if name_threshold_raw > 1.0:
        name_threshold_raw = name_threshold_raw / 100.0
    name_threshold_raw = max(0.0, min(1.0, name_threshold_raw))

    marker_threshold_raw = float(req.get("marker_threshold", legacy_threshold) or legacy_threshold)
    if marker_threshold_raw > 1.0:
        marker_threshold_raw = marker_threshold_raw / 100.0
    marker_threshold_raw = max(0.0, min(1.0, marker_threshold_raw))

    name_threshold_rel = req.get("name_threshold_rel", None)
    if name_threshold_rel is not None:
        try:
            name_threshold_rel = float(name_threshold_rel)
        except Exception:
            name_threshold_rel = None
    if name_threshold_rel is not None:
        name_threshold_rel = max(0.0, min(100.0, name_threshold_rel))

    marker_threshold_rel = req.get("marker_threshold_rel", None)
    if marker_threshold_rel is not None:
        try:
            marker_threshold_rel = float(marker_threshold_rel)
        except Exception:
            marker_threshold_rel = None
    if marker_threshold_rel is not None:
        marker_threshold_rel = max(0.0, min(100.0, marker_threshold_rel))
    latitude = sanitize(req.get("latitude", ""))
    longitude = sanitize(req.get("longitude", ""))
    birdnet_max_audio_min = req.get("birdnet_max_audio_min", 5.0)
    if birdnet_max_audio_min is not None:
        try:
            birdnet_max_audio_min = float(birdnet_max_audio_min)
        except Exception:
            birdnet_max_audio_min = 5.0
        if birdnet_max_audio_min <= 0:
            birdnet_max_audio_min = None

    default_local_birds_file = first_existing_file([DEFAULT_LOCAL_BIRDS_FILE] + FALLBACK_LOCAL_BIRDS_FILES)
    local_birds_file = sanitize(req.get("local_birds_file", default_local_birds_file))
    if not local_birds_file:
        local_birds_file = default_local_birds_file

    if engine == "birdnet":
        if not has_avx_support():
            print(
                "ERROR: BirdNET TensorFlow backend requires AVX CPU support on this machine. "
                "Select Perch v2 engine instead.",
                file=sys.stderr,
            )
            return 2

        _, _, local_preferred_common = load_local_birds(local_birds_file)
        custom_species: Optional[Set[str]] = None
        local_species_requested: Set[str] = set()
        list_mode_used = "aves"
        if list_mode == "local":
            local_species_requested = load_local_species_for_birdnet(local_birds_file)
            if not local_species_requested:
                if strict_local:
                    print(
                        f"ERROR: strict local mode enabled, but local birds file is missing/empty: {local_birds_file}",
                        file=sys.stderr,
                    )
                    return 2
                list_mode_used = "aves_fallback"
                custom_species = None
            else:
                list_mode_used = "local"
                custom_species = local_species_requested
        elif list_mode in {"mammalia", "insecta", "amphibia"}:
            list_mode_used = f"{list_mode}_fallback_aves"
            custom_species = None
            print(
                f"WARN: BirdNET is bird-only; list mode '{list_mode}' falls back to Aves.",
                file=sys.stderr,
            )
        elif list_mode == "all":
            list_mode_used = "all_birdnet_as_aves"
            custom_species = None
        else:
            list_mode_used = "aves"
            custom_species = None

        try:
            birdnet_api_kind, birdnet_model = load_birdnet_model()
        except Exception as ex:
            print("ERROR: BirdNET package/model unavailable.", file=sys.stderr)
            print("Install with: pip3.12 install birdnet", file=sys.stderr)
            print(f"DETAIL: {ex}", file=sys.stderr)
            return 2

        if list_mode == "local" and local_species_requested:
            mapped_species = map_local_species_to_birdnet(local_species_requested, birdnet_model)
            missing_count = max(0, len(local_species_requested) - len(mapped_species))
            if missing_count > 0:
                print(
                    f"WARN: {missing_count} local species not present in BirdNET v2.4 and were ignored.",
                    file=sys.stderr,
                )
            if not mapped_species:
                if strict_local:
                    print(
                        "ERROR: strict local mode enabled, but none of the local species exist in BirdNET v2.4.",
                        file=sys.stderr,
                    )
                    return 2
                list_mode_used = "aves_fallback"
                custom_species = None
            else:
                custom_species = mapped_species

        out_lines: List[str] = []
        for path in paths:
            src = str(path or "")
            if not src or not os.path.isfile(src):
                continue

            try:
                raw_pred = run_birdnet_predict(
                    birdnet_api_kind,
                    birdnet_model,
                    src,
                    custom_species,
                    top_k=top_k,
                    max_audio_duration_min=birdnet_max_audio_min,
                )
            except Exception as ex:
                print(f"WARN: BirdNET inference failed for {src}: {ex}", file=sys.stderr)
                continue

            events = extract_birdnet_events(raw_pred)
            if not events:
                continue
            events.sort(key=lambda x: (x[0], -x[3]))

            per_species: Dict[str, List[float]] = {}
            for _, _, species, conf in events:
                nm = sanitize(species)
                if not nm:
                    continue
                if nm not in per_species:
                    per_species[nm] = []
                per_species[nm].append(float(conf))
            if not per_species:
                continue

            scored_species: List[Tuple[str, float]] = []
            for species, vals in per_species.items():
                arr = np.asarray(vals, dtype=np.float32)
                sc = (0.35 * float(np.mean(arr))) + (0.65 * float(np.max(arr)))
                scored_species.append((species, sc))
            scored_species.sort(key=lambda kv: kv[1], reverse=True)
            scored_species = scored_species[: min(max(3, name_top_n), len(scored_species))]
            if not scored_species:
                continue

            top_species = [s for s, _ in scored_species]
            top_names = [birdnet_display_name(s, local_preferred_common) for s in top_species]
            top_scores = [float(sc) for _, sc in scored_species]

            top_raw = float(top_scores[0]) if top_scores else 0.0
            if name_threshold_rel is not None:
                name_raw_cutoff = top_raw * (name_threshold_rel / 100.0)
                name_mode = "relative"
            else:
                name_raw_cutoff = name_threshold_raw
                name_mode = "raw"

            if marker_threshold_rel is not None:
                marker_raw_cutoff = top_raw * (marker_threshold_rel / 100.0)
                marker_mode = "relative"
            else:
                marker_raw_cutoff = marker_threshold_raw
                marker_mode = "raw"

            kept_for_name = [nm for nm, sc in zip(top_names, top_scores) if sc >= name_raw_cutoff]
            if kept_for_name:
                if name_as_list and len(kept_for_name) > 1:
                    name = ", ".join(kept_for_name[:name_top_n])
                else:
                    name = kept_for_name[0]
            else:
                name = ""

            confidence = max(0.0, min(1.0, top_scores[0] if top_scores else 0.0))

            def fmt_top(i: int) -> str:
                if i >= len(top_names):
                    return ""
                return f"{top_names[i]} | Aves ({top_scores[i]:.3f})"

            def fmt_desc(i: int) -> str:
                if i >= len(top_species):
                    return ""
                return f"Scientific: {birdnet_species_scientific(top_species[i])}"

            audio_for_refine = None
            if onset_snap:
                audio_for_refine = load_audio(src, target_sr=SAMPLE_RATE, max_duration_s=900.0)

            marker_parts: List[str] = []
            prev_name = ""
            prev_t = -9999.0
            for st, _, species, conf in events:
                p = float(conf)
                if p < marker_raw_cutoff:
                    continue
                t = float(st)
                if onset_snap and audio_for_refine is not None:
                    marker_sample = refine_onset_sample(
                        audio_for_refine,
                        center_sample=int(t * SAMPLE_RATE),
                        sr=SAMPLE_RATE,
                        lookback_s=onset_lookback_s,
                        lookahead_s=onset_lookahead_s,
                    )
                    t = float(marker_sample) / float(SAMPLE_RATE)
                nm = birdnet_display_name(species, local_preferred_common)
                if nm == prev_name and (t - prev_t) < marker_gap_s:
                    continue
                prev_name = nm
                prev_t = t
                marker_parts.append(f"{t:.3f}@@{sanitize(nm)} ({p:.3f})")
            scene_markers = "||".join(marker_parts)

            fields = [
                src,
                sanitize(name),
                "Bioacoustic Detection",
                "Aves",
                "",
                f"{confidence:.4f}",
                sanitize(fmt_top(0)),
                sanitize(fmt_top(1)),
                sanitize(fmt_top(2)),
                sanitize(fmt_desc(0)),
                sanitize(fmt_desc(1)),
                sanitize(fmt_desc(2)),
                "birdnet",
                "",
                scene_markers,
                list_mode_used,
                f"{latitude},{longitude}" if (latitude or longitude) else "",
                name_mode,
                f"{name_raw_cutoff:.6f}",
                marker_mode,
                f"{marker_raw_cutoff:.6f}",
                f"{top_raw:.6f}",
                f"{name_threshold_rel:.3f}" if name_threshold_rel is not None else "",
                f"{marker_threshold_rel:.3f}" if marker_threshold_rel is not None else "",
            ]
            out_lines.append("\t".join(fields))

        with open(args.response, "w", encoding="utf-8") as f:
            for ln in out_lines:
                f.write(ln + "\n")
        return 0

    default_model_path = first_existing_file([DEFAULT_PERCH_MODEL_FILE] + FALLBACK_PERCH_MODEL_FILES)
    default_labels_path = first_existing_file([DEFAULT_PERCH_LABELS_FILE] + FALLBACK_PERCH_LABELS_FILES)
    model_path = sanitize(req.get("perch_model", default_model_path))
    labels_path = sanitize(req.get("perch_labels", default_labels_path))
    if not model_path:
        model_path = default_model_path
    if not labels_path:
        labels_path = default_labels_path

    if not os.path.isfile(model_path):
        print(f"ERROR: Perch model not found: {model_path}", file=sys.stderr)
        return 2
    if not os.path.isfile(labels_path):
        print(f"ERROR: Perch labels not found: {labels_path}", file=sys.stderr)
        return 2

    labels = load_labels(labels_path)
    if not labels:
        print("ERROR: labels file empty", file=sys.stderr)
        return 2

    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        print("ERROR: onnxruntime missing in sfx_clap_env", file=sys.stderr)
        return 2

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    inp = sess.get_inputs()[0]
    in_name = inp.name
    out_names = [o.name for o in sess.get_outputs()]
    label_out_name = "label" if "label" in out_names else (out_names[0] if out_names else "")
    if not label_out_name:
        print("ERROR: model has no usable output", file=sys.stderr)
        return 2
    win_samples = pick_window_samples(inp.shape)
    hop_samples = max(1, int(win_samples * DEFAULT_HOP_RATIO))

    if list_mode == "local":
        local_names_raw, local_names_norm, local_preferred_common = load_local_birds(local_birds_file)
    else:
        local_names_raw, local_names_norm, local_preferred_common = set(), set(), {}
    label_idx, list_mode_used = choose_label_indices(
        labels, list_mode, local_names_raw, local_names_norm, strict_local=strict_local
    )
    if list_mode == "local" and strict_local and not label_idx:
        if list_mode_used == "local_missing":
            print(
                f"ERROR: strict local mode enabled, but local birds file is missing or empty: {local_birds_file}",
                file=sys.stderr,
            )
        elif list_mode_used == "local_no_match":
            print(
                "ERROR: strict local mode enabled, but none of the local bird names matched labels.txt "
                f"(file: {local_birds_file})",
                file=sys.stderr,
            )
        else:
            print("ERROR: strict local mode enabled, no local labels available", file=sys.stderr)
        return 2
    if not label_idx:
        label_idx = list(range(len(labels)))
        list_mode_used = "all_fallback"

    out_lines: List[str] = []

    for path in paths:
        src = str(path or "")
        if not src or not os.path.isfile(src):
            continue

        audio = load_audio(src, target_sr=SAMPLE_RATE, max_duration_s=900.0)
        if audio is None:
            continue

        chunks = make_chunks(audio, win_samples=win_samples, hop_samples=hop_samples)
        if not chunks:
            continue

        probs_all: List[np.ndarray] = []
        marker_events: List[Tuple[float, str, float]] = []

        for start_idx, seg in chunks:
            scores_vec = run_session_scores(sess, in_name, label_out_name, seg)
            if scores_vec is None:
                continue

            # Align classes to labels count if model output is different.
            if scores_vec.size != len(labels):
                n = min(scores_vec.size, len(labels))
                if n <= 0:
                    continue
                scores_vec = scores_vec[:n]
            probs_all.append(scores_vec)

            # Marker candidates from selected label subset.
            active_idx = [i for i in label_idx if i < scores_vec.size]
            if not active_idx:
                continue
            sub = scores_vec[active_idx]
            kseg = min(detections_per_segment, int(sub.size))
            if kseg <= 0:
                continue
            order_seg = np.argpartition(-sub, kseg - 1)[:kseg]
            order_seg = order_seg[np.argsort(-sub[order_seg])]
            marker_sample = int(start_idx)
            if onset_snap:
                marker_sample = refine_onset_sample(
                    audio,
                    center_sample=start_idx,
                    sr=SAMPLE_RATE,
                    lookback_s=onset_lookback_s,
                    lookahead_s=onset_lookahead_s,
                )
            t = float(marker_sample) / float(SAMPLE_RATE)
            for oi in order_seg:
                gi = int(active_idx[int(oi)])
                p = float(sub[int(oi)])
                nm = label_name_with_local_preference(labels[gi], local_preferred_common)
                marker_events.append((t, nm, p))

        if not probs_all:
            continue

        mat = np.stack(probs_all, axis=0)
        mean_probs = np.mean(mat, axis=0)
        max_probs = np.max(mat, axis=0)
        score = (0.35 * mean_probs) + (0.65 * max_probs)

        active_idx = [i for i in label_idx if i < score.size]
        if not active_idx:
            continue

        subset_scores = np.asarray([score[i] for i in active_idx], dtype=np.float32)
        order = np.argsort(-subset_scores)
        order = order[: min(max(3, name_top_n), len(order))]
        top_global = [int(active_idx[int(i)]) for i in order]

        top_names = [label_name_with_local_preference(labels[i], local_preferred_common) for i in top_global]
        top_scores = [float(score[i]) for i in top_global]

        top_group = ""
        if top_global:
            top_group = sanitize(labels[top_global[0]].group)

        top_raw = float(top_scores[0]) if top_scores else 0.0
        if name_threshold_rel is not None:
            name_raw_cutoff = top_raw * (name_threshold_rel / 100.0)
            name_mode = "relative"
        else:
            name_raw_cutoff = name_threshold_raw
            name_mode = "raw"

        if marker_threshold_rel is not None:
            marker_raw_cutoff = top_raw * (marker_threshold_rel / 100.0)
            marker_mode = "relative"
        else:
            marker_raw_cutoff = marker_threshold_raw
            marker_mode = "raw"

        kept_for_name = [nm for nm, sc in zip(top_names, top_scores) if sc >= name_raw_cutoff]
        if kept_for_name:
            if name_as_list and len(kept_for_name) > 1:
                name = ", ".join(kept_for_name[:name_top_n])
            else:
                name = kept_for_name[0]
        else:
            name = ""

        confidence = max(0.0, min(1.0, top_scores[0] if top_scores else 0.0))

        def fmt_top(i: int) -> str:
            if i >= len(top_names):
                return ""
            grp = sanitize(labels[top_global[i]].group)
            return f"{top_names[i]} | {grp or 'Bioacoustic'} ({top_scores[i]:.3f})"

        def fmt_desc(i: int) -> str:
            if i >= len(top_names):
                return ""
            lbl = labels[top_global[i]]
            return f"Scientific: {sanitize(lbl.scientific)}"

        # De-duplicate marker events.
        marker_parts: List[str] = []
        prev_name = ""
        prev_t = -9999.0
        for t, nm, p in marker_events:
            if p < marker_raw_cutoff:
                continue
            if nm == prev_name and (t - prev_t) < marker_gap_s:
                continue
            prev_name = nm
            prev_t = t
            marker_parts.append(f"{t:.3f}@@{sanitize(nm)} ({p:.3f})")
        scene_markers = "||".join(marker_parts)

        fields = [
            src,
            sanitize(name),
            "Bioacoustic Detection",
            top_group or "Bioacoustic",
            "",
            f"{confidence:.4f}",
            sanitize(fmt_top(0)),
            sanitize(fmt_top(1)),
            sanitize(fmt_top(2)),
            sanitize(fmt_desc(0)),
            sanitize(fmt_desc(1)),
            sanitize(fmt_desc(2)),
            "perch_v2",
            "",
            scene_markers,
            list_mode_used,
            f"{latitude},{longitude}" if (latitude or longitude) else "",
            name_mode,
            f"{name_raw_cutoff:.6f}",
            marker_mode,
            f"{marker_raw_cutoff:.6f}",
            f"{top_raw:.6f}",
            f"{name_threshold_rel:.3f}" if name_threshold_rel is not None else "",
            f"{marker_threshold_rel:.3f}" if marker_threshold_rel is not None else "",
        ]
        out_lines.append("\t".join(fields))

    with open(args.response, "w", encoding="utf-8") as f:
        for ln in out_lines:
            f.write(ln + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
