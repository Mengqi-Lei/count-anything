import json
import logging
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image as PILImage


def _iter_records(raw) -> Iterable[Tuple[str, Dict]]:
    if isinstance(raw, dict):
        return raw.items()
    if isinstance(raw, list):
        return [(str(i), record) for i, record in enumerate(raw)]
    raise ValueError(f"Unsupported unified json top-level type: {type(raw)}")


def _normalize_path_key(value: str) -> str:
    return str(value).strip().replace("\\", "/")


def _load_split_keys(split_file: Optional[str]) -> Optional[List[str]]:
    if split_file is None:
        return None
    split_path = Path(split_file)
    split_keys = [_normalize_path_key(line) for line in split_path.read_text().splitlines()]
    split_keys = [key for key in split_keys if key]
    return split_keys


def _iter_relative_suffix_keys(path_str: str) -> Iterable[str]:
    normalized_path = _normalize_path_key(path_str)
    path_obj = PurePosixPath(normalized_path)
    path_parts = list(path_obj.parts)
    if path_parts and path_parts[0] == "/":
        path_parts = path_parts[1:]
    for i in range(len(path_parts)):
        yield "/".join(path_parts[i:])


def _safe_int(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _infer_dataset_root(annotation_file: str) -> Path:
    annotation_path = Path(annotation_file).expanduser()
    if not annotation_path.is_absolute():
        annotation_path = annotation_path.resolve()
    if annotation_path.parent.name == "annotations":
        return annotation_path.parent.parent
    return annotation_path.parent


def _resolve_image_path_for_io(image_path: str, dataset_root: Path) -> str:
    path = Path(image_path).expanduser()
    if path.is_absolute():
        return str(path)
    return str(dataset_root / _normalize_path_key(image_path))


class UNIFIED_JSON_FROM_FILE:
    """
    Unified JSON training API for CountAnything-style counting datasets.

    Expected per-sample fields:
      - image_path: absolute path, or path relative to the dataset root
      - annotation[class_name].point: mandatory, absolute pixel coordinates
      - annotation[class_name].bbox: optional, xyxy absolute coordinates
    """

    def __init__(
        self,
        annotation_file,
        split_file=None,
        split_name=None,
        train_ratio=0.8,
        split_seed=123,
        class_name="cars",
        prompt_text=None,
    ):
        self.annotation_file = str(annotation_file)
        self.dataset_root = _infer_dataset_root(self.annotation_file)
        self.split_file = split_file
        self.split_name = split_name
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        ## 新添代码
        # 步骤一：CountAnything 最初的 unified loader 假设整份 JSON 只有一个固定 class_name。
        # 步骤二：像 FSC147 这类数据集，每张图只有一个类，但整个 JSON 会包含很多不同类名。
        # 步骤三：这里增加一个“按样本自动取唯一类名”的模式，避免为了 smoke test 再复制一份 loader。
        class_name_text = None if class_name is None else str(class_name).strip()
        prompt_text_value = None if prompt_text is None else str(prompt_text).strip()
        self.auto_class_name = class_name_text in {None, "", "__auto__", "auto", "dynamic"}
        self.auto_prompt_text = prompt_text_value in {None, "", "__auto__", "auto", "dynamic"}
        self.class_name = None if self.auto_class_name else class_name_text
        self.prompt_text = None if self.auto_prompt_text else prompt_text_value
        ## 新添代码

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        split_keys = _load_split_keys(split_file)

        all_records = []
        records_by_path = {}
        records_by_suffix = {}
        records_by_stem = {}

        for local_idx, (record_key, record) in enumerate(_iter_records(raw)):
            if not isinstance(record, dict):
                continue

            image_path = record.get("image_path", "")
            if not image_path:
                continue

            image_path = str(image_path)
            normalized_image_path = _normalize_path_key(image_path)
            image_path_for_io = _resolve_image_path_for_io(image_path, self.dataset_root)
            image_stem = Path(image_path).stem

            annotation = record.get("annotation", {})
            selected_class_name = self._resolve_class_name(
                record_key=record_key,
                record=record,
                annotation=annotation,
            )
            cls_annotation = annotation[selected_class_name]
            points = cls_annotation.get("point", None)
            if points is None:
                raise ValueError(
                    f"Sample {record_key} is missing mandatory point annotations for class_name={selected_class_name!r}"
                )

            boxes = cls_annotation.get("bbox", [])
            if boxes is None:
                boxes = []

            ## 新添代码
            # 有些 unified JSON（例如 FSC147）会把“无 bbox”写成与 point 等长的空列表数组：
            #   bbox = [[], [], ...]
            # 这种语义仍然应该视为 point-only，而不是把每个 [] 当成一条非法框。
            if len(boxes) > 0 and all(
                (box is None) or (isinstance(box, (list, tuple)) and len(box) == 0)
                for box in boxes
            ):
                boxes = []
            ## 新添代码

            if len(boxes) not in (0, len(points)):
                raise ValueError(
                    "Unified counting contract requires bbox to be either empty "
                    f"or aligned with point. Got {len(points)} points and {len(boxes)} boxes "
                    f"for sample {record_key}."
                )

            # CLOC JSON stores image_path relative to the dataset root, while older
            # local experiments sometimes used absolute paths. Resolve once here so
            # downstream SAM3 code always receives an directly openable path.
            with PILImage.open(image_path_for_io) as image:
                width, height = image.size

            record_data = {
                "record_key": str(record_key),
                "sample_id": _safe_int(record.get("idx"), local_idx),
                "image_path": image_path_for_io,
                "image_path_key": normalized_image_path,
                "stem": image_stem,
                "width": int(width),
                "height": int(height),
                "class_name": selected_class_name,
                "points": points,
                "boxes": boxes,
                "image_from": record.get("image_from", ""),
            }
            all_records.append(record_data)
            records_by_path.setdefault(normalized_image_path, []).append(record_data)
            for suffix_key in _iter_relative_suffix_keys(normalized_image_path):
                records_by_suffix.setdefault(suffix_key, []).append(record_data)
            records_by_stem.setdefault(image_stem, []).append(record_data)

        if split_keys is not None:
            resolved_records = []
            selected_record_keys = set()
            missing_keys = []
            stem_fallback_hits = []
            for split_key in split_keys:
                normalized_split_key = _normalize_path_key(split_key)
                matched_records = records_by_path.get(normalized_split_key, [])
                if not matched_records:
                    suffix_matches = records_by_suffix.get(normalized_split_key, [])
                    suffix_paths = {match["image_path_key"] for match in suffix_matches}
                    if len(suffix_paths) == 1:
                        matched_records = suffix_matches
                    elif len(suffix_paths) > 1:
                        raise ValueError(
                            "Unified JSON split key is ambiguous under path-based matching. "
                            f"split_file={split_file} key={split_key!r} matches="
                            f"{[match['image_path'] for match in suffix_matches[:5]]}"
                        )
                if not matched_records and "/" not in normalized_split_key:
                    stem_key = Path(normalized_split_key).stem
                    stem_matches = records_by_stem.get(stem_key, [])
                    stem_paths = {match["image_path_key"] for match in stem_matches}
                    if len(stem_paths) == 1:
                        matched_records = stem_matches
                        stem_fallback_hits.append(stem_key)
                    elif len(stem_paths) > 1:
                        raise ValueError(
                            "Unified JSON split key is ambiguous under legacy stem matching. "
                            "Please switch split_file entries to relative or absolute image paths. "
                            f"split_file={split_file} key={split_key!r} matches="
                            f"{[match['image_path'] for match in stem_matches[:5]]}"
                        )
                if not matched_records:
                    missing_keys.append(split_key)
                else:
                    for matched_record in matched_records:
                        if matched_record["record_key"] in selected_record_keys:
                            continue
                        selected_record_keys.add(matched_record["record_key"])
                        resolved_records.append(matched_record)

            if stem_fallback_hits:
                logging.warning(
                    "Unified JSON split_file is still using legacy stem entries for some samples. "
                    "Please prefer relative/absolute image paths. split_file=%s stem_fallback_hits=%d sample=%s",
                    split_file,
                    len(stem_fallback_hits),
                    stem_fallback_hits[:10],
                )
            if missing_keys:
                logging.warning(
                    "Unified JSON split does not fully overlap with annotation_file. "
                    "Using split/json intersection only. split_file=%s missing=%d sample=%s",
                    split_file,
                    len(missing_keys),
                    missing_keys[:10],
                )
            self._records = resolved_records
        elif self.split_name is not None:
            import random

            ## 新添代码
            # 步骤一：当没有显式 split_file 时，允许直接基于 unified JSON 做确定性随机划分。
            # 步骤二：train 取前 `train_ratio` 部分，val/test 取剩余部分，保证多次运行结果稳定。
            # 步骤三：这样 Stage 7 就不再依赖外部官方 split 文件，可以直接拿统一 JSON 做 smoke test。
            all_image_keys = sorted(records_by_path.keys())
            local_random = random.Random(self.split_seed)
            local_random.shuffle(all_image_keys)
            train_count = int(len(all_image_keys) * self.train_ratio)
            if len(all_image_keys) > 0:
                if 0.0 < self.train_ratio < 1.0:
                    train_count = max(1, train_count)
                    if len(all_image_keys) > 1:
                        train_count = min(len(all_image_keys) - 1, train_count)
                else:
                    train_count = max(0, min(len(all_image_keys), train_count))
            train_keys = set(all_image_keys[:train_count])
            val_keys = set(all_image_keys[train_count:])
            split_name = str(self.split_name).lower()
            if split_name == "train":
                selected_keys = all_image_keys[:train_count]
            elif split_name in {"val", "test"}:
                selected_keys = [image_key for image_key in all_image_keys if image_key in val_keys]
            else:
                raise ValueError(
                    f"Unsupported split_name={self.split_name!r}. Expected one of: train, val, test."
                )
            self._records = []
            for image_key in selected_keys:
                self._records.extend(records_by_path[image_key])
            logging.info(
                "Unified JSON random split enabled. split_name=%s train_ratio=%.3f split_seed=%d "
                "train_count=%d val_count=%d",
                split_name,
                self.train_ratio,
                self.split_seed,
                len(train_keys),
                len(val_keys),
            )
            ## 新添代码
        else:
            self._records = list(all_records)

        if len(self._records) == 0:
            raise ValueError(
                "Unified JSON loader found 0 usable samples. "
                f"annotation_file={self.annotation_file}, split_file={self.split_file}"
            )

        self._cat_idx_to_text = {
            1: (self.prompt_text if self.prompt_text is not None else "__dynamic_prompt__")
        }

    def _resolve_class_name(self, record_key: str, record: Dict, annotation: Dict) -> str:
        if not isinstance(annotation, dict) or len(annotation) == 0:
            raise ValueError(f"Sample {record_key} has empty annotation dict")

        if not self.auto_class_name:
            if self.class_name not in annotation:
                raise KeyError(
                    f"Sample {record_key} does not contain annotation for class_name={self.class_name!r}"
                )
            return self.class_name

        if len(annotation) == 1:
            return next(iter(annotation.keys()))

        ## 新添代码
        # 步骤一：auto class 模式下，优先尝试读取样本自带的 classes 字段。
        # 步骤二：只有当 classes 里恰好有一个候选，并且它真的存在于 annotation 里时才自动选中。
        # 步骤三：否则直接报错，避免在多类样本里静默选错类。
        sample_classes = record.get("classes", [])
        if isinstance(sample_classes, list):
            valid_classes = [str(x) for x in sample_classes if str(x) in annotation]
            if len(valid_classes) == 1:
                return valid_classes[0]
        ## 新添代码

        raise ValueError(
            "Unified JSON auto class mode requires each sample to expose exactly one usable class. "
            f"record_key={record_key}, annotation_keys={sorted(annotation.keys())}, "
            f"classes={sample_classes}"
        )

    def getDatapointIds(self):
        return list(range(len(self._records)))

    def _xyxy_to_normalized_xywh(
        self, box_xyxy: List[float], image_width: int, image_height: int
    ) -> Tuple[List[float], float]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy]
        x1 = min(max(x1, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        x2 = min(max(x2, 0.0), float(image_width))
        y2 = min(max(y2, 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid xyxy bbox after clamping: {box_xyxy}")
        box_w = x2 - x1
        box_h = y2 - y1
        normalized_box = [
            x1 / float(image_width),
            y1 / float(image_height),
            box_w / float(image_width),
            box_h / float(image_height),
        ]
        normalized_area = normalized_box[2] * normalized_box[3]
        return normalized_box, float(normalized_area)

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        record = self._records[idx]
        width = record["width"]
        height = record["height"]
        class_name = record["class_name"]
        points = record["points"]
        boxes = record["boxes"]

        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }

        annotations = []
        for ann_id, point in enumerate(points):
            annotation = {
                "image_id": 0,
                "bbox": None,
                "area": 0.0,
                "segmentation": None,
                "object_id": ann_id,
                "is_crowd": False,
                "id": ann_id,
                "point": [float(point[0]), float(point[1])],
                "source": record["image_from"],
            }
            if len(boxes) > 0:
                normalized_box, normalized_area = self._xyxy_to_normalized_xywh(
                    boxes[ann_id],
                    image_width=width,
                    image_height=height,
                )
                annotation["bbox"] = normalized_box
                annotation["area"] = normalized_area
            annotations.append(annotation)

        query = query_template.copy()
        query["id"] = 0
        query["original_cat_id"] = 1
        query["query_text"] = self.prompt_text if self.prompt_text is not None else class_name
        query["object_ids_output"] = [ann["id"] for ann in annotations]

        return [query], annotations

    def loadImagesFromDatapoint(self, idx):
        record = self._records[idx]
        return [
            {
                "id": 0,
                "file_name": record["image_path"],
                "original_img_id": record["sample_id"],
                "coco_img_id": record["sample_id"],
            }
        ]
