#!/usr/bin/env python3
"""
RAG Document Extraction with Surya - Optimized for large-scale processing.

Usage:
  python3 text_parser_surya.py /path/to/pdfs -o output.json
  python3 text_parser_surya.py /path/to/pdfs -o output.json --resume
"""
import os
import sys
import gc
import argparse
from pathlib import Path
from typing import List, Dict
from contextlib import contextmanager

import torch
import orjson
from PIL import Image
from tqdm import tqdm
from pdf2image import convert_from_path

if not torch.cuda.is_available():
    print("CUDA required")
    sys.exit(1)

os.environ["TORCH_DEVICE"] = "cuda"
os.environ["DETECTOR_BATCH_SIZE"] = "18"
os.environ["RECOGNITION_BATCH_SIZE"] = "128"
os.environ["LAYOUT_BATCH_SIZE"] = "20"
os.environ["TABLE_REC_BATCH_SIZE"] = "64"

import logging
logging.getLogger('surya').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

from surya.foundation import FoundationPredictor
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor
from surya.layout import LayoutPredictor
from surya.table_rec import TableRecPredictor
from surya.settings import settings

SKIP_TYPES = {"PageFooter", "PageHeader", "Picture"}
PAGE_BATCH = 4
PDF_BATCH = 10


@contextmanager
def gpu_cleanup():
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def is_metadata_table(elem: dict) -> bool:
    if elem.get("type") != "Table" or elem.get("pos") != 0:
        return False
    info = elem.get("info")
    if not info or not info.get("cells"):
        return False
    has_merged = any(c.get("rowspan", 1) > 1 or c.get("colspan", 1) > 1 for c in info["cells"])
    return has_merged and 2 <= info.get("rows", 0) <= 5


def filter_elements(pages: list) -> list:
    return [
        {"page_number": p["page_number"],
         "elements": [e for e in p["elements"] if e["type"] not in SKIP_TYPES and not is_metadata_table(e)]}
        for p in pages
    ]


class SuryaExtractor:
    def __init__(self):
        print("Loading models...")
        self.foundation = FoundationPredictor(device="cuda")
        self.det = DetectionPredictor()
        self.rec = RecognitionPredictor(self.foundation)
        self.layout_foundation = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
        self.layout = LayoutPredictor(self.layout_foundation)
        self.table = TableRecPredictor()
        print("Models loaded")

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        with gpu_cleanup():
            images = convert_from_path(pdf_path, size=(2048, None))
            all_pages = []
            
            for batch_start in range(0, len(images), PAGE_BATCH):
                batch_end = min(batch_start + PAGE_BATCH, len(images))
                batch_images = images[batch_start:batch_end]
                batch_pages = self._process_batch(batch_images, batch_start)
                all_pages.extend(batch_pages)
                del batch_images
                gc.collect()
            
            del images
            return filter_elements(all_pages)

    def _process_batch(self, images: List[Image.Image], start_idx: int) -> List[Dict]:
        layouts = self.layout(images)
        pages = []
        
        for i, (img, layout) in enumerate(zip(images, layouts)):
            page_num = start_idx + i + 1
            elements = []
            table_crops, table_indices = [], []
            
            for box in layout.bboxes:
                elem = {"pos": box.position, "type": box.label, "bbox": list(box.bbox)}
                if box.label == "Table":
                    table_indices.append(len(elements))
                    table_crops.append(img.crop(tuple(map(int, box.bbox))))
                    elem["info"] = None
                else:
                    elem["text"] = None
                elements.append(elem)
            
            if table_crops:
                for idx, tr in zip(table_indices, self.table(table_crops)):
                    if tr.cells:
                        elements[idx]["info"] = {
                            "rows": max(c.row_id for c in tr.cells) + 1,
                            "cols": max(c.col_id for c in tr.cells) + 1,
                            "cells": [{"row": c.row_id, "col": c.col_id, "rowspan": c.rowspan,
                                       "colspan": c.colspan, "is_header": c.is_header,
                                       "bbox": list(c.bbox), "text": None} for c in tr.cells]
                        }
                del table_crops
            
            crops = [img.crop(tuple(map(int, e["bbox"]))) for e in elements]
            if crops:
                ocr_results = self.rec(crops, det_predictor=self.det)
                for elem, ocr in zip(elements, ocr_results):
                    lines = self._sort_lines(ocr.text_lines)
                    if elem["type"] == "Table" and elem.get("info"):
                        self._fill_table_cells(elem["info"]["cells"], lines)
                    else:
                        elem["text"] = " ".join(l.text for l in lines)
                del crops, ocr_results
            
            pages.append({"page_number": page_num, "elements": elements})
        
        del layouts
        return pages

    def _sort_lines(self, lines):
        if not lines:
            return []
        sorted_y = sorted(lines, key=lambda l: (l.bbox[1] + l.bbox[3]) / 2)
        rows, threshold = [], 15
        for line in sorted_y:
            y = (line.bbox[1] + line.bbox[3]) / 2
            if rows and abs(y - rows[-1][0]) < threshold:
                rows[-1][1].append(line)
            else:
                rows.append([y, [line]])
        return [l for _, r in rows for l in sorted(r, key=lambda x: x.bbox[0])]

    def _fill_table_cells(self, cells, lines):
        for cell in cells:
            cx0, cy0, cx1, cy1 = cell["bbox"]
            cell["text"] = " ".join(
                l.text for l in lines
                if cx0 <= (l.bbox[0] + l.bbox[2]) / 2 <= cx1 and cy0 <= (l.bbox[1] + l.bbox[3]) / 2 <= cy1
            )


def get_processed(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    with open(output_path, 'rb') as f:
        return {d["file"] for d in orjson.loads(f.read())}


def save(results: list, path: str):
    with open(path, 'wb') as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('-o', '--output', default='output.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    pdf_files = sorted(Path(args.input).rglob('*.pdf'))
    print(f"Found {len(pdf_files)} PDFs")

    processed = get_processed(args.output) if args.resume else set()
    if processed:
        print(f"Resuming: {len(processed)} done")
    
    results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'rb') as f:
            results = orjson.loads(f.read())

    extractor = SuryaExtractor()
    base = Path(args.input).resolve()

    for idx, pdf in enumerate(tqdm(pdf_files, desc="Processing")):
        rel = str(pdf.relative_to(base))
        if rel in processed:
            continue

        try:
            pages = extractor.process_pdf(str(pdf))
            results.append({"file": rel, "pages": pages})
            
            if (idx + 1) % PDF_BATCH == 0:
                save(results, args.output)
                with gpu_cleanup():
                    pass
                    
        except Exception as e:
            print(f"\nError {pdf}: {e}")
            save(results, args.output)
            continue

    save(results, args.output)
    print(f"\nDone: {len(results)} docs -> {args.output}")


if __name__ == '__main__':
    main()