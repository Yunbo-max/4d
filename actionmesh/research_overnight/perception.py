"""Optional 4D-Bench native QA: concat input versus separately identified views."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile
import time

from .engine import atomic_json, digest, valid_receipt, write_receipt
from .protocol import verify_records


def native_indices(total_frames: int) -> list[int]:
    if total_frames < 6:
        raise ValueError('Require at least six frames in each native view')
    interval = total_frames // 6
    return [i * interval for i in range(6)]


def public_question(row: dict) -> tuple[dict, int, str]:
    public = dict(row)
    label, category = public.pop('Answer index'), public.pop('Category')
    if not isinstance(label, int) or label not in (1, 2, 3, 4):
        raise ValueError('Native answer index must be an integer in [1,4]')
    return public, label, str(category)


def extract_answer_option(text):
    matches = re.findall(r'\(([A-D])\)', text)
    if matches:
        return matches[0]
    matches = re.findall(r'(?:^|[\s\(\.,;:])([A-D])(?:[\s\)\.,;:]|$)', text)
    return matches[0] if matches else None


def native_score(pred_answer, correct_answer_index):
    if pred_answer is None or not isinstance(pred_answer, str):
        return -2, -1
    option = extract_answer_option(pred_answer)
    if not option:
        return -1, -1
    index = {'A': 1, 'B': 2, 'C': 3, 'D': 4}[option]
    return (1 if index == correct_answer_index else 0), index


def native_prompt(public: dict) -> str:
    return f"You are a excellent video analyst. I provide you 18 frames with every six images uniformly sampled from one video, each video captured from a different angle and a question and four choices.             Carefully watch the provided videos and pay attention to every detail. Based on your observations, select the best option that accurately addresses the question. Here is the question and choices: \n {public}.\n             You must return only the option identifier (e.g., '(A)') without any additional text, do not add any additional analysis, just return the correct option identifier."


def worker(protocol_path: Path, question_id: str, unit: Path, signature: str):
    import cv2
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    protocol = json.loads(protocol_path.read_text())
    qa = protocol['perception']
    if question_id not in qa['question_ids']:
        raise ValueError('Question absent from frozen native inventory')
    verify_records(protocol['runner_sources'])
    verify_records(qa['input_records'])
    verify_records(qa['model_records'])
    rows = json.loads(Path(qa['qa_json']).read_text())
    row = rows[question_id]
    public, label, category = public_question(row)
    uid = question_id.split('_')[0]
    unit.mkdir(parents=True, exist_ok=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(qa['model_path'], torch_dtype=torch.float16,
                device_map={'': 0}, attn_implementation='sdpa', local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(qa['model_path'], local_files_only=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix='4dbench-', dir=unit) as temp:
        view_paths, frame_records = [], []
        for view in (1, 8, 16):
            path = Path(qa['video_root']) / uid / f'view_{view}_rgb_white_bg.mp4'
            cap = cv2.VideoCapture(str(path))
            try:
                indices = native_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                paths = []
                for i, index in enumerate(indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'Native frame read failed: {path} frame {index}')
                    jpg = Path(temp) / f'view_{view}_frame_{i}.jpg'
                    if not cv2.imwrite(str(jpg), frame):
                        raise RuntimeError('Frame serialization failed')
                    paths.append('file://' + str(jpg))
                    frame_records.append({'view': view, 'native_frame_index': index,
                                          'jpg_sha256': digest(jpg), 'source_video_sha256': qa['input_records'][str(path.resolve())]['sha256']})
                view_paths.append(paths)
            finally:
                cap.release()
        for arm in ('official_concat', 'separate_views'):
            result_path = unit / f'{arm}.json'
            arm_receipt = unit / f'{arm}.receipt.json'
            if arm_receipt.exists():
                if not valid_receipt(arm_receipt, signature + ':' + arm):
                    raise ValueError('Partial QA output changed')
                results[arm] = json.loads(result_path.read_text())
                continue
            content = []
            if arm == 'official_concat':
                content.append({'type': 'video', 'video': sum(view_paths, [])})
            else:
                for view, paths in zip((1, 8, 16), view_paths):
                    content += [{'type': 'text', 'text': f'View {view}, six frames in native temporal order.'},
                                {'type': 'video', 'video': paths}]
            content.append({'type': 'text', 'text': native_prompt(public)})
            messages = [{'role': 'user', 'content': content}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize()
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            correctness, choice = native_score(raw, label)
            result = {'question_id': question_id, 'uid': uid, 'category': category, 'arm': arm,
                      'raw_answer': raw, 'correctness': correctness, 'answer': choice,
                      'input_tokens': int(inputs.input_ids.shape[-1]), 'output_tokens': int(trimmed[0].numel()),
                      'forward_calls': 1, 'elapsed_seconds': time.monotonic() - started,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                      'runtime': {'dtype': 'float16', 'attention': 'sdpa', 'quantized': False,
                                  'max_new_tokens': 128, 'do_sample': False}, 'frames': frame_records}
            atomic_json(result_path, result)
            write_receipt(arm_receipt, signature + ':' + arm, [result_path])
            results[arm] = result
            del generated, trimmed, inputs, images, videos
            torch.cuda.empty_cache()
    verify_records(qa['input_records'])
    verify_records(qa['model_records'])
    verify_records(protocol['runner_sources'])
    atomic_json(unit / 'pair.json', {'question_id': question_id, 'uid': uid, 'category': category,
                                   'arms': results, 'candidate_method_tested': False, 'formal_gate_pass': False})
    write_receipt(unit / 'receipt.json', signature,
                  [unit / name for name in ('pair.json', 'official_concat.json', 'separate_views.json',
                                            'official_concat.receipt.json', 'separate_views.receipt.json')], family='perception')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--question-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fingerprint', required=True)
    args = parser.parse_args()
    worker(args.protocol.resolve(), args.question_id, args.output.resolve(), args.fingerprint)

if __name__ == '__main__':
    main()
