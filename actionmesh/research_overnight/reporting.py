"""Descriptive reports preserve the complete frozen denominator and raw evidence."""
from __future__ import annotations

import json
from pathlib import Path
import random
import statistics

from .engine import atomic_json, valid_receipt, utc


def paired_summary(values: list[float]) -> dict:
    if not values:
        return {'n_assets': 0, 'mean': None, 'bootstrap_95_ci': None}
    result = {'n_assets': len(values), 'mean': statistics.fmean(values), 'bootstrap_95_ci': None,
              'scope': 'conditional on this development cohort and inference seed; no method verdict'}
    if len(values) >= 8:
        rng = random.Random(271828)
        means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(4000))
        result['bootstrap_95_ci'] = [means[100], means[3899]]
    return result


def collect(output: Path) -> dict:
    protocol = json.loads((output / 'protocol.json').read_text())
    queue = json.loads((output / 'queue.json').read_text()) if (output / 'queue.json').exists() else {'status': 'not_started', 'units': {}}
    signature = json.loads((output / 'window.json').read_text())['fingerprint'] if (output / 'window.json').exists() else None
    report = {'saved_utc': utc().isoformat(), 'queue_status': queue['status'],
              'scientific_gate_pass': False, 'candidate_methods_tested': False,
              'scope': 'bounded baseline qualification/development census',
              'remaining_obligations': ['natural failure interpretation', 'strong simple/published alternatives',
                    'functional collision audit', 'IPCG', 'prospective Gate A', 'independent confirmation']}
    generation = protocol.get('generation') or {}
    uids = generation.get('cohort', {}).get('uids', [])
    complete, pending = [], []
    for uid in uids:
        directory = output / 'units' / ('gen-' + uid)
        entry = queue['units'].get('gen-' + uid, {'status': 'queued'})
        if entry['status'] == 'complete' and signature and valid_receipt(directory / 'receipt.json', signature):
            complete.append(json.loads((directory / 'pair.json').read_text()))
        else:
            pending.append({'uid': uid, 'status': entry['status'], 'attempts': entry.get('attempts', [])})
    report['generation'] = {'declared_assets': len(uids), 'complete_pairs': len(complete),
                            'pending_or_failed': pending, 'pairs': complete,
                            'native_minus_stationary': {k: paired_summary([p['native_minus_stationary'][k] for p in complete])
                                for k in ('cd_3d', 'cd_4d', 'cd_motion')},
                            'inspection_by_native_cd_motion': [p['uid'] for p in sorted(complete, key=lambda p: p['native']['cd_motion'])],
                            'failure_labels_inferred': False}
    qa = protocol.get('perception')
    if qa:
        pairs, missing = [], []
        for qid in qa['question_ids']:
            unit_id = 'qa-' + __import__('hashlib').sha256(qid.encode()).hexdigest()[:16]
            directory = output / 'units' / unit_id
            entry = queue['units'].get(unit_id, {'status': 'queued'})
            if entry['status'] == 'complete' and signature and valid_receipt(directory / 'receipt.json', signature):
                pairs.append(json.loads((directory / 'pair.json').read_text()))
            else:
                missing.append({'question_id': qid, 'status': entry['status']})
        categories = qa['categories']
        scores = {}
        for arm in ('official_concat', 'separate_views'):
            per_category = {}
            for category in categories:
                entries = [p['arms'][arm] for p in pairs if p['category'] == category]
                per_category[category] = {'n_scored': len(entries),
                    'accuracy': sum(p['correctness'] == 1 for p in entries) / len(entries) if entries else None,
                    'invalid_answers': sum(p['correctness'] < 0 for p in entries)}
            values = [v['accuracy'] for v in per_category.values() if v['accuracy'] is not None]
            scores[arm] = {'categories': per_category,
                          'native_macro_accuracy': statistics.fmean(values) if len(values) == len(categories) else None,
                          'macro_over_present_categories': statistics.fmean(values) if values else None,
                          'micro_accuracy': sum(p['arms'][arm]['correctness'] == 1 for p in pairs) / len(pairs) if pairs else None,
                          'selected_denominator_accuracy_lower_bound': sum(p['arms'][arm]['correctness'] == 1 for p in pairs) / len(qa['question_ids'])}
        selected_by_object = {}
        for qid in qa['question_ids']:
            selected_by_object.setdefault(qid.split('_')[0], set()).add(qid)
        completed_ids = {p['question_id'] for p in pairs}
        complete_objects = [uid for uid, ids in selected_by_object.items() if ids <= completed_ids]
        partial_objects = [uid for uid, ids in selected_by_object.items() if ids & completed_ids and not ids <= completed_ids]
        object_scores = {arm: {uid: statistics.fmean(p['arms'][arm]['correctness'] == 1 for p in pairs if p['uid'] == uid)
                               for uid in complete_objects} for arm in ('official_concat', 'separate_views')}
        report['perception'] = {'declared_questions': len(qa['question_ids']), 'complete_pairs': len(pairs),
                               'declared_objects': len(selected_by_object), 'independent_unit': 'object',
                               'complete_objects': len(complete_objects), 'partially_scored_objects': len(partial_objects),
                               'complete_object_accuracy': object_scores,
                               'pending_or_failed': missing,
                               'scores_on_complete_pairs': scores, 'pairs': pairs,
                               'scope': 'question-pair descriptive scores may include partial objects; object scores require every frozen question; same source frames, one generate call each, common FP16/SDPA; token costs may differ',
                               'hallucination_reduction_established': False}
    else:
        report['perception'] = {'status': 'not_scheduled', 'reason': 'Optional native QA data/model not supplied for this generation window'}
    atomic_json(output / 'summary.json', report)
    lines = ['# Eight-hour native-benchmark report', '',
             f"Queue status: {report['queue_status']}",
             f"Generation: {len(complete)}/{len(uids)} complete native/stationary pairs.",
             '', 'This window qualifies baselines and records natural cases. Candidate method efficacy,',
             'novelty and Gate A remain unestablished. Missing/failed comparisons are retained.', '',
             '| Asset | Native CD-3D | Native CD-4D | Native CD-M | Static CD-M |',
             '|---|---:|---:|---:|---:|']
    for pair in complete:
        lines.append(f"| {pair['uid']} | {pair['native']['cd_3d']:.6f} | {pair['native']['cd_4d']:.6f} | {pair['native']['cd_motion']:.6f} | {pair['stationary']['cd_motion']:.6f} |")
    if qa:
        p = report['perception']
        lines += ['', f"Perception: {p['complete_pairs']}/{p['declared_questions']} complete question pairs; "
                  f"{p['complete_objects']}/{p['declared_objects']} complete objects; "
                  f"{p['partially_scored_objects']} partially scored objects.",
                  'Question-level scores are descriptive over completed pairs. Partial objects do not count as complete independent units.']
    lines += ['', 'Full frozen denominators, attempts, raw answers and native receipts: `summary.json`,',
              '`queue.json`, `protocol.json`, `preflight.json` and `units/`.', '',
              'Next review should compare high/low native motion error cases and distinguish geometry,',
              'material correspondence, camera registration and input organization. These ranks are',
              'inspection aids; they are not new failure labels or a substitute for the native metric.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))
    return report
