import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import scrubadub

try:
    import scrubadub_spacy
except ImportError:
    scrubadub_spacy = None


POLICIES = {
    "exact_match": {
        "gold_dir": "datasets/gold_exact_match",
        "results_dir": "results/exact_match",
    },
    "application_oriented": {
        "gold_dir": "datasets/gold_application_oriented",
        "results_dir": "results/application_oriented",
    },
}

LABEL_PRIORITY = {
    "EMAIL_ADDRESS": 100,
    "IBAN_CODE": 95,
    "IP_ADDRESS": 90,
    "URL": 85,
    "DATE_TIME": 80,
    "PHONE_NUMBER": 70,
    "POSTAL_CODE": 60,
    "PERSON": 50,
    "ORGANIZATION": 40,
    "LOCATION": 30,
}


@dataclass(frozen=True)
class Treffer:
    start: int
    ende: int
    label: str
    source: str
    score: float = 0.0
    from_regex: bool = False
    from_ner: bool = False

    def überschneidet(self, other: "Treffer") -> bool:
        return not (self.ende <= other.start or other.ende <= self.start)

    def länge(self) -> int:
        return self.ende - self.start

    def text(self, original_text: str) -> str:
        return original_text[self.start:self.ende]

    def with_flags(
        self,
        *,
        regex: bool | None = None,
        ner: bool | None = None,
    ) -> "Treffer":
        return replace(
            self,
            from_regex=self.from_regex if regex is None else regex,
            from_ner=self.from_ner if ner is None else ner,
        )


@dataclass(frozen=True)
class GoldSpan:
    start: int
    ende: int
    text: str

    def überschneidet(self, pred: Treffer) -> bool:
        return not (self.ende <= pred.start or pred.ende <= self.start)


@dataclass(frozen=True)
class GoldEntity:
    label: str
    spans: list[GoldSpan]

    def hauptspan(self) -> GoldSpan:
        return self.spans[0]


@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        d = self.tp + self.fp
        return (self.tp / d) if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return (self.tp / d) if d else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        d = p + r
        return (2.0 * p * r / d) if d else 0.0


@dataclass(frozen=True)
class DebugEntry:
    dataset_name: str
    kind: str
    label: str
    start: int
    ende: int
    text: str
    source: str
    pred_label: str | None = None
    pred_start: int | None = None
    pred_ende: int | None = None
    pred_text: str | None = None
    pred_source: str | None = None
    pred_score: float | None = None


def load_mapping(mapping_path: Path) -> dict[str, list[str]]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping-Datei nicht gefunden: {mapping_path}")

    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mapping-Datei muss ein JSON-Objekt sein.")

    normalized_mapping: dict[str, list[str]] = {}

    for gold_label, values in payload.items():
        gold_label_norm = str(gold_label).strip().upper()
        if not gold_label_norm:
            continue

        if not isinstance(values, list):
            raise ValueError(f"Mapping für {gold_label_norm} muss eine Liste sein.")

        normalized_values: list[str] = []
        seen: set[str] = set()

        for value in values:
            norm = str(value).strip().upper()
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            normalized_values.append(norm)

        normalized_mapping[gold_label_norm] = normalized_values

    return normalized_mapping


def build_scrubber() -> scrubadub.Scrubber:
    scrubber = scrubadub.Scrubber(locale="de_DE")

    if scrubadub_spacy is not None:
        ner_detector = scrubadub_spacy.detectors.SpacyEntityDetector(
            model="de_core_news_lg",
            named_entities=["PER", "PERSON", "ORG", "LOC", "GPE"],
        )
        scrubber.add_detector(ner_detector)

    return scrubber


def infer_source_from_filth(filth: Any) -> tuple[str, bool, bool]:
    detector_name = str(getattr(filth, "detector_name", "") or "").lower()

    if "spacy" in detector_name:
        return "ner", False, True

    return "regex", True, False


def normalize_external_label(label: str) -> str:
    return str(label or "").strip().upper()


def normalize_gold_label(label: str) -> str:
    return str(label or "").strip().upper()


def filth_to_external_label(filth: Any) -> str:
    filth_type = normalize_external_label(getattr(filth, "type", ""))
    filth_class_name = normalize_external_label(type(filth).__name__)

    mapping = {
        "EMAIL": "EMAIL_ADDRESS",
        "EMAILFILTH": "EMAIL_ADDRESS",
        "PHONE": "PHONE_NUMBER",
        "PHONEFILTH": "PHONE_NUMBER",
        "URL": "URL",
        "URLFILTH": "URL",
        "NAME": "PERSON",
        "NAMEFILTH": "PERSON",
        "PERSON": "PERSON",
        "PERSONFILTH": "PERSON",
        "ORGANIZATION": "ORGANIZATION",
        "ORGANIZATIONFILTH": "ORGANIZATION",
        "LOCATION": "LOCATION",
        "LOCATIONFILTH": "LOCATION",
        "POSTAL_CODE": "POSTAL_CODE",
        "POSTALCODE": "POSTAL_CODE",
        "POSTALCODEFILTH": "POSTAL_CODE",
        "DATEOFBIRTH": "DATE_TIME",
        "DATEOFBIRTHFILTH": "DATE_TIME",
        "DOB": "DATE_TIME",
        "DOBFILTH": "DATE_TIME",
        "DATE_TIME": "DATE_TIME",
        "DATEFILTH": "DATE_TIME",
        "ADDRESS": "ADDRESS",
        "ADDRESSFILTH": "ADDRESS",
        "CREDITCARD": "CREDIT_CARD",
        "CREDITCARDFILTH": "CREDIT_CARD",
        "CREDENTIAL": "CREDENTIAL",
        "CREDENTIALFILTH": "CREDENTIAL",
        "DRIVERSLICENCE": "DRIVERS_LICENCE",
        "DRIVERSLICENCEFILTH": "DRIVERS_LICENCE",
        "SOCIALSECURITYNUMBER": "SOCIAL_SECURITY_NUMBER",
        "SOCIALSECURITYNUMBERFILTH": "SOCIAL_SECURITY_NUMBER",
        "TWITTER": "TWITTER",
        "TWITTERFILTH": "TWITTER",
        "VEHICLELICENCEPLATE": "VEHICLE_LICENCE_PLATE",
        "VEHICLELICENCEPLATEFILTH": "VEHICLE_LICENCE_PLATE",
    }

    if filth_type in mapping:
        return mapping[filth_type]

    if filth_class_name in mapping:
        return mapping[filth_class_name]

    return filth_type or filth_class_name


def filth_to_treffer(filth: Any) -> Treffer:
    source, from_regex, from_ner = infer_source_from_filth(filth)

    start = int(getattr(filth, "beg"))
    ende = int(getattr(filth, "end"))

    return Treffer(
        start=start,
        ende=ende,
        label=filth_to_external_label(filth),
        source=source,
        score=float(getattr(filth, "score", 1.0) or 1.0),
        from_regex=from_regex,
        from_ner=from_ner,
    )


def deduplicate_exact(results: list[Treffer]) -> list[Treffer]:
    seen: set[tuple[int, int, str, str]] = set()
    out: list[Treffer] = []

    for item in results:
        key = (item.start, item.ende, item.label, item.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


def filter_to_mapped_labels(results: list[Treffer], mapping: dict[str, list[str]]) -> list[Treffer]:
    allowed_labels: set[str] = set()

    for pred_labels in mapping.values():
        for pred_label in pred_labels:
            allowed_labels.add(normalize_external_label(pred_label))

    return [item for item in results if item.label in allowed_labels]


def resolve_same_span_label_conflicts(results: list[Treffer]) -> list[Treffer]:
    grouped: dict[tuple[int, int], list[Treffer]] = {}

    for item in results:
        key = (item.start, item.ende)
        grouped.setdefault(key, []).append(item)

    resolved: list[Treffer] = []

    for items in grouped.values():
        best = sorted(
            items,
            key=lambda x: (
                -LABEL_PRIORITY.get(x.label, 0),
                -x.score,
                x.label,
                x.source,
            ),
        )[0]
        resolved.append(best)

    resolved.sort(key=lambda x: (x.start, x.ende, x.label, x.source))
    return resolved


def resolve_overlaps_largest_span_wins(results: list[Treffer]) -> list[Treffer]:
    if not results:
        return []

    candidates = sorted(
        results,
        key=lambda x: (
            -x.länge(),
            -x.score,
            -LABEL_PRIORITY.get(x.label, 0),
            x.start,
            x.ende,
            x.label,
            x.source,
        ),
    )

    selected: list[Treffer] = []

    for candidate in candidates:
        if any(candidate.überschneidet(existing) for existing in selected):
            continue
        selected.append(candidate)

    selected.sort(key=lambda x: (x.start, x.ende, x.label, x.source))
    return selected


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_gold(path: Path, mapping: dict[str, list[str]]) -> list[GoldEntity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities = payload.get("entities", [])

    out: list[GoldEntity] = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        raw_label = entity.get("label")

        if isinstance(raw_label, list):
            if not raw_label:
                continue
            gold_label = normalize_gold_label(str(raw_label[0]))
        else:
            gold_label = normalize_gold_label(str(raw_label))

        if gold_label not in mapping:
            continue

        spans: list[GoldSpan] = []

        start = entity.get("start")
        end = entity.get("end")
        txt = str(entity.get("text", ""))

        if isinstance(start, int) and isinstance(end, int):
            spans.append(GoldSpan(start=int(start), ende=int(end), text=txt))

        alternatives = entity.get("alternatives")
        if isinstance(alternatives, list):
            for alt in alternatives:
                if not isinstance(alt, dict):
                    continue

                alt_start = alt.get("start")
                alt_end = alt.get("end")
                alt_text = str(alt.get("text", ""))

                if not isinstance(alt_start, int) or not isinstance(alt_end, int):
                    continue

                span = GoldSpan(start=int(alt_start), ende=int(alt_end), text=alt_text)
                if span not in spans:
                    spans.append(span)

        if not spans:
            continue

        spans.sort(key=lambda s: (s.start, s.ende))
        out.append(GoldEntity(label=gold_label, spans=spans))

    out.sort(key=lambda g: (g.hauptspan().start, g.hauptspan().ende, g.label))
    return out


def dataset_meta(dataset_name: str) -> tuple[str, str]:
    try:
        index = int(dataset_name.split("_")[-1])
    except Exception:
        return "Unknown", "unknown"

    domains = [
        "Supporttickets",
        "E-Mail",
        "HR-Dokumente",
        "Verträge",
        "Chats",
    ]

    domain_idx = (index - 1) // 6
    within = (index - 1) % 6

    domain = domains[domain_idx] if 0 <= domain_idx < len(domains) else "Unknown"

    if within in (0, 1):
        structure = "structured"
    elif within in (2, 3):
        structure = "regular"
    else:
        structure = "unstructured"

    return domain, structure


def detect_with_existing_scrubber_for_text(
    scrubber: scrubadub.Scrubber,
    text: str,
    mapping: dict[str, list[str]],
) -> list[Treffer]:
    raw_results = list(scrubber.iter_filth(text, document_name=None, run_post_processors=False))
    raw_treffer = [filth_to_treffer(result) for result in raw_results]
    raw_treffer.sort(key=lambda item: (item.start, item.ende, item.label, item.source))

    filtered_treffer = filter_to_mapped_labels(raw_treffer, mapping)
    unique_treffer = deduplicate_exact(filtered_treffer)
    same_span_resolved = resolve_same_span_label_conflicts(unique_treffer)
    final_treffer = resolve_overlaps_largest_span_wins(same_span_resolved)

    return final_treffer


def label_matches(gold_label: str, pred_label: str, mapping: dict[str, list[str]]) -> bool:
    accepted = mapping.get(gold_label, [])
    pred_label_norm = normalize_external_label(pred_label)
    return pred_label_norm in accepted


def evaluate_predictions(
    dataset_name: str,
    preds: list[Treffer],
    golds: list[GoldEntity],
    text: str,
    mapping: dict[str, list[str]],
) -> tuple[EvalCounts, list[DebugEntry]]:
    counts = EvalCounts()
    debug_entries: list[DebugEntry] = []

    matched_pred_indices: set[int] = set()
    partial_pred_indices: set[int] = set()

    for gold in golds:
        exact_match_idx = None
        exact_match_span = None

        for gold_span in gold.spans:
            for pred_idx, pred in enumerate(preds):
                if pred_idx in matched_pred_indices or pred_idx in partial_pred_indices:
                    continue
                if not label_matches(gold.label, pred.label, mapping):
                    continue
                if pred.start == gold_span.start and pred.ende == gold_span.ende:
                    exact_match_idx = pred_idx
                    exact_match_span = gold_span
                    break
            if exact_match_idx is not None:
                break

        if exact_match_idx is not None and exact_match_span is not None:
            counts.tp += 1
            matched_pred_indices.add(exact_match_idx)

            pred = preds[exact_match_idx]
            debug_entries.append(
                DebugEntry(
                    dataset_name=dataset_name,
                    kind="TP",
                    label=gold.label,
                    start=exact_match_span.start,
                    ende=exact_match_span.ende,
                    text=exact_match_span.text or text[exact_match_span.start:exact_match_span.ende],
                    source="gold",
                    pred_label=pred.label,
                    pred_start=pred.start,
                    pred_ende=pred.ende,
                    pred_text=pred.text(text),
                    pred_source=pred.source,
                    pred_score=pred.score,
                )
            )
            continue

        partial_match_idx = None
        partial_match_span = None

        for gold_span in gold.spans:
            for pred_idx, pred in enumerate(preds):
                if pred_idx in matched_pred_indices or pred_idx in partial_pred_indices:
                    continue
                if not label_matches(gold.label, pred.label, mapping):
                    continue
                if not (pred.ende <= gold_span.start or gold_span.ende <= pred.start):
                    partial_match_idx = pred_idx
                    partial_match_span = gold_span
                    break
            if partial_match_idx is not None:
                break

        if partial_match_idx is not None and partial_match_span is not None:
            counts.fp += 1
            counts.fn += 1
            partial_pred_indices.add(partial_match_idx)

            pred = preds[partial_match_idx]
            debug_entries.append(
                DebugEntry(
                    dataset_name=dataset_name,
                    kind="FP/FN",
                    label=gold.label,
                    start=partial_match_span.start,
                    ende=partial_match_span.ende,
                    text=partial_match_span.text or text[partial_match_span.start:partial_match_span.ende],
                    source="gold",
                    pred_label=pred.label,
                    pred_start=pred.start,
                    pred_ende=pred.ende,
                    pred_text=pred.text(text),
                    pred_source=pred.source,
                    pred_score=pred.score,
                )
            )
            continue

        hauptspan = gold.hauptspan()
        counts.fn += 1
        debug_entries.append(
            DebugEntry(
                dataset_name=dataset_name,
                kind="FN",
                label=gold.label,
                start=hauptspan.start,
                ende=hauptspan.ende,
                text=hauptspan.text or text[hauptspan.start:hauptspan.ende],
                source="gold",
            )
        )

    for pred_idx, pred in enumerate(preds):
        if pred_idx in matched_pred_indices or pred_idx in partial_pred_indices:
            continue

        counts.fp += 1
        debug_entries.append(
            DebugEntry(
                dataset_name=dataset_name,
                kind="FP",
                label=pred.label,
                start=pred.start,
                ende=pred.ende,
                text=pred.text(text),
                source=pred.source,
                pred_label=pred.label,
                pred_start=pred.start,
                pred_ende=pred.ende,
                pred_text=pred.text(text),
                pred_source=pred.source,
                pred_score=pred.score,
            )
        )

    debug_entries.sort(key=lambda x: (x.start, x.ende, x.kind, x.label))
    return counts, debug_entries


def format_dataset_block(
    dataset_name: str,
    domain: str,
    policy_name: str,
    counts: EvalCounts,
) -> str:
    return (
        f"{dataset_name:<12} | "
        f"{domain:<14} | "
        f"{policy_name:<20} | "
        f"TP={counts.tp:>3} FP={counts.fp:>3} FN={counts.fn:>3} | "
        f"P={counts.precision():.3f} R={counts.recall():.3f} F1={counts.f1():.3f}"
    )


def format_debug_block(
    dataset_name: str,
    domain: str,
    policy_name: str,
    counts: EvalCounts,
    golds: list[GoldEntity],
    debug_entries: list[DebugEntry],
    text: str,
) -> str:
    lines: list[str] = []

    lines.append(f"DATASET: {dataset_name}")
    lines.append(f"DOMAIN: {domain}")
    lines.append("")
    lines.append(f"POLICY {policy_name}")
    lines.append(f"TP={counts.tp} FP={counts.fp} FN={counts.fn} | P={counts.precision():.3f} R={counts.recall():.3f} F1={counts.f1():.3f}")
    lines.append("")
    lines.append("TARGET LABELS")
    lines.append("----------------------------------------------------------------------")

    for g in golds:
        if len(g.spans) == 1:
            span = g.spans[0]
            shown_text = span.text or text[span.start:span.ende]
            lines.append(f"{g.label:10s} {span.start:4d}:{span.ende:<4d} '{shown_text}'")
        else:
            first = g.spans[0]
            first_text = first.text or text[first.start:first.ende]
            alt_parts: list[str] = []

            for alt in g.spans[1:]:
                alt_text = alt.text or text[alt.start:alt.ende]
                alt_parts.append(f"{alt.start}:{alt.ende} '{alt_text}'")

            lines.append(
                f"{g.label:10s} {first.start:4d}:{first.ende:<4d} '{first_text}' | ALT: " + " OR ".join(alt_parts)
            )

    lines.append("")
    lines.append("EVALUATION")
    lines.append("----------------------------------------------------------------------")

    if not debug_entries:
        lines.append("Keine.")
    else:
        for entry in debug_entries:
            if entry.kind == "TP":
                lines.append(
                    f"TP      {entry.label:10s} {entry.start:4d}:{entry.ende:<4d} '{entry.text}' "
                    f"| pred {entry.pred_label:14s} {entry.pred_start:4d}:{entry.pred_ende:<4d} '{entry.pred_text}'"
                )
            elif entry.kind == "FN":
                lines.append(
                    f"FN      {entry.label:10s} {entry.start:4d}:{entry.ende:<4d} '{entry.text}'"
                )
            elif entry.kind == "FP/FN":
                lines.append(
                    f"FP/FN   {entry.label:10s} {entry.start:4d}:{entry.ende:<4d} '{entry.text}' "
                    f"| pred {entry.pred_label:14s} {entry.pred_start:4d}:{entry.pred_ende:<4d} '{entry.pred_text}'"
                )
            elif entry.kind == "FP":
                lines.append(
                    f"FP      {entry.label:10s} {entry.start:4d}:{entry.ende:<4d} '{entry.text}'"
                )

    lines.append("")
    lines.append("======================================================================")
    lines.append("")

    return "\n".join(lines)


def format_label_report_summary(
    label_entries: dict[str, dict[str, list[DebugEntry]]],
    policy_name: str,
) -> str:
    lines: list[str] = []
    lines.append(f"LABEL REPORT | POLICY: {policy_name}")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("")
    lines.append("LABEL      | TP | PARTIAL | FN | FP")
    lines.append("------------------------------------")

    ordered_labels = sorted(label_entries.keys())

    for label in ordered_labels:
        tp_count = len(label_entries[label]["TP"])
        partial_count = len(label_entries[label]["FP/FN"])
        fn_count = len(label_entries[label]["FN"])
        fp_count = len(label_entries[label]["FP"])

        if tp_count == 0 and partial_count == 0 and fn_count == 0 and fp_count == 0:
            continue

        lines.append(
            f"{label:<10} | {tp_count:>2} | {partial_count:>7} | {fn_count:>2} | {fp_count:>2}"
        )

    lines.append("")
    return "\n".join(lines)


def format_label_report_debug(
    label_entries: dict[str, dict[str, list[DebugEntry]]],
    policy_name: str,
) -> str:
    lines: list[str] = []
    lines.append(f"LABEL REPORT | POLICY: {policy_name}")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("")

    ordered_labels = sorted(label_entries.keys())

    for label in ordered_labels:
        tp_entries = sorted(label_entries[label]["TP"], key=lambda x: (x.dataset_name, x.start, x.ende))
        partial_entries = sorted(label_entries[label]["FP/FN"], key=lambda x: (x.dataset_name, x.start, x.ende))
        fn_entries = sorted(label_entries[label]["FN"], key=lambda x: (x.dataset_name, x.start, x.ende))
        fp_entries = sorted(label_entries[label]["FP"], key=lambda x: (x.dataset_name, x.start, x.ende))

        tp_count = len(tp_entries)
        partial_count = len(partial_entries)
        fn_count = len(fn_entries)
        fp_count = len(fp_entries)

        if tp_count == 0 and partial_count == 0 and fn_count == 0 and fp_count == 0:
            continue

        lines.append(label)
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"TP={tp_count} | PARTIAL={partial_count} | FN={fn_count} | FP={fp_count}")
        lines.append("")

        if tp_count > 0:
            lines.append(f"ERKANNT (TP): {tp_count}")
            for entry in tp_entries:
                pred_source = entry.pred_source or "?"
                lines.append(
                    f"  - {entry.dataset_name} {entry.start}:{entry.ende} [{pred_source}] '{entry.text}'"
                )
            lines.append("")

        if partial_count > 0:
            lines.append(f"NICHT VOLLSTÄNDIG ERKANNT (PARTIAL): {partial_count}")
            for entry in partial_entries:
                pred_source = entry.pred_source or "?"
                pred_label = entry.pred_label or "?"
                pred_start = entry.pred_start if entry.pred_start is not None else -1
                pred_ende = entry.pred_ende if entry.pred_ende is not None else -1
                pred_text = entry.pred_text or ""
                lines.append(
                    f"  - {entry.dataset_name} gold {entry.start}:{entry.ende} [gold] '{entry.text}'"
                    f" | pred {pred_label} {pred_start}:{pred_ende} [{pred_source}] '{pred_text}'"
                )
            lines.append("")

        if fn_count > 0:
            lines.append(f"NICHT ERKANNT (FN): {fn_count}")
            for entry in fn_entries:
                lines.append(
                    f"  - {entry.dataset_name} {entry.start}:{entry.ende} [gold] '{entry.text}'"
                )
            lines.append("")

        if fp_count > 0:
            lines.append(f"UNERWARTET (FP): {fp_count}")
            for entry in fp_entries:
                lines.append(
                    f"  - {entry.dataset_name} {entry.start}:{entry.ende} [{entry.source}] '{entry.text}'"
                )
            lines.append("")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_label_debug_entries(
    mapping: dict[str, list[str]],
    debug_entries: list[DebugEntry],
) -> dict[str, dict[str, list[DebugEntry]]]:
    label_entries: dict[str, dict[str, list[DebugEntry]]] = {
        label: {"TP": [], "FP/FN": [], "FN": [], "FP": []}
        for label in sorted(mapping.keys())
    }

    for entry in debug_entries:
        if entry.label not in label_entries:
            label_entries[entry.label] = {"TP": [], "FP/FN": [], "FN": [], "FP": []}
        if entry.kind in label_entries[entry.label]:
            label_entries[entry.label][entry.kind].append(entry)

    return label_entries


def aggregate_label_counts(
    mapping: dict[str, list[str]],
    debug_entries: list[DebugEntry],
) -> dict[str, dict[str, int]]:
    label_entries = build_label_debug_entries(mapping, debug_entries)
    out: dict[str, dict[str, int]] = {}

    for label in sorted(label_entries.keys()):
        tp_count = len(label_entries[label]["TP"])
        partial_count = len(label_entries[label]["FP/FN"])
        fn_count = len(label_entries[label]["FN"])
        fp_count = len(label_entries[label]["FP"])

        out[label] = {
            "TP": tp_count,
            "PARTIAL": partial_count,
            "FN": fn_count,
            "FP": fp_count,
        }

    return out


def map_policy_name_for_ba(policy_name: str) -> str:
    policy_norm = str(policy_name).strip().lower()

    if policy_norm == "application_oriented":
        return "application_oriented"

    if policy_norm == "exact_match":
        return "exact_match"

    return policy_name


def infer_backend_name() -> str:
    if scrubadub_spacy is not None:
        return "spacy"

    return "unknown"


def infer_model_name() -> str:
    if scrubadub_spacy is not None:
        return "de_core_news_lg"

    return "unknown"


def format_ba_summary(
    policy_name: str,
    global_counts: EvalCounts,
    aggregated_label_counts: dict[str, dict[str, int]],
    label_entries: dict[str, dict[str, list[DebugEntry]]],
) -> str:
    lines: list[str] = []

    lines.append(f"BA SUMMARY | POLICY: {map_policy_name_for_ba(policy_name)}")
    lines.append(f"NER_BACKEND: {infer_backend_name()}")
    lines.append(f"NER_MODEL: {infer_model_name()}")
    lines.append("POSTPROCESSING: off")
    lines.append("")
    lines.append("GESAMTERGEBNIS (micro-aggregated)")
    lines.append("----------------------------------------------------------------------")
    lines.append(f"TP: {global_counts.tp}")
    lines.append(f"FP: {global_counts.fp}")
    lines.append(f"FN: {global_counts.fn}")
    lines.append(f"Precision: {global_counts.precision():.3f}")
    lines.append(f"Recall: {global_counts.recall():.3f}")
    lines.append(f"F1: {global_counts.f1():.3f}")
    lines.append("")
    lines.append("LABELERGEBNISSE")
    lines.append("----------------------------------------------------------------------")

    for label in sorted(aggregated_label_counts.keys()):
        counts = aggregated_label_counts[label]
        tp = counts["TP"]
        fp = counts["FP"] + counts["PARTIAL"]
        fn = counts["FN"] + counts["PARTIAL"]

        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        denominator = precision + recall
        f1 = (2.0 * precision * recall / denominator) if denominator else 0.0

        lines.append(
            f"{label:<12}"
            f" TP={tp:>3} FP={fp:>3} FN={fn:>3} "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

    lines.append("")
    lines.append("VERBLEIBENDE FEHLFÄLLE")
    lines.append("----------------------------------------------------------------------")

    printed_any = False

    for label in sorted(label_entries.keys()):
        partial_entries = sorted(label_entries[label]["FP/FN"], key=lambda x: (x.dataset_name, x.start, x.ende))
        fn_entries = sorted(label_entries[label]["FN"], key=lambda x: (x.dataset_name, x.start, x.ende))
        fp_entries = sorted(label_entries[label]["FP"], key=lambda x: (x.dataset_name, x.start, x.ende))

        if not partial_entries and not fn_entries and not fp_entries:
            continue

        printed_any = True
        lines.append(label)

        if partial_entries:
            lines.append(f"  PARTIAL ({len(partial_entries)}):")
            for entry in partial_entries:
                pred_source = entry.pred_source or "?"
                pred_start = entry.pred_start if entry.pred_start is not None else -1
                pred_ende = entry.pred_ende if entry.pred_ende is not None else -1
                pred_text = entry.pred_text or ""
                lines.append(
                    f"    - {entry.dataset_name} gold:{entry.start}:{entry.ende} [gold] '{entry.text}' "
                    f"<- pred:{pred_source}:{pred_start}:{pred_ende} '{pred_text}'"
                )

        if fn_entries:
            lines.append(f"  FN ({len(fn_entries)}):")
            for entry in fn_entries:
                lines.append(
                    f"    - {entry.dataset_name} {entry.start}:{entry.ende} [gold] '{entry.text}'"
                )

        if fp_entries:
            lines.append(f"  FP ({len(fp_entries)}):")
            for entry in fp_entries:
                lines.append(
                    f"    - {entry.dataset_name} {entry.start}:{entry.ende} [{entry.source}] '{entry.text}'"
                )

        lines.append("")

    if not printed_any:
        lines.append("Keine verbleibenden Fehlfälle.")

    return "\n".join(lines).rstrip() + "\n"


def format_ba_summary_debug(
    policy_name: str,
    global_counts: EvalCounts,
    aggregated_label_counts: dict[str, dict[str, int]],
    label_entries: dict[str, dict[str, list[DebugEntry]]],
) -> str:
    lines: list[str] = []
    lines.append(format_ba_summary(policy_name, global_counts, aggregated_label_counts, label_entries).rstrip())
    lines.append("")
    lines.append("ROHE LABEL-ZUSTÄNDE")
    lines.append("----------------------------------------------------------------------")

    for label in sorted(label_entries.keys()):
        tp_count = len(label_entries[label]["TP"])
        partial_count = len(label_entries[label]["FP/FN"])
        fn_count = len(label_entries[label]["FN"])
        fp_count = len(label_entries[label]["FP"])

        if tp_count == 0 and partial_count == 0 and fn_count == 0 and fp_count == 0:
            continue

        lines.append(
            f"{label:<12} TP={tp_count:>3} PARTIAL={partial_count:>3} FN={fn_count:>3} FP={fp_count:>3}"
        )

    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def warmup(scrubber: scrubadub.Scrubber, warmup_text: str, mapping: dict[str, list[str]]) -> None:
    _ = detect_with_existing_scrubber_for_text(scrubber, warmup_text, mapping)


def run_policy(
    policy_name: str,
    data_dir: Path,
    gold_dir: Path,
    results_dir: Path,
    scrubber: scrubadub.Scrubber,
    mapping: dict[str, list[str]],
) -> None:
    default_dir = results_dir / "default"
    debug_dir = results_dir / "debug"
    ba_dir = results_dir / "ba_results"

    default_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    ba_dir.mkdir(parents=True, exist_ok=True)

    output_path = default_dir / "Results.txt"
    label_output_path = default_dir / "Label_Results.txt"
    debug_output_path = debug_dir / "Results_debug.txt"
    label_debug_output_path = debug_dir / "Label_Results_debug.txt"
    ba_summary_path = ba_dir / "BA_Summary.txt"
    ba_summary_debug_path = ba_dir / "BA_Summary_debug.txt"

    if not gold_dir.exists():
        raise FileNotFoundError(f"Gold directory nicht gefunden: {gold_dir}")

    text_files = sorted(data_dir.glob("Dataset_*.txt"))
    if not text_files:
        raise FileNotFoundError(f"Keine Dataset-Textdateien gefunden in: {data_dir}")

    global_counts = EvalCounts()
    output_blocks: list[str] = []
    debug_blocks: list[str] = []
    all_debug_entries: list[DebugEntry] = []

    label_entries: dict[str, dict[str, list[DebugEntry]]] = {
        label: {"TP": [], "FP/FN": [], "FN": [], "FP": []}
        for label in sorted(mapping.keys())
    }

    print(f"Starte Batch-Evaluation für Policy: {policy_name}")

    for idx, text_file in enumerate(text_files, start=1):
        dataset_name = text_file.stem
        gold_file = gold_dir / f"{dataset_name}.json"

        if not gold_file.exists():
            raise FileNotFoundError(f"Passende Gold-Datei fehlt: {gold_file}")

        text = read_text(text_file)
        gold_entities = read_gold(gold_file, mapping)
        preds = detect_with_existing_scrubber_for_text(scrubber, text, mapping)

        counts, debug_entries = evaluate_predictions(
            dataset_name=dataset_name,
            preds=preds,
            golds=gold_entities,
            text=text,
            mapping=mapping,
        )

        global_counts.tp += counts.tp
        global_counts.fp += counts.fp
        global_counts.fn += counts.fn

        all_debug_entries.extend(debug_entries)

        for entry in debug_entries:
            if entry.label not in label_entries:
                label_entries[entry.label] = {"TP": [], "FP/FN": [], "FN": [], "FP": []}
            if entry.kind in label_entries[entry.label]:
                label_entries[entry.label][entry.kind].append(entry)

        domain, _structure = dataset_meta(dataset_name)

        output_blocks.append(
            format_dataset_block(
                dataset_name=dataset_name,
                domain=domain,
                policy_name=policy_name,
                counts=counts,
            )
        )

        debug_blocks.append(
            format_debug_block(
                dataset_name=dataset_name,
                domain=domain,
                policy_name=policy_name,
                counts=counts,
                golds=gold_entities,
                debug_entries=debug_entries,
                text=text,
            )
        )

        print(f"[{idx:02d}/{len(text_files):02d}] {policy_name} | {dataset_name} fertig")

    summary_lines = [
        "",
        "GLOBAL SUMMARY (micro-averaged over all datasets)",
        "----------------------------------------------------------------------",
        f"POLICY {policy_name:<18} | TP={global_counts.tp:4d} FP={global_counts.fp:4d} FN={global_counts.fn:4d} | P={global_counts.precision():.3f} R={global_counts.recall():.3f} F1={global_counts.f1():.3f}",
        "",
    ]

    final_output_lines = [
        "DATASET      | DOMAIN         | POLICY               | COUNTS                          | METRICS",
        "---------------------------------------------------------------------------------------------",
        *output_blocks,
        *summary_lines,
    ]
    output_path.write_text("\n".join(final_output_lines).rstrip() + "\n", encoding="utf-8")

    final_debug_output = "".join(debug_blocks) + "\n".join(summary_lines)
    debug_output_path.write_text(final_debug_output, encoding="utf-8")

    label_report_summary = format_label_report_summary(label_entries, policy_name=policy_name)
    label_output_path.write_text(label_report_summary, encoding="utf-8")

    label_report_debug = format_label_report_debug(label_entries, policy_name=policy_name)
    label_debug_output_path.write_text(label_report_debug, encoding="utf-8")

    aggregated_label_counts = aggregate_label_counts(mapping, all_debug_entries)

    ba_summary = format_ba_summary(
        policy_name=policy_name,
        global_counts=global_counts,
        aggregated_label_counts=aggregated_label_counts,
        label_entries=label_entries,
    )
    ba_summary_path.write_text(ba_summary, encoding="utf-8")

    ba_summary_debug = format_ba_summary_debug(
        policy_name=policy_name,
        global_counts=global_counts,
        aggregated_label_counts=aggregated_label_counts,
        label_entries=label_entries,
    )
    ba_summary_debug_path.write_text(ba_summary_debug, encoding="utf-8")

    print(f"Ergebnisdatei geschrieben: {output_path}")
    print(f"Label-Datei geschrieben: {label_output_path}")
    print(f"Debug-Datei geschrieben: {debug_output_path}")
    print(f"Label-Debug-Datei geschrieben: {label_debug_output_path}")
    print(f"BA-Summary geschrieben: {ba_summary_path}")
    print(f"BA-Summary-Debug geschrieben: {ba_summary_debug_path}")


def main() -> None:
    data_dir = Path("datasets/data")
    mapping_path = Path("mapping_scrubadub.json")

    mapping = load_mapping(mapping_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory nicht gefunden: {data_dir}")

    print("Initialisiere scrubadub einmalig ...")
    init_start = time.perf_counter()
    scrubber = build_scrubber()
    init_end = time.perf_counter()

    text_files = sorted(data_dir.glob("Dataset_*.txt"))
    if not text_files:
        raise FileNotFoundError(f"Keine Dataset-Textdateien gefunden in: {data_dir}")

    first_text = read_text(text_files[0])
    print("Führe Warmup-Run aus ...")
    warmup_start = time.perf_counter()
    warmup(scrubber, first_text, mapping)
    warmup_end = time.perf_counter()

    print(f"Initialisierung: {(init_end - init_start):.3f} s")
    print(f"Warmup: {(warmup_end - warmup_start):.3f} s")

    for policy_name, policy_cfg in POLICIES.items():
        gold_dir = Path(policy_cfg["gold_dir"])
        results_dir = Path(policy_cfg["results_dir"])

        run_policy(
            policy_name=policy_name,
            data_dir=data_dir,
            gold_dir=gold_dir,
            results_dir=results_dir,
            scrubber=scrubber,
            mapping=mapping,
        )


if __name__ == "__main__":
    main()