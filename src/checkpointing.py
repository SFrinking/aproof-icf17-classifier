"""
Checkpoint helpers for chunked inference runs.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_FILENAME = "manifest.json"
CHUNKS_DIRNAME = "chunks"
FAILED_INPUTS_DIRNAME = "failed_inputs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_checkpoint_dirs(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / CHUNKS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / FAILED_INPUTS_DIRNAME).mkdir(parents=True, exist_ok=True)


def manifest_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / MANIFEST_FILENAME


def chunk_output_path(checkpoint_dir: Path, chunk_number: int) -> Path:
    return checkpoint_dir / CHUNKS_DIRNAME / f"chunk_{chunk_number:06d}.csv"


def failed_input_path(checkpoint_dir: Path, chunk_number: int) -> Path:
    return checkpoint_dir / FAILED_INPUTS_DIRNAME / f"chunk_{chunk_number:06d}_input.csv"


def _atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding=encoding, newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def _atomic_write_json(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
    _atomic_write_text(path, text, encoding="utf-8")


def create_empty_manifest(config: dict) -> dict:
    return {
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "config": config,
        "processed_chunks": {},
        "failed_chunks": {},
    }


def create_or_load_manifest(checkpoint_dir: Path, config: dict, resume: bool) -> dict:
    path = manifest_path(checkpoint_dir)

    if path.exists():
        if not resume:
            raise FileExistsError(
                f'Checkpoint manifest already exists at "{path}". '
                "Use --resume or choose another checkpoint/output path."
            )
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    manifest = create_empty_manifest(config)
    save_manifest(checkpoint_dir, manifest)
    return manifest


def save_manifest(checkpoint_dir: Path, manifest: dict) -> None:
    manifest["updated_at"] = utc_now_iso()
    _atomic_write_json(manifest_path(checkpoint_dir), manifest)


def validate_manifest_config(manifest: dict, current_config: dict) -> None:
    stored = manifest.get("config", {})
    mismatches = []

    for key, current_value in current_config.items():
        stored_value = stored.get(key)
        
        # We ignore exact path mismatches because users often move checkpoint 
        # folders between drives, VMs, or local environments mid-run.
        if key in ["input_file", "output_file"]:
            continue
            
        if stored_value != current_value:
            mismatches.append((key, stored_value, current_value))

    if mismatches:
        mismatch_text = "; ".join(
            f"{key}: stored={stored_value!r}, current={current_value!r}"
            for key, stored_value, current_value in mismatches
        )
        raise ValueError(
            "Current run does not match checkpoint configuration. "
            f"Mismatches: {mismatch_text}"
        )


def register_processed_chunk(
    manifest: dict,
    chunk_number: int,
    row_start: int,
    row_end: int,
    row_count: int,
    stats: dict,
) -> dict:
    #LLM: Successful chunks are recorded in the manifest and become the basis
    #LLM: for resume and final output assembly.
    chunk_key = str(chunk_number)

    manifest["processed_chunks"][chunk_key] = {
        "row_start": int(row_start),
        "row_end": int(row_end),
        "row_count": int(row_count),
        "stats": stats,
        "saved_at": utc_now_iso(),
    }

    manifest["failed_chunks"].pop(chunk_key, None)
    return manifest


def register_failed_chunk(
    manifest: dict,
    chunk_number: int,
    row_start: int,
    row_end: int,
    row_count: int,
    error_message: str,
    traceback_text: str,
) -> dict:
    #LLM: Failed chunks are deliberately not marked as processed,
    #LLM: so a later --resume run will retry them automatically.
    chunk_key = str(chunk_number)
    previous = manifest["failed_chunks"].get(chunk_key, {})
    attempts = int(previous.get("attempts", 0)) + 1

    manifest["failed_chunks"][chunk_key] = {
        "row_start": int(row_start),
        "row_end": int(row_end),
        "row_count": int(row_count),
        "attempts": attempts,
        "last_failed_at": utc_now_iso(),
        "last_error": error_message,
        "traceback": traceback_text,
    }

    return manifest


def count_processed_rows(manifest: dict) -> int:
    return int(sum(chunk_info["row_count"] for chunk_info in manifest["processed_chunks"].values()))


def write_chunk_atomic(df, path: Path, sep: str, encoding: str) -> None:
    #LLM: Every successful chunk is first written to its own atomic checkpoint file.
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding=encoding, newline="") as handle:
        df.to_csv(handle, sep=sep, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def save_failed_input_chunk(checkpoint_dir: Path, chunk_number: int, chunk_df, sep: str, encoding: str) -> None:
    path = failed_input_path(checkpoint_dir, chunk_number)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding=encoding, newline="") as handle:
        chunk_df.to_csv(handle, sep=sep, index=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def assemble_output_from_chunks(
    checkpoint_dir: Path,
    output_path: Path,
    processed_chunk_ids: set[int],
    encoding: str,
) -> None:
    #LLM: The assembled CSV is rebuilt from the authoritative chunk files,
    #LLM: which means it can always be regenerated after a crash.
    if not processed_chunk_ids:
        return

    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    sorted_chunk_ids = sorted(processed_chunk_ids)

    with open(temp_output, "w", encoding=encoding, newline="") as out_handle:
        wrote_header = False

        for chunk_id in sorted_chunk_ids:
            chunk_path = chunk_output_path(checkpoint_dir, chunk_id)
            if not chunk_path.exists():
                raise FileNotFoundError(
                    f"Processed chunk {chunk_id} is listed but file is missing: {chunk_path}"
                )

            with open(chunk_path, "r", encoding=encoding, newline="") as in_handle:
                for line_number, line in enumerate(in_handle):
                    if wrote_header and line_number == 0:
                        continue
                    out_handle.write(line)

            wrote_header = True

        out_handle.flush()
        os.fsync(out_handle.fileno())

    os.replace(temp_output, output_path)