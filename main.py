"""
Optimized batch inference entrypoint for the ICF classifier pipeline.
"""

import argparse
import gc
import logging
import multiprocessing as mp
import queue
import time
from pathlib import Path

import pandas as pd

from src import timer
from src.checkpointing import (
    assemble_output_from_chunks,
    count_processed_rows,
    create_or_load_manifest,
    ensure_checkpoint_dirs,
    save_failed_input_chunk,
    save_manifest,
    register_failed_chunk,
    register_processed_chunk,
    validate_manifest_config,
    write_chunk_atomic,
)
from src.worker import worker_main


log = logging.getLogger(__name__)


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def count_data_rows(csv_path: Path, encoding: str) -> int:
    with open(csv_path, "r", encoding=encoding, errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def start_worker_process(
    ctx,
    runtime_config: dict,
    log_level: str,
) -> tuple[mp.Process, mp.Queue, mp.Queue]:
    #LLM: The heavy models are loaded in a dedicated worker subprocess.
    #LLM: This lets the parent enforce a real timeout by terminating the worker
    #LLM: if a chunk hangs, while still keeping models loaded once during normal operation.
    task_queue = ctx.Queue(maxsize=2)
    result_queue = ctx.Queue(maxsize=2)

    # process = ctx.Process(
    #     target=worker_main,
    #     args=(task_queue, result_queue, runtime_config, log_level),
    #     daemon=True,
    # )
    process = ctx.Process(
        target=worker_main,
        args=(task_queue, result_queue, runtime_config, log_level),
        daemon=False,
    )
    process.start()

    startup_timeout = runtime_config["worker_startup_timeout_seconds"]
    start_time = time.monotonic()

    while True:
        try:
            message = result_queue.get(timeout=1.0)
        except queue.Empty:
            if not process.is_alive():
                raise RuntimeError("Worker process exited during startup.")
            if startup_timeout > 0 and (time.monotonic() - start_time) > startup_timeout:
                process.terminate()
                process.join(timeout=5)
                raise TimeoutError(
                    f"Worker startup exceeded timeout of {startup_timeout} seconds."
                )
            continue

        message_type = message.get("type")

        if message_type == "ready":
            log.info("Worker process is ready.")
            return process, task_queue, result_queue

        if message_type == "startup_error":
            process.join(timeout=5)
            raise RuntimeError(
                "Worker failed during startup.\n"
                f"Error: {message.get('error')}\n"
                f"Traceback:\n{message.get('traceback')}"
            )

        raise RuntimeError(f"Unexpected worker startup message: {message!r}")


def stop_worker_process(
    process: mp.Process | None,
    task_queue,
    result_queue,
    join_timeout: int = 10,
) -> None:
    if process is None:
        return

    try:
        if process.is_alive():
            try:
                task_queue.put({"type": "shutdown"}, timeout=1)
            except Exception:
                pass

            process.join(timeout=join_timeout)

            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    finally:
        try:
            task_queue.close()
        except Exception:
            pass
        try:
            result_queue.close()
        except Exception:
            pass


def run_chunk_with_timeout(
    process: mp.Process,
    task_queue,
    result_queue,
    chunk_number: int,
    chunk_df: pd.DataFrame,
    timeout_seconds: int,
) -> dict:
    task_queue.put(
        {
            "type": "process_chunk",
            "chunk_number": chunk_number,
            "chunk_df": chunk_df,
        }
    )

    start_time = time.monotonic()

    while True:
        try:
            message = result_queue.get(timeout=1.0)
        except queue.Empty:
            if not process.is_alive():
                return {
                    "status": "worker_died",
                    "error": "Worker process died while processing the chunk.",
                    "traceback": "",
                }

            if timeout_seconds > 0 and (time.monotonic() - start_time) > timeout_seconds:
                return {
                    "status": "timeout",
                    "error": f"Chunk timed out after {timeout_seconds} seconds.",
                    "traceback": "",
                }
            continue

        message_type = message.get("type")
        returned_chunk_number = message.get("chunk_number")

        if returned_chunk_number != chunk_number:
            return {
                "status": "protocol_error",
                "error": (
                    f"Expected result for chunk {chunk_number}, "
                    f"but received message for chunk {returned_chunk_number}."
                ),
                "traceback": "",
            }

        if message_type == "result":
            return {
                "status": "success",
                "result_chunk": message["result_chunk"],
                "stats": message["stats"],
            }

        if message_type == "chunk_error":
            return {
                "status": "chunk_error",
                "error": message.get("error", "Unknown worker chunk error."),
                "traceback": message.get("traceback", ""),
            }

        return {
            "status": "protocol_error",
            "error": f"Unexpected worker message type: {message_type}",
            "traceback": "",
        }


@timer
def main(
    in_csv: str,
    text_col: str,
    encoding: str,
    sep: str,
    out_csv: str | None,
    out_sep: str | None,
    chunk_size: int,
    spacy_batch_size: int,
    spacy_n_process: int,
    prediction_batch_size: int,
    progress_every: int,
    cuda_device: int,
    resume: bool,
    log_level: str,
    checkpoint_dir: str | None,
    snapshot_every_n_chunks: int,
    stop_on_chunk_error: bool,
    chunk_timeout_seconds: int,
    worker_startup_timeout_seconds: int,
    domain_token: bool,
) -> None:
    configure_logging(log_level)

    in_csv_path = Path(in_csv)
    assert in_csv_path.exists(), f'The csv file cannot be found in this location: "{in_csv_path}"'

    output_path = Path(out_csv) if out_csv else in_csv_path.parent / f"{in_csv_path.stem}_output.csv"
    output_sep = out_sep if out_sep is not None else sep
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else output_path.parent / f"{output_path.stem}__checkpoints"

    total_rows = count_data_rows(in_csv_path, encoding)

    ensure_checkpoint_dirs(checkpoint_root)

    manifest_config = {
        "input_file": str(in_csv_path.resolve()),
        "output_file": str(output_path.resolve()),
        "text_col": text_col,
        "encoding": encoding,
        "input_sep": sep,
        "output_sep": output_sep,
        "chunk_size": chunk_size,
    }

    manifest = create_or_load_manifest(
        checkpoint_dir=checkpoint_root,
        config=manifest_config,
        resume=resume,
    )
    validate_manifest_config(manifest, manifest_config)

    processed_chunk_ids = {int(chunk_id) for chunk_id in manifest["processed_chunks"].keys()}
    completed_rows = count_processed_rows(manifest)

    log.info("Input file: %s", in_csv_path)
    log.info("Output file: %s", output_path)
    log.info("Checkpoint directory: %s", checkpoint_root)
    log.info("Total input rows detected: %s", total_rows)
    log.info("Already completed rows from checkpoints: %s", completed_rows)

    if completed_rows >= total_rows and total_rows > 0:
        log.info("All rows already processed according to the checkpoint manifest.")
        assemble_output_from_chunks(
            checkpoint_dir=checkpoint_root,
            output_path=output_path,
            processed_chunk_ids=processed_chunk_ids,
            encoding=encoding,
        )
        return

    runtime_config = {
        "text_col": text_col,
        "spacy_model_name": "nl_core_news_lg",
        "spacy_batch_size": spacy_batch_size,
        "spacy_n_process": spacy_n_process,
        "prediction_batch_size": prediction_batch_size,
        "progress_every": progress_every,
        "cuda_device": cuda_device,
        "worker_startup_timeout_seconds": worker_startup_timeout_seconds,
        "domain_token": domain_token,
    }

    ctx = mp.get_context("spawn")
    worker_process = None
    task_queue = None
    result_queue = None

    try:
        worker_process, task_queue, result_queue = start_worker_process(
            ctx=ctx,
            runtime_config=runtime_config,
            log_level=log_level,
        )

        reader = pd.read_csv(
            in_csv_path,
            sep=sep,
            header=0,
            quotechar='"',
            encoding=encoding,
            low_memory=False,
            chunksize=chunk_size,
        )

        rows_seen = 0
        completed_chunks_this_run = 0

        for chunk_number, chunk in enumerate(reader, start=1):
            chunk_row_count = len(chunk)
            row_start = rows_seen + 1
            row_end = rows_seen + chunk_row_count
            rows_seen += chunk_row_count

            if text_col not in chunk.columns:
                raise KeyError(f'Column "{text_col}" was not found in the input CSV.')

            if chunk_number in processed_chunk_ids:
                log.info(
                    "Skipping already completed chunk %s (rows %s-%s).",
                    chunk_number,
                    row_start,
                    row_end,
                )
                continue

            log.info(
                "Processing chunk %s (rows %s-%s of %s).",
                chunk_number,
                row_start,
                row_end,
                total_rows,
            )

            run_result = run_chunk_with_timeout(
                process=worker_process,
                task_queue=task_queue,
                result_queue=result_queue,
                chunk_number=chunk_number,
                chunk_df=chunk,
                timeout_seconds=chunk_timeout_seconds,
            )

            if run_result["status"] == "success":
                result_chunk = run_result["result_chunk"]
                chunk_stats = run_result["stats"]

                chunk_path = checkpoint_root / "chunks" / f"chunk_{chunk_number:06d}.csv"
                write_chunk_atomic(
                    df=result_chunk,
                    path=chunk_path,
                    sep=output_sep,
                    encoding=encoding,
                )

                manifest = register_processed_chunk(
                    manifest=manifest,
                    chunk_number=chunk_number,
                    row_start=row_start,
                    row_end=row_end,
                    row_count=len(result_chunk),
                    stats=chunk_stats,
                )
                save_manifest(checkpoint_root, manifest)

                processed_chunk_ids.add(chunk_number)
                completed_chunks_this_run += 1

                completed_rows = count_processed_rows(manifest)
                log.info(
                    "Finished chunk %s. Total checkpointed rows: %s/%s (%.2f%%).",
                    chunk_number,
                    completed_rows,
                    total_rows,
                    (completed_rows / total_rows * 100.0) if total_rows else 100.0,
                )

                if snapshot_every_n_chunks > 0 and completed_chunks_this_run % snapshot_every_n_chunks == 0:
                    #LLM: The assembled CSV is periodically rebuilt from completed chunk files.
                    #LLM: That keeps a readable output on disk without making it the source of truth.
                    log.info("Refreshing assembled output snapshot.")
                    assemble_output_from_chunks(
                        checkpoint_dir=checkpoint_root,
                        output_path=output_path,
                        processed_chunk_ids=processed_chunk_ids,
                        encoding=encoding,
                    )

                del result_chunk
                gc.collect()
                continue

            error_message = run_result["error"]
            traceback_text = run_result["traceback"]

            save_failed_input_chunk(
                checkpoint_dir=checkpoint_root,
                chunk_number=chunk_number,
                chunk_df=chunk,
                sep=sep,
                encoding=encoding,
            )

            manifest = register_failed_chunk(
                manifest=manifest,
                chunk_number=chunk_number,
                row_start=row_start,
                row_end=row_end,
                row_count=chunk_row_count,
                error_message=error_message,
                traceback_text=traceback_text,
            )
            save_manifest(checkpoint_root, manifest)

            log.error(
                "Chunk %s failed with status '%s'. Rows %s-%s. Error: %s",
                chunk_number,
                run_result["status"],
                row_start,
                row_end,
                error_message,
            )
            if traceback_text:
                log.error("Traceback for chunk %s:\n%s", chunk_number, traceback_text)

            #LLM: On timeout or worker death, we fully restart the worker so the next chunk
            #LLM: gets a fresh process and a fresh CUDA state.
            stop_worker_process(worker_process, task_queue, result_queue)
            worker_process, task_queue, result_queue = start_worker_process(
                ctx=ctx,
                runtime_config=runtime_config,
                log_level=log_level,
            )

            if stop_on_chunk_error:
                log.error("Stopping because --stop_on_chunk_error was set.")
                break

        log.info("Building final assembled output from completed chunk files.")
        assemble_output_from_chunks(
            checkpoint_dir=checkpoint_root,
            output_path=output_path,
            processed_chunk_ids=processed_chunk_ids,
            encoding=encoding,
        )

    finally:
        stop_worker_process(worker_process, task_queue, result_queue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv", default="./example/input.csv")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--sep", default=";")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--out_sep", default=None)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--spacy_batch_size", type=int, default=128)
    parser.add_argument("--spacy_n_process", type=int, default=1)
    parser.add_argument("--prediction_batch_size", type=int, default=32)
    parser.add_argument("--progress_every", type=int, default=250)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--snapshot_every_n_chunks", type=int, default=5)
    parser.add_argument("--stop_on_chunk_error", action="store_true")
    parser.add_argument("--chunk_timeout_seconds", type=int, default=0)
    parser.add_argument("--worker_startup_timeout_seconds", type=int, default=1800)
    parser.add_argument("--domain_token", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True)

    args = parser.parse_args()

    main(
        in_csv=args.in_csv,
        text_col=args.text_col,
        encoding=args.encoding,
        sep=args.sep,
        out_csv=args.out_csv,
        out_sep=args.out_sep,
        chunk_size=args.chunk_size,
        spacy_batch_size=args.spacy_batch_size,
        spacy_n_process=args.spacy_n_process,
        prediction_batch_size=args.prediction_batch_size,
        progress_every=args.progress_every,
        cuda_device=args.cuda_device,
        resume=args.resume,
        log_level=args.log_level,
        checkpoint_dir=args.checkpoint_dir,
        snapshot_every_n_chunks=args.snapshot_every_n_chunks,
        stop_on_chunk_error=args.stop_on_chunk_error,
        chunk_timeout_seconds=args.chunk_timeout_seconds,
        worker_startup_timeout_seconds=args.worker_startup_timeout_seconds,
        domain_token=args.domain_token,
    )