"""
Persistent worker process for chunk inference (icf_17).
"""

import logging
import traceback

import numpy as np
import pandas as pd
import spacy
import torch

from src.icf_classifiers import load_model, predict_domains, predict_levels, domain_token_map
from src.text_processing import preprocess_notes

log = logging.getLogger(__name__)

#TODO: FAC is D450
DOMAINS = ['ENR', 'ATT', 'STM', 'ADM', 'INS', 'MBW', 'FAC', 'ETN', 'BER', 'SOP', 'SLP', 'FML', 'HLC', 'MAE', 'CBP', 'HRN', 'HSP']
DOMAIN_CODES = ['B1300', 'B140', 'B152', 'B440', 'B455', 'B530', 'D450', 'D550', 'D840-D859', 'B280', 'B134', 'D760', 'B164', 'D465', 'D410', 'B230', 'D240']
LEVEL_COLUMNS = [f"{domain}-{code}_lvl" for domain, code in zip(DOMAINS, DOMAIN_CODES)]


def configure_worker_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s",
    )


def load_all_models(prediction_batch_size: int, cuda_device: int, domain_token: bool):
    log.info("Loading domains model and levels model inside worker.")

    # Using icf17 models
    domain_model = load_model(
        "roberta",
        "CLTL/icf17-domains",
        "multi",
    )

    if domain_token:
        level_model = load_model(
            "roberta",
            "CLTL/icf17-levels-domain-token",
            "clf",
        )
    else:
        level_model = load_model(
            "roberta",
            "CLTL/icf17-levels",
            "clf",
        )

    return domain_model, level_model


def add_level_predictions(
    sents: pd.DataFrame,
    level_model: object,
    domain_token: bool,
) -> pd.DataFrame:
    if sents.empty:
        for column in LEVEL_COLUMNS:
            sents[column] = pd.Series(dtype="float64")
        return sents

    predictions_matrix = np.asarray(sents["predictions"].tolist(), dtype=np.uint8)

    for column in LEVEL_COLUMNS:
        sents[column] = np.nan

    if not domain_token:
        # Without domain token, we can just predict once per sentence and apply it to each matched domain
        levels_no_domain = predict_levels(sents["text"], level_model)
        for domain_index, domain in enumerate(DOMAINS):
            mask = predictions_matrix[:, domain_index].astype(bool)
            sentence_count = int(mask.sum())
            if sentence_count == 0:
                continue
                
            sentence_indices = sents.index[mask]
            column = LEVEL_COLUMNS[domain_index]
            sents.loc[sentence_indices, column] = levels_no_domain.loc[sentence_indices].astype(float).to_numpy()
    else:
        # With domain token, we prepend the token before inference per domain match 
        for domain_index, domain in enumerate(DOMAINS):
            column = LEVEL_COLUMNS[domain_index]
            mask = predictions_matrix[:, domain_index].astype(bool)
            sentence_count = int(mask.sum())

            if sentence_count == 0:
                if sentence_count > 0:
                    log.info("No sentences predicted for domain %s in this chunk.", domain)
                continue

            log.info(
                "Predicting levels for domain %s on %s sentences.",
                domain,
                sentence_count,
            )

            sentence_indices = sents.index[mask]
            token = domain_token_map[domain]
            
            modified_texts = token + " " + sents.loc[sentence_indices, "text"].astype(str)
            domain_predictions = predict_levels(
                modified_texts,
                level_model,
            )

            sents.loc[domain_predictions.index, column] = domain_predictions.astype(float).to_numpy()

    return sents


def process_chunk(
    chunk: pd.DataFrame,
    text_col: str,
    nlp,
    domain_model,
    level_model,
    domain_token: bool,
    spacy_batch_size: int,
    spacy_n_process: int,
    progress_every: int,
) -> tuple[pd.DataFrame, dict]:
    result_chunk = chunk.copy()

    for column in LEVEL_COLUMNS:
        if column not in result_chunk.columns:
            result_chunk[column] = np.nan

    valid_mask = result_chunk[text_col].notna() & result_chunk[text_col].astype(str).str.strip().ne("")
    valid_row_count = int(valid_mask.sum())

    stats = {
        "input_rows": int(len(chunk)),
        "valid_text_rows": valid_row_count,
        "sentence_count": 0,
    }

    log.info("Chunk contains %s rows with usable text.", valid_row_count)

    if valid_row_count == 0:
        return result_chunk, stats

    working_chunk = result_chunk.loc[valid_mask].copy()

    sents = preprocess_notes(
        working_chunk[text_col],
        nlp=nlp,
        batch_size=spacy_batch_size,
        n_process=spacy_n_process,
        log_every=progress_every,
    )
    stats["sentence_count"] = int(len(sents))
    log.info("spaCy preprocessing produced %s sentences.", len(sents))

    if sents.empty:
        return result_chunk, stats

    sents["predictions"] = predict_domains(sents["text"], domain_model)
    sents = add_level_predictions(sents, level_model, domain_token)

    note_predictions = sents.groupby("note_index")[LEVEL_COLUMNS].mean()
    result_chunk.loc[note_predictions.index, LEVEL_COLUMNS] = note_predictions[LEVEL_COLUMNS]

    return result_chunk, stats


def worker_main(task_queue, result_queue, runtime_config: dict, log_level: str = "INFO") -> None:
    configure_worker_logging(log_level)

    try:
        text_col = runtime_config["text_col"]
        spacy_model_name = runtime_config["spacy_model_name"]
        spacy_batch_size = runtime_config["spacy_batch_size"]
        spacy_n_process = runtime_config["spacy_n_process"]
        prediction_batch_size = runtime_config["prediction_batch_size"]
        progress_every = runtime_config["progress_every"]
        cuda_device = runtime_config["cuda_device"]
        domain_token = runtime_config.get("domain_token", True)

        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_device)
            log.info("CUDA is available in worker. Using GPU device %s.", cuda_device)
        else:
            log.warning("CUDA is not available in worker. Running on CPU.")

        log.info("Loading spaCy model in worker: %s", spacy_model_name)
        nlp = spacy.load(spacy_model_name)

        domain_model, level_model = load_all_models(
            prediction_batch_size=prediction_batch_size,
            cuda_device=cuda_device,
            domain_token=domain_token,
        )

        result_queue.put({"type": "ready"})

    except Exception as exc:
        result_queue.put(
            {
                "type": "startup_error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        message = task_queue.get()
        message_type = message.get("type")

        if message_type == "shutdown":
            log.info("Worker received shutdown signal.")
            return

        if message_type != "process_chunk":
            result_queue.put(
                {
                    "type": "chunk_error",
                    "chunk_number": message.get("chunk_number"),
                    "error": f"Unknown message type: {message_type}",
                    "traceback": "",
                }
            )
            continue

        chunk_number = message["chunk_number"]
        chunk_df = message["chunk_df"]

        try:
            result_chunk, stats = process_chunk(
                chunk=chunk_df,
                text_col=text_col,
                nlp=nlp,
                domain_model=domain_model,
                level_model=level_model,
                domain_token=domain_token,
                spacy_batch_size=spacy_batch_size,
                spacy_n_process=spacy_n_process,
                progress_every=progress_every,
            )

            result_queue.put(
                {
                    "type": "result",
                    "chunk_number": chunk_number,
                    "result_chunk": result_chunk,
                    "stats": stats,
                }
            )

        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            result_queue.put(
                {
                    "type": "chunk_error",
                    "chunk_number": chunk_number,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
