"""
Functions used in pre-processing of data for the machine learning pipelines.
"""

import logging

import pandas as pd

from src import timer


log = logging.getLogger(__name__)
ANONYMIZE_LABELS = {"PERSON", "GPE"}


def _replace_entities_with_labels(text: str, entities: list[tuple[int, int, str]]) -> str:
    if not entities:
        return text

    parts: list[str] = []
    cursor = 0

    for start_char, end_char, label in sorted(entities, key=lambda item: item[0]):
        if start_char < cursor:
            continue
        parts.append(text[cursor:start_char])
        parts.append(label)
        cursor = end_char

    parts.append(text[cursor:])
    return "".join(parts)


def anonymize_text(txt, nlp):
    """
    Replace entities of type PERSON and GPE with 'PERSON', 'GPE'.
    Return anonymized text.
    """
    doc = nlp(txt)
    entities = [
        (ent.start_char, ent.end_char, ent.label_)
        for ent in doc.ents
        if ent.label_ in ANONYMIZE_LABELS
    ]
    return _replace_entities_with_labels(doc.text, entities)


def _sentence_entities(doc, sent):
    sent_start = sent.start_char
    sent_end = sent.end_char
    return [
        (ent.start_char - sent_start, ent.end_char - sent_start, ent.label_)
        for ent in doc.ents
        if ent.label_ in ANONYMIZE_LABELS and ent.start_char >= sent_start and ent.end_char <= sent_end
    ]


def document_to_anonymized_sentences(doc) -> list[str]:
    #combine both Spacy steps in a single pass
    sentences: list[str] = []
    for sent in doc.sents:
        entities = _sentence_entities(doc, sent)
        sentence_text = _replace_entities_with_labels(sent.text, entities).strip()
        if sentence_text:
            sentences.append(sentence_text)
    return sentences


def _pipes_to_disable(nlp) -> list[str]:
    #remove unrequired pipeline components
    keep = {"ner", "tok2vec", "transformer"}

    for candidate in ("sentencizer", "senter", "parser"):
        if candidate in nlp.pipe_names:
            keep.add(candidate)
            break

    return [pipe_name for pipe_name in nlp.pipe_names if pipe_name not in keep]


@timer
def preprocess_notes(
    notes: pd.Series,
    nlp,
    batch_size: int = 128,
    n_process: int = 1,
    log_every: int = 1000,
) -> pd.DataFrame:
    #removed pandas.apply with reloads the spacy model on each call
    disable_pipes = _pipes_to_disable(nlp)
    records: list[tuple[int, str]] = []

    stream = ((str(text), note_index) for note_index, text in notes.items())

    for processed_count, (doc, note_index) in enumerate(
        nlp.pipe(
            stream,
            as_tuples=True,
            batch_size=batch_size,
            n_process=n_process,
            disable=disable_pipes,
        ),
        start=1,
    ):
        sentences = document_to_anonymized_sentences(doc)
        records.extend((note_index, sentence) for sentence in sentences)

        if log_every and processed_count % log_every == 0:
            log.info(
                "spaCy preprocessing progress: %s notes -> %s sentences",
                processed_count,
                len(records),
            )

    return pd.DataFrame(records, columns=["note_index", "text"])


@timer
def anonymize(notes, nlp):
    anonymize_one = lambda text: anonymize_text(text, nlp)
    return notes.apply(anonymize_one).rename("anonym_text")


@timer
def split_sents(notes, nlp):
    to_sentence = lambda txt: [str(sent) for sent in nlp(txt).sents]
    sents = (
        notes.apply(to_sentence)
        .explode()
        .rename("text")
        .reset_index()
        .rename(columns={"index": "note_index"})
    )
    return sents