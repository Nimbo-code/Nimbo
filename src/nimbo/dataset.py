"""Dataset handling utilities for Nimbo.

Enhanced version of SmoLoRA's dataset module with additional features.
"""

import csv
import json
import logging
import os
from typing import Callable, List, Optional, Union

from datasets import Dataset

logger = logging.getLogger(__name__)


def load_text_data(data_folder: str) -> Dataset:
    """Load text data from a folder containing .txt files into a HuggingFace Dataset.

    Args:
        data_folder: Path to folder containing .txt files

    Returns:
        Dataset with text entries
    """
    texts = read_txt_folder(data_folder)
    data = [{"text": t} for t in texts]
    logger.info(f"Loaded {len(data)} text entries from {data_folder}")
    return Dataset.from_list(data)


def read_txt_folder(folder_path: str) -> List[str]:
    """Read all .txt files in a folder and return a list of lines.

    Args:
        folder_path: Path to the folder containing .txt files

    Returns:
        List of strings, each string is a line from the .txt files
    """
    texts = []
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a directory: {folder_path}")

    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".txt"):
            filepath = os.path.join(folder_path, fname)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    return texts


def read_jsonl(file_path: str, text_field: str = "text") -> List[str]:
    """Read a .jsonl file and extract the specified text field.

    Args:
        file_path: Path to the .jsonl file
        text_field: Field name for text in .jsonl (default: "text")

    Returns:
        List of strings from the specified field
    """
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                if text_field in obj and obj[text_field]:
                    texts.append(str(obj[text_field]).strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    return texts


def read_csv(file_path: str, text_field: str = "text") -> List[str]:
    """Read a .csv file and extract the specified text field.

    Args:
        file_path: Path to the .csv file
        text_field: Field name for text in .csv (default: "text")

    Returns:
        List of strings from the specified field
    """
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if text_field not in (reader.fieldnames or []):
            raise ValueError(
                f"Field '{text_field}' not found. "
                f"Available fields: {reader.fieldnames}"
            )
        for row in reader:
            if text_field in row and row[text_field]:
                texts.append(row[text_field].strip())
    return texts


def read_parquet(file_path: str, text_field: str = "text") -> List[str]:
    """Read a .parquet file and extract the specified text field.

    Args:
        file_path: Path to the .parquet file
        text_field: Field name for text

    Returns:
        List of strings from the specified field
    """
    try:
        import pandas as pd

        df = pd.read_parquet(file_path)
        if text_field not in df.columns:
            raise ValueError(
                f"Field '{text_field}' not found. "
                f"Available fields: {list(df.columns)}"
            )
        return df[text_field].dropna().astype(str).str.strip().tolist()
    except ImportError:
        raise ImportError("pandas and pyarrow required for parquet support")


def chunk_texts(texts: List[str], chunk_size: int = 0) -> List[str]:
    """Split texts into chunks of a specified number of words.

    Args:
        texts: List of strings to be chunked
        chunk_size: Number of words per chunk (0 = no chunking)

    Returns:
        List of chunked strings
    """
    if chunk_size <= 0:
        return texts

    chunks = []
    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk:
                chunks.append(chunk)
    return chunks


def chunk_by_tokens(
    texts: List[str],
    tokenizer: "AutoTokenizer",
    max_tokens: int = 512,
    overlap: int = 0,
) -> List[str]:
    """Split texts into chunks based on token count.

    Args:
        texts: List of strings to be chunked
        tokenizer: HuggingFace tokenizer
        max_tokens: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks

    Returns:
        List of chunked strings
    """
    chunks = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            chunks.append(text)
            continue

        step = max_tokens - overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

    return chunks


def filter_texts(
    texts: List[str],
    min_length: int = 0,
    max_length: int = 0,
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    """Filter texts based on length and custom criteria.

    Args:
        texts: List of strings to filter
        min_length: Minimum character length (0 = no minimum)
        max_length: Maximum character length (0 = no maximum)
        filter_fn: Custom filter function (returns True to keep)

    Returns:
        Filtered list of strings
    """
    filtered = []
    for text in texts:
        if min_length > 0 and len(text) < min_length:
            continue
        if max_length > 0 and len(text) > max_length:
            continue
        if filter_fn is not None and not filter_fn(text):
            continue
        filtered.append(text)

    logger.info(f"Filtered {len(texts)} -> {len(filtered)} texts")
    return filtered


def prepare_dataset(
    source: Union[str, List[str]],
    text_field: str = "text",
    chunk_size: int = 0,
    file_type: Optional[str] = None,
    deduplicate: bool = True,
    min_length: int = 0,
    max_length: int = 0,
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> Dataset:
    """General-purpose dataset preparer.

    Args:
        source: Path to folder/file, or list of text strings
        text_field: Field name for text in .jsonl/.csv/.parquet
        chunk_size: If >0, splits texts into chunks of this many words
        file_type: Force file type: "txt", "jsonl", "csv", "parquet". Auto-detected if None
        deduplicate: Remove duplicate texts
        min_length: Minimum character length filter
        max_length: Maximum character length filter
        filter_fn: Custom filter function

    Returns:
        HuggingFace Dataset with "text" column
    """
    # Handle list input
    if isinstance(source, list):
        texts = source
    else:
        # Auto-detect file type
        if file_type is None:
            if os.path.isdir(source):
                file_type = "txt"
            elif source.endswith(".jsonl"):
                file_type = "jsonl"
            elif source.endswith(".csv"):
                file_type = "csv"
            elif source.endswith(".parquet"):
                file_type = "parquet"
            else:
                raise ValueError(
                    "Cannot infer file type. Please specify file_type parameter."
                )

        # Read texts
        if file_type == "txt":
            texts = read_txt_folder(source)
        elif file_type == "jsonl":
            texts = read_jsonl(source, text_field)
        elif file_type == "csv":
            texts = read_csv(source, text_field)
        elif file_type == "parquet":
            texts = read_parquet(source, text_field)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

    # Apply chunking
    if chunk_size > 0:
        texts = chunk_texts(texts, chunk_size)

    # Apply filtering
    texts = filter_texts(
        texts,
        min_length=min_length,
        max_length=max_length,
        filter_fn=filter_fn,
    )

    # Remove empty texts
    texts = [t for t in texts if t.strip()]

    # Deduplicate
    if deduplicate:
        original_count = len(texts)
        texts = list(dict.fromkeys(texts))
        if len(texts) < original_count:
            logger.info(f"Removed {original_count - len(texts)} duplicate texts")

    data = [{"text": t} for t in texts]
    logger.info(f"Prepared dataset with {len(data)} samples")
    return Dataset.from_list(data)


def prepare_instruction_dataset(
    source: Union[str, List[dict]],
    instruction_field: str = "instruction",
    input_field: str = "input",
    output_field: str = "output",
    template: Optional[str] = None,
) -> Dataset:
    """Prepare instruction-following dataset.

    Args:
        source: Path to .jsonl file or list of dictionaries
        instruction_field: Field name for instruction
        input_field: Field name for input context
        output_field: Field name for expected output
        template: Custom template. Use {instruction}, {input}, {output} placeholders.
                  Default: "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"

    Returns:
        HuggingFace Dataset with "text" column
    """
    if template is None:
        template = (
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        )

    # Load data
    if isinstance(source, str):
        data = []
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        data = source

    # Format texts
    texts = []
    for item in data:
        instruction = item.get(instruction_field, "")
        input_text = item.get(input_field, "")
        output_text = item.get(output_field, "")

        formatted = template.format(
            instruction=instruction,
            input=input_text,
            output=output_text,
        )
        texts.append({"text": formatted})

    logger.info(f"Prepared instruction dataset with {len(texts)} samples")
    return Dataset.from_list(texts)


def prepare_chat_dataset(
    source: Union[str, List[dict]],
    messages_field: str = "messages",
    tokenizer: Optional["AutoTokenizer"] = None,
) -> Dataset:
    """Prepare chat/conversation dataset.

    Args:
        source: Path to .jsonl file or list of conversations
        messages_field: Field name for messages list
        tokenizer: Tokenizer with chat template (optional)

    Returns:
        HuggingFace Dataset with "text" column
    """
    # Load data
    if isinstance(source, str):
        data = []
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        data = source

    texts = []
    for item in data:
        messages = item.get(messages_field, [])

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            # Use tokenizer's chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Simple fallback formatting
            parts = []
            for msg in messages:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            text = "\n\n".join(parts)

        texts.append({"text": text})

    logger.info(f"Prepared chat dataset with {len(texts)} samples")
    return Dataset.from_list(texts)
