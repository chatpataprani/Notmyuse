import os
import re
import time
import secrets
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfFileSystem

# Public Hugging Face Storage Bucket
BUCKET = os.getenv(
    "HF_BUCKET",
    "buckets/CutehackX/icrm-hitek-full-db-mixed-bucket",
)

HF_TOKEN = os.getenv("HF_TOKEN") or None
API_KEY = os.getenv("API_KEY", "")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "25"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4096"))
DEVELOPER = "chatpataprani"

PHONE_RE = re.compile(r"^\d{10,15}$")
AADHAAR_RE = re.compile(r"^\d{12}$")

app = FastAPI(
    title="ICMR/HITEK Fast Lookup API",
    version="3.1",
    description="Fast Parquet index lookup API — Developer: chatpataprani",
)

# HfFileSystem handles hf:// buckets through fsspec.
# IMPORTANT: glob() returns filesystem-relative paths such as
# buckets/CutehackX/... rather than hf:// URLs. PyArrow receives those
# paths together with the HfFileSystem object.
fs = HfFileSystem(token=HF_TOKEN)

PHONE_FILES = sorted(
    fs.glob(f"{BUCKET}/idx_phone.*.parquet")
)

AADHAAR_FILES = sorted(
    fs.glob(f"{BUCKET}/idx_aadhaar.*.parquet")
)

# Support either spelling if the bucket uses idx_aadhar.*.parquet.
if not AADHAAR_FILES:
    AADHAAR_FILES = sorted(
        fs.glob(f"{BUCKET}/idx_aadhar.*.parquet")
    )

if not PHONE_FILES:
    raise RuntimeError(
        "No idx_phone.*.parquet files were found in the Hugging Face bucket."
    )

if not AADHAAR_FILES:
    raise RuntimeError(
        "No idx_aadhaar.*.parquet or idx_aadhar.*.parquet files were found "
        "in the Hugging Face bucket."
    )

# Build Arrow datasets directly on top of HfFileSystem.
# This avoids DuckDB's current rejection of hf://buckets URLs.
phone_dataset = ds.dataset(
    PHONE_FILES,
    filesystem=fs,
    format="parquet",
)

aadhaar_dataset = ds.dataset(
    AADHAAR_FILES,
    filesystem=fs,
    format="parquet",
)


def _verify_api_key(key: str) -> None:
    """Require the URL key to match the API_KEY environment variable."""
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="API key is not configured on the server.",
        )

    if not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )


def _column_type(dataset: ds.Dataset, column: str) -> pa.DataType:
    try:
        return dataset.schema.field(column).type
    except KeyError:
        raise RuntimeError(
            f"Required index column '{column}' was not found. "
            f"Available columns: {dataset.schema.names}"
        )


def _typed_value(dataset: ds.Dataset, column: str, value: str) -> Any:
    """
    Convert the incoming string to the Parquet column's actual type.
    This allows the same code to work whether the index stores the
    identifier as a string or an integer.
    """
    dtype = _column_type(dataset, column)

    if pa.types.is_integer(dtype):
        try:
            return int(value)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"{column} is numeric but the supplied value is invalid.",
            )

    if pa.types.is_floating(dtype):
        try:
            return float(value)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"{column} is numeric but the supplied value is invalid.",
            )

    return value


def _lookup(
    dataset: ds.Dataset,
    column: str,
    value: str,
) -> dict[str, Any]:
    started = time.perf_counter()

    typed = _typed_value(dataset, column, value)

    # Dataset filtering enables Parquet predicate pushdown and row-group
    # pruning, which is important for the large sorted index files.
    filter_expr = ds.field(column) == typed

    rows: list[dict[str, Any]] = []

    scanner = dataset.scanner(
        filter=filter_expr,
        batch_size=BATCH_SIZE,
        use_threads=True,
    )

    for batch in scanner.to_batches():
        batch_rows = batch.to_pylist()

        remaining = MAX_RESULTS - len(rows)
        if remaining <= 0:
            break

        rows.extend(batch_rows[:remaining])

        if len(rows) >= MAX_RESULTS:
            break

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    return {
        "results": rows,
        "count": len(rows),
        "lookup_ms": elapsed_ms,
    }


@app.get("/")
def root():
    return {
        "status": "online",
        "developer": DEVELOPER,
        "authentication": "API key required for lookup endpoints",
        "endpoints": {
            "number": "/key=<API_KEY>/number=<10-15 digit number>",
            "aadhar": "/key=<API_KEY>/aadhar=<12 digit Aadhaar>",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "developer": DEVELOPER,
    }


@app.get("/key={key}/number={number}")
def number_lookup(key: str, number: str):
    _verify_api_key(key)

    number = number.strip()

    if not PHONE_RE.fullmatch(number):
        raise HTTPException(
            status_code=400,
            detail="Number must contain 10 to 15 digits.",
        )

    data = _lookup(
        phone_dataset,
        "phoneNumber",
        number,
    )

    return {
        "status": "success" if data["count"] else "not_found",
        "developer": DEVELOPER,
        "type": "number",
        "number": number,
        **data,
    }


@app.get("/key={key}/aadhar={aadhar}")
def aadhar_lookup(key: str, aadhar: str):
    _verify_api_key(key)

    aadhar = aadhar.strip()

    if not AADHAAR_RE.fullmatch(aadhar):
        raise HTTPException(
            status_code=400,
            detail="Aadhaar value must contain exactly 12 digits.",
        )

    data = _lookup(
        aadhaar_dataset,
        "aadharNumber",
        aadhar,
    )

    return {
        "status": "success" if data["count"] else "not_found",
        "developer": DEVELOPER,
        "type": "aadhar",
        "aadhar": aadhar,
        **data,
    }
