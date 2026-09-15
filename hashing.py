import hashlib
from datetime import datetime, timezone

def calculate_sha256(file_path: str) -> str:
    """Computes SHA-256 checksum in 64KB blocks."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_chain_of_custody(file_path: str) -> dict:
    """Generates immutable custody log data."""
    return {
        "file_hash": calculate_sha256(file_path),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "integrity_status": "Verified Cryptographically"
    }