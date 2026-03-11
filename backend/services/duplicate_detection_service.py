"""
Computes document hash to detect duplicate uploads.
"""

import hashlib


def compute_hash(file_bytes):

    sha = hashlib.sha256()

    sha.update(file_bytes)

    return sha.hexdigest()