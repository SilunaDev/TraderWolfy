"""
ASTRA FUSION QUANT — Build Tools
"""
import hashlib
import json
import os
import sys

def check_parity_hash(filepath: str, expected_hash: str) -> bool:
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return False
    
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
        
    actual = hasher.hexdigest()
    if actual == expected_hash:
        print(f"PASS: {filepath} hash matches expected.")
        return True
    else:
        print(f"FAIL: {filepath} hash mismatch.\nExpected: {expected_hash}\nActual:   {actual}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 2:
        check_parity_hash(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python build.py <file> <expected_sha256>")
