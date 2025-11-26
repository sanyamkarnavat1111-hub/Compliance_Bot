import base64
import json
import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
 
 
def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    """Apply PKCS7 padding to data."""
    padding_len = block_size - (len(data) % block_size)
    return data + bytes([padding_len]) * padding_len
 
 
def pkcs7_unpad(data: bytes) -> bytes:
    """Remove PKCS7 padding from data."""
    padding_len = data[-1]
    if padding_len < 1 or padding_len > 16:
        raise ValueError("Invalid PKCS7 padding.")
    return data[:-padding_len]
 
 
def laravel_decrypt(encrypted_value: str, app_key_base64: str) -> str:
    """Decrypt a Laravel-encrypted string (Crypt::encryptString)."""
    key = base64.b64decode(app_key_base64)
    payload = json.loads(base64.b64decode(encrypted_value))
 
    iv = base64.b64decode(payload['iv'])
    ciphertext = base64.b64decode(payload['value'])
 
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted = cipher.decrypt(ciphertext)
 
    return pkcs7_unpad(decrypted).decode("utf-8")
 
 
def laravel_encrypt(plain_text: str, app_key_base64: str) -> str:
    """Encrypt a string so Laravel can decrypt with Crypt::decryptString."""
    key = base64.b64decode(app_key_base64)
 
    # Step 1: Generate random IV (16 bytes for AES-256-CBC)
    iv = get_random_bytes(16)
 
    # Step 2: Pad plaintext
    padded = pkcs7_pad(plain_text.encode("utf-8"))
 
    # Step 3: Encrypt
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(padded)
 
    # Step 4: Build payload (same as Laravel)
    payload = {
        "iv": base64.b64encode(iv).decode("utf-8"),
        "value": base64.b64encode(ciphertext).decode("utf-8"),
        # Laravel calculates HMAC for tamper check, but it's optional for Crypt::decryptString
        "mac": ""
    }
 
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
 
 
if __name__ == "__main__":
    # ===== CHANGE THESE VALUES =====
    APP_KEY = "base64:jzMfjr8K14jZQDkobv7+zZnfoVSfSxkzWAr1acSQoXw="  
    TEST_STRING = "Hello from Python!"
    
    encrypted_value = laravel_encrypt(TEST_STRING, APP_KEY.replace("base64:", ""))
    print("🔒 Encrypted for Laravel:", encrypted_value)
 
    # Decrypt (round-trip test)
    x = "eyJpdiI6ImhOR3NNdDA0OHNjSVhNaFZNRkRHR0E9PSIsInZhbHVlIjoia3A4dS84bUlYT3ZUZDhUVnA0OFgrU2hmV2I3WjhIRlQ3UVJJN1BkbjR5ZkNNOXJ6a0JPdGFCd2ZZWmc2Um1DV1FPenJMcG5ZZkpQUXFBM085czFva09BOFNjaVc3bkYvZEg2ZnJXMmxOYTBQdk9GNllIdGtoK2NwekJmTGRqNnpNSjA2K0VwTXE2Q2Rxc21UcmwzQWZzU01iZmZxbFZPbDlaamhBNGlPT2dpNHFDWjQ3QzlLbndNV0JHMXFwWjZyIiwibWFjIjoiNDBmMzViMjQ5Yjg1NTk5NDJiMThiODUzOTQ5ZGJhOWI2MTdjMGM4MTQ5ZTMyODM2Yjk4M2NlOWMyMjg3MDVhZCIsInRhZyI6IiJ9"
    decrypted_value = laravel_decrypt(x, APP_KEY.replace("base64:", ""))
    print("🔓 Decrypted back:", decrypted_value)