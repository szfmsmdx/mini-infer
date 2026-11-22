"""Download the Tiny Shakespeare corpus used by nanoGPT's shakespeare_char."""

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen


URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "build" / "gpt-data" / "input.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output

    cached = output.exists()
    if cached:
        data = output.read_bytes()
    else:
        with urlopen(URL, timeout=30) as response:
            data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Tiny Shakespeare hash mismatch: {digest}")

    if not cached:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as directory:
            download = Path(directory) / "input.txt"
            download.write_bytes(data)
            download.replace(output)

    train_tokens = len(data) * 9 // 10
    print(f"Corpus: {output} ({len(data):,} bytes, {len(set(data))} tokens in vocabulary)")
    print(f"SHA-256: {digest}")
    print(f"Train tokens: {train_tokens:,}, validation tokens: {len(data) - train_tokens:,}")


if __name__ == "__main__":
    main()
