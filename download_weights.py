"""Download and verify the pretrained CoForma weights."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

URL = "https://github.com/Dseperater/CoForma/releases/download/v1.0.0/coforma.pt"
SHA256 = "f20353c6751ce59e286ae887ad7bf8c8da182a3fe3e8b11a5c937417a295f085"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights/coforma.pt"))
    args = parser.parse_args()
    if args.output.exists():
        if sha256(args.output) != SHA256:
            parser.error("Existing file has the wrong checksum; choose another output path.")
        print(f"Weights already verified: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with urllib.request.urlopen(URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if sha256(temporary) != SHA256:
            raise RuntimeError("Downloaded weights failed SHA-256 verification.")
        temporary.replace(args.output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Downloaded and verified: {args.output}")


if __name__ == "__main__":
    main()
