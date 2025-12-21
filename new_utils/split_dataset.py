import argparse
import logging
from pathlib import Path

import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def write_first_tokens(input_path: str, output_path: str, num_tokens: int, dtype: str) -> None:
    """Load a memmap file of given dtype, take the first num_tokens, and write them as a new memmap.

    Validates that num_tokens is > 0 and not greater than the total number of tokens in the file.
    """
    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    dtype_np = np.dtype(dtype)
    data = np.memmap(str(input_path_obj), dtype=dtype_np, mode='r')
    total_len = len(data)

    if total_len == 0:
        raise ValueError("Input memmap file is empty.")
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
    if num_tokens > total_len:
        raise ValueError(
            f"Requested num_tokens ({num_tokens}) exceeds file length ({total_len})."
        )

    log.info("Writing first %d/%d tokens from %s to %s (dtype=%s)", num_tokens, total_len, input_path_obj, output_path_obj, dtype_np)

    out_mm = np.memmap(str(output_path_obj), dtype=dtype_np, mode='w+', shape=(num_tokens,))
    out_mm[:] = np.asarray(data[:num_tokens], dtype=dtype_np)
    out_mm.flush()
    del out_mm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Take the first N tokens from a memmap file and write them back as a new memmap.\n\n"
            "Example:\n"
            "  python new_utils/split_dataset.py --input_path=/path/to/input.npy --output_path=/path/to/output_head.npy --num_tokens=1000000\n"
            "  python new_utils/split_dataset.py --input_path=... --output_path=... --num_tokens=... --dtype=int32"
        ),
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to input memmap file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to write the output memmap file")
    parser.add_argument("--num_tokens", type=int, required=True, help="Number of tokens to keep from the start")
    parser.add_argument("--dtype", type=str, default="uint32", help="Data type of memmap (default: uint32)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_first_tokens(args.input_path, args.output_path, args.num_tokens, args.dtype)


if __name__ == "__main__":
    main()


