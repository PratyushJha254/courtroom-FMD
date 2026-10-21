"""Count the rows on which two or more submission files disagree.

Example:
    python compare_submissions.py submissions/a.csv submissions/b.csv --write-ids disputed.txt
"""
import argparse

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--write-ids", help="write the disputed ids here, one per line")
    args = p.parse_args()
    if len(args.files) < 2:
        raise SystemExit("need at least two files")

    labels = pd.concat(
        [pd.read_csv(f).set_index("id")["label"].astype(str).rename(f) for f in args.files], axis=1
    )
    disputed = labels[labels.nunique(axis=1) > 1]
    n = len(labels)
    print(f"{n} rows, {len(disputed)} disputed, {100 * (n - len(disputed)) / n:.2f}% agreement")
    if len(disputed):
        print(disputed.head(20).to_string())

    if args.write_ids:
        with open(args.write_ids, "w") as f:
            f.writelines(f"{i}\n" for i in disputed.index)


if __name__ == "__main__":
    main()
