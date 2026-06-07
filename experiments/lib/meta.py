"""Run-identity helper for the experiment driver.

Two ids per run:
  * run_uid    - UNIQUE per execution: <exp_id>.<UTC timestamp>.<rand6>.
                 Distinguishes every run, including reruns of the same config.
  * config_hash- DETERMINISTIC over the scientific parameters: identical
                 params -> identical hash. Group reruns by config_hash to
                 measure reproducibility (same config, different run_uid).

Usage:
    python experiments/lib/meta.py uid  <exp_id>
    python experiments/lib/meta.py hash <key=val> <key=val> ...
"""
import hashlib
import sys
import time
import uuid


def main():
    cmd = sys.argv[1]
    if cmd == "uid":
        exp_id = sys.argv[2]
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        print(f"{exp_id}.{stamp}.{uuid.uuid4().hex[:6]}")
    elif cmd == "hash":
        # Sort so argument order never changes the hash; hash the params only
        # (no timestamps / uids), so a true rerun reproduces the same hash.
        params = sorted(a for a in sys.argv[2:] if a.strip())
        digest = hashlib.sha1("\n".join(params).encode()).hexdigest()[:12]
        print(digest)
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
