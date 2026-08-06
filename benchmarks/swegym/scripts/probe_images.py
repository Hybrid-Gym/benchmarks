"""Record which SWE-Gym instances have a published xingyaoww image on Docker Hub.

A missing image is an instance no model can ever run, so it must be excluded from the
shared 1500 selection -- otherwise all four rollouts burn retries on the same dead
instances and none of them can reach the supervisor's completion target.

Uses hub.docker.com's repository API, NOT a registry manifest fetch: manifest reads
count against the 200/hr Docker Hub pull budget that the live evals depend on.

Resumable -- results are flushed to the cache file as they land, so a re-run only
probes what is still unknown.
"""

import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, cast

from datasets import load_dataset

from benchmarks.swegym.build_images import get_official_docker_image


if TYPE_CHECKING:
    from datasets import Dataset


CACHE = "eval_outputs/swegym_image_availability.json"


def check(iid: str, tries: int = 4) -> tuple[str, bool | None]:
    name = get_official_docker_image(iid).split("docker.io/")[-1].rsplit(":", 1)[0]
    url = f"https://hub.docker.com/v2/repositories/{name}/tags/latest"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return iid, r.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return iid, False
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return iid, None


def main() -> None:
    ds = cast("Dataset", load_dataset("SWE-Gym/SWE-Gym", split="train"))
    ids = sorted(cast("list[str]", ds["instance_id"]))
    known: dict[str, bool | None] = {}
    if os.path.exists(CACHE):
        known = json.load(open(CACHE))
    todo = [i for i in ids if known.get(i) is None]
    print(
        f"{len(ids)} instances, {len(ids) - len(todo)} cached, {len(todo)} to probe",
        flush=True,
    )

    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(6) as ex:
        for iid, ok in ex.map(check, todo):
            known[iid] = ok
            done += 1
            if done % 100 == 0:
                json.dump(known, open(CACHE, "w"), indent=0)
                rate = done / (time.time() - t0)
                print(
                    f"  {done}/{len(todo)}  {rate:.1f}/s  "
                    f"eta {(len(todo) - done) / max(rate, 1e-9) / 60:.1f}m",
                    flush=True,
                )

    json.dump(known, open(CACHE, "w"), indent=0)
    present = [i for i, v in known.items() if v is True]
    missing = [i for i, v in known.items() if v is False]
    unknown = [i for i, v in known.items() if v is None]
    print(
        f"DONE present={len(present)} missing={len(missing)} unknown={len(unknown)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
