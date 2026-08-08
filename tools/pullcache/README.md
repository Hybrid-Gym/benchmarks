# pullcache — Docker Hub pull-through cache

Docker Hub allows the `gaokaiz2` account **200 pulls/hr** (anonymous would be 100/hr
keyed to this box's IP and shared with the other tenant). The R2E-Gym eval starts a
container per instance and deletes the image immediately after, so it pulls ~340
images/hr — above the ceiling. Every retry, top-up, or disk-pressure re-fetch used to
cost another pull.

A local pull-through cache makes repeat pulls free: only a cache **miss** goes
upstream.

## Setup

```bash
docker run -d --name pullcache --restart unless-stopped \
  -p 127.0.0.1:5000:5000 \
  --env-file ~/.pullcache.env \
  -v /home/gaokaizhang/pullcache:/var/lib/registry \
  registry:2
```

`~/.pullcache.env` (mode 0600, never committed) holds
`REGISTRY_PROXY_REMOTEURL=https://registry-1.docker.io` plus the Docker Hub username
and password, so upstream misses are fetched as `gaokaiz2` and count against the
200/hr account budget rather than the 100/hr anonymous IP budget.

Bind to **127.0.0.1 only** — this box is shared, and an open registry would be
reachable by other tenants.

## Using it

The daemon is *not* configured with `registry-mirrors`, because that needs a dockerd
restart, which kills every running container on the box including another user's.
Instead, callers pull through the cache explicitly and retag to the canonical name, so
the image is found locally and `docker run` never pulls:

```bash
docker pull 127.0.0.1:5000/namanjain12/foo:tag
docker tag  127.0.0.1:5000/namanjain12/foo:tag namanjain12/foo:tag
docker rmi  127.0.0.1:5000/namanjain12/foo:tag   # drops the mirror tag only
```

`benchmarks/r2egym/scripts/batch_eval.sh` does this per batch (`prefetch_batch`).

## Size cap

`cache_guard.sh` (tmux `pullcache-guard`) caps the cache at `CAP_GB` (default 300G).
registry:2 has no size limit — it expires proxied blobs on `REGISTRY_PROXY_TTL` but
nothing bounds the total. The cache is pure derived data, so on exceeding the cap the
guard wipes it and lets it refill rather than attempting an LRU over blobs, which
would need to walk the manifest graph to avoid orphaning layers.

```bash
tmux new-session -d -s pullcache-guard "bash tools/pullcache/cache_guard.sh"
CAP_GB=500 INTERVAL=600 bash tools/pullcache/cache_guard.sh
```
