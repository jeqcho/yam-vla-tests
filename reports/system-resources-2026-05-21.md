# System Resources — 2026-05-21

## Disk

| Mount | Size | Used | Avail | Use% |
|---|---|---|---|---|
| `/` (nvme0n1p5) | 1.2 T | 800 G | 286 G | **74%** |
| `/boot/efi` | 196 M | 38 M | 159 M | 19% |

Root has 286 GB free. Not critical yet, but trending — main growth is under `/home/andon` (628 G).

### Top home consumers

| Path | Size |
|---|---|
| `forecasting-physical-automation` | 124 G |
| `Documents` | 117 G |
| **`yam-tests`** | **91 G** |
| `andon` | 79 G |
| `andon-agent` | 54 G |
| `localize` | 32 G |
| `trlc-dk1` | 30 G |
| `all` | 29 G |
| `1k` | 23 G |
| `miniforge3` | 8.9 G |

### Inside `yam-tests` (91 G)

| Path | Size |
|---|---|
| `molmoact2-setup` | 54 G |
| `grootn1.7 exploration` | 35 G |
| `i2rt` | 1.5 G |
| `dreamzero exploration` | 366 M |

`molmoact2-setup` + `grootn1.7 exploration` account for ~98% of the project. Likely model weights / venvs — candidates for cleanup if those experiments are inactive.

## RAM

| | Total | Used | Avail |
|---|---|---|---|
| Mem | 30 GiB | **25 GiB** | 5.7 GiB |
| Swap | 8 GiB | **4.2 GiB** | 3.8 GiB |

Memory is saturated — actively swapping 4.2 GiB. This degrades latency for any memory-touching workload.

### Top processes

| PID | RSS | Cmd |
|---|---|---|
| 267470 | 1.4 G | `python experimental/rtc/host_server_rtc.py --port 8203 --dtype bfloat16` (running 6m49s) |
| 268955 | 280 M | claude |
| 263758 | 270 M | claude |
| 255301 | 254 M | code |
| 266180 | 233 M | claude |

Three concurrent `claude` processes (~780 M combined) plus VS Code (~600 M across helpers). The RTC host server is the single biggest process but not the main pressure source — the aggregate of dev tooling is.

## GPU (RTX 5090)

| Total | Used | Free | Util |
|---|---|---|---|
| 32.6 GB | 13.1 GB | 19.0 GB | 0% |

13 GB resident, idle. Almost certainly the RTC host server's bfloat16 model. Plenty of headroom.

## Actions worth considering

1. **Free swap pressure**: close one or two of the three `claude` sessions if not in use; reclaim ~500 M.
2. **Disk**: if `molmoact2-setup` (54 G) and `grootn1.7 exploration` (35 G) are stale, archiving them frees ~89 G — roughly 7% of the root partition.
3. **No urgent action** — system is functional, just tight on RAM.
