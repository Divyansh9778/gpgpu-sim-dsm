# Accel-Sim / GPGPU-Sim — Distributed Shared Memory (DSM) Port

This is a fork of Accel-Sim/GPGPU-Sim that adds support for Hopper-style
thread-block clusters and Distributed Shared Memory (DSM), ported from a
research fork called ClusterSim. The upstream project's own README is
preserved unchanged at [`UPSTREAM_README.md`](./UPSTREAM_README.md).

---

## Background

On real Hopper GPUs, a kernel can group several thread blocks into a
*cluster* and let those blocks read and write each other's `__shared__`
memory directly, over an on-chip interconnect, instead of going through
L2 or DRAM. Baseline GPGPU-Sim has no model for this: shared memory is
allocated per SM and a block can only ever see the shared memory of the
SM it happens to be running on. There is no code path for a shared-memory
address that resolves to a *different* SM, and no interconnect model
between SMs to carry such a request if one existed. This port adds both:
a way to resolve a shared-memory address to the SM that owns it, and a
timing model for the request that has to travel there.

---

## What changed

| Layer | What was added | Key files |
|---|---|---|
| PTX front-end | Lexer and grammar support for cluster syntax (`.explicitcluster`, `.reqnctapercluster`, cluster special registers); two new opcodes, `mapa` and `getctarank` | `src/cuda-sim/ptx.l`, `ptx.y`, `opcodes.def`, `ptx_ir.cc/.h` |
| Functional cluster model | `ptx_cluster_info` class tracking which CTAs belong to a cluster; `decode_space()` and the atomics path resolve a shared address to its owning CTA/SM when it falls outside the executing SM's own window | `src/cuda-sim/ptx_sim.cc/.h`, `src/cuda-sim/instructions.cc` |
| CUDA runtime shim | `cudaLaunchKernelEx` reads the cluster dimension from the launch config and threads it into `kernel_info_t` | `libcuda/cuda_runtime_api.cc`, `src/abstract_hardware_model.h/.cc` |
| Timing infrastructure | SM-to-SM network model (`Crossbar`, `Ringbus`, `IdealNetwork`) and the `cluster_shmem_request` message class; SM2SM clock domain; one network per GPC. Remote requests are pushed into the network as an instruction enters the shared-memory pipeline and checked for completion at writeback, so independent remote accesses overlap | `src/gpgpu-sim/sm_2_sm_network.cc/.h`, `src/gpgpu-sim/gpu-sim.cc/.h`, `src/gpgpu-sim/shader.cc` |
| Cluster construction & barriers | Cluster-aware CTA iterator so a cluster's blocks are issued together; `barrier.cluster.arrive`/`wait` support | `src/gpgpu-sim/shader.cc/.h`, `src/cuda-sim/instructions.cc` (`bar_impl`) |
| GPC model & cluster placement | `gpu_processing_cluster` groups SMs (set with `-gpgpu_n_cores_per_gpc`) and owns its SM-to-SM network; a cross-GPC guard aborts with a clear error if a request leaves its GPC; GPC-aware issue places all CTAs of one cluster inside a single GPC (opt-in) | `src/gpgpu-sim/shader.cc/.h`, `src/gpgpu-sim/gpu-sim.cc/.h`, `src/gpgpu-sim/sm_2_sm_network.cc/.h` |
| Statistics | `gpgpu_n_cluster_remote_requests` and `gpgpu_n_cluster_remote_atomic_requests`: warp-level SM-to-SM requests pushed into the network | `src/gpgpu-sim/shader.cc/.h` |

---

## Bugs found and fixed

**1. Wrong SM-id field in address resolution.**
`ptx_cta_info::is_in_generic_shared_memory()` checked a shared address
against `m_sm_idx`, but `mapa_impl()` addresses against `m_shader_id`. A
correctly resolved remote address could fail the check and read garbage.
Fixed to use `m_shader_id`. (`src/cuda-sim/ptx_sim.cc`)

**2. 32-bit truncation of remote shared addresses.**
`ld_exec()` read the source address as `src1_data.u32`; remote cluster
addresses can exceed 32 bits. Widened to `.u64`.
(`src/cuda-sim/instructions.cc`)

**3. Cluster atomics resolved against the wrong SM.**
`atom_callback()` and `atom_impl()` called `generic_to_shared()` against
the executing SM's own id even for remote addresses. Fixed to mirror
`decode_space()`: check `isspace_shared()` first, otherwise resolve the
owning CTA via `ptx_cluster_info`. (`src/cuda-sim/instructions.cc`)

**4. The SM-to-SM network was implemented but never connected.**
Nothing instantiated or advanced a network, so every remote request
bypassed it. Added a `create_sm2sm_network()` factory, connected
`ldst_unit` to push requests into it, and connected the SM2SM clock tick.

**5. Local shared atomics left the target SM id unset.**
The atomics path set `m_last_shared_memory_target_shader_id` only for
remote addresses. A local atomic left it at `UINT_MAX`, so it was
classified as a remote request to SM 4294967295 and the cross-GPC guard
aborted. It is now set to the executing SM. (`src/cuda-sim/instructions.cc`)

**6. Remote atomics updated the wrong CTA's shared memory.**
`atom_callback()` resolved the target CTA but still read and wrote the
executing thread's own `m_shared_mem`. It now uses the target CTA's shared
memory. `Histogram.Cluster` went from 1,225 mismatches to 0.
(`src/cuda-sim/instructions.cc`)

**7. `ideal` and `ringbus` networks never completed requests.**
Only `Crossbar::Advance()` set `cluster_shmem_request::complete`, so any
remote access over `ideal` or `ringbus` deadlocked. Both now mark requests
complete on delivery. (`src/gpgpu-sim/sm_2_sm_network.cc`)

**8. Remote accesses blocked the LD/ST unit for a full round trip.**
The original hook (from fix 4) held `ldst_unit::shared_cycle()` until every
remote request of an instruction completed, so one network round trip was
paid serially per warp instruction. ClusterSim instead lets the instruction
enter the shared-memory pipeline and checks completion at writeback,
parking incomplete instructions at the back of the pipeline. The hook was
changed to match. This roughly halved matmul's network overhead and cut
histogram's by about 78% (see Results). (`src/gpgpu-sim/shader.cc`)

---

## Results

**Setup.** ClusterSim's 64-SM `test.config`, `-gpgpu_n_cores_per_gpc 16,16,16,16`
(GPC-aware issue), PTX functional mode, `gpu_tot_sim_cycle`. Every run below
produced correct output (`MISMATCHES: 0`), and the gtest versions of both
kernels pass with `crossbar`, `ideal` and `ringbus`. Network defaults:
latency 171 cycles, bandwidth 88 bits/cycle. All numbers are for the
overlapped design (fix 8) unless marked otherwise.

### DSM vs conventional shared memory

Matmul (seed 1; matmul was identical across seeds in earlier runs):

| Size | Conventional | DSM `none` | DSM `crossbar` |
|---|---|---|---|
| 64×64 | 13,754 | 11,292 (−17.9%) | 13,703 (−0.4%) |
| 96×96 | 18,103 | 14,348 (−20.7%) | 18,505 (+2.2%) |
| 128×128 | 22,637 | 17,481 (−22.8%) | 23,230 (+2.6%) |

Histogram (6,400 bins, mean of seeds 1, 7, 42):

| Array size | Conventional | DSM `none` | DSM `crossbar` |
|---|---|---|---|
| 1,000 | 11,165 | 6,998 (−37.3%) | 8,484 (−24.0%) |
| 4,000 | 11,470 | 7,099 (−38.1%) | 8,777 (−23.5%) |
| 16,000 | 11,069 | 7,148 (−35.4%) | 9,826 (−11.2%) |
| 32,768 | 11,604 | 7,716 (−33.5%) | 12,766 (+10.0%) |
| 65,536 | 12,279 | 8,639 (−29.6%) | 18,468 (+50.4%) |

- With an idealized network (`none`), DSM always wins (18–38% fewer cycles).
- With the crossbar, matmul is at break-even at every size tested.
- Histogram has a crossover: DSM wins for small inputs and loses beyond
  roughly 16K–32K elements. From the kernel's indexing, that is where all
  16 SMs of the cluster already issue remote atomics, so further growth
  raises the per-SM load. This mechanism is inferred, not measured per SM.

### What the network cost depends on

Extra cycles of `crossbar` over `none`, latency swept at 88 bits/cycle:

| Latency (cycles) | matmul 64 | hist 1,000 | hist 65,536 |
|---|---|---|---|
| 1 | 504 | 1,305 | 9,727 |
| 85 | 1,403 | 1,392 | 9,716 |
| 171 | 2,411 | 1,496 | 9,755 |
| 342 | 4,479 | 1,672 | 10,139 |

Bandwidth swept at latency 171:

| Bandwidth (bits/cycle) | matmul 64 | hist 1,000 | hist 65,536 |
|---|---|---|---|
| 22 | 4,820 | 5,589 | 42,522 |
| 88 | 2,411 | 1,496 | 9,755 |
| 352 | 2,016 | 576 | 2,139 |

- Histogram is bandwidth-bound: its overhead barely changes with latency
  but scales roughly inversely with bandwidth. A likely contributor is that
  `Crossbar::Push` sizes atomic packets at 4× a plain request, and all of
  histogram's remote requests are atomic (hypothesis, not isolated).
- Matmul depends on both: about 11–12 extra cycles per cycle of latency,
  and a smaller bandwidth term.
- The DSM-vs-conventional sign depends on these parameters: matmul breaks
  even near ~175 cycles of latency (interpolated), and histogram (1,000
  elements) is faster at every latency from 1 to 342 cycles but slower at
  22 bits/cycle. The defaults come from ClusterSim and have not been
  checked against hardware.

### Remote-request counts (default size, `crossbar`)

matmul 64×64: 1,536 remote requests (none atomic), growing as N³ with
matrix size (5,184 at 96, 12,288 at 128). Histogram (1,000 elements): 418,
all atomic, growing about linearly with array size. Counts were measured
before fix 8; the counter itself is unaffected by it.

### Earlier blocking design (upper bound)

Before fix 8, every network-cost result was dominated by serialized round
trips: at default settings `crossbar` added 40% (matmul 64), 98% (histogram
1,000) and 326% (histogram 65,536) over `none`, overhead grew linearly with
latency (exactly 32 extra cycles per cycle of latency for histogram 1,000 —
one round trip per warp instruction), and DSM was slower than conventional
on both kernels (+15% matmul, +27% histogram, seed 1). These numbers are an
upper bound produced by the modeling choice, not a property of DSM.

### Sensitivity to CTA placement

The conventional histogram kernel (no remote accesses) takes a mean of
13,503 cycles with the original per-SIMT-cluster issue path and 11,165 with
GPC-aware issue, for the same binary. Compare DSM and conventional only
under the same issue path. The mechanism was not investigated. Changing the
GPC size (64, 32+32, 16×4) moved results by under 0.2% (measured with the
earlier blocking design).

### Regression correctness

The Rodinia 2.0 trace-driven suite (10 benchmarks) gives `gpu_tot_sim_cycle`
identical to unmodified Accel-Sim on the final build. Trace-driven runs
create no remote requests, and GPC-aware issue is opt-in, so the default
path is unchanged.

---

## Build and run

General build and environment setup is unchanged from upstream — see
`UPSTREAM_README.md`.

DSM-specific options:

| Option | Default | Meaning |
|---|---|---|
| `-sm_2_sm_network_type <none\|crossbar\|ideal\|ringbus>` | `none` | `none` charges a flat latency without modeling the interconnect; the others route remote requests through the corresponding network |
| `-sm_2_sm_network_latency` | 171 | Network latency in cycles |
| `-sm_2_sm_network_bw` | 88 | Network bandwidth in bits/cycle |
| `-gpgpu_n_cores_per_gpc <n0,n1,...>` | unset | Literal list of SMs per GPC; must sum to the SM count (e.g. `16,16,16,16` for 64 SMs). Setting it enables GPC-aware cluster issue |
| `-dsmem_{ld,st,atomic}_latency` | 29 | Flat remote latency used in `none` mode |

The RTX 3070 config enables `crossbar`; ClusterSim's `test.config` uses
`none`. A thread-block cluster must fit inside one GPC: if a request crosses
a GPC boundary, the simulator aborts with an error naming the origin SM, the
target SM, and the GPC's range.

To run a cluster kernel, launch it with `cudaLaunchKernelEx` and a
`cudaLaunchAttributeClusterDimension` attribute, as in ClusterSim's own
benchmarks.

**Practical notes**
- Do not put `sm_` anywhere in an application binary's name (e.g.
  `matmul_dsm_p`): the simulator then asks `ptxas` for architecture `sm_0`
  and fails with "ERROR ** while loading PTX". Renaming the binary fixes it.
- The build does not track header dependencies for every object. After
  changing a header, run `make clean` before `make`, and check the
  library timestamp before trusting results.

---

## Limitations

- Only the parallel cluster-placement policy is ported; ClusterSim's
  load-balanced policy is not.
- Cluster-aware execution works in PTX functional mode only. Trace-driven
  mode has no record of which SM owns a shared-memory address.
- The overlapped design follows ClusterSim's structure, but its cycle
  counts have not been compared against ClusterSim itself, and neither has
  been validated against hardware. Network defaults (171 cycles, 88
  bits/cycle) come from ClusterSim.
- Under the overlapped design, remote stores are not waited on, and parked
  instructions are re-checked roughly every `smem_latency` cycles (from
  reading the code; not measured).
- `ringbus` timing was 4–10× `crossbar` under the blocking design and has
  not been re-measured or explained; treat it as unvalidated.
- Results cover two kernels, one 64-SM configuration, and three seeds.
- Upstream CI workflows target the original authors' infrastructure (a
  self-hosted runner and an SST test harness) and are disabled in this fork.

---

## Repo layout

- `src/cuda-sim/` — PTX front-end and functional cluster model (parsing,
  address resolution, cluster construction)
- `src/gpgpu-sim/` — timing model: SM-to-SM network, clock domain, GPC
  model, cluster-aware CTA issue, cluster barriers
- `configs/tested-cfgs/SM86_RTX3070/` — RTX 3070 config
- `gpu-simulator/trace-driven/` — trace-driven integration

---

## Acknowledgements

Built on [GPGPU-Sim 4.0](https://github.com/accel-sim/gpgpu-sim_distribution)
and [Accel-Sim](https://github.com/accel-sim/accel-sim-framework). The DSM
model itself is ported from ClusterSim, a research fork of GPGPU-Sim
(branch `distributed_memory`). Done as a research internship under
Dr. Devashree Tripathy, IIT Bhubaneswar.