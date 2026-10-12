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
| Functional cluster model | `ptx_cluster_info` class tracking which CTAs belong to a cluster; `decode_space()` extended to resolve a shared address to its owning CTA/SM when it falls outside the executing SM's own window | `src/cuda-sim/ptx_sim.cc/.h`, `src/cuda-sim/instructions.cc` |
| CUDA runtime shim | `cudaLaunchKernelEx` reads the cluster dimension from the launch config and threads it into `kernel_info_t`, without which a cluster kernel cannot be launched at all | `src/cuda-sim/cuda-sim.cc`, `src/abstract_hardware_model.h/.cc` |
| Timing infrastructure | SM-to-SM network model (`Crossbar`, `Ringbus`, `IdealNetwork`) and the `cluster_shmem_request` message class; new SM2SM clock domain; one network instance per GPC, wired into `ldst_unit::shared_cycle()` so remote requests actually stall on the network | `src/gpgpu-sim/sm_2_sm_network.cc/.h`, `src/gpgpu-sim/gpu-sim.cc/.h`, `src/gpgpu-sim/shader.cc` |
| Cluster construction & barriers | Cluster-aware CTA iterator so a cluster's blocks are issued together; `barrier.cluster.arrive`/`wait` support | `src/gpgpu-sim/shader.cc/.h`, `src/cuda-sim/instructions.cc` (`bar_impl`) |
| GPC model & cluster placement | `gpu_processing_cluster` groups SMs (set with `-gpgpu_n_cores_per_gpc`) and owns its own SM-to-SM network; a cross-GPC guard aborts with a clear error if a request leaves its GPC; GPC-aware issue places all CTAs of one cluster inside a single GPC (opt-in) | `src/gpgpu-sim/shader.cc/.h`, `src/gpgpu-sim/gpu-sim.cc/.h`, `src/gpgpu-sim/sm_2_sm_network.cc/.h` |

---

## Bugs found and fixed

**1. Wrong SM-id field in address resolution.**
`ptx_cta_info::is_in_generic_shared_memory()` checked a shared address
against `m_sm_idx`, but `mapa_impl()` (the instruction that constructs
cluster-remapped addresses in the first place) addresses against
`m_shader_id`. The two fields diverge, so a correctly-resolved remote
address could still fail this check and read back garbage. Fixed to use
`m_shader_id`, matching `mapa_impl`'s addressing scheme.
(`src/cuda-sim/ptx_sim.cc`)

**2. 32-bit truncation of remote shared addresses.**
`ld_exec()` read the source address as `src1_data.u32`. A local
shared-memory address fits in 32 bits, so this went unnoticed until
cluster addressing was added — a remote address is built by remapping
into another CTA's window and can exceed 32 bits. Widened to `.u64`.
(`src/cuda-sim/instructions.cc`)

**3. Cluster atomics resolved to the wrong SM's shared memory.**
`atom_callback()` and `atom_impl()` both called `generic_to_shared()`
against the executing SM's own id unconditionally — even when the
address belonged to another SM's shared memory under a cluster. This is
the same class of bug `decode_space()` had before `58e1cdda` fixed it
for ordinary loads/stores; the atomics path was missed at the time.
Fixed to mirror that logic in both functions: check `isspace_shared()`
first, and if the address isn't local, resolve the owning CTA via
`ptx_cluster_info` and generate the address against the correct target
SM. (`src/cuda-sim/instructions.cc`)

**4. The SM-to-SM network was fully implemented but never connected.**
`Crossbar`, `Ringbus`, and `IdealNetwork` all existed and compiled, but
nothing in the simulator ever instantiated one or advanced it, so every
remote request bypassed the network entirely. Added a
`create_sm2sm_network()` factory, hooked `ldst_unit::shared_cycle()` to
push requests into it and stall on completion, and connected the
previously dead SM2SM clock-domain tick to `Advance()` the network each
cycle.

**5. Local shared atomics left the target SM id unset.**
`atom_impl()` and `atom_callback()` set `m_last_shared_memory_target_shader_id`
only when an address resolved to *another* CTA. A local shared atomic (for
example `mapa` to the CTA's own rank) left it at its initial `UINT_MAX`, so
`generate_mem_accesses()` classified it as a remote request to SM
4294967295, and the cross-GPC guard aborted. The id is now set to the
executing SM in the local branch. (`src/cuda-sim/instructions.cc`)

**6. Remote atomics updated the wrong CTA's shared memory.**
After resolving a remote address against the target CTA, `atom_callback()`
still read and wrote the *executing* thread's own `m_shared_mem`, so
remote `atomicAdd` updates landed in the wrong CTA's bins. It now uses the
target CTA's shared memory for the read-modify-write.
`Histogram.Cluster` went from 1225 mismatches to 0 in both `none` and
`crossbar`. (`src/cuda-sim/instructions.cc`)

**7. `ideal` and `ringbus` networks never completed requests.**
Only `Crossbar::Advance()` set `cluster_shmem_request::complete` on
delivery. With `ideal` or `ringbus`, any remote access waited forever and
the simulator reported a deadlock. Both now mark the request complete on
delivery; a per-message debug print in `Ringbus::Advance()` was removed.
(`src/gpgpu-sim/sm_2_sm_network.cc`)

---

## Results

All cluster results below are from ClusterSim's 64-SM `test.config`
(four 16-SM GPCs, GPC-aware issue, PTX functional mode) and
`gpu_tot_sim_cycle`. Every run listed produced correct output
(`MISMATCHES: 0`, and the gtest versions pass).

**Network model cost.** Same kernel, same issue path, seed 1.

| Kernel | `none` | `ideal` | `crossbar` | `ringbus` |
|---|---|---|---|---|
| `matmul_gpu_cluster` (64x64, 4x4 cluster) | 11,292 | 15,295 | 15,818 | 64,554 |
| `clusterHistkernel` | 7,001 | 12,751 | 13,834 | 144,049 |

- Modeling the network with `crossbar` adds 40% (matmul) and 98%
  (histogram) over `none`. An idealized flat latency therefore overstates
  what DSM can deliver.
- `ideal` is not free: it still charges the configured latency (+35% and
  +82% over `none`).
- `ringbus` costs about 4x (matmul) and 10x (histogram) more than
  `crossbar`. The cause has not been investigated, so treat `ringbus`
  as unvalidated.
- GPC size (64, 32+32, or 16x4) changed `none` and `crossbar` cycles by
  under 0.2% on both kernels.

**DSM vs. conventional shared memory.** This comparison depends on CTA
placement. On `clusterHistkernel`, the conventional (non-cluster) kernel
takes a mean of 13,503 cycles over seeds 1, 7, 42 (14,384 / 12,858 /
13,266) with the original per-SIMT-cluster issue path, and 11,165
(10,903 / 10,919 / 11,674) with GPC-aware issue: 17% different for the
same binary. DSM with `crossbar` takes about 13,830 cycles in both. DSM
is therefore roughly at parity with conventional under the first path
(about 4% faster on seed 1, about 2% slower against the 3-seed mean) and
about 24% slower under the second. On matmul with GPC-aware issue,
conventional takes 13,754 cycles: DSM is 18% faster with `none` (11,292)
and 15% slower with `crossbar` (15,818).
Compare DSM and conventional only within one issue path. The cost of the
network itself (table above) is the stable result. The mechanism by which
placement affects the conventional kernel was not investigated. The DSM
cycle counts under the original issue path come from runs made before
the atomics fixes above.

**Regression correctness.** A 10-benchmark trace-driven regression suite
(Rodinia 2.0) shows identical `gpu_tot_sim_cycle` to unmodified Accel-Sim
on the final build. The GPC-aware issue path is opt-in (used only when
`-gpgpu_n_cores_per_gpc` is set), so this regression exercises the
original issue path.

---

## Build and run

General build and environment setup (CMake, dependencies, CUDA version) is
unchanged from upstream — see `UPSTREAM_README.md`.

DSM-specific configuration:

```
-sm_2_sm_network_type <none|crossbar|ideal|ringbus>
```

`none` charges a flat configured latency for a remote shared-memory
access without modeling the interconnect; the other three route requests
through the corresponding `SM_2_SM_network` implementation. Check the
value in the config you run with. ClusterSim's `test.config` uses `none`.

```
-gpgpu_n_cores_per_gpc <n0,n1,...>
```

A literal comma-separated list of SMs per GPC; it must sum to the total SM
count, or the simulator aborts at startup. For example, `16,16,16,16` for 64
SMs. A thread-block cluster must fit inside one GPC: if a request crosses a
GPC boundary, the simulator aborts with an error naming the origin SM, the
target SM, and the GPC's range. Setting this flag also switches CTA issue to
the GPC-aware path, which places a cluster's blocks inside a single GPC.
Leaving it unset keeps the original per-SIMT-cluster issue path.

To run a cluster kernel, launch it with `cudaLaunchKernelEx` and a
`cudaLaunchAttributeClusterDimension` attribute, as in ClusterSim's own
benchmarks. No separate flag is needed to enable cluster launch; the flags
above select the network model and GPC layout used for its timing.

---

## Limitations

- Only the parallel cluster-placement policy is ported. ClusterSim's
  load-balanced policy is not.
- Cluster-aware execution works in PTX functional mode. Trace-driven mode has
  no record of which SM owns a shared-memory address, so remote shared-memory
  access is not modelled there.
- Cluster results cover two kernels, one 64-SM configuration, and three
  seeds (matmul is identical across seeds).
- `ringbus` timing is unexplained and unvalidated.
- The SST integration CI job fails because its test harness finds no tests;
  this appears unrelated to the simulator changes.

---

## Repo layout

- `src/cuda-sim/` — PTX front-end and functional cluster model (parsing,
  address resolution, cluster construction)
- `src/gpgpu-sim/` — timing model: SM-to-SM network, clock domain,
  GPC model, cluster-aware CTA issue, cluster barriers
- `configs/tested-cfgs/SM86_RTX3070/` — RTX 3070 config
- `gpu-simulator/trace-driven/` — trace-driven integration

---

## Acknowledgements

Built on [GPGPU-Sim 4.0](https://github.com/accel-sim/gpgpu-sim_distribution)
and [Accel-Sim](https://github.com/accel-sim/accel-sim-framework). The DSM
model itself is ported from ClusterSim, a research fork of GPGPU-Sim
(branch `distributed_memory`). Done as a research internship under
Dr. Devashree Tripathy, IIT Bhubaneswar.