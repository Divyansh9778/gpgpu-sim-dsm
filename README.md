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
nothing in the simulator ever instantiated one or advanced it — running
with `crossbar` vs `none` produced identical cycle counts because every
remote request bypassed the network entirely. Added a
`create_sm2sm_network()` factory invoked from the `gpgpu_sim` constructor,
hooked `ldst_unit::shared_cycle()` to push requests into it and stall on
completion, and connected the previously dead SM2SM clock-domain tick to
`Advance()` the network each cycle. This is the change behind the timing
result below.

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

---

## Results

**Timing model activation.** Once the SM-to-SM network was actually wired
into the critical path (fix #4 above), an isolated DSM test kernel — one
cluster, one remote shared-memory read — moved from 5,347 cycles (network
bypassed) to 5,529 cycles (network modeled): a 182-cycle, 3.4% difference.
Cross-SM shared-memory reads and atomics now also return correct values
rather than the garbage caused by bug #1.

**DSM vs. conventional shared memory.** Across two ClusterSim benchmark
applications, three random seeds, and 100+ runs, DSM's advantage over the
conventional (non-cluster) baseline ranges from **+15.2% to −48.5%**
depending on how realistic the modeled interconnect cost is. The negative
end of that range is the finding, not a failure of the port: an
idealized or absent network cost flatters DSM, since it hides the
latency of the cross-SM traffic DSM depends on. Once that cost is modeled
realistically, DSM can lose to the conventional model on
communication-heavy access patterns — which is the result a supervisor
would expect from a feature whose entire premise is trading DRAM/L2
traffic for on-chip network traffic that isn't free.

**Cluster kernels with per-GPC networks.** Four 16-SM GPCs (64 SMs),
GPC-aware issue, PTX functional mode, ClusterSim's `test.config`. Both
kernels produce correct output.

| Kernel | `none` | `crossbar` | Overhead |
|---|---|---|---|
| `matmul_gpu_cluster` (64x64, 4x4 cluster) | 11,292 | 15,818 | +4,526 (+40%) |
| `clusterHistkernel` | 7,001 | 13,834 | +6,833 (+98%) |

Each number is a single run of one kernel, so these show that the network
cost is workload-dependent, not a general figure. The histogram's higher
overhead has not been investigated.

**Regression correctness.** A 10-benchmark trace-driven regression suite
(Rodinia 2.0) shows 0% cycle-count deviation from unmodified Accel-Sim
when DSM is disabled, confirming the port doesn't change behavior for
non-cluster kernels. The GPC-aware issue path is opt-in (it is used only
when `-gpgpu_n_cores_per_gpc` is set), so this regression exercises the
original issue path and stays unchanged.

---

## Build and run

General build and environment setup (CMake, dependencies, CUDA version) is
unchanged from upstream — see `UPSTREAM_README.md`.

DSM-specific configuration:

```
-sm_2_sm_network_type <none|crossbar|ringbus|ideal>
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
- Cluster results are single runs on two kernels (matmul and histogram).

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