```bash
cd /home/zealot/ResarchWork/PackDT/packetDT

make -C p4 generate
make -C p4 xilinx
make -C p4 bmv2
make -C p4 tofino BF_P4C=/path/to/p4c-barefoot
make -C p4 xilinx-check
make -C p4 check BF_P4C=/path/to/p4c-barefoot
```

`generate_statedt.py` emits the canonical entry layout and type in
`p4/common`. Result headers report `MATCH`, `ALLOCATED`, or
`FALLBACK_COLLISION`; `state_valid` is clear for the fallback case. The four
singleton counters named `statedt_allocations`,
`statedt_fingerprint_mismatches`, `statedt_collisions`, and
`statedt_fallbacks` are readable through each target's normal control plane.
