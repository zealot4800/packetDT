```bash
cd /home/zealot/ResarchWork/PackDT/packetDT

make -C p4 generate
make -C p4 xilinx
make -C p4 bmv2
make -C p4 tofino BF_P4C=/path/to/p4c-barefoot
make -C p4 xilinx-check
make -C p4 check BF_P4C=/path/to/p4c-barefoot
```
