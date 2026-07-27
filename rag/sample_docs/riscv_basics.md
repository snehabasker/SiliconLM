# RISC-V ISA Basics (public-domain summary, for RAG demo purposes)

## 1. Base Integer Instruction Set
RV32I defines 32 general-purpose registers (x0-x31), where x0 is hardwired
to the constant zero. Instructions are fixed-width 32-bit encodings, split
into six formats: R-type, I-type, S-type, B-type, U-type, and J-type.

## 2. Privilege Levels
RISC-V defines three standard privilege levels: Machine (M-mode, mandatory
on every implementation), Supervisor (S-mode, used by operating systems),
and User (U-mode, used by applications). M-mode is the only mode required
by the specification; a minimal RISC-V core can implement M-mode only.

## 3. Control and Status Registers (CSRs)
CSRs are accessed via dedicated CSR instructions (CSRRW, CSRRS, CSRRC, and
their immediate variants). Each CSR is identified by a 12-bit address,
giving up to 4096 addressable registers, partitioned by convention into
read/write and read-only ranges based on the top two bits of the address.

## 4. Extensions
Standard extensions are named with single letters: M (integer
multiply/divide), A (atomics), F (single-precision float), D
(double-precision float), C (compressed 16-bit instructions). The
combination RV32IMAC is common for small embedded cores; RV64GC (G =
IMAFD) is common for application-class Linux-capable cores.
