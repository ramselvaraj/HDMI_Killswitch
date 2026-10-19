"""
cosine_driver.py
================
Python driver for cosine_sim_axi on Pynq-Z2.
Run from Jupyter or a Python 3 script on the board.

Usage
-----
from cosine_driver import CosineSimilarity
import numpy as np

cs = CosineSimilarity()          # loads overlay automatically

a = np.random.randn(128).astype(np.float32)
b = np.random.randn(128).astype(np.float32)

match, cos_val = cs.compare(a, b, threshold=0.80)
print(f"cos={cos_val:.4f}  match={match}")

Register Map (base 0x43C0_0000)
--------------------------------
0x00  CTRL        W   write 1 to start
0x04  STATUS      R   [0]=busy [1]=result_valid [2]=match
0x08  THRESHOLD   RW  Q2.30
0x0C  VEC_SEL     W   0=write A, 1=write B
0x10  VEC_DATA    W   Q1.15 sample (lower 16 bits)
0x14  LOAD_COUNT  R   elements loaded into current vector
"""

import numpy as np
import time

try:
    from pynq import Overlay, MMIO
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("[WARNING] pynq not found – running in simulation mode")

# ---------------------------------------------------------------------------
# Register offsets
# ---------------------------------------------------------------------------
REG_CTRL        = 0x00
REG_STATUS      = 0x04
REG_THRESHOLD   = 0x08
REG_VEC_SEL     = 0x0C
REG_VEC_DATA    = 0x10
REG_LOAD_COUNT  = 0x14

STATUS_BUSY         = (1 << 0)
STATUS_RESULT_VALID = (1 << 1)
STATUS_MATCH        = (1 << 2)

BASE_ADDR  = 0x43C0_0000
ADDR_RANGE = 0x1000        # 4 KB
N          = 128
Q15        = 2**15         # scale for Q1.15


def float_to_q15(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array (range [-1,1]) to int16 Q1.15."""
    clipped = np.clip(arr, -1.0 + 1/Q15, 1.0 - 1/Q15)
    return (clipped * Q15).astype(np.int16)


def float_to_q230(val: float) -> int:
    """Convert a float threshold to Q2.30 uint32."""
    return int(val * (2**30)) & 0xFFFFFFFF


def q230_to_float(raw: int) -> float:
    return raw / (2**30)


# ---------------------------------------------------------------------------
# Hardware driver
# ---------------------------------------------------------------------------
class CosineSimilarity:
    """
    Driver for the cosine_sim_axi IP core.

    Parameters
    ----------
    bitfile : str
        Path to the .bit file (or Overlay .hwh).
        Pass None to use raw MMIO without loading an overlay.
    base_addr : int
        AXI base address of the IP (default 0x43C0_0000).
    """

    def __init__(self, bitfile: str = "cosine_bd.bit", base_addr: int = BASE_ADDR):
        self.base = base_addr

        if not PYNQ_AVAILABLE:
            self._mmio = _SimMMIO()
            print("[SIM] Using software simulation – no hardware")
            return

        if bitfile is not None:
            print(f"Loading overlay: {bitfile}")
            self.overlay = Overlay(bitfile)
            print("Overlay loaded.")

        self._mmio = MMIO(base_addr, ADDR_RANGE)
        # Verify we can read back the default threshold
        thr_raw = self._mmio.read(REG_THRESHOLD)
        print(f"Core alive. Default threshold = {q230_to_float(thr_raw):.4f}  "
              f"(raw=0x{thr_raw:08X})")

    # ------------------------------------------------------------------ low-level
    def _wr(self, offset: int, val: int):
        self._mmio.write(offset, int(val) & 0xFFFFFFFF)

    def _rd(self, offset: int) -> int:
        return self._mmio.read(offset)

    # ------------------------------------------------------------------ helpers
    def set_threshold(self, threshold: float):
        """Set cosine similarity threshold (0 < threshold < 1)."""
        self._wr(REG_THRESHOLD, float_to_q230(threshold))

    def get_threshold(self) -> float:
        return q230_to_float(self._rd(REG_THRESHOLD))

    def _load_vector(self, vec: np.ndarray, slot: int):
        """
        Load a 128-element float32 vector into RAM slot 0 (A) or 1 (B).
        Converts to Q1.15 internally.
        """
        if len(vec) != N:
            raise ValueError(f"Vector must have {N} elements, got {len(vec)}")
        q = float_to_q15(vec)

        # Select which RAM to write
        self._wr(REG_VEC_SEL, slot)
        time.sleep(0.0001)   # let register settle (not strictly needed on real HW)

        # Write elements one at a time
        for sample in q:
            self._wr(REG_VEC_DATA, int(sample) & 0xFFFF)

        # Verify count
        loaded = self._rd(REG_LOAD_COUNT)
        if loaded != 0:   # count resets to 0 after last element
            raise RuntimeError(f"Load incomplete: {loaded}/{N} elements in slot {slot}")

    def _wait_result(self, timeout_ms: float = 10.0) -> int:
        """Poll STATUS until result_valid. Returns raw STATUS register."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            status = self._rd(REG_STATUS)
            if status & STATUS_RESULT_VALID:
                return status
            time.sleep(1e-6)
        raise TimeoutError(f"No result after {timeout_ms} ms. STATUS=0x{self._rd(REG_STATUS):08X}")

    # ------------------------------------------------------------------ main API
    def compare(self, a: np.ndarray, b: np.ndarray,
                threshold: float = None) -> tuple:
        """
        Compute cosine similarity between vectors a and b.

        Parameters
        ----------
        a, b      : float32 numpy arrays of length 128, values in [-1, 1]
        threshold : override threshold for this call (optional)

        Returns
        -------
        (match: bool, cos_approx: float)
            cos_approx is a software-computed reference value for verification.
        """
        if threshold is not None:
            self.set_threshold(threshold)

        # Load vectors
        self._load_vector(a, slot=0)
        self._load_vector(b, slot=1)

        # Fire!
        self._wr(REG_CTRL, 1)

        # Wait
        status = self._wait_result()
        match  = bool(status & STATUS_MATCH)

        # Software reference (for debug/verification)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-9 or b_norm < 1e-9:
            cos_ref = 0.0
        else:
            cos_ref = float(np.dot(a, b) / (a_norm * b_norm))

        return match, cos_ref

    def batch_compare(self, pairs: list, threshold: float = 0.80) -> list:
        """
        Compare multiple (a, b) pairs.

        Parameters
        ----------
        pairs     : list of (a, b) tuples
        threshold : single threshold applied to all pairs

        Returns
        -------
        list of (match: bool, cos_ref: float)
        """
        self.set_threshold(threshold)
        return [self.compare(a, b) for a, b in pairs]


# ---------------------------------------------------------------------------
# Simulation back-end (used when pynq is not available)
# ---------------------------------------------------------------------------
class _SimMMIO:
    """Pure-Python cosine similarity for offline testing of this driver."""

    def __init__(self):
        self._regs  = {
            REG_THRESHOLD: float_to_q230(0.80),
            REG_STATUS:    0,
        }
        self._ram_a = np.zeros(N, dtype=np.int16)
        self._ram_b = np.zeros(N, dtype=np.int16)
        self._vec_sel   = 0
        self._cnt_a     = 0
        self._cnt_b     = 0

    def write(self, offset, val):
        if offset == REG_THRESHOLD:
            self._regs[REG_THRESHOLD] = val
        elif offset == REG_VEC_SEL:
            self._vec_sel = val & 1
            if val & 1 == 0: self._cnt_a = 0
            else:             self._cnt_b = 0
        elif offset == REG_VEC_DATA:
            sample = np.int16(val & 0xFFFF)
            if self._vec_sel == 0 and self._cnt_a < N:
                self._ram_a[self._cnt_a] = sample
                self._cnt_a += 1
            elif self._vec_sel == 1 and self._cnt_b < N:
                self._ram_b[self._cnt_b] = sample
                self._cnt_b += 1
        elif offset == REG_CTRL and (val & 1):
            self._run_sim()

    def read(self, offset):
        return self._regs.get(offset, 0)

    def _run_sim(self):
        a = self._ram_a.astype(np.float64) / (2**15)
        b = self._ram_b.astype(np.float64) / (2**15)
        dot = np.dot(a, b)
        na  = np.linalg.norm(a)
        nb  = np.linalg.norm(b)
        thr = q230_to_float(self._regs[REG_THRESHOLD])
        if na < 1e-9 or nb < 1e-9:
            cos = 0.0
        else:
            cos = dot / (na * nb)
        match = int(cos >= thr)
        self._regs[REG_STATUS] = STATUS_RESULT_VALID | (match << 2)
        # reset counts so load_count returns 0 (done)
        self._cnt_a = 0
        self._cnt_b = 0


# ---------------------------------------------------------------------------
# Quick self-test  –  run this file directly to verify the driver
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print(" CosineSimilarity driver self-test")
    print("=" * 55)

    cs = CosineSimilarity(bitfile=None)   # sim mode if no PYNQ
    rng = np.random.default_rng(42)

    tests = [
        ("Identical",      lambda: (np.ones(128), np.ones(128)),           0.80, True),
        ("Anti-parallel",  lambda: (np.ones(128), -np.ones(128)),          0.80, False),
        ("Orthogonal",     lambda: (np.eye(128)[0], np.eye(128)[1]),       0.80, False),
        ("High sim 0.97",  lambda: _make_partial(rng, frac=0.015),         0.80, True),
        ("Low sim 0.75",   lambda: _make_partial(rng, frac=0.25),          0.80, False),
        ("Random",         lambda: (rng.standard_normal(128),
                                    rng.standard_normal(128)),             0.80, None),
    ]

    def _make_partial(rng, frac):
        a = rng.standard_normal(128).astype(np.float32)
        b = a.copy()
        flip = rng.choice(128, int(128*frac), replace=False)
        b[flip] *= -1
        return a, b

    passed = failed = 0
    for name, gen, thr, expected in tests:
        a, b = gen()
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        match, cos_ref = cs.compare(a, b, threshold=thr)
        ok = (expected is None) or (match == expected)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<20}  cos={cos_ref:+.4f}  "
              f"match={int(match)}  expected={expected}")
        if ok: passed += 1
        else:  failed += 1

    print(f"\n  {passed} passed / {failed} failed")
