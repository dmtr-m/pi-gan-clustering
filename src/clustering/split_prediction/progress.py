"""Step progress that reads well both on a terminal and in a redirected log.

A tqdm bar redraws with carriage returns.  On a terminal that is a live bar; in a
log file it is one enormous line of stacked refreshes, and ``tail -f`` renders it
with stale characters left behind wherever a redraw is shorter than the last.
So: a bar when stderr is a TTY, and plain newline-terminated lines every
``every`` steps (plus the last) when it is not.
"""
import sys
import time
from typing import Dict

from tqdm import tqdm


class StepProgress:
    def __init__(self, total: int, desc: str, every: int = 10) -> None:
        self.total, self.desc, self.every = total, desc, max(1, every)
        self.n = 0
        self.t0 = time.time()
        self.post: Dict[str, object] = {}
        self.bar = (tqdm(total=total, desc=desc, unit="step", mininterval=1.0,
                         dynamic_ncols=True) if sys.stderr.isatty() else None)

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.bar is not None:
            self.bar.update(n)
        elif self.n % self.every == 0 or self.n == self.total:
            elapsed = time.time() - self.t0
            per = elapsed / max(self.n, 1)
            eta = per * (self.total - self.n) / 60.0
            extra = "  ".join(f"{k}={v}" for k, v in self.post.items())
            print(f"[{self.desc}] step {self.n}/{self.total} ({100 * self.n / self.total:.0f}%)"
                  f"  {per:.2f} s/step  eta {eta:.1f} min  {extra}", flush=True)

    def set_postfix(self, refresh: bool = False, **kw) -> None:
        self.post = kw
        if self.bar is not None:
            self.bar.set_postfix(refresh=refresh, **kw)

    def write(self, msg: str) -> None:
        (tqdm.write if self.bar is not None else lambda m: print(m, flush=True))(msg)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
