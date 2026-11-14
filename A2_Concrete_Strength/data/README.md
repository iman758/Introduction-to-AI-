## A2 — Concrete Compressive Strength Dataset

**Source:** UCI Machine Learning Repository, Dataset ID 165
**URL:** https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
**Citation:** I-Cheng Yeh, "Modeling of strength of high-performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797–1808, 1998.

### Download Options

**Option A — Automatic (recommended):**
The notebook loads the dataset automatically via `ucimlrepo`:
```python
from ucimlrepo import fetch_ucirepo
concrete = fetch_ucirepo(id=165)
```
Install with: `pip install ucimlrepo`

**Option B — Manual:**
1. Visit https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
2. Click "Download" and save the zip
3. Extract `Concrete_Data.xls` to this `data/` folder
4. Set `USE_LOCAL = True` in notebook Cell 2

### File Not Committed
Per assignment rules, the dataset file (`.xls`) is excluded via `.gitignore`.
The notebook will auto-download it via `ucimlrepo` on first run.

