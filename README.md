# macro-stress-optimizer

A macro financial stress pipeline with an ML weight optimizer. The pipeline ingests market and FRED data, computes a composite stress score across 16 leading indicators, and writes the result to parquet. The optimizer learns per-indicator weights that maximize AUC between the weighted stress score and realized SPY drawdown labels.

The stress score is a forward-looking risk indicator built from indicators with demonstrated leading properties. Reactive and coincident indicators (VIX, SKEW, CPI, USD/CNY, gold/equity ratio) were excluded. Optimized weights improve historical fit; they are not a trading signal.

## Indicators

**Yield curve (FRED)**
| Series | Indicator |
|---|---|
| `T10Y2Y` | 10Y-2Y Treasury spread |
| `T10Y3M` | 10Y-3M Treasury spread |
| `T30Y10Y` | 30Y-10Y Treasury spread (computed: `DGS30 - DGS10`) |

**Macro leading (FRED)**
| Series | Indicator |
|---|---|
| `USALOLITOAASTSAM` | OECD composite leading indicator |
| `UMCSENT` | University of Michigan consumer sentiment |
| `PERMIT` | Building permits |
| `NEWORDER` | Manufacturers new orders |

**Labor and credit (FRED)**
| Series | Indicator |
|---|---|
| `ICSA` | Initial jobless claims |
| `DRCCLACBS` | Credit card delinquency rate |
| `BAMLH0A0HYM2` | ICE BofA HY OAS spread |

**Market (yfinance)**
| Ticker | Indicator |
|---|---|
| `XLK` / `XLV` | Tech vs defensive rotation |
| `TLT` | Long-duration Treasury ETF |
| `HG=F` | Copper futures (growth proxy) |
| `CL=F` | Crude oil futures (rate-of-change; dual stress regime) |
| `EEM` | Emerging markets ETF |
| `DX=F` | DXY dollar index |

## Architecture

```
Pipeline
  fetch_data.py      pulls yfinance + FRED
  process_data.py    merges, resamples, computes ratios
  features.py        rolling percentile rank, direction flip, composite score
  pipeline.py        orchestration, writes stress_score.parquet

        |
        | data/processed/stress_score.parquet
        v

ML
  labels.py          derives binary SPY drawdown labels from parquet
  optimizer.py       SLSQP weight optimization, alpha sweep, CV evaluation
```

Both packages are installed from `src/` via `pyproject.toml`. The ML package does not import from the pipeline package; the parquet file is the only interface.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-username/macro-stress-optimizer.git
cd macro-stress-optimizer
uv sync
```

FRED requires a free API key. Create a `.env` file in the project root:

```
FRED_API_KEY=your_key_here
```

Get a key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

## Usage

```bash
uv run stress-pipeline   # fetch data, compute stress score
uv run stress-optimize   # read parquet, optimize weights
```

Outputs:

- `data/raw/market_raw.csv`: raw yfinance closes
- `data/raw/fred_raw.csv`: raw FRED series
- `data/processed/stress_score.parquet`: stress score with all ranked indicators and SPY
- `data/processed/optimized_weights.json`: per-indicator weights, AUC comparison, and run metadata

## Results

On 5,000+ trading days (2006-2026), SLSQP improves mean CV test AUC from **0.838** (equal weights) to **0.853** (optimized, alpha=3.0) on a 5-fold time series split. The improvement is intentionally modest — L2 regularization toward equal weights prevents overfitting to specific stress regimes on a single 20-year timeline with sparse drawdown events (~26% of trading days). The optimized weights are descriptive: they reflect which indicators historically led SPY drawdowns, not a forward-looking signal.

## Notebooks

`notebooks/stress_score_eda.ipynb` visualizes the pipeline output. Run the pipeline first, then open in Jupyter or VS Code.

**Stress score vs SPY**

![Stress score vs SPY](docs/stress_vs_spy.png)

**Stress score vs SPY drawdown (normalized)**

![Stress score vs SPY drawdown](docs/stress_vs_drawdown.png)

**Indicator breakdown: most recent observation**

![Indicator breakdown](docs/indicator_breakdown.png)

**Individual indicator time series**

![Individual indicators](docs/indicator_grid.png)

`notebooks/weight_optimizer.ipynb` covers the full ML workflow: label construction, SLSQP optimization, alpha sweep via cross-validation, weight stability, ROC curves, and optimized score visualization.

**Optimized vs equal-weight indicator weights**

![Weight stability](docs/weight_stability.png)

**ROC curves: equal weight vs optimized**

![ROC curves](docs/roc_curves.png)

**Optimized vs equal-weight score overlaid on SPY**

![Optimized vs equal weight vs SPY](docs/optimized_vs_equalweight_spy.png)

**Optimized vs equal-weight score overlaid on SPY drawdown**

![Optimized vs equal weight vs drawdown](docs/optimized_vs_equalweight_drawdown.png)

## Development

```bash
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
```
