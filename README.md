
# Portfolio Decision & Risk Dashboard — Version 6

All performance and risk statistics are now calculated directly from the
supplied daily portfolio-return file.

This resolves naming and lookup issues affecting the tuned models.

## Metrics calculated from daily returns

- Total return
- Annualised return
- Annualised volatility
- Sharpe ratio using a 2% annual risk-free rate
- Maximum drawdown
- 63-day rolling volatility
- 63-day rolling Sharpe ratio
- 2020 stress-period statistics

## Launch

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```


## Version 7 fix

The allocation-table column names are now normalised using the same mapping as
the daily-return columns. Previously, tuned portfolio pages failed at the asset
allocation chart, which prevented Streamlit from rendering all charts below it.


## Version 8 update

All cumulative-performance charts are now indexed to 100 at the start of the
backtest. A final index value of 168.65 therefore represents a cumulative return
of 68.65%.
