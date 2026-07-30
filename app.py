
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="Portfolio Decision & Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

RISK_FREE_RATE = 0.02
TRADING_DAYS = 252
ROLLING_WINDOW = 63

MODEL_NAME_MAP = {
    "Decision Tree (tuned)": "Decision Tree (Tuned)",
    "Random Forest (tuned)": "Random Forest (Tuned)",
    "Gradient boosting (tuned)": "Gradient boosting (Tuned)",
    "Gradient Boosting (tuned)": "Gradient boosting (Tuned)",
    "Gradient Boosting (Tuned)": "Gradient boosting (Tuned)",
    "Gradient Boosting": "Gradient boosting",
}

@st.cache_data
def load_data():
    allocation = pd.read_csv(
        DATA_DIR / "average_allocation.csv",
        index_col=0,
    ).rename(columns=MODEL_NAME_MAP)

    allocation_2020 = pd.read_csv(
        DATA_DIR / "average_allocation_2020.csv",
        index_col=0,
    ).rename(columns=MODEL_NAME_MAP)

    daily_returns = pd.read_csv(DATA_DIR / "daily_returns.csv")
    date_column = daily_returns.columns[0]
    daily_returns[date_column] = pd.to_datetime(
        daily_returns[date_column]
    )
    daily_returns = (
        daily_returns
        .set_index(date_column)
        .sort_index()
        .rename(columns=MODEL_NAME_MAP)
    )
    daily_returns.index.name = "Date"

    return allocation, allocation_2020, daily_returns

allocation, allocation_2020, daily_returns = load_data()
models = list(daily_returns.columns)

MODEL_DESCRIPTIONS = {
    "MVO": (
        "Classical mean–variance optimisation that maximises the Sharpe ratio "
        "using historical returns and covariances."
    ),
    "Decision Tree": (
        "A decision-tree model trained to predict monthly portfolio weights "
        "from technical, factor and risk features."
    ),
    "Decision Tree (Tuned)": (
        "A hyperparameter-tuned decision tree for monthly portfolio-weight prediction."
    ),
    "Random Forest": (
        "An ensemble of decision trees used to predict monthly portfolio allocations."
    ),
    "Random Forest (Tuned)": (
        "A tuned random-forest allocation model selected through time-series-aware validation."
    ),
    "Gradient boosting": (
        "A gradient-boosting model that sequentially improves monthly allocation predictions."
    ),
    "Gradient boosting (Tuned)": (
        "A tuned gradient-boosting model for monthly portfolio allocation."
    ),
}

def calculate_metrics(
    returns,
    risk_free_rate=RISK_FREE_RATE,
):
    returns = returns.dropna().astype(float)

    cumulative_growth = (1 + returns).prod()
    total_return = cumulative_growth - 1

    elapsed_days = (
        returns.index[-1] - returns.index[0]
    ).days
    years = elapsed_days / 365.25

    annualized_return = cumulative_growth ** (1 / years) - 1
    annualized_volatility = (
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    sharpe_ratio = (
        annualized_return - risk_free_rate
    ) / annualized_volatility

    cumulative_series = (1 + returns).cumprod()
    drawdown = (
        cumulative_series / cumulative_series.cummax()
    ) - 1
    maximum_drawdown = drawdown.min()

    return {
        "Total return": total_return * 100,
        "Annualized return": annualized_return * 100,
        "Annualized volatility": annualized_volatility * 100,
        "Sharpe ratio": sharpe_ratio,
        "Maximum drawdown": maximum_drawdown * 100,
    }

def calculate_metric_table(return_frame):
    return pd.DataFrame({
        model: calculate_metrics(return_frame[model])
        for model in return_frame.columns
    })

performance = calculate_metric_table(daily_returns)

returns_2020 = daily_returns.loc["2020-01-01":"2020-12-31"]
performance_2020 = calculate_metric_table(returns_2020)

with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Select page",
        ["Overview"] + models,
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        "Historical research dashboard calculated directly from "
        "the frozen daily portfolio returns."
    )

def styled_performance_table(frame):
    table = frame.T.copy()
    table.index.name = "Portfolio"
    return table.style.format({
        "Total return": "{:.2f}%",
        "Annualized return": "{:.2f}%",
        "Annualized volatility": "{:.2f}%",
        "Sharpe ratio": "{:.2f}",
        "Maximum drawdown": "{:.2f}%",
    })

def show_metric_cards(model):
    values = performance[model]
    labels = [
        "Total return",
        "Annualized return",
        "Annualized volatility",
        "Sharpe ratio",
        "Maximum drawdown",
    ]

    columns = st.columns(5)

    for column, label in zip(columns, labels):
        value = float(values.loc[label])
        shown = (
            f"{value:.2f}"
            if label == "Sharpe ratio"
            else f"{value:.2f}%"
        )
        column.metric(label, shown)

def cumulative_growth(returns):
    """Return portfolio value indexed to 100 at the start."""
    return (1 + returns).cumprod() * 100

def drawdown_series(returns):
    growth = cumulative_growth(returns)
    return growth / growth.cummax() - 1

def rolling_volatility(
    returns,
    window=ROLLING_WINDOW,
):
    return (
        returns.rolling(window).std()
        * np.sqrt(TRADING_DAYS)
        * 100
    )

def rolling_sharpe(
    returns,
    window=ROLLING_WINDOW,
    risk_free_rate=RISK_FREE_RATE,
):
    daily_rf = (
        (1 + risk_free_rate) ** (1 / TRADING_DAYS)
    ) - 1

    excess_returns = returns - daily_rf

    annualised_excess_return = (
        excess_returns.rolling(window).mean()
        * TRADING_DAYS
    )
    annualised_volatility = (
        returns.rolling(window).std()
        * np.sqrt(TRADING_DAYS)
    )

    return annualised_excess_return / annualised_volatility

def create_cumulative_chart(return_frame, title):
    cumulative = cumulative_growth(return_frame)

    plot_frame = (
        cumulative
        .reset_index()
        .melt(
            id_vars="Date",
            var_name="Portfolio",
            value_name="Growth",
        )
    )

    fig = px.line(
        plot_frame,
        x="Date",
        y="Growth",
        color="Portfolio",
        title=title,
        labels={
            "Growth": "Portfolio value, indexed to 100",
            "Date": "",
        },
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Date: %{x|%d %b %Y}<br>"
            "Index value: %{y:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=55, b=10),
        legend_title_text="Portfolio",
        hovermode="x unified",
    )
    return fig

def create_allocation_pie(model, period):
    source = (
        allocation
        if period == "Full backtest"
        else allocation_2020
    )

    if model not in source.columns:
        available = ", ".join(source.columns)
        raise KeyError(
            f"Allocation data for '{model}' are unavailable. "
            f"Available columns: {available}"
        )

    series = source[model]
    frame = (
        series[series > 0.0001]
        .rename("Weight")
        .reset_index()
    )
    frame.columns = ["ETF", "Weight"]
    frame["Weight (%)"] = frame["Weight"] * 100

    fig = px.pie(
        frame,
        names="ETF",
        values="Weight (%)",
        hole=0.42,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="label+percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Weight: %{value:.2f}%<extra></extra>"
        ),
    )
    fig.update_layout(
        height=470,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig

def create_portfolio_risk_return_bubbles(selected_model):
    frame = (
        performance.T
        .reset_index()
        .rename(columns={"index": "Portfolio"})
    )

    other_frame = frame[
        frame["Portfolio"] != selected_model
    ]
    selected_frame = frame[
        frame["Portfolio"] == selected_model
    ]

    fig = px.scatter(
        other_frame,
        x="Annualized volatility",
        y="Annualized return",
        size="Sharpe ratio",
        text="Portfolio",
        labels={
            "Annualized volatility": "Annualised volatility (%)",
            "Annualized return": "Annualised return (%)",
        },
        hover_data={
            "Portfolio": True,
            "Sharpe ratio": ":.2f",
        },
        size_max=36,
    )

    fig.update_traces(
        marker=dict(
            color="#8A8F98",
            opacity=0.55,
            line=dict(width=1, color="#666A70"),
        ),
        textposition="top center",
        textfont=dict(color="#777B82"),
        name="Other portfolios",
    )

    fig.add_scatter(
        x=selected_frame["Annualized volatility"],
        y=selected_frame["Annualized return"],
        mode="markers+text",
        text=selected_frame["Portfolio"],
        textposition="top center",
        marker=dict(
            size=28,
            color="#F28E2B",
            opacity=1.0,
            line=dict(width=2, color="#B85F00"),
        ),
        customdata=selected_frame[
            ["Sharpe ratio"]
        ].to_numpy(),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Annualised volatility: %{x:.2f}%<br>"
            "Annualised return: %{y:.2f}%<br>"
            "Sharpe ratio: %{customdata[0]:.2f}"
            "<extra></extra>"
        ),
        name="Selected portfolio",
    )

    fig.update_layout(
        height=470,
        margin=dict(l=10, r=10, t=20, b=10),
        legend_title_text="",
    )
    return fig

def create_drawdown_chart(model):
    drawdown = (
        drawdown_series(daily_returns[model]) * 100
    )
    frame = (
        drawdown
        .rename("Drawdown (%)")
        .reset_index()
    )

    fig = px.area(
        frame,
        x="Date",
        y="Drawdown (%)",
        labels={
            "Date": "",
            "Drawdown (%)": "Drawdown (%)",
        },
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig

def create_rolling_volatility_chart(model):
    series = rolling_volatility(daily_returns[model])
    frame = (
        series
        .rename("Rolling volatility (%)")
        .reset_index()
    )

    fig = px.line(
        frame,
        x="Date",
        y="Rolling volatility (%)",
        labels={
            "Date": "",
            "Rolling volatility (%)":
                "63-day annualised volatility (%)",
        },
    )
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig

def create_rolling_sharpe_chart(model):
    series = rolling_sharpe(daily_returns[model])
    frame = (
        series
        .rename("Rolling Sharpe ratio")
        .reset_index()
    )

    fig = px.line(
        frame,
        x="Date",
        y="Rolling Sharpe ratio",
        labels={
            "Date": "",
            "Rolling Sharpe ratio":
                "63-day rolling Sharpe ratio",
        },
    )
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig

if page == "Overview":
    st.title("Portfolio Decision & Risk Dashboard")
    st.markdown(
        """
        This project compares a classical **mean–variance optimisation (MVO)**
        portfolio with Decision Tree, Random Forest and Gradient Boosting models
        that predict monthly ETF allocations.

        The study uses ten equity and fixed-income ETFs, monthly rebalancing and
        an expanding-window backtest from **January 2015 to June 2021**.
        """
    )

    st.subheader("Cumulative return of the portfolios")
    st.plotly_chart(
        create_cumulative_chart(
            daily_returns,
            "Cumulative portfolio performance",
        ),
        use_container_width=True,
    )

    st.subheader("Performance and risk summary")
    st.dataframe(
        styled_performance_table(performance),
        use_container_width=True,
    )

    st.subheader("Risk–return positioning")

    overview_frame = (
        performance.T
        .reset_index()
        .rename(columns={"index": "Portfolio"})
    )

    overview_fig = px.scatter(
        overview_frame,
        x="Annualized volatility",
        y="Annualized return",
        size="Sharpe ratio",
        text="Portfolio",
        labels={
            "Annualized volatility":
                "Annualised volatility (%)",
            "Annualized return":
                "Annualised return (%)",
        },
        hover_data={"Sharpe ratio": ":.2f"},
    )
    overview_fig.update_traces(
        textposition="top center"
    )
    overview_fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(
        overview_fig,
        use_container_width=True,
    )

else:
    model = page

    st.title(model)
    st.write(MODEL_DESCRIPTIONS.get(model, ""))
    show_metric_cards(model)

    allocation_period = st.radio(
        "Allocation period",
        ["Full backtest", "2020 stress period"],
        horizontal=True,
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Asset allocation")
        st.plotly_chart(
            create_allocation_pie(
                model,
                allocation_period,
            ),
            use_container_width=True,
        )

    with right:
        st.subheader("Risk–return profile")
        st.plotly_chart(
            create_portfolio_risk_return_bubbles(model),
            use_container_width=True,
        )

    st.subheader("Performance over time")
    st.plotly_chart(
        create_cumulative_chart(
            daily_returns[[model]],
            f"{model} cumulative performance",
        ),
        use_container_width=True,
    )

    st.subheader("Risk dashboard")

    risk_left, risk_right = st.columns(2)

    with risk_left:
        st.markdown("#### Drawdown")
        st.plotly_chart(
            create_drawdown_chart(model),
            use_container_width=True,
        )

    with risk_right:
        st.markdown("#### Rolling volatility")
        st.plotly_chart(
            create_rolling_volatility_chart(model),
            use_container_width=True,
        )

    st.markdown("#### Rolling Sharpe ratio")
    st.plotly_chart(
        create_rolling_sharpe_chart(model),
        use_container_width=True,
    )

    st.subheader("2020 stress-period summary")

    stress_values = performance_2020[model]

    stress_table = pd.DataFrame({
        "Metric": [
            "Total return",
            "Annualized return",
            "Annualized volatility",
            "Sharpe ratio",
            "Maximum drawdown",
        ],
        "Value": [
            f"{stress_values.loc['Total return']:.2f}%",
            f"{stress_values.loc['Annualized return']:.2f}%",
            f"{stress_values.loc['Annualized volatility']:.2f}%",
            f"{stress_values.loc['Sharpe ratio']:.2f}",
            f"{stress_values.loc['Maximum drawdown']:.2f}%",
        ],
    })

    st.dataframe(
        stress_table,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Methodology and limitations"):
        st.markdown(
            """
            - Historical simulation from January 2015 to June 2021.
            - Monthly portfolio rebalancing.
            - Ten equity and fixed-income ETFs.
            - Sharpe ratios use a 2% annual risk-free-rate assumption.
            - Rolling risk charts use a 63-trading-day window.
            - Transaction costs, taxes and market impact are excluded.
            - All performance and risk metrics are calculated directly
              from the frozen daily portfolio returns.
            """
        )

st.caption(
    "Historical research project · "
    "Not an investment recommendation"
)
