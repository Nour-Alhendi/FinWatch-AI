"""
FinWatch AI — Chart Renderers
==============================
Plotly chart components: price candlestick, RSI, and SPX overview.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


_RS_BUTTONS = [
    dict(count=1,  label="1M", step="month", stepmode="backward"),
    dict(count=3,  label="3M", step="month", stepmode="backward"),
    dict(count=6,  label="6M", step="month", stepmode="backward"),
    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
    dict(step="all", label="All"),
]

_TICKFORMATSTOPS = [
    dict(dtickrange=[None,    86400000], value="%d %b"),
    dict(dtickrange=[86400000, "M1"],   value="%d %b '%y"),
    dict(dtickrange=["M1",    "M12"],   value="%b '%y"),
    dict(dtickrange=["M12",   None],    value="%Y"),
]

_HOVERLABEL = dict(
    bgcolor="#0d1117", bordercolor="#314158",
    font=dict(size=11, family="IBM Plex Mono", color="#cdd9e5"),
    namelength=0,
    align="left",
)

_PERIOD_OFFSETS = {
    "1M":  pd.DateOffset(months=1),
    "3M":  pd.DateOffset(months=3),
    "6M":  pd.DateOffset(months=6),
    "1Y":  pd.DateOffset(years=1),
    "All": None,
}


def render_price_chart(det_df: pd.DataFrame, ticker: str, period: str = "1M", clicked_date: str = None):
    """Render the main price candlestick + EMA20 + anomaly dots chart."""
    if det_df is None or "Close" not in det_df.columns:
        st.markdown(
            '<div style="height:300px;display:flex;align-items:center;'
            'justify-content:center;color:#5c7080;font-size:12px;'
            'font-family:IBM Plex Mono">No price data</div>',
            unsafe_allow_html=True,
        )
        return

    # EMA on full data so it's accurate, then filter by period
    full_df   = det_df.copy()
    ema20_all = full_df["Close"].ewm(span=20).mean()

    x_end  = full_df["Date"].iloc[-1]
    offset = _PERIOD_OFFSETS.get(period)
    if offset is not None:
        mask    = full_df["Date"] >= (x_end - offset)
        plot_df = full_df[mask].reset_index(drop=True)
        ema20   = ema20_all[mask].reset_index(drop=True)
    else:
        plot_df = full_df
        ema20   = ema20_all

    if len(plot_df) == 0:
        st.markdown('<div style="color:#5c7080;font-size:11px;font-family:IBM Plex Mono">No data for selected period</div>', unsafe_allow_html=True)
        return

    has_ohlc   = all(c in plot_df.columns for c in ["Open", "High", "Low"])
    close_last = plot_df["Close"].iloc[-1]
    ema20_last = ema20.iloc[-1]

    # Y range calculated from the filtered (visible) data
    if has_ohlc:
        y_lo = min(plot_df["Low"].min(), ema20.min()) * 0.98
        y_hi = max(plot_df["High"].max(), ema20.max()) * 1.02
    else:
        y_lo = min(plot_df["Close"].min(), ema20.min()) * 0.98
        y_hi = max(plot_df["Close"].max(), ema20.max()) * 1.02

    fig = go.Figure()

    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=plot_df["Date"],
            open=plot_df["Open"], high=plot_df["High"],
            low=plot_df["Low"],   close=plot_df["Close"],
            increasing_line_color="#1de9b6", decreasing_line_color="#f85149",
            increasing_fillcolor="#1de9b6",  decreasing_fillcolor="#f85149",
            name="Price", whiskerwidth=0.4,
            hoverinfo="none",
        ))
        _hover_cd = np.column_stack([
            plot_df["Open"].round(2), plot_df["High"].round(2),
            plot_df["Low"].round(2),  plot_df["Close"].round(2),
            ema20.round(2),
        ])
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=plot_df["Close"],
            mode="markers", marker=dict(opacity=0, size=28),
            customdata=_hover_cd, showlegend=False, name="",
            hovertemplate=(
                "<b>%{x|%d %b '%y}</b><br>"
                "─────────────────<br>"
                "O&nbsp;&nbsp;$%{customdata[0]:,.2f}<br>"
                "H&nbsp;&nbsp;$%{customdata[1]:,.2f}<br>"
                "L&nbsp;&nbsp;&nbsp;$%{customdata[2]:,.2f}<br>"
                "C&nbsp;&nbsp;$%{customdata[3]:,.2f}<br>"
                "─────────────────<br>"
                "EMA&nbsp;$%{customdata[4]:,.2f}"
                "<extra></extra>"
            ),
        ))
    else:
        prev_c     = plot_df["Close"].shift(1).fillna(plot_df["Close"])
        day_colors = ["#1de9b6" if c >= p else "#f85149"
                      for c, p in zip(plot_df["Close"], prev_c)]
        fig.add_trace(go.Scatter(
            x=plot_df["Date"], y=plot_df["Close"].round(2),
            mode="markers",
            marker=dict(symbol="line-ew-open", color=day_colors,
                        size=8, line=dict(width=2.5)),
            name="Close",
            hovertemplate="<b>%{x|%d %b '%y}</b><br>Close $%{y:,.2f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=plot_df["Date"], y=ema20.round(2),
        mode="lines", name="EMA20",
        line=dict(color="#a371f7", width=1.2, dash="dot"),
        hoverinfo="skip",
    ))

    _det_cols = ["z_anomaly", "z_anomaly_60", "if_anomaly", "ae_anomaly"]
    _available = [c for c in _det_cols if c in plot_df.columns]
    if _available:
        _confirmed = plot_df[[*_available]].apply(
            lambda row: sum(bool(v) for v in row), axis=1
        )
        anom_multi = plot_df[_confirmed >= 2]
        if len(anom_multi) > 0:
            _det_labels = anom_multi.apply(
                lambda r: ", ".join(
                    n for c, n in zip(
                        _det_cols,
                        ["Z-30D", "Z-60D", "IsoForest", "LSTM-AE"]
                    ) if c in r.index and bool(r.get(c))
                ),
                axis=1,
            )
            fig.add_trace(go.Scatter(
                x=anom_multi["Date"], y=anom_multi["Close"],
                mode="markers",
                marker=dict(color="#FFD700", size=8, symbol="circle",
                            line=dict(color="#b8960c", width=1)),
                name="Confirmed Anomaly",
                customdata=_det_labels,
                hovertemplate="<b>%{x|%d %b '%y}</b><br>⬤ Anomaly confirmed by: %{customdata}<extra></extra>",
            ))

    label_gap    = abs(close_last - ema20_last)
    close_yshift = (+8 if close_last >= ema20_last else -8) if label_gap / (y_hi - y_lo + 1e-9) < 0.04 else 0
    ema20_yshift = -close_yshift

    fig.add_annotation(x=1.01, xref="paper", y=close_last, yref="y",
                       text=f"${close_last:.2f}",
                       xanchor="left", showarrow=False,
                       font=dict(size=9, color="#1de9b6", family="IBM Plex Mono"),
                       yshift=close_yshift)
    fig.add_annotation(x=1.01, xref="paper", y=ema20_last, yref="y",
                       text=f"EMA20 ${ema20_last:.2f}",
                       xanchor="left", showarrow=False,
                       font=dict(size=9, color="#a371f7", family="IBM Plex Mono"),
                       yshift=ema20_yshift)

    if clicked_date:
        try:
            fig.add_vline(
                x=pd.Timestamp(clicked_date),
                line=dict(color="#FFD700", width=1, dash="dot"),
                layer="below",
            )
        except Exception:
            pass

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(
            showgrid=True, gridcolor="#111b27", color="#5c7080",
            tickfont=dict(size=10, family="IBM Plex Mono"),
            tickformatstops=_TICKFORMATSTOPS,
        ),
        yaxis=dict(showgrid=True, gridcolor="#111b27", color="#5c7080",
                   tickfont=dict(size=10, family="IBM Plex Mono"), tickprefix="$",
                   side="right", range=[y_lo, y_hi]),
        hoverlabel=_HOVERLABEL,
        margin=dict(t=30, b=10, l=10, r=90), height=400,
        showlegend=False, hovermode=False,
        clickmode="event+select",
    )
    return st.plotly_chart(
        fig, use_container_width=True, config={"displayModeBar": False},
        key="price_chart", on_select="rerun", selection_mode="points",
    )


def render_rsi_chart(det_df: pd.DataFrame, period: str = "1M") -> None:
    """Render RSI + RSI MA(14) chart filtered to the active period."""
    if det_df is None or "rsi" not in det_df.columns:
        return

    full_df    = det_df.copy()
    rsi_ma_all = full_df["rsi"].rolling(window=14, min_periods=1).mean()

    x_end  = full_df["Date"].iloc[-1]
    offset = _PERIOD_OFFSETS.get(period)
    if offset is not None:
        mask   = full_df["Date"] >= (x_end - offset)
        rsi_df = full_df[mask].reset_index(drop=True)
        rsi_ma = rsi_ma_all[mask].reset_index(drop=True)
    else:
        rsi_df = full_df
        rsi_ma = rsi_ma_all

    rsi_last   = rsi_df["rsi"].iloc[-1]
    rsima_last = rsi_ma.iloc[-1]

    fig2 = go.Figure()
    fig2.add_hline(y=70, line=dict(color="#f85149", width=0.6, dash="dot"))
    fig2.add_hline(y=30, line=dict(color="#3fb950", width=0.6, dash="dot"))
    fig2.add_hline(y=50, line=dict(color="#2d3748", width=0.5, dash="dot"))

    fig2.add_trace(go.Scatter(
        x=rsi_df["Date"], y=rsi_ma.round(2),
        mode="lines", name="RSI MA14",
        line=dict(color="#58a6ff", width=1.2, dash="dot"),
        hovertemplate="RSI MA14: %{y:.1f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=rsi_df["Date"], y=rsi_df["rsi"].round(2),
        mode="lines", name="RSI",
        line=dict(color="#e3b341", width=1.5),
        hovertemplate="RSI: %{y:.1f}<extra></extra>",
    ))

    gap          = abs(rsi_last - rsima_last)
    rsi_yshift   = (+6 if rsi_last >= rsima_last else -6)  if gap < 5 else 0
    rsima_yshift = (-6 if rsi_last >= rsima_last else +6)  if gap < 5 else 0

    fig2.add_annotation(x=1.01, xref="paper", y=rsi_last, yref="y",
                        text=f"<b>RSI {rsi_last:.1f}</b>",
                        xanchor="left", showarrow=False,
                        font=dict(size=8, color="#e3b341", family="IBM Plex Mono"),
                        yshift=rsi_yshift)
    fig2.add_annotation(x=1.01, xref="paper", y=rsima_last, yref="y",
                        text=f"MA {rsima_last:.1f}",
                        xanchor="left", showarrow=False,
                        font=dict(size=8, color="#58a6ff", family="IBM Plex Mono"),
                        yshift=rsima_yshift)

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(showgrid=False, color="#5c7080",
                   tickfont=dict(size=9, family="IBM Plex Mono")),
        yaxis=dict(showgrid=False, color="#5c7080",
                   tickfont=dict(size=9, family="IBM Plex Mono"), range=[0, 100]),
        margin=dict(t=5, b=5, l=10, r=70), height=130,
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


def render_spx_chart(spx_df: pd.DataFrame, period: str = "1M") -> None:
    """Render S&P 500 overview chart filtered to the active period."""
    if spx_df is None:
        st.markdown(
            '<div style="height:300px;display:flex;align-items:center;'
            'justify-content:center;color:#5c7080;font-size:12px;'
            'font-family:IBM Plex Mono">No SPX data available</div>',
            unsafe_allow_html=True,
        )
        return

    full_spx = spx_df.copy()
    x_end    = full_spx["Date"].iloc[-1]
    offset   = _PERIOD_OFFSETS.get(period)
    if offset is not None:
        mask     = full_spx["Date"] >= (x_end - offset)
        plot_spx = full_spx[mask].reset_index(drop=True)
    else:
        plot_spx = full_spx

    if len(plot_spx) == 0:
        return

    spx_prev   = plot_spx["Close"].shift(1).fillna(plot_spx["Close"])
    spx_colors = ["#1de9b6" if c >= p else "#f85149"
                  for c, p in zip(plot_spx["Close"], spx_prev)]

    fig = go.Figure()
    has_ohlc_spx = all(c in plot_spx.columns for c in ["Open", "High", "Low"])
    if has_ohlc_spx:
        fig.add_trace(go.Candlestick(
            x=plot_spx["Date"],
            open=plot_spx["Open"], high=plot_spx["High"],
            low=plot_spx["Low"],   close=plot_spx["Close"],
            increasing_line_color="#1de9b6", decreasing_line_color="#f85149",
            increasing_fillcolor="#1de9b6",  decreasing_fillcolor="#f85149",
            name="SPX", whiskerwidth=0.4,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_spx["Date"], y=plot_spx["Close"].round(2),
            mode="markers",
            marker=dict(symbol="line-ew-open", color=spx_colors,
                        size=6, line=dict(width=2)),
            hovertemplate="%{x|%d %b %Y}<br><b>%{y:,.2f}</b><extra></extra>",
        ))

    y_lo = plot_spx["Low"].min()  * 0.98 if has_ohlc_spx else plot_spx["Close"].min() * 0.98
    y_hi = plot_spx["High"].max() * 1.02 if has_ohlc_spx else plot_spx["Close"].max() * 1.02

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
        xaxis=dict(
            showgrid=True, gridcolor="#111b27", color="#5c7080",
            tickfont=dict(size=10, family="IBM Plex Mono"),
            tickformatstops=_TICKFORMATSTOPS,
        ),
        yaxis=dict(showgrid=True, gridcolor="#111b27", color="#5c7080",
                   tickfont=dict(size=10, family="IBM Plex Mono"),
                   range=[y_lo, y_hi]),
        hoverlabel=_HOVERLABEL,
        margin=dict(t=10, b=10, l=10, r=10), height=300,
        showlegend=False, hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
