---
title: UrbanFlow Traffic Forecast Demo
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.58.0"
app_file: app/demo/streamlit_app.py
pinned: false
license: mit
---

# UrbanFlow — Probabilistic Traffic Forecast Demo

Pick an urban junction and see a spatio-temporal traffic forecast with a calibrated
90% prediction interval, served from an XGBoost Quantile Regression model trained
on real hourly traffic-volume data (4 junctions, Nov 2015 – Jun 2017).

Full project, code, and benchmark comparisons (classical baselines, GRU, Temporal
Fusion Transformer, MC Dropout, quantile regression): [github.com/NSVEGUR/urban-flow](https://github.com/NSVEGUR/urban-flow)
