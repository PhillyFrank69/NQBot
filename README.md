# 🤖 NQBot — AI-Powered Futures Trading with NQ (Nasdaq 100)

NQBot is a modular machine learning pipeline designed to train and backtest predictive models for day trading the E-Mini Nasdaq-100 futures (`NQ`). It includes a clean ETL pipeline, feature engineering with technical indicators and order book microstructure, and integration with backtesting frameworks.

---

## 🧠 Key Features

- 📊 Support for minute and tick-level data
- ⚙️ Pluggable model architecture (e.g., XGBoost, TimesNet)
- 📈 Feature engineering with custom indicators and market microstructure
- 🔁 Walk-forward backtesting support
- 🎯 Designed for high-frequency & intraday trading on the NQ futures market

---

## 📁 Project Structure

NQBot/
├── train_model.py # Orchestrates the full pipeline
├── configs/
│ └── config.yaml # Data paths, hyperparameters
├── data/
│ └── loader.py # Polars/Dask-based ingestion
├── features/
│ ├── indicators.py # Technical indicators (VWAP, BB, Stochastics, etc.)
│ └── microstructure.py # Order book & trade flow features
├── models/
│ ├── xgb_head.py # XGBoost model wrapper
│ └── timesnet.py # Time series deep learning (optional)
├── backtest/
│ ├── vector_bt.py # Vectorized backtesting (FastQuant-style)
│ └── walk_forward.py # Rolling window strategy testing
└── pipeline.py # Dataflow and orchestration

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/NQBot.git
cd NQBot
2. Set Up Environment
Using Conda:
bash
Copy
Edit
conda create -n nqbot python=3.10
conda activate nqbot
pip install -r requirements.txt
📌 Make sure torch is installed with GPU support if you're using deep learning.

3. Configure
Edit configs/config.yaml to set paths to your data and model parameters.

⚙️ Running the Pipeline
Training a Model:
bash
Copy
Edit
python train_model.py --config configs/config.yaml --model xgb --niter 20
Running Backtest:
bash
Copy
Edit
python backtest/walk_forward.py --config configs/config.yaml --model xgb
📌 Requirements
Python 3.10+

Polars

Pandas, NumPy

scikit-learn, xgboost

PyTorch (for TimesNet or other deep learning models)

tqdm, pyyaml

📉 Data Sources
This repo is designed to work with:

Tick data (e.g., from NinjaTrader, Rithmic, or FirstRateData)

1-minute bars (2008+ preferred)

Ensure your data is properly formatted before use. Data loaders may need light customization.

📜 License
MIT License — see LICENSE for details.

🧠 Author
## 🧠 Author

Built by [PhillyFrank69](https://github.com/PhillyFrank69)


☕ Support
If you find this project helpful, give it a ⭐️ or share it with others in the trading/AI community!


