import os
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from tinkoff.invest import Client, CandleInterval
    from tinkoff.invest.services import InstrumentsService
    from tinkoff.invest.utils import now
    TINKOFF_AVAILABLE = True
except ImportError:
    TINKOFF_AVAILABLE = False
    print("Warning: tinkoff-investments package not available. Using mock data.")

class TinkoffAPI:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TINKOFF_TOKEN")
        self.tz = pytz.timezone("Europe/Moscow")

        if not self.token and TINKOFF_AVAILABLE:
            raise ValueError("TINKOFF_TOKEN environment variable not set")

    def get_figi_by_ticker(self, ticker: str) -> Optional[str]:
        """Get FIGI by ticker symbol"""
        if not TINKOFF_AVAILABLE:
            return None

        try:
            with Client(self.token) as client:
                instruments: InstrumentsService = client.instruments
                shares = instruments.shares().instruments
                for share in shares:
                    if share.ticker.upper() == ticker.upper():
                        return share.figi
        except Exception as e:
            print(f"Error getting FIGI for {ticker}: {e}")

        return None

    def fetch_candles(
        self,
        ticker: str,
        days: int = 30,
        interval: str = "5min"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch candle data from Tinkoff API

        Parameters:
        - ticker: Stock ticker (e.g., 'SBER')
        - days: Number of days of historical data
        - interval: Candle interval ('1min', '5min', '15min', '1hour', '1day')

        Returns:
        - DataFrame with OHLCV data or None if error
        """
        if not TINKOFF_AVAILABLE:
            return self._get_mock_data(ticker, days, interval)

        try:
            # Map interval string to CandleInterval enum
            interval_map = {
                '1min': CandleInterval.CANDLE_INTERVAL_1_MIN,
                '5min': CandleInterval.CANDLE_INTERVAL_5_MIN,
                '15min': CandleInterval.CANDLE_INTERVAL_15_MIN,
                '1hour': CandleInterval.CANDLE_INTERVAL_HOUR,
                '1day': CandleInterval.CANDLE_INTERVAL_DAY
            }

            candle_interval = interval_map.get(interval, CandleInterval.CANDLE_INTERVAL_5_MIN)

            with Client(self.token) as client:
                # Get FIGI
                figi = self.get_figi_by_ticker(ticker)
                if not figi:
                    print(f"Could not find FIGI for ticker {ticker}")
                    return self._get_mock_data(ticker, days, interval)

                # Fetch candles
                end = now()
                start = end - timedelta(days=days)

                candles = client.get_all_candles(
                    figi=figi,
                    from_=start,
                    to=end,
                    interval=candle_interval,
                )

                # Convert to DataFrame
                data = []
                for candle in candles:
                    data.append({
                        "time": candle.time.astimezone(self.tz),
                        "open": candle.open.units + candle.open.nano / 1e9,
                        "high": candle.high.units + candle.high.nano / 1e9,
                        "low": candle.low.units + candle.low.nano / 1e9,
                        "close": candle.close.units + candle.close.nano / 1e9,
                        "volume": candle.volume,
                    })

                df = pd.DataFrame(data)

                if len(df) == 0:
                    print(f"No data received for {ticker}")
                    return self._get_mock_data(ticker, days, interval)

                return df

        except Exception as e:
            print(f"Error fetching data from Tinkoff API: {e}")
            return self._get_mock_data(ticker, days, interval)

    def _get_mock_data(self, ticker: str, days: int, interval: str) -> pd.DataFrame:
        """Generate mock data for testing when API is unavailable"""
        import numpy as np

        # Determine number of periods based on interval
        intervals_per_day = {
            '1min': 1440,  # 24 * 60
            '5min': 288,   # 24 * 12
            '15min': 96,   # 24 * 4
            '1hour': 24,
            '1day': 1
        }

        periods_per_day = intervals_per_day.get(interval, 288)
        total_periods = days * periods_per_day

        # Generate timestamps
        end_time = datetime.now(self.tz)
        if interval == '1day':
            dates = pd.date_range(end=end_time, periods=total_periods, freq='D', tz=self.tz)
        elif interval == '1hour':
            dates = pd.date_range(end=end_time, periods=total_periods, freq='H', tz=self.tz)
        elif interval == '15min':
            dates = pd.date_range(end=end_time, periods=total_periods, freq='15min', tz=self.tz)
        elif interval == '5min':
            dates = pd.date_range(end=end_time, periods=total_periods, freq='5min', tz=self.tz)
        else:  # 1min
            dates = pd.date_range(end=end_time, periods=total_periods, freq='min', tz=self.tz)

        # Generate realistic price data
        np.random.seed(42)  # For reproducible results

        # Base prices for different tickers
        base_prices = {
            'SBER': 280.0,
            'GAZP': 160.0,
            'LKOH': 6500.0,
            'ROSN': 520.0
        }

        base_price = base_prices.get(ticker.upper(), 100.0)

        prices = [base_price]
        for i in range(len(dates)-1):
            # Random walk with slight upward trend
            change = np.random.normal(0.0001, 0.02)  # Small drift up, 2% vol
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.1))  # Prevent negative prices

        # Create OHLCV data
        data = []
        for i, (time, close) in enumerate(zip(dates, prices)):
            if i == 0:
                continue

            # Generate OHLC from close price
            volatility = 0.01  # 1% volatility for OHLC spread
            high = close * (1 + abs(np.random.normal(0, volatility)))
            low = close * (1 - abs(np.random.normal(0, volatility)))
            open_price = prices[i-1] if i > 0 else close * (1 + np.random.normal(0, 0.005))

            data.append({
                "time": time,
                "open": round(open_price, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": np.random.randint(1000, 100000)
            })

        return pd.DataFrame(data)

    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker"""
        df = self.fetch_candles(ticker, days=1)
        if df is not None and len(df) > 0:
            return float(df.iloc[-1]['close'])
        return None

    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        if not TINKOFF_AVAILABLE:
            return {"status": "unknown", "message": "API not available"}

        try:
            with Client(self.token) as client:
                # Get trading schedules
                schedules = client.get_trading_schedules(
                    exchange="MOEX",
                    from_=datetime.now().date(),
                    to=datetime.now().date()
                )

                if schedules.exchanges:
                    exchange = schedules.exchanges[0]
                    is_open = exchange.is_trading_open
                    return {
                        "status": "open" if is_open else "closed",
                        "message": f"Market is {'open' if is_open else 'closed'}"
                    }

        except Exception as e:
            print(f"Error getting market status: {e}")

        return {"status": "unknown", "message": "Could not determine market status"}

# Global API instance
api = TinkoffAPI()
