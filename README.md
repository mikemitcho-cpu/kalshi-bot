# kalshi-bot
Need help completing bot
"""
Kalshi AI Trading Bot
---------------------
Uses Claude AI for probability estimation, EV filtering,
and Quarter Kelly position sizing on Kalshi prediction markets.

Requirements:
    pip install kalshi-python anthropic python-dotenv

Setup:
    1. Copy .env.example to .env and fill in your credentials
    2. Run demo first:  python bot.py --demo
    3. Go live:         python bot.py --live
"""

import asyncio
import json
import logging
import math
import os
import uuid
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
import kalshi_api_v2_external
from kalshi_api_v2_external import ApiClient, Configuration, PortfolioApi, MarketsApi
from kalshi_api_v2_external.models import CreateOrderRequest
import anthropic

load_dotenv()

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", mode="a"),
    ],
)
log = logging.getLogger("kalshi_bot")


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    anthropic_api_key: str = ""

    # Strategy
    min_ev_threshold: float = 0.05    # Skip markets with EV < 5%
    kelly_fraction: float = 0.25      # Quarter Kelly
    max_position_usd: float = 150.0   # Hard cap per trade
    scan_interval_seconds: int = 180  # Scan every 3 minutes

    demo_mode: bool = True            # True = paper trade only

    @classmethod
    def from_env(cls, demo: bool = True) -> "BotConfig":
        return cls(
            kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
            kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            demo_mode=demo,
        )


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class MarketSignal:
    ticker: str
    title: str
    yes_price: float
    ev: float
    claude_prob: float
    confidence: str
    reasoning: str
    kelly_frac: float
    size_usd: float

@dataclass
class Position:
    ticker: str
    title: str
    side: str
    entry_price: float
    size_usd: float
    contracts: int
    order_id: str
    opened_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ─── Kalshi client ────────────────────────────────────────────────────────────

class KalshiClient:
    def __init__(self, config: BotConfig):
        self.config = config
        self.api_client: Optional[ApiClient] = None
        self.portfolio_api: Optional[PortfolioApi] = None
        self.markets_api: Optional[MarketsApi] = None

    def connect(self):
        cfg = Configuration()
        if self.config.demo_mode:
            cfg.host = "https://demo-api.kalshi.co/trade-api/v2"
            log.info("[AUTH] Connecting to Kalshi DEMO environment")
        else:
            cfg.host = "https://trading-api.kalshi.com/trade-api/v2"
            log.info("[AUTH] Connecting to Kalshi PRODUCTION environment")

        self.api_client = ApiClient(cfg)
        self.api_client.set_kalshi_auth(self.config.kalshi_api_key_id, self.config.kalshi_private_key_path)
        
        self.portfolio_api = PortfolioApi(self.api_client)
        self.markets_api = MarketsApi(self.api_client)
        
        log.info("[AUTH] Kalshi session established")

    def get_balance(self) -> float:
        try:
            resp = self.portfolio_api.get_balance()
            return resp.balance / 100.0  # Kalshi returns cents
        except Exception as e:
            # In demo mode, use default balance. In live, return 0 to be safe
            if self.config.demo_mode:
                log.warning(f"[BALANCE] Failed, using demo default: {e}")
                return 10000.0  # Default demo balance
            else:
                log.error(f"[BALANCE] Failed: {e}")
                return 0.0

    def get_markets(self, limit: int = 100) -> list:
        try:
            resp = self.markets_api.get_markets(limit=limit, status="open")
            return resp.markets or []
        except Exception as e:
            log.error(f"[SCAN] Failed to fetch markets: {e}")
            return []

    def place_order(self, ticker: str, side: str, price_cents: int, count: int) -> Optional[str]:
        order_id = str(uuid.uuid4())
        try:
            self.portfolio_api.create_order(CreateOrderRequest(
                ticker=ticker,
                action="buy",
                type="limit",
                yes_price=price_cents if side == "yes" else 100 - price_cents,
                count=count,
                client_order_id=order_id,
                side=side,
            ))
            log.info(f"[TRADE] Order placed: {ticker} {side} x{count} @ {price_cents}c | id={order_id}")
            return order_id
        except Exception as e:
            log.error(f"[TRADE] Order failed for {ticker}: {e}")
            return None


# ─── Claude AI brain ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert prediction market analyst.
Estimate the true probability a market resolves YES, considering base rates,
recent news, and historical patterns. Return ONLY valid JSON."""

USER_TEMPLATE = """Analyze this prediction market:

Title: {title}
Current market price (implied probability): {price:.1%}

Return JSON only, no extra text:
{{"probability": <float 0.0-1.0>, "confidence": "<high|medium|low>", "reasoning": "<2-3 sentences>"}}"""


class ClaudeBrain:
    def __init__(self, config: BotConfig):
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def analyze(self, title: str, price: float) -> Optional[dict]:
        try:
            msg = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(title=title, price=price)}],
            )
            raw = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"[CLAUDE] JSON parse error: {e}")
            return None
        except Exception as e:
            log.error(f"[CLAUDE] API error: {e}")
            return None


# ─── Math engine ──────────────────────────────────────────────────────────────

def compute_ev(claude_prob: float, market_price: float) -> float:
    """EV = P(win)*profit - P(lose)*loss for a YES bet."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 - market_price) / market_price
    return claude_prob * b - (1.0 - claude_prob)

def compute_kelly(claude_prob: float, market_price: float, fraction: float = 0.25) -> float:
    """Quarter Kelly bet fraction."""
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 - market_price) / market_price
    full_kelly = (claude_prob * b - (1.0 - claude_prob)) / b
    return max(0.0, full_kelly * fraction)

def log_return(p1: float, p0: float) -> float:
    """Log return for accurate P&L tracking."""
    if p0 <= 0:
        return 0.0
    return math.log(p1 / p0)


# ─── Position tracker ─────────────────────────────────────────────────────────

class PositionTracker:
    """In-memory tracker. Swap for SQLite for persistence across restarts."""

    def __init__(self):
        self.positions: dict[str, Position] = {}

    def add(self, pos: Position):
        self.positions[pos.ticker] = pos
        log.info(f"[POS] Opened: {pos.ticker} {pos.side} ${pos.size_usd:.2f} @ {pos.entry_price:.2f}")

    def has(self, ticker: str) -> bool:
        return ticker in self.positions

    def summary(self) -> str:
        if not self.positions:
            return "  (none)"
        return "\n".join(
            f"  {p.ticker}: {p.side} ${p.size_usd:.2f} @ {p.entry_price:.2f}"
            for p in self.positions.values()
        )


# ─── Bot core ─────────────────────────────────────────────────────────────────

class KalshiBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.kalshi = KalshiClient(config)
        self.brain = ClaudeBrain(config)
        self.tracker = PositionTracker()
        self.scan_count = 0

    def startup(self):
        self.kalshi.connect()
        balance = self.kalshi.get_balance()
        log.info(f"[BOOT] USD balance: ${balance:.2f}")
        log.info(f"[BOOT] Mode: {'DEMO (paper trading)' if self.config.demo_mode else 'LIVE'}")
        log.info(f"[BOOT] Min EV: {self.config.min_ev_threshold:.0%} | Kelly: {self.config.kelly_fraction} | Max size: ${self.config.max_position_usd:.0f}")

    def analyze_market(self, market) -> Optional[MarketSignal]:
        ticker = market.ticker
        title = getattr(market, "title", ticker)

        # Get yes price in cents, convert to decimal
        yes_price_cents = getattr(market, "yes_ask", None) or getattr(market, "last_price", 50)
        if yes_price_cents is None:
            return None
        yes_price = yes_price_cents / 100.0

        # Skip extreme prices — little edge near 0 or 1
        if yes_price < 0.05 or yes_price > 0.95:
            return None

        # Skip markets we already hold
        if self.tracker.has(ticker):
            return None

        # Claude probability estimate
        result = self.brain.analyze(title=title, price=yes_price)
        if not result:
            return None

        claude_prob = float(result.get("probability", yes_price))
        confidence = result.get("confidence", "low")
        reasoning = result.get("reasoning", "")

        # Skip low-confidence calls
        if confidence == "low":
            return None

        ev = compute_ev(claude_prob, yes_price)

        # EV filter — the single most important rule
        if ev < self.config.min_ev_threshold:
            log.info(f"[SKIP] {ticker} — EV {ev:.1%} < {self.config.min_ev_threshold:.0%} threshold")
            return None

        balance = self.kalshi.get_balance()
        kelly = compute_kelly(claude_prob, yes_price, self.config.kelly_fraction)
        size_usd = min(balance * kelly, self.config.max_position_usd)
        size_usd = max(size_usd, 1.0)

        log.info(f"[SIGNAL] {ticker} | price={yes_price:.2f} | claude={claude_prob:.2f} | EV={ev:.1%} | conf={confidence} | size=${size_usd:.2f}")
        return MarketSignal(
            ticker=ticker, title=title, yes_price=yes_price,
            ev=ev, claude_prob=claude_prob, confidence=confidence,
            reasoning=reasoning, kelly_frac=kelly, size_usd=size_usd,
        )

    def execute_signal(self, signal: MarketSignal):
        contracts = max(1, int(signal.size_usd / signal.yes_price))
        price_cents = int(signal.yes_price * 100)

        if self.config.demo_mode:
            order_id = f"DEMO-{uuid.uuid4().hex[:8]}"
            log.info(f"[DEMO] Paper trade: {signal.ticker} YES x{contracts} @ {price_cents}c | ${signal.size_usd:.2f}")
        else:
            order_id = self.kalshi.place_order(
                ticker=signal.ticker,
                side="yes",
                price_cents=price_cents,
                count=contracts,
            )
            if not order_id:
                return

        self.tracker.add(Position(
            ticker=signal.ticker, title=signal.title, side="yes",
            entry_price=signal.yes_price, size_usd=signal.size_usd,
            contracts=contracts, order_id=order_id,
        ))

    def run_scan(self):
        self.scan_count += 1
        log.info(f"[SCAN] === Cycle #{self.scan_count} ===")

        balance = self.kalshi.get_balance()
        log.info(f"[BALANCE] ${balance:.2f}")
        if balance < 1.0 and not self.config.demo_mode:
            log.warning("[BALANCE] Insufficient funds — skipping scan")
            return

        markets = self.kalshi.get_markets(limit=100)
        log.info(f"[SCAN] {len(markets)} open markets fetched")

        signals = 0
        for market in markets:
            try:
                signal = self.analyze_market(market)
                if signal:
                    signals += 1
                    self.execute_signal(signal)
            except Exception as e:
                log.error(f"[ERROR] {e}", exc_info=True)

        log.info(f"[SCAN] Cycle #{self.scan_count} done — {signals} signal(s)")
        log.info(f"[POS] Open positions:\n{self.tracker.summary()}")

    async def run_async(self):
        self.startup()
        log.info(f"[BOOT] Scanning every {self.config.scan_interval_seconds}s. Ctrl+C to stop.")
        while True:
            try:
                self.run_scan()
            except Exception as e:
                log.error(f"[ERROR] Scan crashed: {e}", exc_info=True)
            await asyncio.sleep(self.config.scan_interval_seconds)

    def run(self):
        asyncio.run(self.run_async())


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kalshi AI Trading Bot")
    parser.add_argument("--live", action="store_true", help="Trade with real money")
    args = parser.parse_args()

    demo = not args.live

    if not demo:
        print("\n*** LIVE MODE — real money will be used ***")
        confirm = input("Type CONFIRM to proceed: ")
        if confirm.strip() != "CONFIRM":
            print("Aborted.")
            return

    config = BotConfig.from_env(demo=demo)

    missing = [v for v, k in [("KALSHI_API_KEY_ID", config.kalshi_api_key_id),
                               ("KALSHI_PRIVATE_KEY_PATH", config.kalshi_private_key_path),
                               ("ANTHROPIC_API_KEY", config.anthropic_api_key)] if not k]
    if missing:
        print(f"Missing .env values: {', '.join(missing)}")
        print("See .env.example for setup instructions.")
        return

    KalshiBot(config).run()


if __name__ == "__main__":
    main()
