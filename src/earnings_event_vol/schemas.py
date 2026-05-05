from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class AnnouncementTiming(StrEnum):
    BMO = "BMO"
    AMC = "AMC"
    DMH = "DMH"
    UNKNOWN = "UNKNOWN"


class OptionRight(StrEnum):
    CALL = "call"
    PUT = "put"


class OptionSide(StrEnum):
    LONG = "long"
    SHORT = "short"


class QuoteSource(StrEnum):
    NBBO = "nbbo"
    CONSOLIDATED = "consolidated"
    SINGLE_EXCHANGE = "single_exchange"
    VENDOR = "vendor"
    UNKNOWN = "unknown"


class TimeConvention(StrEnum):
    ACT_365 = "ACT/365"
    TRADING_252 = "TRADING/252"
    VENDOR = "VENDOR"


class IVARFailureReason(StrEnum):
    NO_TWO_EVENT_COVERING_EXPIRIES = "no_two_event_covering_expiries"
    NONPOSITIVE_TIME_GAP = "nonpositive_time_gap"
    NEGATIVE_EXTRACTED_IVAR = "negative_extracted_ivar"
    STALE_OR_MISSING_IV = "stale_or_missing_iv"
    NONPOSITIVE_TOTAL_VARIANCE = "nonpositive_total_variance"
    NONMONOTONE_TOTAL_VARIANCE = "nonmonotone_total_variance"


class CostModel(StrEnum):
    MID = "mid"
    HALF_SPREAD = "half_spread"
    FULL_SPREAD_CROSSING = "full_spread_crossing"


class OptionQuote(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    quote_date: date
    expiration: date
    strike: float = Field(gt=0)
    right: OptionRight
    bid: float = Field(ge=0)
    ask: float = Field(gt=0)
    volume: int = Field(ge=0, default=0)
    open_interest: int = Field(ge=0, default=0)
    timestamp: datetime | None = None
    implied_vol: float | None = Field(default=None, gt=0)
    vendor_iv: float | None = Field(default=None, gt=0)
    local_iv: float | None = Field(default=None, gt=0)
    delta: float | None = None
    gamma: float | None = None
    vega: float | None = None
    quote_source: QuoteSource = QuoteSource.UNKNOWN
    option_multiplier: int = Field(default=100, gt=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @computed_field  # type: ignore[prop-decorator]
    @property
    def spread_over_mid(self) -> float:
        return float("inf") if self.mid <= 0 else self.spread / self.mid


class UnderlyingBar(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)
    vendor_halt_flag: bool | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def proxy_halt_flag(self) -> bool:
        unchanged = self.open == self.high == self.low == self.close
        return self.volume == 0 and unchanged


class EarningsEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    announcement_date: date
    announcement_timing: AnnouncementTiming
    source: str
    source_timestamp: datetime | None = None
    sector: str | None = None

    @field_validator("announcement_timing", mode="before")
    @classmethod
    def normalize_timing(cls, value: object) -> AnnouncementTiming:
        text = str(value).strip().upper()
        return (
            AnnouncementTiming(text)
            if text in AnnouncementTiming.__members__
            else AnnouncementTiming.UNKNOWN
        )


class EventWindow(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    announcement_date: date
    announcement_timing: AnnouncementTiming
    entry_date: date
    exit_date: date
    feature_cutoff_date: date
    event_entry_timestamp: datetime
    source: str
    sector: str | None = None
    exclusion_reason: str | None = None


class VarianceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    event_date: date
    rvar_event: float | None = None
    ivar_event: float | None = None
    edge_var: float | None = None
    time_convention: TimeConvention = TimeConvention.ACT_365
    failure_reason: IVARFailureReason | None = None


class FeatureRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    ticker: str
    event_date: date
    feature_asof_timestamp: datetime
    event_entry_timestamp: datetime


class SignalRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    event_date: date
    strategy: str
    forecast_rvar_event: float = Field(ge=0)
    ivar_event: float = Field(ge=0)
    edge_var: float
    expected_strategy_value_usd: float
    market_entry_cost_usd: float
    expected_strategy_edge_usd: float
    estimated_transaction_cost_usd: float = Field(ge=0)
    threshold_multiplier: float = Field(default=1.5, gt=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def should_trade(self) -> bool:
        threshold = self.threshold_multiplier * self.estimated_transaction_cost_usd
        return self.expected_strategy_edge_usd > threshold


class TradeLeg(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    expiration: date
    strike: float = Field(gt=0)
    right: OptionRight
    side: OptionSide
    contracts: float
    filled_price: float = Field(ge=0)
    filled_timestamp: datetime
    quote_source: QuoteSource = QuoteSource.UNKNOWN
    option_multiplier: int = Field(default=100, gt=0)


class StrategyTrade(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    event_date: date
    strategy: str
    sector: str | None = None
    expected_net_edge_usd: float
    max_theoretical_loss_usd: float = Field(gt=0)
    legs: tuple[TradeLeg, ...]
    fractional_contracts: bool = True
    integer_contracts: int | None = None
    zero_rounded: bool = False


class PnLRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    event_date: date
    strategy: str
    pnl_usd: float
    transaction_cost_usd: float = Field(ge=0)
    max_theoretical_loss_usd: float = Field(gt=0)
    fractional_contracts: bool = True
    zero_rounded: bool = False
