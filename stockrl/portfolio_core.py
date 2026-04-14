from __future__ import annotations

from dataclasses import dataclass, replace


ACTION_SELL = 0
ACTION_HOLD = 1
ACTION_BUY = 2

POSITION_FLAT = 0
POSITION_LONG = 1


@dataclass(frozen=True)
class PortfolioState:
    cash: float
    shares: float
    position: int
    portfolio_value: float
    trade_count: int


def make_initial_state(initial_cash: float) -> PortfolioState:
    return PortfolioState(
        cash=initial_cash,
        shares=0.0,
        position=POSITION_FLAT,
        portfolio_value=initial_cash,
        trade_count=0,
    )


def apply_action(
    state: PortfolioState,
    action: int,
    execution_price: float,
    mark_price: float,
    fee_rate: float,
) -> PortfolioState:
    """
    Apply the action at the execution price, then mark the portfolio at mark_price.

    State machine:
      FLAT --buy--> LONG
      LONG --sell--> FLAT
      repeated buy/sell/hold -> no-op
    """
    next_state = state

    if action == ACTION_BUY and state.position == POSITION_FLAT:
        next_state = buy_all_cash(state=state, execution_price=execution_price, fee_rate=fee_rate)
    elif action == ACTION_SELL and state.position == POSITION_LONG:
        next_state = sell_all_shares(
            state=state,
            execution_price=execution_price,
            fee_rate=fee_rate,
        )

    portfolio_value = next_state.cash + (next_state.shares * mark_price)
    return replace(next_state, portfolio_value=portfolio_value)


def buy_all_cash(
    state: PortfolioState,
    execution_price: float,
    fee_rate: float,
) -> PortfolioState:
    if execution_price <= 0:
        raise ValueError("execution_price must be positive.")

    if state.cash <= 0:
        return state

    spendable_cash = state.cash / (1 + fee_rate)
    shares = spendable_cash / execution_price
    remaining_cash = state.cash - (shares * execution_price * (1 + fee_rate))

    return PortfolioState(
        cash=remaining_cash,
        shares=shares,
        position=POSITION_LONG,
        portfolio_value=state.portfolio_value,
        trade_count=state.trade_count + 1,
    )


def sell_all_shares(
    state: PortfolioState,
    execution_price: float,
    fee_rate: float,
) -> PortfolioState:
    if execution_price <= 0:
        raise ValueError("execution_price must be positive.")

    if state.shares <= 0:
        return state

    gross_cash = state.shares * execution_price
    net_cash = gross_cash * (1 - fee_rate)

    return PortfolioState(
        cash=state.cash + net_cash,
        shares=0.0,
        position=POSITION_FLAT,
        portfolio_value=state.portfolio_value,
        trade_count=state.trade_count + 1,
    )
