from __future__ import annotations

import pytest

from stockrl.portfolio_core import (
    ACTION_BUY,
    ACTION_SELL,
    POSITION_FLAT,
    POSITION_LONG,
    apply_action,
    make_initial_state,
)


def test_buy_from_flat_invests_cash_and_marks_long_position() -> None:
    state = make_initial_state(100.0)
    next_state = apply_action(
        state=state,
        action=ACTION_BUY,
        execution_price=10.0,
        mark_price=12.0,
        fee_rate=0.0,
    )

    assert next_state.position == POSITION_LONG
    assert next_state.shares == pytest.approx(10.0)
    assert next_state.cash == pytest.approx(0.0)
    assert next_state.portfolio_value == pytest.approx(120.0)


def test_sell_while_flat_is_a_safe_no_op() -> None:
    state = make_initial_state(100.0)
    next_state = apply_action(
        state=state,
        action=ACTION_SELL,
        execution_price=10.0,
        mark_price=10.0,
        fee_rate=0.001,
    )

    assert next_state.position == POSITION_FLAT
    assert next_state.cash == pytest.approx(100.0)
    assert next_state.shares == pytest.approx(0.0)


def test_sell_from_long_returns_to_cash() -> None:
    bought = apply_action(
        state=make_initial_state(100.0),
        action=ACTION_BUY,
        execution_price=10.0,
        mark_price=10.0,
        fee_rate=0.0,
    )
    sold = apply_action(
        state=bought,
        action=ACTION_SELL,
        execution_price=11.0,
        mark_price=11.0,
        fee_rate=0.0,
    )

    assert sold.position == POSITION_FLAT
    assert sold.shares == pytest.approx(0.0)
    assert sold.cash == pytest.approx(110.0)
    assert sold.portfolio_value == pytest.approx(110.0)


def test_repeated_buy_while_long_is_a_safe_no_op() -> None:
    bought = apply_action(
        state=make_initial_state(100.0),
        action=ACTION_BUY,
        execution_price=10.0,
        mark_price=10.0,
        fee_rate=0.0,
    )
    rebought = apply_action(
        state=bought,
        action=ACTION_BUY,
        execution_price=11.0,
        mark_price=11.0,
        fee_rate=0.0,
    )

    assert rebought.position == POSITION_LONG
    assert rebought.shares == pytest.approx(bought.shares)
