"""State machine conformance to Appendix A."""

from __future__ import annotations

import pytest

from core.state_machine import (
    ACTIVE_TASK_STATES,
    BROADCASTING,
    MOVEMENT_STATES,
    TRANSITIONS,
    Event,
    IllegalTransition,
    State,
    StateMachine,
    validate_table,
)


class TestTable:
    def test_table_is_well_formed(self) -> None:
        """Every state reachable, every state leavable, FAULT always available."""
        validate_table()

    def test_all_nine_appendix_a_states_exist(self) -> None:
        assert {s.value for s in State} == {
            "IDLE", "BIDDING", "PLANNING", "MOVING", "YIELD",
            "REPLAN", "AT_DROP", "CHARGING", "FAULT",
        }

    def test_yield_and_replan_return_to_moving(self) -> None:
        """Appendix A: they are excursions from MOVING, not fault states."""
        assert TRANSITIONS[State.YIELD][Event.CONFLICT_CLEARED] is State.MOVING
        assert TRANSITIONS[State.REPLAN][Event.ROUTE_REPAIRED] is State.MOVING
        assert State.YIELD in MOVEMENT_STATES

    def test_yield_is_a_moving_state(self) -> None:
        """FR-5.6: a yielding robot sheds speed, it does not stop. If YIELD were
        not a movement state the metrics would book it as stopped time."""
        assert State.YIELD in MOVEMENT_STATES
        assert State.CHARGING not in MOVEMENT_STATES

    def test_states_holding_work_are_identified(self) -> None:
        """FR-6.5 re-announces the task of a peer lost while holding work."""
        assert ACTIVE_TASK_STATES == {
            State.PLANNING, State.MOVING, State.YIELD, State.REPLAN, State.AT_DROP
        }
        assert State.IDLE not in ACTIVE_TASK_STATES
        assert State.BIDDING not in ACTIVE_TASK_STATES

    def test_every_state_declares_what_it_broadcasts(self) -> None:
        """A robot that stops sending INTENT is indistinguishable from a dead one
        after PEER_TIMEOUT_MS (FR-1.5), so no state may be silent."""
        for state in State:
            assert BROADCASTING[state], f"{state.value} broadcasts nothing"

    def test_unreachable_task_returns_to_idle_not_fault(self) -> None:
        """FR-3.7 / FR-6.2: the map changed, not the robot."""
        assert TRANSITIONS[State.PLANNING][Event.ROUTE_UNREACHABLE] is State.IDLE
        assert TRANSITIONS[State.REPLAN][Event.ROUTE_UNREACHABLE] is State.IDLE


class TestTransitions:
    def test_nominal_task_cycle(self) -> None:
        """The path a task actually takes, end to end."""
        machine = StateMachine()
        sequence = [
            (Event.TASK_ANNOUNCED, State.BIDDING),
            (Event.AUCTION_WON, State.PLANNING),
            (Event.ROUTE_READY, State.MOVING),
            (Event.LEG_COMPLETE, State.PLANNING),   # reached pickup
            (Event.ROUTE_READY, State.MOVING),
            (Event.ARRIVED_AT_DROP, State.AT_DROP),
            (Event.TASK_REPORTED, State.IDLE),
        ]
        for tick, (event, expected) in enumerate(sequence):
            assert machine.fire(event, tick * 20) is expected
        assert machine.transition_count == len(sequence)

    def test_losing_an_auction_returns_to_idle(self) -> None:
        machine = StateMachine()
        machine.fire(Event.TASK_ANNOUNCED, 0)
        assert machine.fire(Event.AUCTION_LOST, 300) is State.IDLE

    def test_yield_excursion_and_return(self) -> None:
        machine = StateMachine(state=State.MOVING)
        assert machine.fire(Event.CONFLICT_LOST, 100) is State.YIELD
        assert machine.fire(Event.CONFLICT_CLEARED, 900) is State.MOVING

    def test_yield_can_escalate_to_a_replan(self) -> None:
        """Appendix B AVOID: if speed shedding cannot absorb the shift, replan."""
        machine = StateMachine(state=State.YIELD)
        assert machine.fire(Event.EDGE_BLOCKED, 100) is State.REPLAN

    def test_illegal_transition_raises_rather_than_being_ignored(self) -> None:
        """An ignored event means state silently stops describing reality."""
        machine = StateMachine()
        with pytest.raises(IllegalTransition, match="not accepted in IDLE"):
            machine.fire(Event.ARRIVED_AT_DROP, 0)
        assert machine.state is State.IDLE

    def test_error_message_lists_what_is_accepted(self) -> None:
        machine = StateMachine(state=State.MOVING)
        with pytest.raises(IllegalTransition, match="ARRIVED_AT_DROP"):
            machine.fire(Event.CHARGED, 0)

    def test_fire_if_possible_reports_rather_than_raises(self) -> None:
        """BR-4: a BATTERY_LOW arriving mid-task is remembered, not forced -- an
        AMR below reserve finishes work it already holds."""
        machine = StateMachine(state=State.MOVING)
        assert machine.fire_if_possible(Event.BATTERY_LOW, 0) is False
        assert machine.state is State.MOVING
        assert machine.fire_if_possible(Event.ARRIVED_AT_DROP, 0) is True

    def test_fault_is_reachable_from_every_working_state(self) -> None:
        for state in ACTIVE_TASK_STATES | {State.IDLE, State.BIDDING, State.CHARGING}:
            machine = StateMachine(state=state)
            assert machine.fire(Event.FAULT_DETECTED, 0) is State.FAULT

    def test_fault_recovers_to_idle(self) -> None:
        machine = StateMachine(state=State.FAULT)
        assert machine.fire(Event.RECOVERED, 0) is State.IDLE

    def test_charging_cycle(self) -> None:
        machine = StateMachine()
        assert machine.fire(Event.BATTERY_LOW, 0) is State.CHARGING
        assert machine.fire(Event.CHARGED, 60_000) is State.IDLE


class TestHistory:
    def test_history_records_cause_and_time(self) -> None:
        """NFR-4.4: a safety decision must be reconstructable from its inputs."""
        machine = StateMachine()
        machine.fire(Event.TASK_ANNOUNCED, 1234)
        record = machine.last_transition
        assert record is not None
        assert (record.at_ms, record.from_state, record.to_state, record.event) == (
            1234, State.IDLE, State.BIDDING, Event.TASK_ANNOUNCED
        )

    def test_history_is_bounded_but_the_count_is_not(self) -> None:
        """A long scale100 run must not grow history without limit."""
        machine = StateMachine(history_limit=8)
        for tick in range(40):
            machine.fire(Event.BATTERY_LOW if tick % 2 == 0 else Event.CHARGED, tick)
        assert len(machine.history) == 8
        assert machine.transition_count == 40

    def test_recent_returns_the_tail(self) -> None:
        machine = StateMachine()
        machine.fire(Event.TASK_ANNOUNCED, 0)
        machine.fire(Event.AUCTION_WON, 300)
        recent = machine.recent(1)
        assert len(recent) == 1
        assert recent[0].event is Event.AUCTION_WON

    def test_reset_clears_everything(self) -> None:
        machine = StateMachine()
        machine.fire(Event.TASK_ANNOUNCED, 0)
        machine.reset()
        assert machine.state is State.IDLE
        assert machine.history == []
        assert machine.transition_count == 0


class TestDerivedProperties:
    def test_is_moving_tracks_movement_states(self) -> None:
        assert StateMachine(state=State.MOVING).is_moving
        assert StateMachine(state=State.YIELD).is_moving
        assert not StateMachine(state=State.IDLE).is_moving

    def test_holds_task_tracks_active_states(self) -> None:
        assert StateMachine(state=State.REPLAN).holds_task
        assert not StateMachine(state=State.BIDDING).holds_task

    def test_broadcasts_matches_appendix_a(self) -> None:
        assert "BID" in StateMachine(state=State.BIDDING).broadcasts
        assert "RESERVE" in StateMachine(state=State.MOVING).broadcasts
        assert "COMPLETE" in StateMachine(state=State.AT_DROP).broadcasts
