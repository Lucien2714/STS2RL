import pytest

from sts2rl.env import mcp_client
from sts2rl.env.mcp_client import STS2Client, STS2ClientError


class FakeResponse:
    def __init__(self, data, status_code=200, text=""):
        self.data = data
        self.status_code = status_code
        self.text = text

    def json(self):
        return self.data


class FakeSession:
    def __init__(self, response=None):
        self.response = response or FakeResponse({"state_type": "menu"})
        self.requests = []
        self.closed = False

    def request(self, **kwargs):
        self.requests.append(kwargs)
        return self.response

    def close(self):
        self.closed = True


def test_client_builds_request_from_complete_base_url():
    session = FakeSession()
    client = STS2Client(
        base_url="http://localhost:16666/api/v1/",
        timeout=3.5,
        session=session,
    )

    state = client.get_state()

    assert state == {"state_type": "menu"}
    assert session.requests == [
        {
            "method": "GET",
            "url": "http://localhost:16666/api/v1/singleplayer",
            "params": {"format": "json"},
            "json": None,
            "timeout": 3.5,
        }
    ]


def test_client_does_not_close_injected_session():
    session = FakeSession()
    client = STS2Client(session=session)

    client.close()

    assert session.closed is False


def test_context_manager_closes_owned_session(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(mcp_client.requests, "Session", lambda: session)

    with STS2Client() as client:
        assert client.session is session
        assert session.closed is False

    assert session.closed is True


def test_http_error_with_non_object_json_has_clear_message():
    session = FakeSession(FakeResponse(["backend failure"], status_code=500))
    client = STS2Client(session=session)

    try:
        client.get_state()
    except STS2ClientError as exc:
        assert str(exc) == "HTTP 500: ['backend failure']"
    else:
        raise AssertionError("Expected STS2ClientError")


def test_rejected_actions_carry_the_unchanged_state():
    """The API returns HTTP 200 with status=error and the state it did not change."""
    session = FakeSession(
        FakeResponse(
            {
                "status": "error",
                "error": "card_index 99 out of range",
                "state": {"state_type": "monster"},
            }
        )
    )

    with pytest.raises(STS2ClientError, match="out of range") as caught:
        STS2Client(session=session).play_card(99)

    assert caught.value.state == {"state_type": "monster"}


def test_errors_without_a_state_carry_none():
    session = FakeSession(FakeResponse({"error": "Not found"}, status_code=404))

    with pytest.raises(STS2ClientError) as caught:
        STS2Client(session=session).get_state()

    assert caught.value.state is None


def test_accepted_actions_pause_before_the_next_request(monkeypatch):
    """The next action must not reach a game still resolving the last one."""
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    client = STS2Client(session=FakeSession(), action_delay_seconds=0.1)

    client.play_card(0)
    client.end_turn()

    assert slept == [0.1, 0.1]


def test_reading_state_does_not_pause(monkeypatch):
    """Only actions change the game, so only actions are worth waiting on."""
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    client = STS2Client(session=FakeSession(), action_delay_seconds=0.1)

    client.get_state()

    assert slept == []


def test_rejected_actions_do_not_pause(monkeypatch):
    """A rejected action changed nothing, so there is nothing to settle."""
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    session = FakeSession(
        FakeResponse({"status": "error", "error": "no", "state": {"state_type": "map"}})
    )
    client = STS2Client(session=session, action_delay_seconds=0.1)

    with pytest.raises(STS2ClientError):
        client.play_card(0)

    assert slept == []


def test_a_zero_delay_skips_the_call_entirely(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(mcp_client.time, "sleep", slept.append)
    client = STS2Client(session=FakeSession(), action_delay_seconds=0.0)

    client.end_turn()

    assert slept == []


def test_a_negative_delay_is_rejected():
    with pytest.raises(ValueError, match="action_delay_seconds"):
        STS2Client(action_delay_seconds=-0.1)
