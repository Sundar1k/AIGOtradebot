"""close_position must cancel resting brackets — the naked-position fix.

A market close (/Position/closeContract) does NOT fire a bracket leg, so the OCO
never auto-cancels and the protective stop is left WORKING. If price later returns
to it, it fills and opens a brand-new NAKED position. close_position now sweeps
/Order/searchOpen and cancels every working order for the contract.
"""
import broker


class StubClient(broker.TopstepXClient):
    """A TopstepXClient with the network stubbed: records every _post call and
    returns canned gateway responses."""

    def __init__(self, open_orders):
        super().__init__("u", "k")
        self._token = "t"                  # pretend authenticated
        self._open_orders = open_orders
        self.calls = []

    def _post(self, path, payload, auth=True):
        self.calls.append((path, payload))
        if path == "/Order/searchOpen":
            return {"orders": self._open_orders}
        return {"success": True}           # closeContract + cancel succeed

    def cancelled_ids(self):
        return [p["orderId"] for path, p in self.calls if path == "/Order/cancel"]


def test_close_cancels_only_this_contracts_working_orders():
    orders = [
        {"id": 1, "contractId": "CON.F.US.NQ.U6", "type": 4, "status": 1},  # our SL
        {"id": 2, "contractId": "CON.F.US.NQ.U6", "type": 1, "status": 1},  # our TP
        {"id": 3, "contractId": "CON.F.US.ES.U6", "type": 4, "status": 1},  # other contract
    ]
    c = StubClient(orders)
    c.close_position(123, "CON.F.US.NQ.U6")
    # both NQ legs cancelled, the ES order is left alone
    assert c.cancelled_ids() == [1, 2]


def test_close_flattens_before_cancelling():
    c = StubClient([{"id": 1, "contractId": "NQ", "type": 4, "status": 1}])
    c.close_position(123, "NQ")
    paths = [path for path, _ in c.calls]
    assert paths[0] == "/Position/closeContract"   # flatten first
    assert "/Order/cancel" in paths                # then cancel the orphan


def test_close_succeeds_even_if_searchopen_fails():
    # a best-effort cancel must never turn a successful flatten into a failure
    c = StubClient([])

    real_post = c._post

    def flaky(path, payload, auth=True):
        if path == "/Order/searchOpen":
            raise RuntimeError("gateway hiccup")
        return real_post(path, payload, auth)

    c._post = flaky
    assert c.close_position(123, "NQ").get("success") is True


def test_no_orphan_left_when_no_brackets():
    c = StubClient([])               # nothing resting
    c.close_position(123, "NQ")
    assert c.cancelled_ids() == []
