from lekiwi_node._envelope import parse_cmd_request, build_cmd_response, InvalidEnvelope


def test_parse_minimal_request():
    req = parse_cmd_request({"verb": "vendor.dora_nav.base.stop", "request_id": "r1"})
    assert req.verb == "vendor.dora_nav.base.stop"
    assert req.request_id == "r1"
    assert req.params == {}


def test_parse_missing_verb_raises():
    try:
        parse_cmd_request({"request_id": "r1"})
        assert False
    except InvalidEnvelope:
        pass


def test_build_response_shape():
    req = parse_cmd_request({"verb": "v", "request_id": "r2", "trace_id": "t"})
    resp = build_cmd_response(req, ok=True, code="0", data={"k": 1})
    assert resp["ok"] is True and resp["code"] == "0"
    assert resp["request_id"] == "r2" and resp["trace_id"] == "t"
    assert resp["data"] == {"k": 1} and "ts" in resp
