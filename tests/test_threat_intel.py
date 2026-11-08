"""Tests for threat-intelligence and sender checks.

All network calls are replaced with fakes, so these tests run offline and
never contact OpenPhish or RDAP.
"""

import json
import urllib.error
from datetime import datetime, timezone

import pytest

from src.link_check import HIGH, LOW, MEDIUM, NO_FLAGS, check_links
from src.threat_intel import (
    add_online_findings,
    check_sender,
    domain_age_finding,
    fetch_blocklist,
    lookup_domain_age,
    parse_feed,
)

FEED = "https://evil-login.example/verify\nhttp://phish.example/a/\nhttps://docs.google.com/forms/d/abc\n"
TODAY = datetime(2026, 11, 8, tzinfo=timezone.utc)


def rdap_response(registered: str):
    return lambda url: json.dumps({"events": [{"eventAction": "registration", "eventDate": registered}]}).encode()


def raise_http(code):
    def get(url):
        raise urllib.error.HTTPError(url, code, "error", {}, None)
    return get


# ---------- blocklist ----------

def test_blocklist_matches_listed_urls_and_hosts():
    blocklist = parse_feed(FEED)
    assert blocklist.matches("https://evil-login.example/verify")
    assert blocklist.matches("http://phish.example/a")                 # trailing slash ignored
    assert blocklist.matches("https://evil-login.example/other-page")  # same dedicated host
    assert not blocklist.matches("https://www.paypal.com")


def test_shared_hosts_need_an_exact_url_match():
    blocklist = parse_feed(FEED)
    assert blocklist.matches("https://docs.google.com/forms/d/abc")
    assert not blocklist.matches("https://docs.google.com/document/d/something-else")


def test_fetch_blocklist_uses_the_feed():
    blocklist = fetch_blocklist(get=lambda url: FEED.encode())
    assert len(blocklist.urls) == 3


# ---------- domain age ----------

def test_domain_age_lookup():
    age = lookup_domain_age("example.com", get=rdap_response("2026-11-01T00:00:00Z"), today=TODAY)
    assert age == {"status": "ok", "registered": "2026-11-01", "age_days": 7}


def test_domain_age_not_found_and_errors():
    assert lookup_domain_age("nope.example", get=raise_http(404))["status"] == "not_found"
    assert lookup_domain_age("x.example", get=raise_http(503))["status"] == "error"

    def offline(url):
        raise urllib.error.URLError("no network")
    assert lookup_domain_age("x.example", get=offline)["status"] == "error"


@pytest.mark.parametrize("age_days, severity", [(7, HIGH), (90, MEDIUM), (4000, None)])
def test_domain_age_findings(age_days, severity):
    finding = domain_age_finding("x.example", {"status": "ok", "age_days": age_days, "registered": "2026-01-01"})
    assert (finding.severity if finding else None) == severity


def test_missing_registration_record_is_only_low_risk():
    assert domain_age_finding("x.example", {"status": "not_found"}).severity == LOW
    assert domain_age_finding("x.example", {"status": "error", "error": "timeout"}) is None


# ---------- combining with the Link Safety Check ----------

def test_online_findings_are_added_to_link_reports():
    reports = check_links("Go to https://evil-login.example/verify or https://www.bbc.co.uk")
    ages = {"evil-login.example": {"status": "ok", "age_days": 3, "registered": "2026-11-05"},
            "bbc.co.uk": {"status": "ok", "age_days": 11000, "registered": "1994-12-13"}}
    status = add_online_findings(reports, parse_feed(FEED), age_lookup=lambda d: ages[d])

    assert status == {"blocklist": "checked", "domain_age": {"evil-login.example": "ok", "bbc.co.uk": "ok"}}
    titles = [f.title for f in reports[0].findings]
    assert titles[0] == "Listed as an active phishing site"
    assert "Very new domain (3 days old)" in titles
    assert reports[1].risk_level == NO_FLAGS


def test_unavailable_blocklist_and_ip_links_are_handled():
    reports = check_links("http://192.168.0.1/login")
    status = add_online_findings(reports, blocklist=None,
                                 age_lookup=lambda d: pytest.fail("IP addresses have no domain to look up"))
    assert status["blocklist"] == "unavailable"


def test_shared_platforms_skip_domain_age():
    reports = check_links("https://someone.github.io/page")
    add_online_findings(reports, None, age_lookup=lambda d: pytest.fail("github.io's age says nothing"))


# ---------- sender check ----------

@pytest.mark.parametrize("sender, title", [
    ("09061701461", "Premium-rate number (UK 09…)"),
    ("+44 906 170 1461", "Premium-rate number (UK 09…)"),
    ("1-900-555-0199", "Premium-rate number (US 900/976)"),
    ("80086", "Short code"),
    ("promo@deals-mail.com", "Sent from an email address"),
    ("AMAZON", "Sender name instead of a number"),
    ("??", "Unusual sender format"),
])
def test_sender_warning_signs(sender, title):
    assert [f.title for f in check_sender(sender)] == [title]


@pytest.mark.parametrize("sender", ["+1 555 123 4567", "07700 900123", "", None])
def test_normal_or_empty_senders_have_no_findings(sender):
    assert check_sender(sender) == []
