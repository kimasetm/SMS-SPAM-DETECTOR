"""Tests for the rule-based Link Safety Check."""

import pytest

from src.link_check import HIGH, LOW, MEDIUM, NO_FLAGS, analyze_url, check_links, extract_urls, registered_domain


def titles(url):
    return [f.title for f in analyze_url(url).findings]


# ---------- finding links ----------

@pytest.mark.parametrize("text, expected", [
    ("Reschedule here: http://parcel-redelivery.info", ["http://parcel-redelivery.info"]),
    ("visit www.bbc.co.uk/news.", ["www.bbc.co.uk/news"]),          # trailing period removed
    ("bit.ly/3xYzAbc win now", ["bit.ly/3xYzAbc"]),                 # no http:// or www.
    ("(see example.com)", ["example.com"]),
    ("a http://x.com b http://x.com", ["http://x.com"]),             # duplicates removed
])
def test_extract_urls(text, expected):
    assert extract_urls(text) == expected


@pytest.mark.parametrize("text", [
    "my email is john.doe@gmail.com thanks",    # email addresses aren't links
    "Meet at 5.30 ok? I'm at the U.S. office, e.g. tomorrow",
    "",
    None,
])
def test_text_without_links(text):
    assert extract_urls(text) == []


def test_registered_domain():
    assert registered_domain("login.paypal.com") == "paypal.com"
    assert registered_domain("www.bbc.co.uk") == "bbc.co.uk"
    assert registered_domain("accounts.google.com.security-check.top") == "security-check.top"


# ---------- warning signs ----------

def test_ip_address_link_is_high_risk():
    report = analyze_url("http://192.168.4.20/claim")
    assert report.risk_level == HIGH
    assert "Raw IP address instead of a domain name" in titles(report.url)


def test_at_sign_trick_reveals_real_destination():
    report = analyze_url("http://www.paypal.com@evil-site.ru/login")
    assert report.host == "evil-site.ru"
    assert report.risk_level == HIGH


def test_brand_on_unofficial_domain():
    assert analyze_url("https://paypal-account-verify.xyz/login").risk_level == HIGH
    assert any("'google'" in t for t in titles("https://accounts.google.com.security-check.top/x"))


def test_punycode_lookalike():
    assert "Look-alike characters (punycode)" in titles("https://xn--pple-43d.com")


def test_shortener_risky_tld_and_phishing_words():
    assert "Link shortener hides the real destination" in titles("bit.ly/3xYz")
    assert "Frequently abused domain ending (.xyz)" in titles("https://free-stuff.xyz")
    assert "Phishing-style words in the domain" in titles("https://secure-bank-login.com")


def test_unusual_port_and_http():
    assert "Unusual port number (8080)" in titles("https://example.com:8080/a")
    assert "Not encrypted (http instead of https)" in titles("http://example.com")


def test_risk_levels():
    assert analyze_url("http://example.com").risk_level == LOW                         # one low finding
    assert analyze_url("https://free-stuff.xyz").risk_level == MEDIUM                  # one medium finding
    assert analyze_url("http://parcel-redelivery.info").risk_level == HIGH             # two mediums


# ---------- no false alarms on legitimate links ----------

@pytest.mark.parametrize("url", [
    "https://www.paypal.com/signin",
    "https://www.amazon.com/orders",
    "https://www.ups.com/track",       # "ups" must match the official domain
    "https://zoom.us/j/123",
    "www.bbc.co.uk/news",
    "https://www.google.com/maps",
])
def test_legitimate_links_have_no_flags(url):
    assert analyze_url(url).risk_level == NO_FLAGS


def test_brand_name_inside_another_word_is_not_flagged():
    # "groups" contains "ups" but isn't impersonating UPS
    assert analyze_url("https://groups.example.org").risk_level == NO_FLAGS


def test_check_links_analyzes_every_link():
    reports = check_links("Pay at http://192.168.0.1/pay or https://www.paypal.com")
    assert [r.risk_level for r in reports] == [HIGH, NO_FLAGS]
    assert check_links("no links here") == []


def test_links_are_never_visited(monkeypatch):
    """The checker must not make network requests."""
    import socket

    def fail(*args, **kwargs):
        raise AssertionError("Link check tried to use the network")

    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket, "getaddrinfo", fail)
    check_links("http://parcel-redelivery.info and bit.ly/abc and https://www.paypal.com")
