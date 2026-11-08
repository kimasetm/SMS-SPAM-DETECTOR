"""Online threat-intelligence checks for links, and offline sender checks.

These complement the rule-based Link Safety Check (src/link_check.py):

  1. Phishing blocklist: is the link on the OpenPhish community feed, a free
     list of active phishing URLs updated about every 12 hours?
     https://openphish.com (community feed: non-commercial use)
  2. Domain age (RDAP): when was the domain registered? Scam sites are often
     only days or weeks old. RDAP is the official, free successor to WHOIS.
  3. Sender check: does the sender look like a premium-rate number, an email
     address posing as an SMS sender, or a spoofable sender name?

Privacy: checks 1 and 2 send the link's address or domain to a third party,
so the app only runs them when the user opts in. No API keys are needed, so
there are no secrets to store. Network errors never crash the app; the
result is reported as "couldn't check".
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit

from src.link_check import HIGH, LOW, MEDIUM, Finding, registered_domain

OPENPHISH_FEED_URL = "https://openphish.com/feed.txt"
RDAP_URL = "https://rdap.org/domain/{domain}"
TIMEOUT_SECONDS = 5
USER_AGENT = "SmartSpam/3.0 (student project; link safety check)"

# Hosts that serve many unrelated users' content. A phishing page on one of
# them doesn't make every link on that host phishing, so only exact URL
# matches count for these.
SHARED_HOSTS = {"docs.google.com", "drive.google.com", "sites.google.com", "forms.gle",
                "storage.googleapis.com", "github.io", "dropbox.com", "www.dropbox.com",
                "onedrive.live.com", "1drv.ms", "bit.ly", "tinyurl.com"}

NEW_DOMAIN_DAYS = 30        # registered less than a month ago: high risk
YOUNG_DOMAIN_DAYS = 180     # less than six months: medium risk


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read()


def _normalize_url(url: str) -> str:
    url = url.strip().lower()
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parts = urlsplit(url)
    return f"{parts.hostname or ''}{parts.path.rstrip('/')}"


# ---------- 1. Phishing blocklist ----------

@dataclass
class Blocklist:
    urls: set
    hosts: set
    source: str = "OpenPhish community feed"

    def matches(self, url: str) -> bool:
        normalized = _normalize_url(url)
        host = normalized.split("/", 1)[0]
        if normalized in self.urls:
            return True
        return host in self.hosts and host not in SHARED_HOSTS and registered_domain(host) not in SHARED_HOSTS


def parse_feed(text: str) -> Blocklist:
    urls = {_normalize_url(line) for line in text.splitlines() if line.strip()}
    return Blocklist(urls=urls, hosts={u.split("/", 1)[0] for u in urls})


def fetch_blocklist(get=_get) -> Blocklist:
    """Download the OpenPhish community feed. Raises an exception if it can't be reached."""
    return parse_feed(get(OPENPHISH_FEED_URL).decode("utf-8", errors="replace"))


# ---------- 2. Domain age ----------

def lookup_domain_age(domain: str, get=_get, today=None) -> dict:
    """Look up when a domain was registered, using RDAP.

    Returns {"status": "ok", "age_days": n, "registered": "YYYY-MM-DD"},
    {"status": "not_found"} if the registry has no record, or
    {"status": "error", "error": "..."} if the lookup failed.
    """
    today = today or datetime.now(timezone.utc)
    try:
        data = json.loads(get(RDAP_URL.format(domain=domain)))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {"status": "not_found"}
        return {"status": "error", "error": f"HTTP {error.code}"}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        return {"status": "error", "error": str(error)}

    for event in data.get("events", []):
        if event.get("eventAction") == "registration" and event.get("eventDate"):
            registered = datetime.fromisoformat(event["eventDate"].replace("Z", "+00:00"))
            if registered.tzinfo is None:
                registered = registered.replace(tzinfo=timezone.utc)
            return {"status": "ok", "registered": registered.date().isoformat(),
                    "age_days": (today - registered).days}
    return {"status": "error", "error": "no registration date in RDAP record"}


def domain_age_finding(domain: str, age: dict):
    """Turn a domain-age lookup into a Finding (or None when there's nothing to report)."""
    if age["status"] == "ok":
        days = age["age_days"]
        if days < NEW_DOMAIN_DAYS:
            return Finding(HIGH, f"Very new domain ({days} days old)",
                           f"'{domain}' was registered on {age['registered']}. Scam sites are often only days old.")
        if days < YOUNG_DOMAIN_DAYS:
            return Finding(MEDIUM, f"Young domain ({days} days old)",
                           f"'{domain}' was registered on {age['registered']}, less than six months ago.")
        return None
    if age["status"] == "not_found":
        return Finding(LOW, "No registration record found",
                       f"The domain registry has no RDAP record for '{domain}'. It may not exist, or the "
                       "registry may not publish this information.")
    return None


# ---------- Combining online checks with the Link Safety Check ----------

def add_online_findings(reports, blocklist=None, age_lookup=lookup_domain_age) -> dict:
    """Add blocklist and domain-age findings to LinkReports from src.link_check.

    blocklist: a Blocklist, or None if it couldn't be downloaded.
    Returns a status summary, e.g. {"blocklist": "checked", "domain_age": {...}}.
    """
    status = {"blocklist": "checked" if blocklist else "unavailable", "domain_age": {}}
    for report in reports:
        if blocklist and blocklist.matches(report.url):
            report.findings.insert(0, Finding(HIGH, "Listed as an active phishing site",
                                              f"This link appears on the {blocklist.source}."))
        if not report.host or re.fullmatch(r"[\d.]+", report.host):
            continue   # IP addresses have no domain registration
        domain = registered_domain(report.host)
        if domain in SHARED_HOSTS or report.host in SHARED_HOSTS:
            continue   # e.g. github.io: the platform's age says nothing about one user's page
        age = age_lookup(domain)
        status["domain_age"][domain] = age["status"]
        finding = domain_age_finding(domain, age)
        if finding:
            report.findings.append(finding)
    return status


# ---------- 3. Sender check (offline) ----------

def check_sender(sender: str) -> list:
    """Warning signs in the sender of a text message. Returns a list of Findings."""
    sender = (sender or "").strip()
    if not sender:
        return []
    findings = []
    digits = re.sub(r"[\s\-().]", "", sender)

    if re.fullmatch(r"\S+@\S+\.\S+", sender):
        findings.append(Finding(MEDIUM, "Sent from an email address",
                                "Texts that come from an email address instead of a phone number are a common way to "
                                "send bulk spam cheaply."))
    elif re.fullmatch(r"(\+44|0044|0)9\d{8,9}", digits):
        findings.append(Finding(HIGH, "Premium-rate number (UK 09…)",
                                "Calling or replying to UK numbers starting with 09 can cost several pounds per minute."))
    elif re.fullmatch(r"(\+1|1)?(900|976)\d{7}", digits):
        findings.append(Finding(HIGH, "Premium-rate number (US 900/976)",
                                "US numbers starting with 900 or 976 charge high fees per call."))
    elif re.fullmatch(r"\d{4,6}", digits):
        findings.append(Finding(LOW, "Short code",
                                "Short codes are used by banks and delivery companies, but also by premium-rate "
                                "subscription scams. Don't reply with STOP or YES unless you recognize the service."))
    elif re.search(r"[A-Za-z]", sender) and not re.search(r"@", sender):
        findings.append(Finding(LOW, "Sender name instead of a number",
                                f"Sender names like '{sender}' can be faked (spoofed), so the name alone doesn't prove "
                                "who sent the message."))
    elif not re.fullmatch(r"\+?\d{7,15}", digits):
        findings.append(Finding(MEDIUM, "Unusual sender format",
                                "This doesn't look like a normal phone number, short code or sender name."))
    return findings
