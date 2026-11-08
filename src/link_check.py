"""Link Safety Check: look for common phishing warning signs in URLs.

This is a separate, rule-based check that complements the neural network.
The model only sees every link as the placeholder word "urltoken", so it
can't tell a safe link from a dangerous one. These rules look at the link
itself.

Important:
  - Links are NEVER opened or visited. Only the text of the URL is analyzed,
    so it is safe to check real scam messages.
  - Finding no warning signs does NOT mean a link is safe. Real security tools
    also check threat-intelligence lists, domain age and the actual web page.

Usage:
    from src.link_check import check_links
    for report in check_links("Verify now at http://paypal-secure.xyz/login"):
        print(report.url, report.risk_level, [f.title for f in report.findings])
"""

import ipaddress
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

HIGH, MEDIUM, LOW = "high", "medium", "low"
NO_FLAGS = "none found"

# Domain endings that are cheap to register and frequently abused for scams
RISKY_TLDS = {"xyz", "top", "info", "click", "zip", "mov", "icu", "buzz", "live", "rest",
              "cam", "loan", "work", "support", "online", "site", "shop", "vip", "cyou"}
# Common endings, used to recognize links written without "http://" or "www."
COMMON_TLDS = {"com", "net", "org", "edu", "gov", "io", "co", "us", "uk", "ca", "au", "de",
               "me", "ly", "gl", "app", "dev", "biz", "ai", "in", "at", "cc"} | RISKY_TLDS
# Two-part endings where the registered name is the third label from the right
TWO_PART_SUFFIXES = {"co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "co.nz", "co.in"}

URL_SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
                  "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "t.ly", "s.id"}

# Brand name -> official registered domains. A domain that contains the brand
# name but isn't one of these is a common impersonation trick.
BRANDS = {
    "paypal": {"paypal.com"}, "amazon": {"amazon.com", "amazon.co.uk"}, "apple": {"apple.com"},
    "icloud": {"icloud.com"}, "microsoft": {"microsoft.com"}, "google": {"google.com"},
    "netflix": {"netflix.com"}, "usps": {"usps.com"}, "fedex": {"fedex.com"}, "ups": {"ups.com"},
    "dhl": {"dhl.com"}, "chase": {"chase.com"}, "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"}, "irs": {"irs.gov"}, "venmo": {"venmo.com"},
    "coinbase": {"coinbase.com"}, "facebook": {"facebook.com"}, "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com"},
}

# Words often found in phishing domain names
PHISHING_WORDS = {"login", "signin", "verify", "verification", "secure", "account", "update",
                  "unlock", "suspend", "suspended", "confirm", "billing", "redelivery",
                  "delivery", "refund", "wallet", "support", "reward", "prize"}

URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+"                                   # http://... or www....
    r"|(?<![@\w.-])(?:[a-z0-9-]+\.)+(?:"                                 # (not part of an email address)
    r"" + "|".join(sorted(COMMON_TLDS, key=len, reverse=True)) + r")"
    r"(?::\d+)?(?:/[^\s<>\"']*)?(?=[\s.,!?)]|$)",                       # bare domain like bit.ly/abc
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    severity: str      # "high", "medium" or "low"
    title: str
    explanation: str


@dataclass
class LinkReport:
    url: str
    host: str
    findings: list = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        severities = [f.severity for f in self.findings]
        if HIGH in severities or severities.count(MEDIUM) >= 2:
            return HIGH
        if MEDIUM in severities:
            return MEDIUM
        if LOW in severities:
            return LOW
        return NO_FLAGS


def extract_urls(text: str) -> list:
    """Find links in a message, without duplicates, in the order they appear."""
    if not isinstance(text, str):
        return []
    urls = []
    for match in URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(".,!?;:)")
        if url.lower() not in [u.lower() for u in urls]:
            urls.append(url)
    return urls


def registered_domain(host: str) -> str:
    """The part of a host name that's actually registered, e.g. 'login.paypal.com' -> 'paypal.com'."""
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in TWO_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def analyze_url(url: str) -> LinkReport:
    """Check one URL for phishing warning signs. The URL is never visited."""
    has_scheme = url.lower().startswith(("http://", "https://"))
    parts = urlsplit(url if has_scheme else "http://" + url)
    netloc = parts.netloc.lower()
    host = (parts.hostname or "").lower().rstrip(".")
    report = LinkReport(url=url, host=host)
    add = report.findings.append

    if "@" in netloc:
        add(Finding(HIGH, "Hidden destination (@ trick)",
                    f"Everything before the '@' is ignored by browsers; this link actually goes to '{host}'."))

    try:
        ipaddress.ip_address(host)
        add(Finding(HIGH, "Raw IP address instead of a domain name",
                    "Legitimate companies use domain names. Links to bare IP addresses are a classic phishing sign."))
        return report   # the domain checks below don't apply to IP addresses
    except ValueError:
        pass

    if "xn--" in host:
        add(Finding(HIGH, "Look-alike characters (punycode)",
                    "The domain uses characters from other alphabets that can imitate a real brand's name."))

    domain = registered_domain(host)
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    name_labels = re.split(r"[.-]", host[: -len(tld) - 1] if tld else host)

    for brand, official in BRANDS.items():
        mentions_brand = brand in name_labels or (len(brand) >= 5 and brand in host)
        if mentions_brand and domain not in official:
            add(Finding(HIGH, f"Uses the name '{brand}' on an unofficial domain",
                        f"The registered domain is '{domain}', not {' or '.join(sorted(official))}. "
                        "Scammers put brand names in their domains to look trustworthy."))
            break

    if domain in URL_SHORTENERS or host in URL_SHORTENERS:
        add(Finding(MEDIUM, "Link shortener hides the real destination",
                    f"'{domain}' redirects to another site, so you can't see where the link really goes."))

    if tld in RISKY_TLDS:
        add(Finding(MEDIUM, f"Frequently abused domain ending (.{tld})",
                    f"'.{tld}' domains are cheap to register and are often used for scam sites."))

    words = sorted({w for w in PHISHING_WORDS if w in host})
    if words:
        add(Finding(MEDIUM, "Phishing-style words in the domain",
                    f"The domain contains {', '.join(repr(w) for w in words)}, words scammers use to create urgency or "
                    "imitate account pages."))

    if parts.port and parts.port not in (80, 443):
        add(Finding(MEDIUM, f"Unusual port number ({parts.port})",
                    "Normal websites don't put a port number in the link."))

    if url.lower().startswith("http://"):
        add(Finding(LOW, "Not encrypted (http instead of https)",
                    "Information sent to this page isn't encrypted. Sites that ask for logins should use https."))

    if host.count(".") >= 4 or len(url) > 100:
        add(Finding(LOW, "Unusually long or complex link",
                    "Very long links or many subdomains are often used to hide the real domain."))

    return report


def check_links(text: str) -> list:
    """Analyze every link in a message. Returns a list of LinkReport (empty if there are no links)."""
    return [analyze_url(url) for url in extract_urls(text)]
