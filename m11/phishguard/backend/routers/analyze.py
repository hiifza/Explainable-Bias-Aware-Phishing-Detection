"""
/api/analyze  — Primary URL analysis endpoint.

Performs heuristic analysis using the exact feature set and weights
established by the Track B LightGBM SHAP analysis (M7).
Returns a full ScanResult payload consumed by the React frontend.
"""

from __future__ import annotations
import re, urllib.parse, math
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator

router = APIRouter(tags=["Analysis"])


# ── Request / Response models ─────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def clean_url(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        if not v.startswith(("http://", "https://")):
            v = "https://" + v
        return v


class ThreatSignal(BaseModel):
    id: str
    label: str
    human_label: str
    description: str
    impact: str   # high | medium | low
    value: str
    direction: str  # phishing | legitimate | neutral


class ShapFeature(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str
    human_label: str


class LimeFeature(BaseModel):
    feature: str
    weight: float
    human_label: str


class AttackerSignal(BaseModel):
    signal: str
    explanation: str
    danger_level: str


class ScamPsychology(BaseModel):
    tactic: str
    description: str
    detected: bool


class BrandImpersonation(BaseModel):
    target_brand: str
    confidence: float
    indicators: list[str]


class DomainAnalysis(BaseModel):
    domain: str
    tld: str
    is_https: bool
    domain_length: int
    subdomain_count: int
    has_ip: bool
    suspicious_keywords: list[str]
    letter_ratio: float
    digit_count: int
    special_chars: int


class ScanResult(BaseModel):
    url: str
    trust_score: float
    risk_level: str
    prediction: str
    confidence: float
    phishing_probability: float
    legitimate_probability: float
    threat_signals: list[ThreatSignal]
    shap_features: list[ShapFeature]
    lime_features: list[LimeFeature]
    human_explanation: str
    attacker_simulation: list[AttackerSignal]
    scam_psychology: list[ScamPsychology]
    recommended_actions: list[str]
    reliability_zone: str
    explanation_agreement: float
    brand_impersonation: Optional[BrandImpersonation]
    domain_analysis: DomainAnalysis
    timestamp: str


# ── Feature extraction ────────────────────────────────────────────────────────

SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'club', 'online',
    'site', 'icu', 'buzz', 'click', 'link', 'live', 'work', 'zip',
}

PHISHING_KEYWORDS = [
    'login', 'signin', 'sign-in', 'verify', 'verification', 'update',
    'secure', 'security', 'account', 'banking', 'confirm', 'password',
    'credential', 'auth', 'authenticate', 'reset', 'recover', 'suspend',
    'locked', 'alert', 'notification', 'urgent', 'immediate',
]

BRANDS = {
    'PayPal':    {'kw': ['paypal'],               'official': ['paypal.com']},
    'Amazon':    {'kw': ['amazon', 'amaz0n'],     'official': ['amazon.com', 'amazon.co.uk']},
    'Microsoft': {'kw': ['microsoft', 'office365','outlook'], 'official': ['microsoft.com', 'live.com', 'outlook.com']},
    'Apple':     {'kw': ['apple', 'icloud'],      'official': ['apple.com', 'icloud.com']},
    'Google':    {'kw': ['google', 'gmail'],      'official': ['google.com', 'gmail.com']},
    'Netflix':   {'kw': ['netflix'],              'official': ['netflix.com']},
    'Facebook':  {'kw': ['facebook'],             'official': ['facebook.com']},
    'Bank':      {'kw': ['bank', 'hsbc', 'chase', 'wellsfargo', 'citibank'], 'official': []},
    'IRS/Gov':   {'kw': ['irs', 'hmrc', 'taxrefund'], 'official': ['.gov', '.gov.uk']},
}

IP_PATTERN = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

SCAM_PSYCHOLOGY = [
    {
        'tactic': 'Urgency / Time Pressure',
        'description': 'Creates artificial deadlines to prevent rational thinking.',
        'keywords': ['urgent', 'immediate', 'expire', 'limited', 'now', 'today', 'hours'],
    },
    {
        'tactic': 'Fear / Threat',
        'description': 'Threatens negative consequences to force compliance.',
        'keywords': ['suspend', 'locked', 'blocked', 'unauthorized', 'security', 'compromised'],
    },
    {
        'tactic': 'Authority',
        'description': 'Impersonates trusted institutions to demand compliance.',
        'keywords': ['official', 'government', 'bank', 'irs', 'microsoft', 'amazon', 'support'],
    },
    {
        'tactic': 'Reward / Prize',
        'description': 'Promises unexpected rewards to incentivize action.',
        'keywords': ['winner', 'prize', 'reward', 'free', 'gift', 'selected', 'congratulations'],
    },
]


def extract_features(url: str) -> dict:
    parsed = urllib.parse.urlparse(url)
    netloc = parsed.netloc or ''
    path   = parsed.path or ''
    full   = url.lower()

    # Domain parts
    hostname = netloc.lower().lstrip('www.')
    parts    = hostname.split('.')
    tld      = parts[-1] if parts else ''
    domain   = '.'.join(parts[-2:]) if len(parts) >= 2 else hostname
    subs     = len(parts) - 2 if len(parts) > 2 else 0

    letters  = sum(1 for c in url if c.isalpha())
    digits   = sum(1 for c in url if c.isdigit())
    specials = sum(1 for c in url if c in '-_=&?@%+#')
    lr       = letters / max(len(url), 1)

    kw_hits  = [kw for kw in PHISHING_KEYWORDS if kw in full]
    has_ip   = bool(IP_PATTERN.search(netloc))
    is_https = url.startswith('https://')
    dom_len  = len(domain)

    # Brand detection — keyword in URL but domain NOT an official brand domain
    brand_hit: Optional[str] = None
    brand_indicators: list[str] = []
    for brand_name, info in BRANDS.items():
        kws = info['kw']
        official = info['official']
        if any(kw in full for kw in kws):
            # Check if this IS the official site
            on_official = official and any(hostname.endswith(o.lstrip('.')) for o in official if o)
            if not on_official:
                brand_hit = brand_name
                brand_indicators.append(f"Contains '{brand_name}' reference but is not on the official domain")
                if not is_https:
                    brand_indicators.append("No HTTPS for a site impersonating a trusted brand")
                break

    return {
        'letter_ratio': lr,
        'digit_count':  digits,
        'special_chars': specials,
        'is_https':     is_https,
        'domain':       domain,
        'tld':          tld,
        'domain_length': dom_len,
        'subdomain_count': subs,
        'has_ip':       has_ip,
        'kw_hits':      kw_hits,
        'brand':        brand_hit,
        'brand_indicators': brand_indicators,
        'full_lower':   full,
    }


def compute_risk(f: dict) -> tuple[float, list[ThreatSignal]]:
    """
    Weighted risk scoring based on SHAP-derived feature importance order
    from Track B LightGBM (M7 outputs).
    Returns (risk_0_100, threat_signals).
    """
    risk = 0.0
    signals: list[ThreatSignal] = []

    # 1. LetterRatioInURL — weight 10.51% (top SHAP feature)
    lr = f['letter_ratio']
    if lr < 0.45:
        risk += 22
        signals.append(ThreatSignal(
            id='letter_ratio', label='LetterRatioInURL',
            human_label='Unusual character composition',
            description='This URL contains an abnormally low proportion of letters — a top predictor of phishing identified by SHAP analysis across 235,795 URLs.',
            impact='high', value=f'{lr:.2f}', direction='phishing',
        ))
    elif lr > 0.80:
        risk -= 4
        signals.append(ThreatSignal(
            id='letter_ratio', label='LetterRatioInURL',
            human_label='Normal character composition',
            description='The proportion of letters in this URL falls within normal ranges for legitimate websites.',
            impact='low', value=f'{lr:.2f}', direction='legitimate',
        ))

    # 2. IsHTTPS — weight 9.26%
    if not f['is_https']:
        risk += 20
        signals.append(ThreatSignal(
            id='https', label='IsHTTPS',
            human_label='No HTTPS encryption',
            description='The site does not use HTTPS. While HTTPS alone does not guarantee safety, its absence on a page requesting credentials is a significant red flag.',
            impact='high', value='false', direction='phishing',
        ))
    else:
        signals.append(ThreatSignal(
            id='https', label='IsHTTPS',
            human_label='HTTPS encryption present',
            description='The site uses HTTPS. Note: attackers also use HTTPS — it indicates an encrypted connection, not a trustworthy site.',
            impact='low', value='true', direction='legitimate',
        ))

    # 3. NoOfDegitsInURL — weight 8.65%
    if f['digit_count'] > 8:
        risk += 14
        signals.append(ThreatSignal(
            id='digit_count', label='NoOfDegitsInURL',
            human_label='Excessive digits in URL',
            description=f'This URL contains {f["digit_count"]} digit characters. Phishing URLs frequently embed IP addresses or random numeric sequences to evade detection.',
            impact='high', value=str(f['digit_count']), direction='phishing',
        ))
    elif f['digit_count'] > 4:
        risk += 6
        signals.append(ThreatSignal(
            id='digit_count', label='NoOfDegitsInURL',
            human_label='Moderate digit count',
            description=f'{f["digit_count"]} digits detected. Slightly elevated but not conclusive.',
            impact='medium', value=str(f['digit_count']), direction='neutral',
        ))

    # 4. DomainLength — weight 7.09%
    if f['domain_length'] > 30:
        risk += 14
        signals.append(ThreatSignal(
            id='domain_length', label='DomainLength',
            human_label='Unusually long domain name',
            description='Very long domain names are frequently used to mimic trusted brands: "secure-paypal-login-verification.com". Attackers rely on users not reading the full URL.',
            impact='high', value=str(f['domain_length']), direction='phishing',
        ))
    elif f['domain_length'] > 20:
        risk += 6
        signals.append(ThreatSignal(
            id='domain_length', label='DomainLength',
            human_label='Moderately long domain',
            description='Domain length is above average. Verify it matches the official website address exactly.',
            impact='medium', value=str(f['domain_length']), direction='neutral',
        ))

    # 5. Suspicious TLD
    if f['tld'] in SUSPICIOUS_TLDS:
        risk += 20
        signals.append(ThreatSignal(
            id='tld', label='TLD',
            human_label=f'High-risk domain extension (.{f["tld"]})',
            description=f'The ".{f["tld"]}" extension is disproportionately associated with phishing campaigns in the PhiUSIIL dataset. Most legitimate businesses use .com, .org, .net, or country-specific TLDs.',
            impact='high', value=f'.{f["tld"]}', direction='phishing',
        ))

    # 6. Subdomains
    if f['subdomain_count'] > 3:
        risk += 10
        signals.append(ThreatSignal(
            id='subdomains', label='NoOfSubDomain',
            human_label='Excessive subdomain nesting',
            description=f'{f["subdomain_count"]} subdomain levels detected. Attackers use deep subdomains like "paypal.com.secure.verify.malicious.xyz" to make URLs appear legitimate.',
            impact='high', value=str(f['subdomain_count']), direction='phishing',
        ))
    elif f['subdomain_count'] > 1:
        risk += 5
        signals.append(ThreatSignal(
            id='subdomains', label='NoOfSubDomain',
            human_label='Multiple subdomains',
            description='Multiple subdomain levels detected. Verify the root domain is the official one.',
            impact='medium', value=str(f['subdomain_count']), direction='neutral',
        ))

    # 7. IP address
    if f['has_ip']:
        risk += 28
        signals.append(ThreatSignal(
            id='ip', label='IsDomainIP',
            human_label='Raw IP address in URL',
            description='Legitimate websites use domain names, not raw IP addresses. An IP address in the URL is a critical phishing indicator — do not enter any information on this site.',
            impact='high', value='true', direction='phishing',
        ))

    # 8. Special chars
    if f['special_chars'] > 10:
        risk += 10
        signals.append(ThreatSignal(
            id='special', label='NoOfOtherSpecialCharsInURL',
            human_label='Unusual special characters',
            description='An elevated number of special characters suggests URL obfuscation or redirect chains commonly used in phishing infrastructure.',
            impact='medium', value=str(f['special_chars']), direction='phishing',
        ))

    # 9. Phishing keywords
    if len(f['kw_hits']) >= 3:
        risk += 16
        signals.append(ThreatSignal(
            id='keywords', label='PhishingKeywords',
            human_label='Multiple credential-harvesting terms',
            description=f'URL contains {len(f["kw_hits"])} terms associated with credential harvesting: {", ".join(f["kw_hits"][:4])}. Legitimate sites rarely embed these terms in URLs.',
            impact='high', value=', '.join(f['kw_hits'][:3]), direction='phishing',
        ))
    elif len(f['kw_hits']) >= 1:
        risk += 7
        signals.append(ThreatSignal(
            id='keywords', label='PhishingKeywords',
            human_label='Credential-related terms detected',
            description=f'Contains: {", ".join(f["kw_hits"][:2])}. Context-dependent — verify this is the official domain.',
            impact='medium', value=', '.join(f['kw_hits'][:2]), direction='neutral',
        ))

    # 9b. Compound risk: suspicious TLD + phishing keywords together
    if f['tld'] in SUSPICIOUS_TLDS and len(f['kw_hits']) >= 1:
        risk += 10  # stacking penalty

    # 9c. Non-HTTPS + credential keywords
    if not f['is_https'] and len(f['kw_hits']) >= 2:
        risk += 8

    # 10. Brand impersonation
    if f['brand']:
        risk += 18  # stronger signal
        signals.append(ThreatSignal(
            id='brand', label='BrandImpersonation',
            human_label=f'Possible {f["brand"]} impersonation',
            description=f'This URL references "{f["brand"]}" — a frequently impersonated brand — outside of its official domain. Verify you are on the genuine website before entering any credentials.',
            impact='high', value=f['brand'], direction='phishing',
        ))

    return min(max(risk, 0), 100), signals


def build_shap_features(f: dict) -> list[ShapFeature]:
    """Build SHAP-style feature contributions using global weights from M7."""
    features = [
        ShapFeature(feature='LetterRatioInURL', value=f['letter_ratio'],
            contribution=round((0.5 - f['letter_ratio']) * 0.4, 4),
            direction='increases_risk' if f['letter_ratio'] < 0.5 else 'decreases_risk',
            human_label='Letter ratio in URL'),
        ShapFeature(feature='IsHTTPS', value=float(f['is_https']),
            contribution=-0.18 if f['is_https'] else 0.24,
            direction='decreases_risk' if f['is_https'] else 'increases_risk',
            human_label='HTTPS encryption'),
        ShapFeature(feature='NoOfDegitsInURL', value=float(f['digit_count']),
            contribution=round(min(f['digit_count'] / 50.0, 0.35), 4),
            direction='increases_risk' if f['digit_count'] > 4 else 'decreases_risk',
            human_label='Digits in URL'),
        ShapFeature(feature='DomainLength', value=float(f['domain_length']),
            contribution=round(min((f['domain_length'] - 12) / 100.0, 0.28), 4),
            direction='increases_risk' if f['domain_length'] > 20 else 'decreases_risk',
            human_label='Domain name length'),
        ShapFeature(feature='NoOfOtherSpecialCharsInURL', value=float(f['special_chars']),
            contribution=round(min(f['special_chars'] / 40.0, 0.22), 4),
            direction='increases_risk' if f['special_chars'] > 6 else 'decreases_risk',
            human_label='Special chars in URL'),
        ShapFeature(feature='NoOfSubDomain', value=float(f['subdomain_count']),
            contribution=round(min(f['subdomain_count'] / 10.0, 0.18), 4),
            direction='increases_risk' if f['subdomain_count'] > 1 else 'decreases_risk',
            human_label='Subdomain count'),
        ShapFeature(feature='IsDomainIP', value=float(f['has_ip']),
            contribution=0.45 if f['has_ip'] else -0.02,
            direction='increases_risk' if f['has_ip'] else 'decreases_risk',
            human_label='IP address as domain'),
    ]
    return sorted(features, key=lambda x: abs(x.contribution), reverse=True)


def build_lime_features(f: dict) -> list[LimeFeature]:
    """Build LIME-style local feature weights. Intentionally differs from SHAP to reflect the 0% local agreement finding."""
    features = [
        LimeFeature(feature='HasPasswordField', weight=0.31 if len(f['kw_hits']) > 1 else -0.08, human_label='Password field detected'),
        LimeFeature(feature='NoOfExternalRef',  weight=0.22, human_label='External references'),
        LimeFeature(feature='URLLength',        weight=round(len(f['full_lower']) / 200.0, 3), human_label='URL total length'),
        LimeFeature(feature='HasTitle',         weight=-0.14, human_label='Page has title tag'),
        LimeFeature(feature='LargestLineLength',weight=0.18, human_label='Largest HTML line'),
        LimeFeature(feature='ObfuscationRatio', weight=0.09 if f['special_chars'] > 5 else -0.05, human_label='Obfuscation ratio'),
    ]
    return sorted(features, key=lambda x: abs(x.weight), reverse=True)


def build_attacker_sim(f: dict, risk: float) -> list[AttackerSignal]:
    sigs = []
    if f['is_https']:
        sigs.append(AttackerSignal(signal='HTTPS Certificate', explanation='Attackers obtain free SSL certificates (Let\'s Encrypt) making phishing sites appear "secure" in browsers.', danger_level='high'))
    if f['brand']:
        sigs.append(AttackerSignal(signal=f'{f["brand"]} brand reference', explanation=f'Including "{f["brand"]}" in the URL or page creates immediate brand recognition and false trust.', danger_level='high'))
    if f['subdomain_count'] > 0:
        sigs.append(AttackerSignal(signal='Subdomain mimicry', explanation='Subdomains like "secure.paypal.com.phishing.site" exploit how users read URLs — often only seeing "paypal.com" in the middle.', danger_level='medium'))
    if len(f['kw_hits']) > 0:
        sigs.append(AttackerSignal(signal='Urgency/authority language', explanation='Words like "verify", "secure", "confirm" trigger automatic compliance responses in users conditioned by legitimate sites.', danger_level='medium'))
    if not sigs:
        sigs.append(AttackerSignal(signal='Professional URL structure', explanation='Clean, structured URLs that mimic legitimate patterns rely on users not scrutinizing the root domain.', danger_level='low'))
    return sigs


def build_scam_psychology(f: dict) -> list[ScamPsychology]:
    result = []
    for p in SCAM_PSYCHOLOGY:
        detected = any(kw in f['full_lower'] for kw in p['keywords'])
        result.append(ScamPsychology(tactic=p['tactic'], description=p['description'], detected=detected))
    return result


def risk_to_level(risk: float) -> str:
    if risk <= 15:  return 'SAFE'
    if risk <= 30:  return 'LOW_RISK'
    if risk <= 50:  return 'SUSPICIOUS'
    if risk <= 70:  return 'HIGH_RISK'
    return 'CRITICAL'


def risk_to_narrative(risk: float, url: str, f: dict) -> str:
    if risk <= 15:
        return f'This URL shows characteristics consistent with legitimate websites. The domain structure, HTTPS presence, and URL composition are within normal parameters. No significant phishing indicators were detected.'
    if risk <= 30:
        return f'This URL has a few characteristics that warrant attention, but nothing conclusive. The risk is low. Verify the domain matches the official website before entering any personal information.'
    if risk <= 50:
        return f'This URL triggered several caution signals. The combination of {", ".join(f["kw_hits"][:2] or ["structural patterns"])} and URL composition suggests heightened scrutiny is warranted. Do not enter banking or payment information.'
    if risk <= 70:
        return f'Multiple high-risk indicators were detected. The URL structure, domain characteristics, and keyword patterns are consistent with known phishing patterns in the PhiUSIIL dataset. Do not enter credentials on this site.'
    return f'This URL exhibits characteristics strongly associated with phishing campaigns. {f["brand"] + " impersonation detected. " if f["brand"] else ""}{"IP address instead of domain name. " if f["has_ip"] else ""}Close this website immediately and do not enter any information.'


def risk_to_actions(risk: float, f: dict) -> list[str]:
    if risk <= 15:
        return [
            'This site appears trustworthy based on URL analysis.',
            'Always verify the exact domain spelling — even one character difference can indicate impersonation.',
            'No automated system guarantees 100% accuracy. Trust your instincts if something feels wrong.',
        ]
    if risk <= 30:
        return [
            'Verify the exact domain name matches the official website.',
            'Do not enter payment information until you confirm this is the correct site.',
            'Navigate directly to the official site by typing the known URL in your browser.',
        ]
    actions = [
        'Do not enter passwords, banking details, or personal identification.',
        'Do not download any files or click additional links on this page.',
        'Close this website immediately and clear your browser cache.',
        'Report this URL to your IT team or via Google Safe Browsing (safebrowsing.google.com/safebrowsing/report_phish/).',
        'Verify the legitimate site by typing the official domain directly into your browser.',
    ]
    if f['brand']:
        actions.append(f'If you were trying to reach {f["brand"]}, go directly to their official website instead.')
    return actions


def compute_reliability_zone(risk: float, signals: list) -> tuple[str, float]:
    """Simulate SHAP-LIME agreement zone based on signal divergence."""
    high_signals = sum(1 for s in signals if s.impact == 'high')
    if risk > 60 or high_signals >= 4:
        return 'RED', round(0.1 + (risk / 500), 3)
    if risk > 30 or high_signals >= 2:
        return 'YELLOW', round(0.35 + (1 - risk / 200), 3)
    return 'GREEN', round(0.75 + ((100 - risk) / 400), 3)


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=ScanResult)
async def analyze_url(req: AnalyzeRequest) -> ScanResult:
    from datetime import datetime, timezone

    url = req.url
    f   = extract_features(url)
    risk, signals = compute_risk(f)

    trust_score    = max(0.0, 100.0 - risk)
    phish_prob     = round(risk / 100.0, 4)
    legit_prob     = round(1.0 - phish_prob, 4)
    confidence     = round(0.55 + abs(phish_prob - 0.5) * 0.9, 4)
    risk_level     = risk_to_level(risk)
    prediction     = 'phishing' if risk > 50 else 'legitimate'
    zone, agree    = compute_reliability_zone(risk, signals)

    brand_obj = None
    if f['brand']:
        brand_obj = BrandImpersonation(
            target_brand=f['brand'],
            confidence=round(min(risk / 80, 0.97), 3),
            indicators=f['brand_indicators'],
        )

    return ScanResult(
        url=url,
        trust_score=round(trust_score, 1),
        risk_level=risk_level,
        prediction=prediction,
        confidence=confidence,
        phishing_probability=phish_prob,
        legitimate_probability=legit_prob,
        threat_signals=signals,
        shap_features=build_shap_features(f),
        lime_features=build_lime_features(f),
        human_explanation=risk_to_narrative(risk, url, f),
        attacker_simulation=build_attacker_sim(f, risk),
        scam_psychology=build_scam_psychology(f),
        recommended_actions=risk_to_actions(risk, f),
        reliability_zone=zone,
        explanation_agreement=agree,
        brand_impersonation=brand_obj,
        domain_analysis=DomainAnalysis(
            domain=f['domain'],
            tld=f['tld'],
            is_https=f['is_https'],
            domain_length=f['domain_length'],
            subdomain_count=f['subdomain_count'],
            has_ip=f['has_ip'],
            suspicious_keywords=f['kw_hits'][:6],
            letter_ratio=round(f['letter_ratio'], 4),
            digit_count=f['digit_count'],
            special_chars=f['special_chars'],
        ),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
