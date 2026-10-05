"""Block signups from known disposable / temporary-email providers.

This is a lightweight first line of defense against free-tier abuse (someone
spinning up throwaway accounts to farm free runs). It only catches the lazy
case — temp-mail websites. It does not verify that a real address exists; that
needs email verification with a sending provider.

The list covers the most common throwaway services. Matching is done on the
registered domain and any subdomain of it, case-insensitively.
"""

# Common disposable / temporary email domains. Kept deliberately curated rather
# than exhaustive — the goal is to stop casual abuse, not to win an arms race.
DISPOSABLE_DOMAINS: frozenset[str] = frozenset({
    "0clock.com", "0-mail.com", "10minutemail.com", "10minutemail.net",
    "20minutemail.com", "33mail.com", "guerrillamail.com", "guerrillamail.net",
    "guerrillamail.org", "guerrillamail.biz", "guerrillamailblock.com",
    "sharklasers.com", "grr.la", "spam4.me", "mailinator.com", "mailinator.net",
    "mailinator2.com", "notmailinator.com", "reconmail.com", "sogetthis.com",
    "trashmail.com", "trashmail.net", "trashmail.me", "trash-mail.com",
    "trbvm.com", "tempmail.com", "temp-mail.org", "temp-mail.io", "tempmail.net",
    "tempmailo.com", "tempr.email", "tmpmail.org", "tmpmail.net", "tmpeml.com",
    "tmails.net", "dispostable.com", "mailnesia.com", "mailcatch.com",
    "maildrop.cc", "mailnull.com", "getnada.com", "nada.email", "inboxbear.com",
    "throwawaymail.com", "throwam.com", "fakeinbox.com", "fakemailgenerator.com",
    "yopmail.com", "yopmail.net", "yopmail.fr", "cool.fr.nf", "jetable.fr.nf",
    "nospam.ze.tc", "mega.zik.dj", "speed.1s.fr", "moncourrier.fr.nf",
    "monemail.fr.nf", "monmail.fr.nf", "emailondeck.com", "emailtemporario.com.br",
    "mohmal.com", "mytemp.email", "burnermail.io", "33mail.com", "spambox.us",
    "maileater.com", "mailexpire.com", "mintemail.com", "mytrashmail.com",
    "spamgourmet.com", "spamgourmet.net", "spamgourmet.org", "incognitomail.org",
    "mailtothis.com", "deadaddress.com", "discard.email", "discardmail.com",
    "discardmail.de", "wegwerfmail.de", "wegwerfmail.net", "wegwerfmail.org",
    "einrot.com", "fleckens.hu", "harakirimail.com", "cuvox.de", "dayrep.com",
    "gustr.com", "jourrapide.com", "rhyta.com", "superrito.com", "teleworm.us",
    "armyspy.com", "mvrht.com", "0-mail.com", "crazymailing.com", "mailforspam.com",
    "tempinbox.com", "emlhub.com", "emlpro.com", "emltmp.com", "luxusmail.org",
    "mail-temp.com", "minuteinbox.com", "mail7.io", "generator.email",
    "mailpoof.com", "inboxkitten.com", "temporary-mail.net", "linshiyouxiang.net",
    "1secmail.com", "1secmail.net", "1secmail.org", "kzccv.com", "qiott.com",
    "wuuvo.com", "xojxe.com", "yoggm.com",
})


def is_disposable_email(email: str) -> bool:
    """True if the email's domain is a known disposable/temporary provider.

    Matches the exact domain and any subdomain of a blocked domain
    (e.g. foo.mailinator.com is blocked because mailinator.com is).
    """
    if not email or "@" not in email:
        return False
    domain = email.rsplit("@", 1)[-1].strip().lower().rstrip(".")
    if not domain:
        return False
    if domain in DISPOSABLE_DOMAINS:
        return True
    # Block subdomains of any listed domain.
    return any(domain.endswith("." + blocked) for blocked in DISPOSABLE_DOMAINS)
