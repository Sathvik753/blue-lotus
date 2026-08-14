import React from "react";
import { Link } from "react-router-dom";
import Logo from "../components/Logo";

/* ─────────────────────────────────────────────────────────────────────────
   FILL THESE IN before relying on these documents:
     ENTITY  → your registered legal entity name once the LLC is formed
     STATE   → the US state whose law governs (usually your state of formation)
   These are strong starting drafts. Have a fintech/securities attorney review
   them (esp. the investment-adviser posture) before taking firm customers.
   ───────────────────────────────────────────────────────────────────────── */
const COMPANY = "Blue Lotus Labs";
const ENTITY = "Blue Lotus Labs";      // ← update to "<Your LLC>, LLC"
const STATE = "Michigan";              // governing-law state
const EFFECTIVE = "July 2026";

function LegalShell({ title, children }) {
  return (
    <div className="fade-in" style={{ maxWidth: 760, margin: "0 auto", padding: "40px 24px 80px" }}>
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 40 }}>
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Logo size={34} />
          <span className="gradient-text" style={{ fontFamily: "Syne, sans-serif", fontWeight: 800 }}>Blue Lotus</span>
        </Link>
        <Link to="/" style={{ color: "var(--muted)", fontSize: 13 }}>← Home</Link>
      </header>

      <h1 style={{ fontSize: 30, marginBottom: 6 }} className="gradient-text">{title}</h1>
      <p style={{ color: "var(--muted)", fontSize: 13, marginBottom: 8 }}>Effective date: {EFFECTIVE}</p>

      <div style={{ height: 1, background: "var(--border-soft)", margin: "18px 0 28px" }} />

      <div style={{ color: "var(--light)", fontSize: 14.5, lineHeight: 1.72 }}>{children}</div>

      <footer style={{ marginTop: 48, paddingTop: 20, borderTop: "1px solid var(--border-soft)",
        display: "flex", gap: 18, flexWrap: "wrap", fontSize: 12.5, color: "var(--muted)" }}>
        <Link to="/terms" style={{ color: "var(--muted)" }}>Terms</Link>
        <Link to="/privacy" style={{ color: "var(--muted)" }}>Privacy</Link>
        <Link to="/disclaimer" style={{ color: "var(--muted)" }}>Disclaimer</Link>
      </footer>
    </div>
  );
}

const H = ({ children }) => (
  <h2 style={{ fontSize: 17, fontWeight: 700, color: "var(--white)", margin: "28px 0 8px" }}>{children}</h2>
);
const P = ({ children }) => <p style={{ margin: "0 0 12px" }}>{children}</p>;
const UL = ({ children }) => (
  <ul style={{ margin: "0 0 12px", paddingLeft: 20, display: "flex", flexDirection: "column", gap: 6 }}>{children}</ul>
);
const Strong = ({ children }) => <strong style={{ color: "var(--white)" }}>{children}</strong>;

// ─────────────────────────────────────────────────────────── DISCLAIMER
export function Disclaimer() {
  return (
    <LegalShell title="Disclaimer">
      <div style={{ background: "rgba(224,86,63,0.08)", border: "1px solid rgba(224,86,63,0.28)",
        borderRadius: 10, padding: "16px 18px", marginBottom: 20 }}>
        <P><Strong>Blue Lotus is a risk-analytics tool, not an investment adviser.</Strong> It is provided for
          informational and educational purposes only. Nothing on this site or produced by the software is
          investment, financial, legal, tax, or accounting advice, or a recommendation to buy, sell, or hold any
          security, instrument, or strategy.</P>
      </div>

      <H>Not investment advice</H>
      <P>{COMPANY} is not a registered investment adviser, broker-dealer, or commodity trading advisor, and does
        not act as a fiduciary to you. Use of the software does not create any advisory or fiduciary relationship.
        You are solely responsible for your own investment and trading decisions, and you should consult a licensed
        professional before acting on any output.</P>

      <H>Outputs are model estimates, not predictions</H>
      <P>The software produces <Strong>probability distributions</Strong> of possible outcomes (such as drawdown,
        tail loss, and recovery) that are conditional on the modelling assumptions and on the historical data
        supplied to it. These outputs are estimates produced by statistical models. They are not forecasts,
        guarantees, or assurances of any result, and they can be materially wrong — particularly during market
        crises and other events without close historical precedent, where models calibrated to past data are known
        to under-state risk. <Strong>Past performance and historical model calibration do not guarantee future
        results.</Strong></P>

      <H>No warranty of accuracy</H>
      <P>Market and reference data may be sourced from third parties and may be delayed, incomplete, or inaccurate.
        {" "}{COMPANY} does not warrant the accuracy, completeness, timeliness, or fitness for any purpose of any
        data or output. All use is at your own risk. See the <Link to="/terms" style={{ color: "var(--gold)" }}>
        Terms of Service</Link> for the full limitation of liability.</P>
    </LegalShell>
  );
}

// ─────────────────────────────────────────────────────────── TERMS
export function Terms() {
  return (
    <LegalShell title="Terms of Service">
      <P>These Terms of Service (“Terms”) are a binding agreement between you (“you”) and {ENTITY} (“{COMPANY}”,
        “we”, “us”) governing your access to and use of the {COMPANY} website, software, and services (the
        “Service”). By creating an account or using the Service, you agree to these Terms. If you do not agree, do
        not use the Service.</P>

      <H>1. Eligibility and accounts</H>
      <P>You must be at least 18 years old and able to form a binding contract. If you use the Service on behalf of
        an organization, you represent that you are authorized to bind it, and “you” includes that organization.
        You are responsible for the accuracy of your account information, for keeping your credentials secure, and
        for all activity under your account.</P>

      <H>2. The Service — what it is and is not</H>
      <P>The Service is a software tool that computes probabilistic risk analytics from return series and other
        market data that you supply or that we retrieve from third-party sources. <Strong>The Service does not
        provide investment advice and is not an investment adviser, broker-dealer, or fiduciary.</Strong> Outputs
        are informational only. See the <Link to="/disclaimer" style={{ color: "var(--gold)" }}>Disclaimer</Link>,
        which is incorporated into these Terms. You are solely responsible for any decision you make.</P>

      <H>3. Acceptable use</H>
      <P>You agree not to: (a) use the Service for any unlawful purpose or in violation of any regulation; (b)
        upload data you do not have the right to use; (c) reverse engineer, decompile, or attempt to extract the
        source code, models, or methods of the Service; (d) resell, sublicense, or provide the Service to third
        parties except as expressly permitted; (e) scrape, overload, probe, or interfere with the Service or its
        infrastructure; or (f) use the Service to build a competing product.</P>

      <H>4. Subscriptions, billing, and cancellation</H>
      <P>Paid plans are billed in advance on a recurring basis through our payment processor (Stripe). By
        subscribing you authorize recurring charges until you cancel. <Strong>You may cancel at any time</Strong>,
        effective at the end of the current billing period; unless required by law, fees already paid are
        non-refundable and partial periods are not prorated. We may change prices or plan features on prospective
        notice. Failure to pay may result in suspension or termination.</P>

      <H>5. Your data and content</H>
      <P>You retain ownership of the return series and other data you upload (“Your Data”). You grant {COMPANY} a
        limited, non-exclusive license to host, process, and analyze Your Data solely to provide and improve the
        Service. We do not sell Your Data. Our handling of personal information is described in the{" "}
        <Link to="/privacy" style={{ color: "var(--gold)" }}>Privacy Policy</Link>. You are responsible for
        maintaining your own copies of Your Data.</P>

      <H>6. Intellectual property</H>
      <P>The Service — including its software, engine, statistical models, methodology, interfaces, and
        documentation — is owned by {COMPANY} and protected by intellectual-property laws. We grant you a limited,
        revocable, non-transferable license to use the Service in accordance with these Terms and your plan. All
        rights not expressly granted are reserved.</P>

      <H>7. Third-party data and services</H>
      <P>The Service may retrieve market data from, and rely on, third-party sources and providers (including
        hosting, payment, and data vendors). Such data and services are provided “as is”, and {COMPANY} is not
        responsible for their accuracy, availability, or acts and omissions.</P>

      <H>8. Disclaimer of warranties</H>
      <P><Strong>THE SERVICE AND ALL OUTPUTS ARE PROVIDED “AS IS” AND “AS AVAILABLE”, WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED,</Strong> including any implied warranties of merchantability, fitness for a particular
        purpose, accuracy, non-infringement, or uninterrupted or error-free operation. {COMPANY} does not warrant
        that any output is accurate, reliable, or suitable for any purpose. Some jurisdictions do not allow the
        exclusion of implied warranties, so parts of this section may not apply to you.</P>

      <H>9. Limitation of liability</H>
      <P><Strong>To the maximum extent permitted by law, {COMPANY} and its owners will not be liable for any
        indirect, incidental, special, consequential, exemplary, or punitive damages, or for any lost profits,
        lost data, or trading or investment losses,</Strong> arising out of or relating to the Service, even if
        advised of the possibility. Our total aggregate liability for all claims relating to the Service will not
        exceed the greater of (a) the amount you paid us in the twelve months before the event giving rise to the
        claim, or (b) US $100. Some jurisdictions do not allow certain limitations, so parts of this section may
        not apply to you.</P>

      <H>10. Indemnification</H>
      <P>You agree to indemnify and hold harmless {COMPANY} and its owners from any claims, damages, and expenses
        (including reasonable legal fees) arising from your use of the Service, Your Data, or your violation of
        these Terms or of any law or third-party right.</P>

      <H>11. Termination</H>
      <P>You may stop using the Service at any time. We may suspend or terminate your access if you violate these
        Terms, fail to pay, or if we discontinue the Service. Sections that by their nature should survive
        termination (including ownership, disclaimers, limitation of liability, indemnification, and dispute
        resolution) will survive.</P>

      <H>12. Governing law and dispute resolution</H>
      <P>These Terms are governed by the laws of the State of {STATE}, without regard to conflict-of-laws rules.
        <Strong> Any dispute arising out of or relating to these Terms or the Service will be resolved by binding
        individual arbitration</Strong>, and not in court, except that either party may bring an individual claim
        in small-claims court. <Strong>You and {COMPANY} waive any right to a jury trial and to participate in a
        class, collective, or representative action.</Strong> If the class-action waiver is found unenforceable,
        the arbitration agreement is void as to that claim and it proceeds in court in {STATE}.</P>

      <H>13. Changes</H>
      <P>We may update these Terms; material changes will be posted here with a new effective date and, where
        appropriate, notice to you. Continued use after changes take effect constitutes acceptance.</P>

      <H>14. Contact</H>
      <P>Questions about these Terms may be directed to {COMPANY} through the {COMPANY} website.</P>
    </LegalShell>
  );
}

// ─────────────────────────────────────────────────────────── PRIVACY
export function Privacy() {
  return (
    <LegalShell title="Privacy Policy">
      <P>This Privacy Policy explains how {ENTITY} (“{COMPANY}”, “we”, “us”) collects, uses, and protects
        information when you use the {COMPANY} website and services (the “Service”).</P>

      <H>1. Information we collect</H>
      <UL>
        <li><Strong>Account information</Strong> — your name, email address, organization name, and hashed
          password. We never store your password in plaintext.</li>
        <li><Strong>Content you provide</Strong> — the return series and parameters you submit for analysis, and
          the results we generate for you.</li>
        <li><Strong>Payment information</Strong> — processed by our payment provider, Stripe. We do not receive or
          store your full card number; we retain only limited billing metadata (such as plan and status).</li>
        <li><Strong>Usage and technical data</Strong> — log data, IP address, timestamps, and basic diagnostics
          used to operate, secure, and improve the Service. We use only essential cookies necessary for
          authentication and sessions.</li>
      </UL>

      <H>2. How we use information</H>
      <P>We use information to provide and operate the Service, authenticate you, process payments, provide
        support, maintain security and prevent abuse, meter usage against your plan, and improve the Service. We
        may use aggregated or de-identified data that does not identify you.</P>

      <H>3. How we share information</H>
      <P>We do not sell your personal information. We share it only with service providers that help us run the
        Service, under confidentiality obligations, including: <Strong>Stripe</Strong> (payments),
        <Strong> Google Cloud</Strong> (hosting and database), and our transactional email provider. We may also
        disclose information if required by law or to protect our rights, or in connection with a business
        transfer.</P>

      <H>4. Your data and tenant isolation</H>
      <P>The return series and results in your account are private to your organization and are logically isolated
        from other customers. We use them only to provide the Service to you and do not sell them or use them to
        train models for third parties.</P>

      <H>5. Data retention</H>
      <P>We retain account and content data for as long as your account is active and as needed to provide the
        Service, comply with legal obligations, resolve disputes, and enforce agreements. You may request deletion
        of your account and associated data as described below.</P>

      <H>6. Security</H>
      <P>We protect information using encryption in transit (HTTPS), access controls, per-tenant isolation, and
        secret management. No method of transmission or storage is completely secure, and we cannot guarantee
        absolute security.</P>

      <H>7. Your rights</H>
      <P>You may access, correct, or request deletion of your personal information by contacting us. Depending on
        where you live, you may have additional rights under laws such as the California Consumer Privacy Act
        (CCPA/CPRA) or, for individuals in the EEA/UK, the GDPR — including rights to access, correct, delete, and
        object to certain processing. We will honor applicable rights on verified request.</P>

      <H>8. Children</H>
      <P>The Service is not directed to and may not be used by anyone under 18. We do not knowingly collect
        information from children.</P>

      <H>9. International users</H>
      <P>The Service is operated from the United States and information is processed and stored in the United
        States. By using the Service you consent to this processing.</P>

      <H>10. Changes</H>
      <P>We may update this Policy; changes will be posted here with a new effective date.</P>

      <H>11. Contact</H>
      <P>Privacy questions or requests may be directed to {COMPANY} through the {COMPANY} website.</P>
    </LegalShell>
  );
}
