"""Subscription plans, usage metering, and Stripe integration.

Stripe is optional: if STRIPE_SECRET_KEY is unset the module runs in *mock mode*,
where checkout returns a local URL and plans can be switched manually for demos.
Set the Stripe env vars to make billing live without any code change.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Organization, Run, PlanTier, SubscriptionStatus

# --- Plan catalogue -------------------------------------------------------------
# monthly_runs = metered quota; None = unlimited. price_usd is display only.
PLANS = {
    PlanTier.free: {
        "name": "Free",
        "price_usd": 0,
        "monthly_runs": 25,
        "blurb": "Evaluation and light single-asset use.",
        "features": ["25 stress runs / month", "JSON + PDF export", "Single seat"],
        "stripe_price_env": None,
    },
    PlanTier.pro: {
        "name": "Pro",
        "price_usd": 1000,
        "monthly_runs": 1500,
        "blurb": "Desk-grade risk for a small fund or trading team.",
        "features": [
            "1,500 stress runs / month",
            "Full API access + API keys",
            "Priority support",
            "Up to 10 seats",
        ],
        "stripe_price_env": "STRIPE_PRICE_PRO",
    },
    PlanTier.enterprise: {
        "name": "Enterprise",
        "price_usd": None,   # "Contact us"
        "monthly_runs": None,
        "blurb": "Unlimited runs, SSO, and a dedicated environment.",
        "features": [
            "Unlimited runs",
            "SSO / SAML",
            "Dedicated deployment + SLA",
            "Custom scenarios & onboarding",
        ],
        "stripe_price_env": "STRIPE_PRICE_ENTERPRISE",
    },
}

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:3000")

STRIPE_ENABLED = bool(STRIPE_SECRET_KEY)


def _plan(org: Organization) -> dict:
    tier = org.plan if org.plan in PLANS else PlanTier.free
    return PLANS[tier]


def month_key(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y-%m")


async def runs_this_month(db: AsyncSession, org_id: str) -> int:
    """Source of truth for metering: count of runs created in the current UTC month."""
    start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    q = await db.execute(
        select(func.count(Run.id)).where(Run.org_id == org_id, Run.created_at >= start)
    )
    return int(q.scalar() or 0)


async def usage_summary(db: AsyncSession, org: Organization) -> dict:
    used = await runs_this_month(db, org.id)
    limit = _plan(org)["monthly_runs"]
    return {
        "plan": org.plan,
        "plan_name": _plan(org)["name"],
        "subscription_status": org.subscription_status,
        "period": month_key(),
        "runs_used": used,
        "runs_limit": limit,
        "runs_remaining": (None if limit is None else max(0, limit - used)),
        "stripe_enabled": STRIPE_ENABLED,
    }


async def enforce_quota(db: AsyncSession, org: Organization, user=None) -> None:
    """Raise 402 if the org has exhausted its monthly run quota.

    Developer accounts are exempt: they exist to test and operate the system,
    and metering them just produces support noise."""
    if user is not None:
        from api.auth import is_developer
        if is_developer(user):
            return
    limit = _plan(org)["monthly_runs"]
    if limit is None:
        return
    used = await runs_this_month(db, org.id)
    if used >= limit:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Monthly run limit reached ({used}/{limit} on the "
                f"{_plan(org)['name']} plan). Upgrade to continue."
            ),
        )


async def create_checkout_session(db: AsyncSession, org: Organization, tier: str) -> dict:
    """Create a Stripe Checkout session, or a mock in dev mode."""
    if tier not in PLANS or tier == PlanTier.free:
        raise HTTPException(status_code=400, detail="Choose a paid plan.")

    price_env = PLANS[tier]["stripe_price_env"]

    if not STRIPE_ENABLED:
        # Mock mode: flip the plan immediately so the flow is demoable end-to-end.
        org.plan = tier
        org.subscription_status = SubscriptionStatus.active
        await db.commit()
        return {
            "mode": "mock",
            "checkout_url": f"{APP_BASE_URL}/billing?mock_upgraded={tier}",
            "message": "Stripe not configured — plan switched in mock mode.",
        }

    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    price_id = os.environ.get(price_env or "", "")
    if not price_id:
        raise HTTPException(status_code=500, detail=f"Missing {price_env} configuration.")

    if not org.stripe_customer_id:
        customer = stripe.Customer.create(metadata={"org_id": org.id}, name=org.name)
        org.stripe_customer_id = customer.id
        await db.commit()

    session = stripe.checkout.Session.create(
        mode="subscription",
        customer=org.stripe_customer_id,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{APP_BASE_URL}/billing?checkout=success",
        cancel_url=f"{APP_BASE_URL}/billing?checkout=cancel",
        metadata={"org_id": org.id, "tier": tier},
    )
    return {"mode": "live", "checkout_url": session.url}


async def create_portal_session(org: Organization) -> dict:
    """Stripe billing portal for managing/canceling a subscription."""
    if not STRIPE_ENABLED or not org.stripe_customer_id:
        return {"mode": "mock", "portal_url": f"{APP_BASE_URL}/billing"}
    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    session = stripe.billing_portal.Session.create(
        customer=org.stripe_customer_id,
        return_url=f"{APP_BASE_URL}/billing",
    )
    return {"mode": "live", "portal_url": session.url}


async def handle_webhook(db: AsyncSession, payload: bytes, sig_header: str) -> dict:
    """Verify and apply a Stripe webhook event to the matching organization."""
    if not STRIPE_ENABLED:
        return {"received": True, "mode": "mock"}

    import stripe
    stripe.api_key = STRIPE_SECRET_KEY
    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        else:
            event = stripe.Event.construct_from(
                stripe.util.json.loads(payload.decode()), stripe.api_key
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid webhook: {e}")

    etype = event["type"]
    obj = event["data"]["object"]

    async def _org_by_customer(cust_id):
        r = await db.execute(select(Organization).where(Organization.stripe_customer_id == cust_id))
        return r.scalar_one_or_none()

    if etype == "checkout.session.completed":
        org_id = (obj.get("metadata") or {}).get("org_id")
        tier = (obj.get("metadata") or {}).get("tier", PlanTier.pro)
        r = await db.execute(select(Organization).where(Organization.id == org_id))
        org = r.scalar_one_or_none()
        if org:
            org.plan = tier
            org.subscription_status = SubscriptionStatus.active
            org.stripe_subscription_id = obj.get("subscription")
            await db.commit()

    elif etype in ("customer.subscription.updated", "customer.subscription.deleted"):
        org = await _org_by_customer(obj.get("customer"))
        if org:
            status_map = {
                "active": SubscriptionStatus.active,
                "trialing": SubscriptionStatus.trialing,
                "past_due": SubscriptionStatus.past_due,
                "canceled": SubscriptionStatus.canceled,
            }
            org.subscription_status = status_map.get(obj.get("status"), SubscriptionStatus.inactive)
            if etype == "customer.subscription.deleted" or obj.get("status") == "canceled":
                org.plan = PlanTier.free
            await db.commit()

    return {"received": True, "type": etype}
