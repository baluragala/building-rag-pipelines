# Acme Cloud Service Level Agreement (SLA)

This SLA describes the uptime commitments and service credits for Acme Cloud.

## Uptime Commitment

Acme commits to the following monthly uptime percentages by tier:

- Starter: 99.5% uptime
- Growth: 99.9% uptime
- Enterprise: 99.99% uptime

Uptime is measured as the percentage of minutes in a calendar month during which
the Acme Cloud API is reachable and returns successful responses, excluding
scheduled maintenance.

## Service Credits

If Acme fails to meet the committed uptime, affected customers receive service
credits against the following month's bill:

- Below the commitment but at or above 99.0%: 10% credit
- Below 99.0% but at or above 95.0%: 25% credit
- Below 95.0%: 50% credit

Service credits must be requested within 30 days of the affected month by opening
a support ticket. Credits do not apply to the free trial or to outages caused by
customer misconfiguration.

## Scheduled Maintenance

Acme performs scheduled maintenance during a weekly window on Sundays from 02:00
to 04:00 UTC. Customers are notified at least 48 hours in advance. Scheduled
maintenance does not count against the uptime commitment.

## Support Response Targets

- Starter: community forum only
- Growth: email support, next-business-day response
- Enterprise: 24/7 support with a 1-hour response target for severity-1 incidents
