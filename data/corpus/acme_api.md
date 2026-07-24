# Acme Cloud API Reference (Overview)

The Acme Cloud API is a REST API served over HTTPS at `https://api.acme.cloud/v2`.

## Authentication

Requests are authenticated with a bearer token passed in the `Authorization`
header. Tokens are created in the developer console and are scoped to a single
project. Tokens can be revoked at any time and expire after 12 months if unused.

## Rate Limits

Rate limits depend on the subscription tier:

- Starter: 10 requests per second, 100,000 requests per month
- Growth: 100 requests per second, 5,000,000 requests per month
- Enterprise: negotiated per contract

When a rate limit is exceeded, the API returns HTTP 429 with a `Retry-After`
header indicating how many seconds to wait. Clients should implement exponential
backoff with jitter.

## Pagination

List endpoints return at most 100 items per page. Use the `cursor` query
parameter with the `next_cursor` value from the previous response to page through
results. Cursors expire after 24 hours.

## Errors

The API uses standard HTTP status codes. Error responses include a JSON body with
`error.code` and `error.message`. A `request_id` is included in every response
header and should be quoted when contacting support.

## Webhooks

Acme can send webhook events to a URL you configure. Each webhook payload is
signed with an HMAC-SHA256 signature in the `X-Acme-Signature` header, computed
using your webhook signing secret. Verify the signature before trusting a payload.
