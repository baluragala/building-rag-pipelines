# Getting Started with Acme Cloud

This guide walks a new team through onboarding onto Acme Cloud.

## Step 1: Create an Account

Sign up at acme.cloud with a work email address. Personal email domains such as
gmail.com are not accepted for organisation accounts. You will be asked to choose
a data residency region during sign-up; this choice is permanent.

## Step 2: Invite Your Team

From the Members page, invite colleagues by email. Each invited member counts
toward your plan's seat limit. Roles are Owner, Admin, Developer, and Viewer.
Only Owners can change the billing plan or delete the organisation.

## Step 3: Create a Project

Projects isolate resources, API tokens, and billing usage. Most teams create one
project per environment (for example, `production` and `staging`). Storage and
API quotas are shared across all projects in an organisation.

## Step 4: Generate an API Token

Open the Developer console, select a project, and click "Create token". Copy the
token immediately — it is shown only once. Store it in a secrets manager, never
in source control.

## Step 5: Make Your First Request

Use the token to call the health endpoint:

    curl -H "Authorization: Bearer $TOKEN" https://api.acme.cloud/v2/health

A successful response returns HTTP 200 with a JSON body `{"status": "ok"}`.

## Support

Growth and Enterprise customers can email support@acme.cloud. Starter customers
should use the community forum at community.acme.cloud.
