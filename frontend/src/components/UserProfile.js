import React from "react";
import {
  SignedIn,
  UserButton,
  OrganizationSwitcher,
  useUser,
  useOrganization,
} from "@clerk/clerk-react";

const CLERK_KEY = process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

export default function UserProfile() {
  if (!CLERK_KEY) return null;

  return (
    <SignedIn>
      <UserInfo />
    </SignedIn>
  );
}

function UserInfo() {
  const { user, isLoaded } = useUser();

  // Role of the current user in the ACTIVE org. Clerk roles arrive as
  // "org:admin" / "org:member". Only admins may manage the org (invite
  // members, change roles, edit settings) — so we hide the "Manage
  // organization" action for everyone else.
  const { membership } = useOrganization();
  const isOrgAdmin = membership?.role === "org:admin";

  const email =
    isLoaded && user
      ? user.primaryEmailAddress?.emailAddress || user.fullName || "User"
      : "Loading...";

  const displayName =
    isLoaded && user
      ? user.fullName || user.firstName || email.split("@")[0]
      : "";

  // OrganizationSwitcher calls Clerk's setActive() under the hood, which
  // re-issues the session JWT with the new active-org claim.
  // afterSelectOrganizationUrl="/" forces a navigation so OrgContext
  // re-fetches /organizations/me/active and the app rescopes to the chosen
  // org. hidePersonal — this app has no personal-account mode.
  //
  // Org creation is DISABLED for everyone here — members AND org admins.
  // Orgs are provisioned by the platform team only (via Clerk dashboard /
  // backend), never self-served from the app. We hide the "Create
  // organization" button so the switcher is switch-only. This is the
  // UI belt; the real enforcement is the Clerk Dashboard setting
  // "Allow users to create organizations" = off.
  return (
    <div className="border-t" style={{ borderColor: "var(--border-color)" }}>
      <div className="px-4 pt-3">
        <OrganizationSwitcher
          hidePersonal
          afterSelectOrganizationUrl="/"
          appearance={{
            elements: {
              rootBox: "w-full",
              organizationSwitcherTrigger:
                "w-full justify-between focus:shadow-none",
              // Hide the "Create organization" action inside the switcher
              // popover for all users (members AND admins). The popover
              // action button — NOT `createOrganizationButton`, which
              // targets the standalone create flow.
              organizationSwitcherPopoverActionButton__createOrganization: {
                display: "none",
              },
              // Members must NOT manage the org — hide "Manage
              // organization" for non-admins. Admins keep it (invite,
              // roles, settings). This is the UI belt; Clerk's role
              // permissions are the real server-side enforcement.
              ...(isOrgAdmin
                ? {}
                : {
                    organizationSwitcherPopoverActionButton__manageOrganization: {
                      display: "none",
                    },
                  }),
            },
          }}
        />
      </div>

      <div className="flex items-center gap-3 px-4 py-3">
        <UserButton
          appearance={{
            elements: {
              avatarBox: "w-8 h-8",
              userButtonTrigger: "focus:shadow-none",
            },
          }}
          afterSignOutUrl="/"
        />
        <div className="min-w-0 flex-1">
          {displayName && (
            <p className="text-xs t-text font-medium truncate">{displayName}</p>
          )}
          <p className="text-[11px] t-text-muted truncate">{email}</p>
        </div>
      </div>
    </div>
  );
}
