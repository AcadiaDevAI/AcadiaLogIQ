import React from "react";
import {
  SignedIn,
  UserButton,
  OrganizationSwitcher,
  useUser,
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
  return (
    <div className="border-t" style={{ borderColor: "var(--border-color)" }}>
      <div className="px-4 pt-3">
        <OrganizationSwitcher
          hidePersonal
          afterSelectOrganizationUrl="/"
          afterCreateOrganizationUrl="/"
          appearance={{
            elements: {
              rootBox: "w-full",
              organizationSwitcherTrigger:
                "w-full justify-between focus:shadow-none",
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
