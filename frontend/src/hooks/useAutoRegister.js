import { useEffect, useRef } from "react";
import { useUser } from "@clerk/clerk-react";
import { registerOrLogin } from "../services/api";

const CLERK_KEY = process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

/**
 * useAutoRegister
 *
 * Fires once after Clerk sign-in AND after auth interceptor is ready.
 * Calls POST /auth/register-or-login to save user in the DB.
 *
 * Includes retry logic: if the first attempt fails (e.g., token
 * timing), it retries once after a short delay.
 */
export default function useAutoRegister(isReady = false) {
  if (!CLERK_KEY) {
    return;
  }

  return useAutoRegisterInner(isReady);
}

function useAutoRegisterInner(isReady) {
  const { user, isLoaded } = useUser();
  const registered = useRef(false);

  useEffect(() => {
    // Wait for ALL conditions: Clerk user loaded + auth interceptor ready
    if (!isLoaded || !user || !isReady || registered.current) return;
    registered.current = true;

    const doRegister = async (attempt = 1) => {
      try {
        const res = await registerOrLogin({
          email: user.primaryEmailAddress?.emailAddress || null,
          full_name: user.fullName || null,
          avatar_url: user.imageUrl || null,
        });

        if (res.data?.is_new) {
          console.log("[AutoRegister] New user created:", res.data.user_id);
        } else {
          console.log("[AutoRegister] Returning user:", res.data.user_id);
        }
      } catch (err) {
        console.error(`[AutoRegister] Attempt ${attempt} failed:`, err?.response?.status || err.message);

        // Retry once after 1.5 seconds
        if (attempt < 2) {
          console.log("[AutoRegister] Retrying in 1.5s...");
          setTimeout(() => doRegister(attempt + 1), 1500);
        }
      }
    };

    doRegister();
  }, [isLoaded, user, isReady]);
}
