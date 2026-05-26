// useSidebarPeek — enables the blocks-screen sidebar behaviour
// (collapsed by default + hover-to-peek expand) on any screen that
// imports this hook.
//
// Why a hook?
//   The Resolution Journey was the first screen to want this — it
//   dispatches SET_SIDEBAR(false) + SET_SIDEBAR_HOVER_PEEK(true) on
//   mount and restores both on unmount. The Ticket Filter and
//   ServiceNow flows want identical behaviour, so this hook bundles
//   the four dispatches into one reusable lifecycle. Any future
//   right-pane takeover that wants the same UX just calls
//   useSidebarPeek() at the top of its render function.
//
// What the hook does NOT do:
//   - Does not own the back navigation. Callers still render their
//     own back / return button and wire it to their unmount path
//     (clearing the right-pane flag in App.js, etc.).
//   - Does not animate. The Sidebar component owns the 650 ms
//     gentle slide-out keyframe — the hook just flips the gate.

import { useEffect } from "react";

import { useChat } from "./ChatContext";


export default function useSidebarPeek() {
  const { dispatch } = useChat();

  useEffect(() => {
    // Collapse + enable hover-peek on entry.
    dispatch({ type: "SET_SIDEBAR", payload: false });
    dispatch({ type: "SET_SIDEBAR_HOVER_PEEK", payload: true });
    // Restore the expanded default + drop the gate on exit. Matches
    // ResolutionJourney's cleanup so navigating away from any
    // peek-enabled screen lands on landing/chat with an expanded
    // sidebar, regardless of what the engineer did while inside.
    return () => {
      dispatch({ type: "SET_SIDEBAR_HOVER_PEEK", payload: false });
      dispatch({ type: "SET_SIDEBAR", payload: true });
    };
  }, [dispatch]);
}
