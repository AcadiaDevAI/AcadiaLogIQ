import React from "react";
import Tier1Workspace from "../../components/Tier1Copilot/Tier1Workspace";
import USPharmaResolutionJourney from "./journey/ResolutionJourney";

export default function USPharmaWorkspace(props) {
  return <Tier1Workspace {...props} JourneyComponent={USPharmaResolutionJourney} />;
}
