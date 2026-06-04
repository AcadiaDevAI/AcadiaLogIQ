// Category -> leaf mapping for the Escalation Procedure picker.
//
// `section` here is the id the backend uses to filter the KB. Keep in
// sync with backend/escalation/sections.py.

export const ESCALATION_KB_FILENAME = "Escalation_Procedures_KB.pdf";

export const ESCALATION_CATEGORIES = [
  {
    key: "oem_vendor",
    label: "OEM Vendor",
    leaves: [
      { section: "cisco", label: "Cisco" },
      { section: "microsoft", label: "Microsoft" },
    ],
  },
  {
    key: "telco",
    label: "Telco",
    leaves: [
      { section: "verizon", label: "Verizon" },
      { section: "att", label: "AT&T" },
    ],
  },
  {
    key: "third_party",
    label: "Third-party Coordinations",
    leaves: [
      { section: "vendor_dispatch", label: "Vendor Dispatch" },
    ],
  },
];

export const SECTION_LABELS = {
  cisco: "Cisco",
  microsoft: "Microsoft",
  verizon: "Verizon",
  att: "AT&T",
  vendor_dispatch: "Vendor Dispatch",
};
