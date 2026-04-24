import React from "react";
import { AutoComplete, Input } from "antd";

/**
 * Sprint 8 — AssetAutocomplete
 *
 * Spec §6.2 defers the backend `/tier1/suggest` endpoint to Sprint 9.
 * For now this component wraps AntD AutoComplete with a static
 * `recent` suggestion list supplied by the parent (Tier1IntakeForm).
 * When the list is empty it behaves as a plain Input — zero regression
 * versus Sprint 6's free-text field.
 */
export default function AssetAutocomplete({
  value,
  onChange,
  onBlur,
  placeholder,
  maxLength = 200,
  suggestions = [],
  size = "large",
}) {
  const options = (suggestions || []).map((s) => ({ value: s, label: s }));

  const handleSearch = (next) => {
    if (onChange) onChange(next);
  };

  return (
    <AutoComplete
      value={value || ""}
      options={options}
      onChange={handleSearch}
      onBlur={onBlur}
      style={{ width: "100%" }}
      filterOption={(input, option) =>
        (option && option.value ? option.value : "")
          .toLowerCase()
          .includes((input || "").toLowerCase())
      }
    >
      <Input
        size={size}
        placeholder={placeholder}
        maxLength={maxLength}
        allowClear
      />
    </AutoComplete>
  );
}
