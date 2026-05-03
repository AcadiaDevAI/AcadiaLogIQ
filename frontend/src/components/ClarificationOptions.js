import React, { useState } from "react";
import { Button, Input } from "antd";
import { CheckCircleOutlined } from "@ant-design/icons";

const { TextArea } = Input;

/**
 * Renders clarification options as clickable buttons inline in a chat message.
 * When the user clicks one, onSelect(optionId, freeText) is called.
 *
 * Props:
 *   options     [{id, label, refined_query, record_ref}] — must include opt_other
 *   selectedId  string|null  — option already chosen (disables all buttons)
 *   onSelect    (optionId: string, freeText?: string) => void
 *   disabled    bool         — true while a request is in flight
 */
export default function ClarificationOptions({ options, selectedId, onSelect, disabled }) {
  const [otherText, setOtherText] = useState("");
  const [otherOpen, setOtherOpen] = useState(false);

  if (!options || !options.length) return null;

  const handleClick = (opt) => {
    if (disabled || selectedId) return;
    if (opt.id === "opt_other") {
      setOtherOpen(true);
      return;
    }
    onSelect(opt.id);
  };

  const submitOther = () => {
    const text = otherText.trim();
    if (!text) return;
    onSelect("opt_other", text);
  };

  return (
    <div className="mt-3 flex flex-col gap-2">
      {options.map((opt) => {
        const isSelected = selectedId === opt.id;
        const isDimmed = !!selectedId && !isSelected;
        return (
          <Button
            key={opt.id}
            type={isSelected ? "primary" : "default"}
            onClick={() => handleClick(opt)}
            disabled={disabled || !!selectedId}
            icon={isSelected ? <CheckCircleOutlined /> : null}
            style={{
              textAlign: "left",
              whiteSpace: "normal",
              height: "auto",
              padding: "8px 12px",
              opacity: isDimmed ? 0.5 : 1,
              borderColor: isSelected ? "var(--acadia-primary)" : "var(--border-color)",
              backgroundColor: isSelected ? "var(--acadia-primary)" : "transparent",
              color: isSelected ? "#fff" : "var(--text-primary)",
            }}
          >
            {opt.label}
          </Button>
        );
      })}

      {otherOpen && !selectedId && (
        <div className="mt-2 flex flex-col gap-2">
          <TextArea
            rows={2}
            placeholder="Let us know what you meant..."
            value={otherText}
            onChange={(e) => setOtherText(e.target.value)}
            autoFocus
            maxLength={500}
            disabled={disabled}
          />
          <div className="flex gap-2">
            <Button
              type="primary"
              onClick={submitOther}
              disabled={disabled || !otherText.trim()}
              style={{ backgroundColor: "var(--acadia-primary)", borderColor: "var(--acadia-primary)" }}
            >
              Send
            </Button>
            <Button
              onClick={() => {
                setOtherOpen(false);
                setOtherText("");
              }}
              disabled={disabled}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
