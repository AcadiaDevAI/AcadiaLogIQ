import React, { useState } from "react";
import { Upload, Button, Progress, message, Tag, Select } from "antd";
import {
  InboxOutlined,
  BookOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  CloseCircleOutlined,
} from "@ant-design/icons";
import { useChat } from "../hooks/ChatContext";
import { uploadFile, uploadFileV2, getUploadStatus } from "../services/api";

// Phase 1 — direct-to-S3 upload toggle. When true, the panel calls
// uploadFileV2 (presign → PUT to S3 → finalize). When false (default),
// it keeps the legacy multipart POST /upload path. Flip by setting
// REACT_APP_UPLOAD_VIA_S3=true in frontend/.env and rebuilding.
const USE_S3_UPLOAD =
  (process.env.REACT_APP_UPLOAD_VIA_S3 || "false").toLowerCase() === "true";

const { Dragger } = Upload;

const DOC_ACCEPT = ".txt,.md,.json,.pdf,.docx,.log,.csv,.xlsx,.xls";

// Sprint 3-PREP-B — doc_kind dropdown options. Values MUST match the
// backend VALID_DOC_KINDS frozenset in backend/config.py. When the
// frontend flag is off, the dropdown is disabled and defaults to
// 'ticket' so behavior is byte-identical to post-PREP-A.
const DOC_KIND_OPTIONS = [
  { value: "ticket",           label: "Ticket history (JSON)" },
  { value: "sop",              label: "SOP / Runbook" },
  { value: "kb",               label: "KB article" },
  { value: "contact_customer", label: "Customer contact directory" },
  { value: "contact_vendor",   label: "Vendor contact directory" },
  { value: "vendor_case",      label: "Past vendor case" },
];

const BULK_INGEST_ENABLED =
  (process.env.REACT_APP_LOGIQ_BULK_INGEST_FRONTEND || "false").toLowerCase() === "true";

export default function UploadPanel({ onUploadComplete }) {
  const { state, dispatch } = useChat();
  const [docFiles, setDocFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [jobStatuses, setJobStatuses] = useState({});
  // Sprint 3-PREP-B — selected doc_kind for this batch. Flag-off keeps
  // this fixed at 'ticket' since the dropdown is disabled.
  const [docKind, setDocKind] = useState("ticket");

  const handleUpload = async () => {
    if (!docFiles.length) {
      message.warning("Select document files first");
      return;
    }

    dispatch({ type: "SET_UPLOADING", payload: true });

    for (const file of docFiles) {
      const fileKey = `doc-${file.name}`;
      try {
        setUploadProgress((p) => ({ ...p, [fileKey]: 0 }));
        setJobStatuses((s) => ({ ...s, [fileKey]: "uploading" }));

        const _uploader = USE_S3_UPLOAD ? uploadFileV2 : uploadFile;
        const res = await _uploader(
          file,
          "kb",
          (pct) => {
            setUploadProgress((p) => ({ ...p, [fileKey]: pct }));
          },
          BULK_INGEST_ENABLED ? docKind : undefined,   // Sprint 3-PREP-B
        );

        const { job_id, file_id } = res.data;
        setUploadProgress((p) => ({ ...p, [fileKey]: 100 }));
        setJobStatuses((s) => ({ ...s, [fileKey]: "processing" }));

        dispatch({
          type: "ADD_FILE",
          payload: {
            id: file_id,
            name: file.name,
            file_type: "kb",
            size_mb: file.size / (1024 * 1024),
            status: "processing",
            job_id,
            uploaded_at: new Date().toISOString(),
          },
        });

        await pollJob(job_id, fileKey, file_id);
      } catch (err) {
        const detail = err?.response?.data?.error || err?.message || "Upload failed";
        message.error(`${file.name}: ${detail}`);
        setJobStatuses((s) => ({ ...s, [fileKey]: "failed" }));
      }
    }

    dispatch({ type: "SET_UPLOADING", payload: false });
    setDocFiles([]);
    onUploadComplete?.();
  };

  const pollJob = async (jobId, fileKey, fileId) => {
    const maxAttempts = 120;
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const res = await getUploadStatus(jobId);
        const { status, error } = res.data;
        if (status === "done") {
          setJobStatuses((s) => ({ ...s, [fileKey]: "done" }));
          dispatch({ type: "UPDATE_FILE_STATUS", payload: { id: fileId, status: "indexed" } });
          message.success("Indexed successfully");
          return;
        }
        // Sprint 2.9 — "error" covers index_file_job exception path AND
        // the new JSON-validator rejection path. When an error detail is
        // present (e.g. "Invalid JSON at line 3 col 1: ..."), surface it
        // in the toast so the admin doesn't have to hunt in logs.
        if (status === "failed" || status === "error") {
          setJobStatuses((s) => ({ ...s, [fileKey]: "failed" }));
          dispatch({ type: "UPDATE_FILE_STATUS", payload: { id: fileId, status: "failed" } });
          message.error(error ? `Indexing failed — ${error}` : "Indexing failed", 8);
          onUploadComplete?.();
          return;
        }
      } catch { /* continue */ }
      await new Promise((r) => setTimeout(r, 1500));
    }
    setJobStatuses((s) => ({ ...s, [fileKey]: "timeout" }));
  };

  const statusIcon = (status) => {
    switch (status) {
      case "done": return <CheckCircleOutlined style={{ color: "#10b981" }} />;
      case "failed": case "timeout": return <CloseCircleOutlined style={{ color: "#ef4444" }} />;
      case "processing": case "uploading": return <LoadingOutlined style={{ color: "#6366f1" }} spin />;
      default: return null;
    }
  };

  const beforeUpload = (file) => {
    setDocFiles((prev) => {
      // Prevent duplicates by file name
      if (prev.some((f) => f.name === file.name)) return prev;
      return [...prev, file];
    });
    return false;
  };

  return (
    <div className="flex flex-col gap-4 overflow-y-auto max-h-[calc(100vh-380px)] pb-4">
      <div>
        <div className="flex items-center gap-2 mb-2">
          <BookOutlined style={{ color: "#6366f1" }} />
          <span className="text-xs font-semibold t-text-secondary">Upload Documents</span>
          <Tag color="blue" className="text-[10px] ml-auto">.pdf .docx .txt .md .log .json .csv .xlsx .xls</Tag>
        </div>
        {/* Sprint 3-PREP-B — doc_kind picker. Disabled until
            REACT_APP_LOGIQ_BULK_INGEST_FRONTEND=true; flag-off state keeps
            the control locked on 'ticket' and the field is omitted from
            the POST, so the backend default path (post-PREP-A) runs. */}
        <div className="flex items-center gap-2 mb-2">
          <span className="text-[11px] t-text-muted" style={{ minWidth: 90 }}>
            Document type
          </span>
          <Select
            size="small"
            value={docKind}
            onChange={setDocKind}
            disabled={!BULK_INGEST_ENABLED || state.isUploading}
            options={DOC_KIND_OPTIONS}
            style={{ flex: 1 }}
          />
        </div>
        <Dragger
          accept={DOC_ACCEPT}
          multiple
          fileList={[]}
          beforeUpload={beforeUpload}
          disabled={state.isUploading}
          className="rounded-lg"
          style={{ padding: "8px" }}
        >
          <p className="t-text-muted text-xs">
            <InboxOutlined className="text-2xl mb-1 block" style={{ color: "#6366f1" }} />
            Drop files here or click to browse
          </p>
        </Dragger>
        {docFiles.length > 0 && (
          <div className="mt-2 space-y-1">
            {docFiles.map((f) => {
              const key = `doc-${f.name}`;
              return (
                <div key={key} className="flex items-center gap-2 text-xs t-text-muted px-2 py-1 t-bg-tertiary rounded">
                  {statusIcon(jobStatuses[key])}
                  <span className="truncate flex-1">{f.name}</span>
                  {uploadProgress[key] !== undefined && uploadProgress[key] < 100 && (
                    <Progress percent={uploadProgress[key]} size="small" className="w-16" showInfo={false} />
                  )}
                </div>
              );
            })}
            <Button
              type="primary"
              size="small"
              icon={<BookOutlined />}
              onClick={handleUpload}
              loading={state.isUploading}
              block
              className="mt-1 rounded-lg"
            >
              Upload {docFiles.length} Document{docFiles.length > 1 ? "s" : ""}
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}