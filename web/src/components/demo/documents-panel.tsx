"use client";

import { FileTextIcon, Loader2Icon, SparklesIcon, Trash2Icon, UploadIcon } from "lucide-react";
import { useRef, useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError, api } from "@/lib/api";
import type { DocumentInfo, Info } from "@/lib/types";
import { cn } from "@/lib/utils";

function sizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** Client-side check before uploading; the server enforces the same rules. */
export function validateFile(file: File, info: Info): string | null {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!info.limits.supported_file_types.includes(extension)) {
    return `${file.name}: only ${info.limits.supported_file_types.map((t) => t.toUpperCase()).join(", ")} files are supported.`;
  }
  if (file.size > info.limits.max_upload_size_mb * 1024 * 1024) {
    return `${file.name} is larger than ${info.limits.max_upload_size_mb} MB.`;
  }
  if (file.size === 0) return `${file.name} is empty.`;
  return null;
}

export function DocumentsPanel({
  info,
  sessionId,
  documents,
  onChange,
}: {
  info: Info;
  sessionId: string;
  documents: DocumentInfo[];
  onChange: (documents: DocumentInfo[]) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const samplesLoaded =
    info.sample_documents.length > 0 &&
    info.sample_documents.every((name) => documents.some((d) => d.filename === name));

  async function run(label: string, action: () => Promise<void>) {
    setBusy(label);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(null);
    }
  }

  async function refresh() {
    onChange((await api.listDocuments(sessionId)).documents);
  }

  function upload(files: FileList | File[]) {
    const list = Array.from(files);
    if (list.length === 0) return;
    const problem = list.map((f) => validateFile(f, info)).find(Boolean);
    if (problem) {
      setError(problem);
      return;
    }
    void run(list.length === 1 ? `Reading ${list[0].name}…` : `Reading ${list.length} files…`, async () => {
      try {
        for (const file of list) await api.upload(sessionId, file);
      } finally {
        await refresh();
      }
    });
  }

  return (
    <Card className="gap-4">
      <CardHeader>
        <CardTitle>Documents</CardTitle>
        <CardDescription>
          Up to {info.limits.max_documents_per_session} of your own files (
          {info.limits.supported_file_types.map((t) => t.toUpperCase()).join(", ")}, max{" "}
          {info.limits.max_upload_size_mb} MB), kept for {info.limits.document_retention_hours} hours.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div
          role="button"
          tabIndex={0}
          aria-label="Upload documents: drop files here or press to choose"
          aria-disabled={busy !== null}
          onClick={() => busy === null && inputRef.current?.click()}
          onKeyDown={(event) => {
            if ((event.key === "Enter" || event.key === " ") && busy === null) {
              event.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            if (busy === null) upload(event.dataTransfer.files);
          }}
          className={cn(
            "flex cursor-pointer flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 text-center text-sm transition-colors outline-none hover:border-primary/50 hover:bg-accent/40 focus-visible:ring-[3px] focus-visible:ring-ring/50",
            dragging && "border-primary bg-accent/60",
            busy !== null && "cursor-wait opacity-60",
          )}
        >
          <UploadIcon className="size-5 text-muted-foreground" aria-hidden />
          <span>
            <span className="font-medium text-primary">Choose files</span> or drag them here
          </span>
          <input
            ref={inputRef}
            type="file"
            multiple
            className="sr-only"
            tabIndex={-1}
            accept={info.limits.supported_file_types.map((t) => `.${t}`).join(",")}
            onChange={(event) => {
              if (event.target.files) upload(event.target.files);
              event.target.value = "";
            }}
          />
        </div>

        {info.sample_documents.length > 0 && !samplesLoaded && (
          <Button
            variant="secondary"
            disabled={busy !== null}
            onClick={() =>
              void run("Loading sample documents…", async () => {
                onChange((await api.loadSamples(sessionId)).documents);
              })
            }
          >
            <SparklesIcon /> Load {info.sample_documents.length} sample documents
          </Button>
        )}

        <div aria-live="polite" className="contents">
          {busy && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2Icon className="size-4 animate-spin" aria-hidden /> {busy}
            </p>
          )}
          {error && (
            <Alert variant="destructive">
              <AlertDescription className="text-destructive">{error}</AlertDescription>
            </Alert>
          )}
        </div>

        {documents.length === 0 ? (
          <p className="text-sm text-muted-foreground">No documents yet.</p>
        ) : (
          <ul className="flex flex-col gap-1" aria-label="Your documents">
            {documents.map((doc) => (
              <li key={doc.document_id} className="group flex items-center gap-2 rounded-md px-2 py-1.5 hover:bg-muted">
                <FileTextIcon className="size-4 shrink-0 text-muted-foreground" aria-hidden />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm" title={doc.filename}>
                    {doc.filename}
                  </span>
                  <span className="text-xs text-muted-foreground">{sizeLabel(doc.size_bytes)}</span>
                </span>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label={`Delete ${doc.filename}`}
                  disabled={busy !== null}
                  onClick={() =>
                    void run(`Deleting ${doc.filename}…`, async () => {
                      await api.deleteDocument(sessionId, doc.document_id);
                      await refresh();
                    })
                  }
                >
                  <Trash2Icon />
                </Button>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
