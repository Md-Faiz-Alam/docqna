"use client";

import { CloudOffIcon, Loader2Icon, MoonStarIcon, ShieldAlertIcon } from "lucide-react";
import { useEffect, useState } from "react";

import { Chat } from "@/components/demo/chat";
import { DocumentsPanel } from "@/components/demo/documents-panel";
import { useServer } from "@/components/demo/use-server";
import { useSessionId } from "@/components/demo/use-session-id";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import type { DocumentInfo } from "@/lib/types";

export function DemoApp() {
  const { state, info, retry } = useServer();
  const [sessionId, resetSession] = useSessionId();
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [chatKey, setChatKey] = useState(0);

  useEffect(() => {
    if (state !== "ready" || !sessionId) return;
    api
      .listDocuments(sessionId)
      .then((list) => setDocuments(list.documents))
      .catch(() => setDocuments([]));
  }, [state, sessionId]);

  if (state === "checking" || !sessionId) {
    return (
      <div className="grid gap-6 lg:grid-cols-[20rem_1fr]" aria-busy>
        <Skeleton className="h-72" />
        <Skeleton className="h-[32rem]" />
      </div>
    );
  }

  if (state === "waking") {
    return (
      <Alert className="mx-auto max-w-xl py-5" role="status">
        <MoonStarIcon />
        <AlertTitle>Waking up the server…</AlertTitle>
        <AlertDescription>
          The free server sleeps when nobody is using it. It usually wakes in a minute or two; this
          page will continue on its own.
          <Loader2Icon className="mt-3 size-5 animate-spin text-primary" aria-hidden />
        </AlertDescription>
      </Alert>
    );
  }

  if (state === "down" || !info) {
    return (
      <Alert variant="destructive" className="mx-auto max-w-xl py-5">
        <CloudOffIcon />
        <AlertTitle>The server isn’t responding</AlertTitle>
        <AlertDescription>
          It may be restarting or over its free limits. Please try again in a few minutes.
          <Button variant="outline" size="sm" className="mt-3" onClick={retry}>
            Try again
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {info.external_llm && (
        <Alert variant="warning">
          <ShieldAlertIcon />
          <AlertTitle>Public demo: please don’t upload private documents</AlertTitle>
          <AlertDescription>
            Answers are written by {info.llm_model} (Google Gemini), so your questions and the
            matching passages are sent to Google, which may use them to improve its products.
            Uploads are deleted after {info.limits.document_retention_hours} hours or when the
            server restarts.
          </AlertDescription>
        </Alert>
      )}
      <div className="grid items-start gap-6 lg:grid-cols-[20rem_1fr]">
        <div className="flex flex-col gap-3">
          <DocumentsPanel info={info} sessionId={sessionId} documents={documents} onChange={setDocuments} />
          <Button
            variant="ghost"
            size="sm"
            className="self-start text-muted-foreground"
            onClick={() => {
              resetSession();
              setDocuments([]);
              setChatKey((k) => k + 1);
            }}
          >
            Start over with an empty workspace
          </Button>
        </div>
        <Chat
          key={`${sessionId}-${chatKey}`}
          sessionId={sessionId}
          maxChars={info.limits.max_question_chars}
          hasDocuments={documents.length > 0}
          samplesAvailable={info.sample_documents.length > 0}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Answer model: {info.llm_model} · Search: {info.embedding_model.split("/").pop()} +{" "}
        {info.reranker_model.split("/").pop()}
      </p>
    </div>
  );
}
