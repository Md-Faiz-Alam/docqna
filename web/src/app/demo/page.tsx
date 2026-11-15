import type { Metadata } from "next";

import { DemoApp } from "@/components/demo/demo-app";
import { PageHeader } from "@/components/page-header";

export const metadata: Metadata = {
  title: "Demo",
  description: "Upload a document or load the samples, then ask questions and check every source.",
};

export default function DemoPage() {
  return (
    <>
      <PageHeader eyebrow="Live demo" title="Ask your documents">
        Add documents, ask questions, and click any [number] to see the exact source passage.
      </PageHeader>
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <DemoApp />
      </div>
    </>
  );
}
