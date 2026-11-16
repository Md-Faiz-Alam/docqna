import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import HomePage from "@/app/page";
import { AnswerCard } from "@/components/demo/answer";
import { historyFrom, type Message } from "@/components/demo/chat";
import { validateFile } from "@/components/demo/documents-panel";
import type { Citation, Info, QAResponse } from "@/lib/types";

vi.mock("next/navigation", () => ({ usePathname: () => "/" }));

const source: Citation = {
  citation_id: 1,
  chunk_id: "c1",
  rank: 1,
  filename: "northwind_employee_handbook.txt",
  file_type: "txt",
  page_start: null,
  page_end: null,
  section: null,
  distance: 0.1,
  text: "The standard working week is 38 hours.",
};

const answered: QAResponse = {
  answer: "The working week is 38 hours [1].",
  abstained: false,
  cited_source_ids: [1],
  sources: [source],
  standalone_question: null,
};

describe("AnswerCard", () => {
  it("makes each citation open its source", async () => {
    const onCite = vi.fn();
    render(<AnswerCard response={answered} onCite={onCite} onRate={() => undefined} />);

    await userEvent.click(screen.getByRole("button", { name: /Open source 1/ }));
    expect(onCite).toHaveBeenCalledWith(source);
    expect(screen.getByRole("list", { name: "Sources" })).toHaveTextContent(source.filename);
  });

  it("shows an abstention without citations and records one rating", async () => {
    const onRate = vi.fn();
    const abstained = { ...answered, answer: "Not enough information.", abstained: true, cited_source_ids: [] };
    const { rerender } = render(<AnswerCard response={abstained} onCite={() => undefined} onRate={onRate} />);

    expect(screen.queryByRole("button", { name: /Open source/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Helpful answer" }));
    expect(onRate).toHaveBeenCalledWith("up");

    rerender(<AnswerCard response={abstained} rating="up" onCite={() => undefined} onRate={onRate} />);
    expect(screen.getByRole("button", { name: "Unhelpful answer" })).toBeDisabled();
  });
});

describe("chat history", () => {
  const signed = (content: string): QAResponse => ({ ...answered, answer: content, answer_signature: `sig-${content}` });

  it("keeps completed turns only, oldest first, at most 12", () => {
    const messages: Message[] = Array.from({ length: 14 }, (_, i) => ({
      id: i,
      role: i % 2 ? "assistant" : "user",
      content: `turn ${i}`,
      response: i % 2 ? signed(`turn ${i}`) : undefined,
    }));
    messages.push({ id: 99, role: "assistant", content: "", pending: true });
    messages.push({ id: 100, role: "assistant", content: "", error: "failed" });

    const history = historyFrom(messages);
    expect(history).toHaveLength(12);
    expect(history[0]).toEqual({ role: "user", content: "turn 2" });
    expect(history.at(-1)).toEqual({ role: "assistant", content: "turn 13", signature: "sig-turn 13" });
  });

  it("leaves out assistant turns the API did not sign", () => {
    const messages: Message[] = [
      { id: 1, role: "user", content: "Leave?" },
      { id: 2, role: "assistant", content: "24 days [1].", response: answered },
      { id: 3, role: "user", content: "And hours?" },
      { id: 4, role: "assistant", content: "38 hours [1].", response: signed("38 hours [1].") },
    ];

    expect(historyFrom(messages)).toEqual([
      { role: "user", content: "Leave?" },
      { role: "user", content: "And hours?" },
      { role: "assistant", content: "38 hours [1].", signature: "sig-38 hours [1]." },
    ]);
  });
});

describe("validateFile", () => {
  const info = {
    limits: { supported_file_types: ["docx", "pdf", "txt"], max_upload_size_mb: 1 },
  } as Info;

  it("accepts supported files and rejects others before uploading", () => {
    expect(validateFile(new File(["x"], "notes.TXT"), info)).toBeNull();
    expect(validateFile(new File(["x"], "image.png"), info)).toMatch(/only DOCX, PDF, TXT/);
    expect(validateFile(new File([], "empty.pdf"), info)).toMatch(/empty/);
    expect(validateFile(new File([new Uint8Array(2 * 1024 * 1024)], "big.pdf"), info)).toMatch(/1 MB/);
  });
});

describe("landing page", () => {
  it("shows the measured numbers and links to the demo", () => {
    render(<HomePage />);

    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/Ask your documents/);
    expect(screen.getAllByText("98.3%").length).toBeGreaterThan(0);
    expect(screen.getByText("/ 28")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Try the live demo/ })).toHaveAttribute("href", "/demo");
  });
});
