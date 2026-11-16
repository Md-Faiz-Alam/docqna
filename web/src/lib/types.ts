/** Types mirroring the FastAPI schemas in src/docqna/api/schemas.py. */

export interface InfoLimits {
  max_upload_size_mb: number;
  max_question_chars: number;
  max_documents_per_session: number;
  document_retention_hours: number;
  supported_file_types: string[];
}

export interface Info {
  app: string;
  version: string;
  llm_provider: string;
  llm_model: string;
  external_llm: boolean;
  embedding_model: string;
  reranker_model: string;
  sample_documents: string[];
  limits: InfoLimits;
}

export interface DocumentInfo {
  document_id: string;
  filename: string;
  file_type: string;
  version: number;
  content_hash: string;
  size_bytes: number;
  registered_at: string;
}

export interface DocumentList {
  count: number;
  documents: DocumentInfo[];
}

export interface Citation {
  citation_id: number;
  chunk_id: string;
  rank: number;
  filename: string;
  file_type: string;
  page_start: number | null;
  page_end: number | null;
  section: string | null;
  distance: number;
  text: string;
}

export interface QAResponse {
  answer: string;
  abstained: boolean;
  cited_source_ids: number[];
  sources: Citation[];
  standalone_question: string | null;
  /** Sent back as `signature` when this answer is part of later history. */
  answer_signature?: string;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  /** For assistant turns: the answer's `answer_signature`; unsigned ones are ignored. */
  signature?: string;
}
