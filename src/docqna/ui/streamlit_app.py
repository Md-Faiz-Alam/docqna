import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import streamlit as st

DEFAULT_API_URL = os.getenv("DOCQNA_API_URL", "http://127.0.0.1:8000").rstrip("/")
SUPPORTED_UPLOAD_TYPES = ("pdf", "docx", "txt")
PRODUCTION_TOP_K = 5
MAX_QUESTION_CHARS = int(os.getenv("DOCQNA_MAX_QUESTION_CHARS", "2000"))
SESSION_HEADER_NAME = "X-DocQNA-Session-ID"
SESSION_COOKIE_NAME = "docqna_session_id"
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


def _decode_json_bytes(payload: bytes) -> dict[str, Any]:
    """Decode an API JSON object."""

    data = json.loads(payload.decode("utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Backend response was not a JSON object.")

    return data


def _http_error_detail(exc: HTTPError) -> str:
    """Extract a useful FastAPI error detail."""

    try:
        payload = _decode_json_bytes(exc.read())

    except Exception:
        return f"Backend returned HTTP {exc.code}."

    detail = payload.get("detail")

    if detail is None:
        detail = payload.get("error")

    if isinstance(detail, list):
        return json.dumps(detail, ensure_ascii=False)

    if detail is None:
        return f"Backend returned HTTP {exc.code}."

    return str(detail)


def get_api_health(api_url: str) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Check whether the DocQ&A FastAPI backend is reachable."""

    request = Request(url=f"{api_url}/health", headers={"Accept": "application/json"}, method="GET")

    try:
        with urlopen(request, timeout=5) as response:
            data = _decode_json_bytes(response.read())

            return (response.status == 200, data, None)

    except HTTPError as exc:
        return (False, None, _http_error_detail(exc))

    except URLError as exc:
        return (False, None, f"Could not connect to backend: {exc.reason}")

    except Exception as exc:
        return (False, None, f"Unexpected backend error: {exc}")


def list_documents(api_url: str, session_id: str) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch active indexed documents from FastAPI."""

    request = Request(
        url=f"{api_url}/documents",
        headers={"Accept": "application/json", SESSION_HEADER_NAME: session_id},
        method="GET",
    )

    try:
        with urlopen(request, timeout=30) as response:
            data = _decode_json_bytes(response.read())

    except HTTPError as exc:
        return ([], _http_error_detail(exc))

    except URLError as exc:
        return ([], f"Could not connect to backend: {exc.reason}")

    except Exception as exc:
        return ([], f"Could not load documents: {exc}")

    documents = data.get("documents", [])

    if not isinstance(documents, list):
        return ([], "Backend returned an invalid document list.")

    normalized: list[dict[str, Any]] = []

    for document in documents:
        if isinstance(document, dict):
            normalized.append(document)

    return (normalized, None)


def build_multipart_file_payload(
    *, filename: str, content: bytes, content_type: str, boundary: str | None = None
) -> tuple[bytes, str]:
    """
    Build a multipart/form-data request body for one uploaded file.

    FastAPI currently indexes one file per POST /documents request.
    The Streamlit UI sends multiple selected files sequentially.
    """

    boundary = boundary or f"----DocQNA{uuid4().hex}"

    safe_filename = Path(filename).name.replace("\r", "").replace("\n", "").replace('"', "")

    if not safe_filename:
        raise ValueError("Uploaded file must have a filename.")

    safe_content_type = content_type.strip() or "application/octet-stream"

    prefix = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; "
        f'name="file"; filename="{safe_filename}"\r\n'
        f"Content-Type: {safe_content_type}\r\n"
        "\r\n"
    ).encode()

    suffix = (f"\r\n--{boundary}--\r\n").encode()

    return (prefix + content + suffix, (f"multipart/form-data; boundary={boundary}"))


def upload_document(
    api_url: str, *, session_id: str, filename: str, content: bytes, content_type: str | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Upload and index one document through FastAPI."""

    extension = Path(filename).suffix.lower().lstrip(".")

    if extension not in (SUPPORTED_UPLOAD_TYPES):
        return (None, ("Unsupported file type. Use PDF, DOCX, or TXT."))

    if not content:
        return (None, "Uploaded file is empty.")

    resolved_content_type = (
        content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    )

    body, multipart_content_type = build_multipart_file_payload(
        filename=filename, content=content, content_type=(resolved_content_type)
    )

    request = Request(
        url=f"{api_url}/documents",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": (multipart_content_type),
            SESSION_HEADER_NAME: session_id,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=600) as response:
            return (_decode_json_bytes(response.read()), None)

    except HTTPError as exc:
        return (None, _http_error_detail(exc))

    except URLError as exc:
        return (None, f"Could not connect to backend: {exc.reason}")

    except Exception as exc:
        return (None, f"Document upload failed: {exc}")


def delete_document(api_url: str, document_id: str, session_id: str) -> tuple[bool, str | None]:
    """Delete one logical document from the active index."""

    request = Request(
        url=(f"{api_url}/documents/{document_id}"),
        headers={"Accept": "application/json", SESSION_HEADER_NAME: session_id},
        method="DELETE",
    )

    try:
        with urlopen(request, timeout=120) as response:
            data = _decode_json_bytes(response.read())

            return (bool(data.get("deleted", False)), None)

    except HTTPError as exc:
        return (False, _http_error_detail(exc))

    except URLError as exc:
        return (False, f"Could not connect to backend: {exc.reason}")

    except Exception as exc:
        return (False, f"Document deletion failed: {exc}")


def build_qa_payload(*, question: str, top_k: int, document_id: str | None) -> dict[str, Any]:
    """
    Build the /qa request.

    A document_id restricts retrieval to one selected document.
    None means search across all active indexed documents.
    """

    payload: dict[str, Any] = {"question": question, "top_k": top_k}

    if document_id is not None:
        payload["filters"] = {"document_id": document_id}

    return payload


def ask_api(
    api_url: str, question: str, top_k: int, *, session_id: str, document_id: str | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Send a grounded QA request to FastAPI."""

    payload = json.dumps(
        build_qa_payload(question=question, top_k=top_k, document_id=document_id)
    ).encode("utf-8")

    request = Request(
        url=f"{api_url}/qa",
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": ("application/json"),
            SESSION_HEADER_NAME: session_id,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            return (_decode_json_bytes(response.read()), None)

    except HTTPError as exc:
        return (None, _http_error_detail(exc))

    except URLError as exc:
        return (None, f"Could not connect to backend: {exc.reason}")

    except Exception as exc:
        return (None, f"Unexpected request error: {exc}")


def format_document_label(document: dict[str, Any]) -> str:
    """Return a concise human-readable document label."""

    filename = str(document.get("filename", "Unknown document"))

    version = document.get("version")

    file_type = str(document.get("file_type", "")).upper()

    version_text = "" if version is None else f" · v{version}"

    type_text = f" · {file_type}" if file_type else ""

    return f"{filename}{version_text}{type_text}"


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render cited document sources below an answer."""

    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            citation_id = source.get("citation_id", "?")

            filename = source.get("filename", "Unknown document")

            page_start = source.get("page_start")

            page_end = source.get("page_end")

            section = source.get("section")

            text = str(source.get("text", ""))

            st.markdown(f"### [{citation_id}] {filename}")

            metadata_parts: list[str] = []

            if page_start is not None:
                if page_end is not None and page_end != page_start:
                    metadata_parts.append(f"Pages {page_start}–{page_end}")
                else:
                    metadata_parts.append(f"Page {page_start}")

            if section:
                metadata_parts.append(f"Section: {section}")

            if metadata_parts:
                st.caption(" · ".join(metadata_parts))

            if text:
                st.markdown("> " + text.replace("\n", "\n> "))

            st.divider()


def _valid_session_id(value: str | None) -> bool:
    """Return whether a browser session ID is safe to reuse."""

    if value is None:
        return False

    return bool(_SESSION_ID_PATTERN.fullmatch(value.strip()))


def _persist_session_cookie(session_id: str) -> None:
    """
    Persist the anonymous session namespace in the browser.

    The cookie is intentionally client-side and non-authenticated.
    SameSite=Lax limits cross-site sending, and Secure is added
    automatically when the app is served over HTTPS.
    """

    cookie_name = json.dumps(SESSION_COOKIE_NAME)
    cookie_value = json.dumps(session_id)

    st.html(
        f"""
        <script>
        (() => {{
            const name = {cookie_name};
            const value = {cookie_value};

            let cookie = (
                name
                + "="
                + encodeURIComponent(value)
                + "; Path=/; SameSite=Lax"
            );

            if (window.location.protocol === "https:") {{
                cookie += "; Secure";
            }}

            document.cookie = cookie;
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _initialize_session_state() -> None:
    """Initialize Streamlit state and stable browser session identity."""

    browser_session_id = st.context.cookies.get(SESSION_COOKIE_NAME)

    if "session_id" not in st.session_state:
        if isinstance(browser_session_id, str) and _valid_session_id(browser_session_id):
            st.session_state.session_id = browser_session_id.strip()

        else:
            st.session_state.session_id = uuid4().hex

    if "session_cookie_initialized" not in st.session_state:
        session_id = str(st.session_state.session_id)

        if not isinstance(browser_session_id, str) or browser_session_id.strip() != session_id:
            _persist_session_cookie(session_id)

        st.session_state.session_cookie_initialized = True

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_scope" not in st.session_state:
        st.session_state.document_scope = None


def _render_chat_history() -> None:
    """Render conversation history."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                if message.get("abstained"):
                    st.caption(
                        "The answer was rejected because "
                        "the selected indexed documents "
                        "did not provide sufficient "
                        "supporting evidence."
                    )

                else:
                    sources = message.get("sources", [])

                    if isinstance(sources, list):
                        render_sources(sources)


def main() -> None:
    """Run the DocQ&A Streamlit application."""

    st.set_page_config(page_title="DocQ&A", page_icon="📚", layout="centered")

    _initialize_session_state()

    session_id = str(st.session_state.session_id)

    st.title("📚 DocQ&A")

    st.caption("Upload PDF, DOCX, or TXT files and ask questions grounded in their contents.")

    # -----------------------------------------------------
    # Backend
    # -----------------------------------------------------

    api_url = DEFAULT_API_URL

    top_k = PRODUCTION_TOP_K

    st.sidebar.caption("Retrieval depth: production top-k = 5")

    (is_healthy, health_data, health_error) = get_api_health(api_url)

    if is_healthy and health_data is not None:
        st.sidebar.success("API connected")

        st.sidebar.caption(f"{health_data.get('app', 'DocQ&A')} v{health_data.get('version', '?')}")

    else:
        st.sidebar.error("API unavailable")

        if health_error:
            st.sidebar.caption(health_error)

    # -----------------------------------------------------
    # Document management
    # -----------------------------------------------------

    st.sidebar.divider()
    st.sidebar.header("Documents")

    documents: list[dict[str, Any]] = []

    document_error: str | None = None

    if is_healthy:
        (documents, document_error) = list_documents(api_url, session_id)

    if document_error:
        st.sidebar.error(document_error)

    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=list(SUPPORTED_UPLOAD_TYPES),
        accept_multiple_files=True,
        disabled=not is_healthy,
        help=("Select one or several PDF, DOCX, or TXT files. Files are indexed sequentially."),
    )

    if st.sidebar.button(
        "Upload & index", disabled=(not is_healthy or not uploaded_files), use_container_width=True
    ):
        successful = 0
        failed = 0

        progress = st.sidebar.progress(0)

        for index, uploaded_file in enumerate(uploaded_files, start=1):
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                response, error = upload_document(
                    api_url,
                    session_id=session_id,
                    filename=(uploaded_file.name),
                    content=(uploaded_file.getvalue()),
                    content_type=getattr(uploaded_file, "type", None),
                )

            if error is not None:
                failed += 1
                st.sidebar.error(f"{uploaded_file.name}: {error}")

            elif response is not None:
                successful += 1

                chunk_count = response.get("chunk_count", "?")

                st.sidebar.success(f"{uploaded_file.name}: indexed ({chunk_count} chunks)")

            progress.progress(int((index / len(uploaded_files)) * 100))

        st.sidebar.caption(f"Upload complete: {successful} succeeded, {failed} failed.")

        (documents, document_error) = list_documents(api_url, session_id)

        if document_error:
            st.sidebar.error(document_error)

    st.sidebar.caption(f"Indexed documents: {len(documents)}")

    selected_document_id: str | None = None

    selected_document: dict[str, Any] | None = None

    search_mode = st.sidebar.radio(
        "Search scope",
        options=["Selected document", "All documents"],
        disabled=(not is_healthy or not documents),
    )

    if documents and search_mode == "Selected document":
        document_by_id = {
            str(document.get("document_id", "")): document
            for document in documents
            if document.get("document_id")
        }

        document_ids = list(document_by_id)

        selected_document_id = st.sidebar.selectbox(
            "Active document",
            options=document_ids,
            format_func=lambda doc_id: (format_document_label(document_by_id[doc_id])),
        )

        if selected_document_id is not None:
            selected_document = document_by_id[selected_document_id]

    # -----------------------------------------------------
    # Delete selected document
    # -----------------------------------------------------

    if is_healthy and selected_document_id is not None and selected_document is not None:
        st.sidebar.divider()

        confirm_delete = st.sidebar.checkbox(
            f"Confirm delete {selected_document.get('filename', '')}"
        )

        if st.sidebar.button(
            "Delete selected document", disabled=not confirm_delete, use_container_width=True
        ):
            deleted, error = delete_document(api_url, selected_document_id, session_id)

            if error is not None:
                st.sidebar.error(error)

            elif deleted:
                st.session_state.messages = []
                st.session_state.document_scope = None
                st.sidebar.success("Document deleted.")
                st.rerun()

    # -----------------------------------------------------
    # Conversation scope
    # -----------------------------------------------------

    if search_mode == "All documents":
        scope_key = "all_documents" if documents else "no_documents"

    else:
        scope_key = selected_document_id or "no_document_selected"

    if st.session_state.document_scope != scope_key:
        st.session_state.messages = []
        st.session_state.document_scope = scope_key

    if st.sidebar.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # -----------------------------------------------------
    # Main-page state
    # -----------------------------------------------------

    if not is_healthy:
        st.warning("The DocQ&A backend is unavailable. Start FastAPI and refresh the page.")

    elif not documents:
        st.info("Upload at least one PDF, DOCX, or TXT document to start asking questions.")

    elif search_mode == "All documents":
        st.info(f"Search scope: all {len(documents)} indexed documents.")

    elif selected_document is not None:
        st.info(f"Search scope: {format_document_label(selected_document)}")

    # -----------------------------------------------------
    # Conversation
    # -----------------------------------------------------

    _render_chat_history()

    chat_disabled = (
        not is_healthy
        or not documents
        or (search_mode == "Selected document" and selected_document_id is None)
    )

    question = st.chat_input(
        "Ask a question about your documents...",
        max_chars=MAX_QUESTION_CHARS,
        disabled=chat_disabled,
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                response, error = ask_api(
                    api_url=api_url,
                    question=question,
                    top_k=top_k,
                    session_id=session_id,
                    document_id=(
                        selected_document_id if search_mode == "Selected document" else None
                    ),
                )

            if error is not None:
                st.error(error)

            elif response is not None:
                answer = str(response.get("answer", "No answer returned."))

                abstained = bool(response.get("abstained", False))

                cited_source_ids = response.get("cited_source_ids", [])

                sources = response.get("sources", [])

                if not isinstance(cited_source_ids, list):
                    cited_source_ids = []

                if not isinstance(sources, list):
                    sources = []

                st.markdown(answer)

                if abstained:
                    st.caption(
                        "The answer was rejected because "
                        "the selected indexed documents "
                        "did not provide sufficient "
                        "supporting evidence."
                    )

                else:
                    render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "abstained": abstained,
                        "cited_source_ids": (cited_source_ids),
                        "sources": sources,
                    }
                )


if __name__ == "__main__":
    main()
