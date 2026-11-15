/**
 * Anonymous browser session: a random ID the backend uses to keep each
 * visitor's documents separate. It is not a login.
 */

const STORAGE_KEY = "docqna_session_id";
const VALID = /^[A-Za-z0-9_-]{32,128}$/;

function randomId(): string {
  const bytes = new Uint8Array(24);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

export function getSessionId(): string {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored && VALID.test(stored)) return stored;
    const created = randomId();
    window.localStorage.setItem(STORAGE_KEY, created);
    return created;
  } catch {
    // Storage blocked (private mode): a per-page session still works.
    return randomId();
  }
}

/** Start over with an empty document space. */
export function resetSessionId(): string {
  const created = randomId();
  try {
    window.localStorage.setItem(STORAGE_KEY, created);
  } catch {
    // ignore
  }
  return created;
}
