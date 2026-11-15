import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Node 25+ ships its own experimental `localStorage` global, which is
// undefined unless Node gets a storage file, and it hides jsdom's. Give the
// tests a working in-memory Storage, like a browser's.
if (typeof window !== "undefined" && !window.localStorage) {
  const data = new Map<string, string>();
  const storage: Storage = {
    get length() {
      return data.size;
    },
    clear: () => data.clear(),
    getItem: (key) => data.get(key) ?? null,
    key: (index) => [...data.keys()][index] ?? null,
    removeItem: (key) => void data.delete(key),
    setItem: (key, value) => void data.set(key, String(value)),
  };
  Object.defineProperty(window, "localStorage", { configurable: true, value: storage });
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value: storage });
}

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});
