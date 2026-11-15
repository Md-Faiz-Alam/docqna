import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ThemePicker } from "@/components/theme-picker";
import { ACCENT_INIT_SCRIPT, ACCENT_STORAGE_KEY, ACCENTS } from "@/lib/accents";

const setTheme = vi.fn();
vi.mock("next-themes", () => ({ useTheme: () => ({ theme: "system", setTheme }) }));

describe("theme picker", () => {
  it("switches colour theme and mode, and remembers the colour", async () => {
    render(<ThemePicker />);
    await userEvent.click(screen.getByRole("button", { name: "Change theme" }));

    const colourNames = new RegExp(`^(${ACCENTS.map((a) => a.label).join("|")}) theme$`);
    expect(screen.getAllByRole("button", { name: colourNames })).toHaveLength(ACCENTS.length);

    await userEvent.click(screen.getByRole("button", { name: "Teal theme" }));
    expect(document.documentElement.dataset.accent).toBe("teal");
    expect(window.localStorage.getItem(ACCENT_STORAGE_KEY)).toBe("teal");
    expect(screen.getByRole("button", { name: "Teal theme" })).toHaveAttribute("aria-pressed", "true");

    await userEvent.click(screen.getByRole("button", { name: "Dark" }));
    expect(setTheme).toHaveBeenCalledWith("dark");
  });

  it("applies a saved colour before paint and ignores unknown values", () => {
    window.localStorage.setItem(ACCENT_STORAGE_KEY, "rose");
    new Function(ACCENT_INIT_SCRIPT)();
    expect(document.documentElement.dataset.accent).toBe("rose");

    window.localStorage.setItem(ACCENT_STORAGE_KEY, "<script>");
    new Function(ACCENT_INIT_SCRIPT)();
    expect(document.documentElement.dataset.accent).toBe("indigo");
  });
});
