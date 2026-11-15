"use client";

import { CheckIcon, MonitorIcon, MoonIcon, PaletteIcon, SunIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { Popover } from "radix-ui";
import { useSyncExternalStore } from "react";

import { setAccent, useAccent } from "@/components/use-accent";
import { Button } from "@/components/ui/button";
import { ACCENTS } from "@/lib/accents";
import { cn } from "@/lib/utils";

const MODES = [
  { id: "light", label: "Light", icon: SunIcon },
  { id: "dark", label: "Dark", icon: MoonIcon },
  { id: "system", label: "System", icon: MonitorIcon },
] as const;

const subscribeNothing = () => () => undefined;

/** Light / dark / system mode plus a colour theme, in one popover. */
export function ThemePicker() {
  const { theme, setTheme } = useTheme();
  const accent = useAccent();
  // The saved mode is only known in the browser; avoid a hydration mismatch.
  const mounted = useSyncExternalStore(subscribeNothing, () => true, () => false);

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <Button variant="ghost" size="icon" aria-label="Change theme">
          <PaletteIcon />
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={8}
          className="z-50 w-72 rounded-xl border bg-popover/95 p-4 text-popover-foreground shadow-2xl backdrop-blur-xl outline-none data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95"
        >
          <fieldset>
            <legend className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
              Mode
            </legend>
            <div className="grid grid-cols-3 gap-1 rounded-lg bg-muted p-1">
              {MODES.map((mode) => {
                const active = mounted && theme === mode.id;
                return (
                  <button
                    key={mode.id}
                    type="button"
                    aria-pressed={active}
                    onClick={() => setTheme(mode.id)}
                    className={cn(
                      "flex items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-medium text-muted-foreground transition-all outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
                      active && "bg-background text-foreground shadow-sm",
                    )}
                  >
                    <mode.icon className="size-3.5" aria-hidden />
                    {mode.label}
                  </button>
                );
              })}
            </div>
          </fieldset>

          <fieldset className="mt-4">
            <legend className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">
              Colour
            </legend>
            <div className="grid grid-cols-4 gap-2">
              {ACCENTS.map((option) => {
                const active = accent === option.id;
                return (
                  <button
                    key={option.id}
                    type="button"
                    aria-pressed={active}
                    aria-label={`${option.label} theme`}
                    title={option.label}
                    onClick={() => setAccent(option.id)}
                    className={cn(
                      "group flex flex-col items-center gap-1 rounded-lg p-1.5 text-[11px] text-muted-foreground transition-colors outline-none hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50",
                      active && "text-foreground",
                    )}
                  >
                    <span
                      className={cn(
                        "flex size-7 items-center justify-center rounded-full ring-2 ring-transparent ring-offset-2 ring-offset-background transition-all group-hover:scale-110",
                        active && "ring-foreground/70",
                      )}
                      style={{ background: option.swatch }}
                    >
                      {active && <CheckIcon className="size-3.5 text-white" aria-hidden />}
                    </span>
                    {option.label}
                  </button>
                );
              })}
            </div>
          </fieldset>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
