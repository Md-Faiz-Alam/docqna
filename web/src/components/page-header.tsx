/** Title band shared by inner pages, with the same soft glow as the hero. */
export function PageHeader({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <section className="relative isolate overflow-hidden border-b">
      <div aria-hidden className="absolute inset-0 -z-10">
        <div className="bg-grid absolute inset-0 opacity-70" />
        <div className="absolute -top-32 left-1/4 size-[26rem] animate-aurora rounded-full bg-glow-1 opacity-20 blur-[100px] dark:opacity-30" />
        <div className="absolute -top-20 right-1/5 size-[20rem] animate-aurora rounded-full bg-glow-2 opacity-15 blur-[100px] [animation-delay:-8s]" />
      </div>
      <div className="mx-auto max-w-6xl animate-in px-4 pt-12 pb-10 fade-in-0 slide-in-from-bottom-2 duration-700 sm:px-6">
        <p className="text-sm font-semibold tracking-wide text-primary uppercase">{eyebrow}</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
        {children && <div className="mt-3 max-w-3xl text-muted-foreground text-pretty">{children}</div>}
      </div>
    </section>
  );
}
