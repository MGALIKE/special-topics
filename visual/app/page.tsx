import Dashboard from "@/components/Dashboard";
import type { SimulationData } from "@/lib/demoData";

// This is a Server Component — no "use client" directive
// Attempts to pre-fetch results from the FDTD backend at render time
// for maximum perceived speed (no loading spinner for initial data)

async function getInitialData(): Promise<SimulationData | null> {
  try {
    const res = await fetch("http://localhost:4000/results", {
      cache: "no-store",
      signal: AbortSignal.timeout(2000), // 2s timeout
    });
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Backend not running — that's fine, Dashboard will use demo data
  }
  return null;
}

export default async function Home() {
  const initialData = await getInitialData();

  return <Dashboard initialData={initialData} />;
}
