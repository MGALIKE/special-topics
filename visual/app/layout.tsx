import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "FDTD Antenna Simulator — Visualization Dashboard",
  description:
    "Real-time 3D visualization of Inverted-F Antenna FDTD simulation results. View S-parameters, radiation patterns, and time-domain waveforms.",
  keywords: [
    "FDTD",
    "antenna",
    "simulation",
    "electromagnetic",
    "IFA",
    "S-parameters",
    "radiation pattern",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable}`}>
      <body style={{ fontFamily: "var(--font-inter), system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
