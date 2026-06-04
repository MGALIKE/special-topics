import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// Local Helvetica family (self-hosted from /public). Swiss / International
// Typographic Style leans on Helvetica with heavy use of the Bold weight.
const helvetica = localFont({
  variable: "--font-helvetica",
  display: "swap",
  src: [
    { path: "../public/Helvetica.ttf", weight: "400", style: "normal" },
    { path: "../public/Helvetica-Oblique.ttf", weight: "400", style: "italic" },
    { path: "../public/Helvetica-Bold.ttf", weight: "700", style: "normal" },
    {
      path: "../public/Helvetica-BoldOblique.ttf",
      weight: "700",
      style: "italic",
    },
  ],
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
    <html lang="en" className={helvetica.variable}>
      <body>{children}</body>
    </html>
  );
}
