import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "ClaimSense", description: "Adjuster dashboard" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="top">
          <Link href="/" className="brand">ClaimSense</Link>
          <span className="muted">Adjuster dashboard · synthetic data</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
