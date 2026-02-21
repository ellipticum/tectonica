import type { Metadata } from "next";
import { JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

const monoFont = JetBrains_Mono({
  variable: "--font-mono-debug",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Planetary Causality Engine",
  description: "Procedural geophysics simulator with causal layers and event-driven updates.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${spaceGrotesk.variable} ${monoFont.variable} antialiased`}>
        <TooltipProvider>
          <div className="relative min-h-screen overflow-x-hidden bg-slate-950 text-foreground">
            {children}
          </div>
        </TooltipProvider>
      </body>
    </html>
  );
}
