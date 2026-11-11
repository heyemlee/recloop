import type { Metadata } from "next";
import { Suspense } from "react";
import { Geist_Mono, Inter } from "next/font/google";

import { Rail } from "@/components/rail";
import { TopBar } from "@/components/top-bar";
import { UserProvider } from "@/lib/user";

import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RecLoop",
  description: "Personalised recommendations for an office-supplies store (RecLoop demo)",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${inter.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full bg-background">
        <UserProvider>
          {/* The rail on the left (on top on a phone), the page next to it. */}
          <div className="mx-auto flex min-h-dvh w-full max-w-[1320px] gap-4 px-4 pt-5 pb-20 max-md:flex-col md:gap-6 md:px-6">
            {/* Both read the path, which a page with a dynamic segment only has at request time. */}
            <Suspense fallback={<div className="md:w-12" />}>
              <Rail />
            </Suspense>
            <div className="flex min-w-0 flex-1 flex-col">
              <Suspense fallback={<div className="min-h-14" />}>
                <TopBar />
              </Suspense>
              {children}
            </div>
          </div>
        </UserProvider>
      </body>
    </html>
  );
}
