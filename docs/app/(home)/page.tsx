import type { Metadata } from "next";
import { DocsBody } from "fumadocs-ui/layouts/notebook/page";
import { getMDXComponents } from "@/components/mdx";
import ReadMe from "@/home/read-me.mdx";
import HomeLayout from "@/src/layouts/HomeLayout";
import HomeHeroSection from "@/src/sections/home/HomeHeroSection";
import { siteTitle } from "@/lib/shared";

// Homepage sets `title.absolute` so the root layout's `%s | …` template
// doesn't double up the site name. The tagline here mirrors the old
// Docusaurus `tagline` ("Red Teaming Framework for LLMs") expanded into
// a proper meta-description sentence.
export const metadata: Metadata = {
  title: { absolute: siteTitle },
  description:
    "DeepTeam is the open-source LLM red teaming framework for testing and securing LLM applications — adversarial attacks, agentic vulnerabilities, guardrails, and compliance frameworks.",
  alternates: { canonical: "/" },
};

export default function HomePage() {
  return (
    <HomeLayout
      leftContent={<HomeHeroSection />}
      rightContent={
        <div className="docs-page-surface">
          <DocsBody>
            <ReadMe components={getMDXComponents()} />
          </DocsBody>
        </div>
      }
    />
  );
}
