import { type ReactNode, useMemo } from "react";
import Layout from "@theme-original/DocItem/Layout";
import type LayoutType from "@theme/DocItem/Layout";
import type { WrapperProps } from "@docusaurus/types";
import { useDoc } from "@docusaurus/plugin-content-docs/client";
import SchemaInjector from "@site/src/components/SchemaInjector/SchemaInjector";
import {
  buildArticleSchema,
  buildBreadcrumbSchema,
} from "@site/src/utils/schema-helpers";
import CopyPageButton from "@site/src/components/CopyPageButton";

type Props = WrapperProps<typeof LayoutType>;

export default function LayoutWrapper(props: Props): ReactNode {
  const { frontMatter, metadata } = useDoc();

  const breadcrumbs = (props as any).route?.context?.breadcrumbs || [];
  const breadcrumbTrail = breadcrumbs.map((crumb: any) => ({
    name: crumb.label,
    url: crumb.href,
  }));
  const breadcrumbSchema = useMemo(
    () => buildBreadcrumbSchema(breadcrumbTrail),
    [breadcrumbs]
  );

  const mainSchema = useMemo(() => {
    return buildArticleSchema({
      title: metadata.title,
      description: (frontMatter.description as string) || metadata.description,
      url: metadata.permalink,
      datePublished: (frontMatter as any).date
        ? new Date((frontMatter as any).date as string).toISOString()
        : undefined,
      authors: (frontMatter as any).authors as string[] | undefined,
      image: frontMatter.image as string | undefined,
    });
  }, [metadata, frontMatter]);

  return (
    <>
      <SchemaInjector schema={breadcrumbSchema} />
      <SchemaInjector schema={mainSchema} />
      <Layout {...props} />
      <CopyPageButton />
    </>
  );
}
