import React, { type ReactNode, useMemo } from 'react';
import BlogPostPage from '@theme-original/BlogPostPage';
import type BlogPostPageType from '@theme/BlogPostPage';
import type { WrapperProps } from '@docusaurus/types';
import SchemaInjector from '@site/src/components/SchemaInjector/SchemaInjector';
import { buildArticleSchema } from '@site/src/utils/schema-helpers';

type Props = WrapperProps<typeof BlogPostPageType>;

export default function BlogPostPageWrapper(props: Props): ReactNode {
  const { metadata, frontMatter } = props.content;

  const articleSchema = useMemo(() => {

    const authorNames = metadata.authors?.map((author: any) => author.name) || frontMatter.authors;

    return buildArticleSchema({
      title: metadata.title,
      description: frontMatter.description as string || metadata.description,
      url: metadata.permalink,
      datePublished: metadata.date,
      authors: authorNames as string[] | undefined,
      image: frontMatter.image as string | undefined,
    });
  }, [metadata, frontMatter]);

  return (
    <>
      <SchemaInjector schema={articleSchema} />
      <BlogPostPage {...props} />
    </>
  );
}
