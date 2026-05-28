import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.ctfassets.net',
      },
      // Authored MDX references S3-hosted images directly (e.g.
      // `<img src="https://deepteam-docs.s3…png">`) and Next's MDX
      // pipeline lowers those to `next/image`, which rejects unknown
      // hosts. Allow the buckets explicitly rather than reaching for
      // `unoptimized: true`, so images still get optimized.
      //
      // `deepteam-docs.s3{,.us-east-1}.amazonaws.com` hosts the
      // DeepTeam doc imagery (both URL forms appear in authored MDX);
      // `deepeval-docs.s3.us-east-1.amazonaws.com` is still referenced
      // by a couple of ported blog covers and home-section assets.
      {
        protocol: 'https',
        hostname: 'deepteam-docs.s3.amazonaws.com',
      },
      {
        protocol: 'https',
        hostname: 'deepteam-docs.s3.us-east-1.amazonaws.com',
      },
      {
        protocol: 'https',
        hostname: 'deepeval-docs.s3.us-east-1.amazonaws.com',
      },
    ],
  },
};

export default withMDX(config);
