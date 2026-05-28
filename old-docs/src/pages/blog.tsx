import React, { type ReactNode } from "react";
import styles from "./blog.module.css";
import LayoutProvider from "@theme/Layout/Provider";
import Footer from "@theme/Footer";
import Navbar from "@theme/Navbar";
import Link from "@docusaurus/Link";
import SchemaInjector from "../components/SchemaInjector/SchemaInjector";
import { buildBlogHomeSchema } from "@site/src/utils/schema-helpers";

interface AuthorData {
  name: string;
  title: string;
  url: string;
  image_url: string;
}

interface AuthorsMap {
  [key: string]: AuthorData;
}

interface Blog {
  title: string;
  description: string;
  slug: string;
  authors: string[];
  date: string;
}

interface BlogCardProps {
  title: string;
  description: string;
  date: string;
  authors: string[];
  slug: string;
  authorsData: AuthorsMap;
}

class BlogCard extends React.Component<BlogCardProps> {
  formatDate(dateString: string): string {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  }

  renderAuthors(authors: string[], authorsData: AuthorsMap): ReactNode {
    if (!authors || authors.length === 0) {
      return <span className={styles.author}>Anonymous</span>;
    }
    
    return authors.map((authorKey, index) => {
      const author = authorsData[authorKey];
      if (!author) {
        return <span key={index} className={styles.author}>{authorKey}</span>;
      }
      
      return (
        <div key={index} className={styles.authorItem}>
          <img 
            src={author.image_url} 
            alt={author.name}
            className={styles.authorImage}
          />
          <span className={styles.author}>{author.name}</span>
        </div>
      );
    });
  }

  render(): ReactNode {
    const { title, description, date, authors, slug, authorsData } = this.props;
    const blogUrl = `/blog/${slug}`;

    return (
      <Link to={blogUrl} className={styles.blogCard}>
        <div className={styles.blogCardHeader}>
          <div className={styles.title}>{title}</div>
          <div className={styles.date}>{this.formatDate(date)}</div>
        </div>
        <p className={styles.description}>{description}</p>
        <div className={styles.meta}>
          <div className={styles.authors}>
            {this.renderAuthors(authors, authorsData)}
          </div>
          <div className={styles.readMore}>
            Read more
            <img src="icons/right-arrow.svg" alt="arrow" />
          </div>
        </div>
      </Link>
    );
  }
}

interface IndexState {
  blogs: Blog[];
  authors: AuthorsMap;
  loading: boolean;
  error: string | null;
}

class Index extends React.Component<Record<string, unknown>, IndexState> {
  constructor(props: Record<string, unknown>) {
    super(props);
    this.state = {
      blogs: [],
      authors: {},
      loading: true,
      error: null
    };
  }

  async componentDidMount(): Promise<void> {
    try {
      const blogData: Blog[] = [
        {
          title: "Breaking Gemini 2.5 Pro using DeepTeam",
          description: "An in-depth analysis of Gemini 2.5 Pro's vulnerabilities using DeepTeam, revealing how different attack strategies can bypass AI safety measures",
          slug: "breaking-gemini-pro-deepteam",
          authors: ["penguine"],
          date: "2025-05-24"
        },
        {
          title: "Comparing Jailbreaking Techniques",
          description: "Comparative evaluation of Linear, Tree, and Crescendo jailbreaking techniques against six contemporary AI models reveals significant performance disparities and model-specific vulnerability patterns",
          slug: "multi-turn-jailbreaking-analysis-deepteam",
          authors: ["kritinv"],
          date: "2025-06-05"
        },
        {
          title: "Claude Fortress Analysis",
          description: "A comprehensive analysis of Claude's security mechanisms and potential vulnerabilities when subjected to various red teaming techniques",
          slug: "claude-sonnet-4-defense-analysis-deepteam-v2",
          authors: ["penguine"],
          date: "2025-04-15"
        },
        {
          title: "Bypassing Claude with Advanced Techniques",
          description: "Exploring sophisticated methods to bypass Claude's safety measures using multi-turn conversations and contextual manipulation",
          slug: "shakespeare-claude-jailbreak-deepteam",
          authors: ["kritinv"],
          date: "2025-03-20"
        },
        {
          title: "The Intensive Alignment Paradox",
          description: "An exploration of how intensive alignment training can sometimes create unexpected vulnerabilities in large language models",
          slug: "ai-safety-paradox-deepteam",
          authors: ["sid", "penguine"],
          date: "2025-02-10"
        }
      ];

      const authorsData: AuthorsMap = {
        penguine: {
          name: "Penguine",
          title: "DeepTeam Wizard",
          url: "https://github.com/penguine-ip",
          image_url: "https://github.com/penguine-ip.png"
        },
        kritinv: {
          name: "Kritin",
          title: "DeepTeam Guru",
          url: "https://github.com/kritinv",
          image_url: "https://github.com/kritinv.png"
        },
        sid: {
          name: "Sid",
          title: "DeepTeamer",
          url: "https://github.com/sid-sredharan",
          image_url: "https://avatars.githubusercontent.com/u/133195670?s=400&u=2f0ec53bdb20a06391b4ae992d96da7b539b08fe&v=4"
        }
      };

      const sortedBlogs = blogData.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

      this.setState({
        blogs: sortedBlogs,
        authors: authorsData,
        loading: false
      });
    } catch (error) {
      this.setState({
        error: 'Failed to load blog posts',
        loading: false
      });
    }
  }

  render(): ReactNode {
    const { blogs, authors, loading, error } = this.state;

    const blogHomeSchema = blogs.length > 0 ? buildBlogHomeSchema(blogs) : null;

    return (
      <div className={styles.blogHomeContainer}>
        {blogHomeSchema && <SchemaInjector schema={blogHomeSchema} />}
        <div className={styles.blogHeroCard}>
          <div className={styles.content}>
            <span className={styles.tag}>Blog</span>
            <h1>Where AI Safety Begins</h1>
              <p>
                Insights, research, and discoveries in LLM red teaming, AI safety, 
                and adversarial testing from the DeepTeam community.
              </p>
              <a 
                href="https://github.com/confident-ai/deepteam" 
                target="_blank" 
                rel="noopener noreferrer"
                className={styles.githubButton}
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
                  <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                Contribute on GitHub
              </a>
            </div>
        </div>
        
        <div className={styles.blogContainer}>
          {loading && (
            <div className={styles.loadingContainer}>
              <div className={styles.loadingText}>Loading blog posts...</div>
            </div>
          )}
          
          {error && (
            <div className={styles.errorContainer}>
              <div className={styles.errorText}>{error}</div>
            </div>
          )}
          
          {!loading && !error && blogs.length === 0 && (
            <div className={styles.noBlogsContainer}>
              <div className={styles.noBlogsText}>No blog posts available.</div>
            </div>
          )}
          
          {!loading && !error && blogs.length > 0 && blogs.map((blog, index) => (
            <BlogCard
              key={index}
              title={blog.title}
              description={blog.description}
              date={blog.date}
              authors={blog.authors}
              slug={blog.slug}
              authorsData={authors}
            />
          ))}
        </div>
      </div>
    );
  }
}

export default function BlogHome(props: Record<string, unknown>): ReactNode {
  return (
    <LayoutProvider>
      <Navbar />
      <Index {...props} />
      <Footer />
    </LayoutProvider>
  );
}
