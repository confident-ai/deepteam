import { visit, SKIP } from "unist-util-visit";
import { toString as mdastToString } from "mdast-util-to-string";
import type { Root } from "mdast";
import type { ContainerDirective, LeafDirective, TextDirective } from "mdast-util-directive";

const ADMONITION_TYPES = new Set([
  "note",
  "info",
  "tip",
  "success",
  "important",
  "warning",
  "caution",
  "danger",
  "error",
  "secondary",
]);

/**
 * `remark-directive` parses every `:name` and `::name` occurrence as a
 * text / leaf directive, even when the author meant a literal colon
 * (e.g. `(LLM01:2025)` in headings, `Step :1` in prose). Since we only
 * use *container* directives (`:::type[title]` for admonitions —
 * handled by `remarkAdmonitions` above), every text + leaf directive
 * the parser produces is unintended and ends up either silently
 * dropped or rendered as an empty span, mangling the surrounding
 * prose.
 *
 * This plugin converts orphan text + leaf directive nodes back to
 * their original text representation so authors can write
 * `Foo:bar`-style content freely without escaping every colon. Runs
 * AFTER `remarkAdmonitions` so legitimate container directives are
 * already consumed.
 */
export function remarkRestoreUnusedDirectives() {
  return (tree: Root) => {
    visit(tree, (node, index, parent) => {
      if (!parent || index == null) return;
      if (node.type !== "textDirective" && node.type !== "leafDirective") return;

      const directive = node as TextDirective | LeafDirective;
      const prefix = node.type === "leafDirective" ? "::" : ":";

      // Reconstruct the literal source. Attributes / labels on these
      // orphan directives are vanishingly rare in our content; if they
      // appear we still get readable text by stringifying children.
      const childrenText = mdastToString(directive).trim();
      const reconstructed = childrenText
        ? `${prefix}${directive.name}[${childrenText}]`
        : `${prefix}${directive.name}`;

      parent.children.splice(index, 1, { type: "text", value: reconstructed });
      return [SKIP, index + 1];
    });
  };
}

/**
 * Converts Docusaurus-style `:::type[title]` container directives into
 * `<Callout type="..." title="...">` MDX JSX elements. Requires
 * `remark-directive` to run before this plugin.
 */
export function remarkAdmonitions() {
  return (tree: Root) => {
    visit(tree, "containerDirective", (node: ContainerDirective, index, parent) => {
      if (!ADMONITION_TYPES.has(node.name)) return;
      if (!parent || index == null) return;

      // The label (from `:::note[My Title]`) lives as the first child
      // paragraph with `data.directiveLabel` — pluck it out.
      let title: string | undefined;
      const children = [...(node.children ?? [])];
      const labelIdx = children.findIndex(
        (child) =>
          child.type === "paragraph" && (child as { data?: { directiveLabel?: boolean } }).data?.directiveLabel,
      );
      if (labelIdx !== -1) {
        const [label] = children.splice(labelIdx, 1);
        title = mdastToString(label).trim();
      }

      const attributes: Array<{
        type: "mdxJsxAttribute";
        name: string;
        value: string;
      }> = [{ type: "mdxJsxAttribute", name: "type", value: node.name }];
      if (title) {
        attributes.push({ type: "mdxJsxAttribute", name: "title", value: title });
      }

      const replacement = {
        type: "mdxJsxFlowElement" as const,
        name: "Callout",
        attributes,
        children,
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      parent.children.splice(index, 1, replacement as any);
    });
  };
}

export default remarkAdmonitions;
