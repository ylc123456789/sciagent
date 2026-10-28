"""Private, bounded Bash AST views for source evidence, never shell execution.

Tree-sitter owns syntax recognition. This adapter only extracts literal words
and top-level linear command relationships; opaque expressions stay opaque.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from tree_sitter import Language, Node, Parser
import tree_sitter_bash


MAX_COMMAND_BYTES = 64 * 1024
MAX_AST_NODES = 8192
MAX_AST_DEPTH = 64


@dataclass(frozen=True)
class BashStructure:
    commands: list[dict[str, Any]]
    gaps: list[dict[str, Any]]


def _unescape(text: str, *, double_quoted: bool = False) -> str | None:
    """Remove literal Bash quoting escapes, without evaluating expansions."""
    result: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 1
            if index >= len(text):
                return None
            following = text[index]
            if following != "\n":
                if double_quoted and following not in '\\"$`':
                    result.append("\\")
                result.append(following)
        else:
            # These unquoted characters can expand without a dedicated AST
            # node (notably globs and brace expansion). Quoted literals remain
            # supported; this intentionally does not implement expansion.
            if not double_quoted and character in "*?[]{}~$`":
                return None
            result.append(character)
        index += 1
    return "".join(result)


def parse_bash_structure(source: str) -> BashStructure:
    """Extract only finite linear Bash statements and retain local unknowns."""
    commands: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    try:
        raw = source.encode("utf-8")
    except UnicodeError:
        return BashStructure([], [{"code": "shell_encoding_unavailable"}])
    if len(raw) > MAX_COMMAND_BYTES:
        return BashStructure([], [{"code": "shell_command_bytes_limit", "limit": MAX_COMMAND_BYTES}])
    if not source.strip() or "\x00" in source:
        return BashStructure([], [{"code": "unsupported_shell_syntax"}])
    try:
        root = Parser(Language(tree_sitter_bash.language())).parse(raw).root_node
    except (ValueError, RuntimeError, OverflowError):
        return BashStructure([], [{"code": "shell_parse_failed"}])
    has_error = root.has_error
    stack = [(root, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_AST_NODES:
            return BashStructure([], [{"code": "shell_ast_nodes_limit", "limit": MAX_AST_NODES}])
        if depth > MAX_AST_DEPTH:
            return BashStructure([], [{"code": "shell_ast_depth_limit", "limit": MAX_AST_DEPTH}])
        if node.is_error or node.is_missing:
            has_error = True
        stack.extend((child, depth + 1) for child in node.children)
    if has_error:
        return BashStructure([], [{"code": "unsupported_shell_syntax", "reason": "parse_error_or_missing_node"}])

    def text(node: Node) -> str:
        return raw[node.start_byte:node.end_byte].decode("utf-8")

    def span(node: Node) -> list[int]:
        return [node.start_byte, node.end_byte]

    def literal(node: Node) -> str | None:
        value = text(node)
        if node.type in {"word", "number"}:
            return _unescape(value)
        if node.type == "raw_string":
            return value[1:-1]
        if node.type == "string":
            if any(child.type != "string_content" for child in node.named_children):
                return None
            return _unescape(value[1:-1], double_quoted=True)
        if node.type in {"concatenation", "command_name"}:
            pieces = [literal(child) for child in node.named_children]
            return "".join(pieces) if pieces and all(piece is not None for piece in pieces) else None
        return None

    def opaque(node: Node) -> dict[str, Any]:
        pending = [node]
        evaluated: set[str] = set()
        while pending:
            child = pending.pop()
            if child.type in {"command_substitution", "process_substitution", "arithmetic_expansion", "expansion"}:
                evaluated.add(child.type)
            pending.extend(child.named_children)
        return {"span": span(node), "text": text(node), "node_type": node.type,
                "evaluated_expressions": sorted(evaluated)}

    def view(node: Node, *, conditional: bool, relation: str,
             redirects: tuple[Node, ...] = ()) -> None:
        original = node
        if node.type == "redirected_statement":
            body = node.child_by_field_name("body")
            redirects = (*redirects, *(child for child in node.named_children
                                      if "redirect" in child.type))
            if body is None or body.type != "command":
                unsupported(original, conditional=conditional, relation=relation)
                return
            node = body
        if node.type != "command":
            unsupported(original, conditional=conditional, relation=relation)
            return
        name = node.child_by_field_name("name")
        arguments = list(node.children_by_field_name("argument"))
        assignments = [opaque(child) for child in node.named_children
                       if child.type == "variable_assignment"]
        redirects = tuple(sorted((*redirects, *node.children_by_field_name("redirect")),
                                 key=lambda child: child.start_byte))
        command_span = [original.start_byte, max([original.end_byte,
                                                 *(child.end_byte for child in redirects)])]
        command_text = raw[command_span[0]:command_span[1]].decode("utf-8")
        redirect_views: list[dict[str, Any]] = []
        redirect_starts = {child.start_byte for child in redirects}
        for redirect in redirects:
            if redirect.type != "file_redirect":
                redirect_views.append({**opaque(redirect), "kind": "unsupported"})
                continue
            destinations = list(redirect.children_by_field_name("destination"))
            descriptor = redirect.child_by_field_name("descriptor")
            operators = [child.type for child in redirect.children if not child.is_named]
            operator = operators[0] if len(operators) == 1 else None
            destination = literal(destinations[0]) if destinations else None
            if (operator in {">&", "<&"} and destination is not None
                    and re.fullmatch(r"(?:[0-9]+-?|-)", destination)):
                kind = "fd_duplication"
            elif operator in {">", ">>", ">|", "&>", "&>>", "<>", ">&"}:
                kind = "file_write"
            elif operator == "<":
                kind = "file_read"
            else:
                kind = "unsupported"
            redirect_views.append({"span": span(redirect), "text": text(redirect),
                                   "operator": operator, "kind": kind,
                                   "descriptor": text(descriptor) if descriptor else None,
                                   "destination": destination,
                                   "evaluated_expressions": sorted({
                                       value for child in destinations
                                       for value in opaque(child)["evaluated_expressions"]}),
                                   })
            # Bash allows argv after a redirect. The grammar keeps those words
            # as subsequent destination fields; the first is the actual target.
            arguments.extend(child for child in destinations[1:]
                             if not (text(child).isdecimal() and child.end_byte in redirect_starts))
        arguments.sort(key=lambda child: child.start_byte)
        words = ([name] if name is not None else []) + arguments
        values = [literal(child) for child in words]
        uncertain_boundaries: set[int] = set()
        for index in range(1, len(words)):
            between = raw[words[index - 1].end_byte:words[index].start_byte]
            if not between.replace(b"\\\n", b""):
                # The grammar can separate words around a line continuation
                # even though Bash removes it before tokenization. Do not
                # invent argv or a replacement filename by joining them here.
                uncertain_boundaries.update({index - 1, index})
        for index in uncertain_boundaries:
            values[index] = None
        unknown = [opaque(child) for child, value in zip(words, values) if value is None]
        if uncertain_boundaries:
            gaps.append({"code": "unsupported_shell_syntax", "reason": "unestablished_word_boundary",
                         "span": command_span})
        item = {"span": command_span, "text": command_text, "conditional": conditional,
                "relation": relation, "argv": values, "opaque_words": unknown,
                "assignments": assignments, "redirections": redirect_views,
                "node_type": node.type}
        commands.append(item)
        if unknown:
            gaps.append({"code": "unsupported_shell_syntax", "reason": "opaque_word",
                         "span": command_span, "words": unknown})
        for redirect in redirect_views:
            if redirect["kind"] != "fd_duplication":
                gaps.append({"code": "shell_redirection", **redirect})

    def unsupported(node: Node, *, conditional: bool, relation: str) -> None:
        commands.append({"span": span(node), "text": text(node), "conditional": conditional,
                         "relation": relation, "argv": [], "node_type": node.type,
                         "status": "unsupported_shell_structure"})
        gaps.append({"code": "unsupported_shell_syntax", "node_type": node.type,
                     "span": span(node)})

    def linear(node: Node, *, conditional: bool = False, relation: str = "sequence",
               redirects: tuple[Node, ...] = ()) -> None:
        if node.type == "redirected_statement":
            body = node.child_by_field_name("body")
            if body is not None and body.type == "list":
                # tree-sitter-bash wraps a linear &&/|| list for redirections
                # written after its last command. Bash binds those redirects
                # only to that last command, not to the preceding list.
                attached = tuple(child for child in node.named_children if "redirect" in child.type)
                linear(body, conditional=conditional, relation=relation,
                       redirects=(*redirects, *attached))
                return
        if node.type in {"program", "list"}:
            pending = relation
            next_conditional = conditional
            children = node.children
            last_statement = max((index for index, child in enumerate(children)
                                  if child.is_named and child.type != "comment"), default=-1)
            for index, child in enumerate(children):
                if child.type in {"&&", "||"}:
                    pending, next_conditional = child.type, True
                elif child.type == ";":
                    pending, next_conditional = "sequence", conditional
                elif child.type == "comment" or not child.is_named:
                    continue
                elif index + 1 < len(children) and children[index + 1].type == "&":
                    unsupported(child, conditional=next_conditional, relation="background")
                else:
                    linear(child, conditional=next_conditional, relation=pending,
                           redirects=redirects if index == last_statement else ())
                    pending, next_conditional = "sequence", conditional
        else:
            view(node, conditional=conditional, relation=relation, redirects=redirects)

    linear(root)
    return BashStructure(commands, gaps)
