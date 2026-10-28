"""Minimal POSIX line entry with immediate Esc, using the standard library.

Non-terminal input retains Python's normal line input. Terminal attributes are
restored after every line, EOF or error. This owns only line editing; control
requests are still published and consumed by the normal Run mechanism.
"""

from __future__ import annotations

import codecs
import os
import select
import sys


def terminal_input(prompt: str = "", *, stream=None) -> str:
    if os.name != "posix" or not sys.stdin.isatty():
        return input(prompt)
    import termios
    import tty

    stream = stream or sys.stdout
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")("replace")
    text: list[str] = []
    cursor = 0
    history: list[str] = []
    try:
        import readline
        history = [readline.get_history_item(i) or "" for i in range(1, readline.get_current_history_length() + 1)]
    except ImportError:
        readline = None
    history_position = len(history)
    draft = ""

    def redraw() -> None:
        stream.write("\r\x1b[2K" + prompt + "".join(text))
        if len(text) > cursor:
            stream.write(f"\x1b[{len(text) - cursor}D")
        stream.flush()

    try:
        tty.setcbreak(descriptor)
        stream.write(prompt)
        stream.flush()
        while True:
            raw = os.read(descriptor, 1)
            if not raw:
                raise EOFError
            if raw in {b"\r", b"\n"}:
                stream.write("\n")
                stream.flush()
                result = "".join(text)
                if readline is not None and result:
                    readline.add_history(result)
                return result
            if raw == b"\x04":
                if not text:
                    raise EOFError
                if cursor < len(text):
                    del text[cursor]
            elif raw in {b"\x7f", b"\x08"}:
                if cursor:
                    cursor -= 1
                    del text[cursor]
            elif raw == b"\x01":
                cursor = 0
            elif raw == b"\x05":
                cursor = len(text)
            elif raw == b"\x15":
                del text[:cursor]
                cursor = 0
            elif raw == b"\x0b":
                del text[cursor:]
            elif raw == b"\x1b":
                if not select.select([descriptor], [], [], 0.1)[0]:
                    stream.write("\n")
                    stream.flush()
                    return "/interrupt"
                sequence = os.read(descriptor, 1)
                if sequence in {b"[", b"O"}:
                    while select.select([descriptor], [], [], 0.1)[0]:
                        following = os.read(descriptor, 1)
                        sequence += following
                        if following.isalpha() or following == b"~":
                            break
                if sequence in {b"[D", b"OD"}:
                    cursor = max(0, cursor - 1)
                elif sequence in {b"[C", b"OC"}:
                    cursor = min(len(text), cursor + 1)
                elif sequence in {b"[H", b"OH", b"[1~"}:
                    cursor = 0
                elif sequence in {b"[F", b"OF", b"[4~"}:
                    cursor = len(text)
                elif sequence == b"[3~" and cursor < len(text):
                    del text[cursor]
                elif sequence in {b"[A", b"[B"}:
                    if history_position == len(history):
                        draft = "".join(text)
                    history_position = max(0, min(len(history), history_position + (-1 if sequence == b"[A" else 1)))
                    text = list(draft if history_position == len(history) else history[history_position])
                    cursor = len(text)
            else:
                decoded = decoder.decode(raw)
                for character in decoded:
                    if character.isprintable():
                        text.insert(cursor, character)
                        cursor += 1
            redraw()
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)
