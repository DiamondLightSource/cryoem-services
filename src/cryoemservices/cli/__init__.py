from __future__ import annotations

import argparse
import re
import textwrap


class LineWrapHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """
    A helper class for formatting the help messages the CLIs nicely. This implementation
    will preserve indents at the start of a line and interpret newline metacharacters
    accordingly.

    Credits: https://stackoverflow.com/a/35925919
    """

    def _add_whitespace(self, idx, wspace_idx, text):
        if idx == 0:
            return text
        return (" " * wspace_idx) + text

    def _split_lines(self, text, width):
        text_rows = text.splitlines()
        for idx, line in enumerate(text_rows):
            search = re.search(r"\s*[0-9\-]{0,}\.?\s*", line)
            if line.strip() == "":
                text_rows[idx] = " "
            elif search:
                wspace_line = search.end()
                lines = [
                    self._add_whitespace(i, wspace_line, x)
                    for i, x in enumerate(textwrap.wrap(line, width))
                ]
                text_rows[idx] = lines
        return [item for sublist in text_rows for item in sublist]
