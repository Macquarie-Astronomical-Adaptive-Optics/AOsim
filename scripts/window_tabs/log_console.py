"""Qt dock widget that captures logging + stdout/stderr.

Usage:
    from scripts.window_tabs.log_console import add_console
    logger = add_console(main_window)

`add_console`:
- adds a dock to the given main window
- wires a logging.Handler to send records to the dock
- redirects sys.stdout/sys.stderr into the logger (so print() shows up)
- installs exception hooks
"""

from __future__ import annotations

import html
import io
import logging
import sys
import threading
import traceback
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFontDatabase, QFontMetrics, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QDockWidget, QMainWindow, QTextEdit


class ConsoleDock(QDockWidget):
    def __init__(self, parent: Optional[QMainWindow] = None):
        super().__init__('Console', parent)

        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.document().setMaximumBlockCount(5000)
        self.setWidget(self.view)
        self.setMinimumHeight(180)

        # Monospace + nicer tab width
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.view.setFont(font)
        self.view.setTabStopDistance(QFontMetrics(font).horizontalAdvance(' ') * 4)

    def append_text(self, text: str, *, color: str = '#e6e6e6') -> None:
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))

        cursor.insertText(text, fmt)
        if not text.endswith('\n'):
            cursor.insertText('\n', fmt)

        self.view.setTextCursor(cursor)
        self.view.ensureCursorVisible()

    def append_html(self, line_html: str) -> None:
        self.view.moveCursor(QTextCursor.End)
        self.view.insertHtml(line_html + '<br/>')
        self.view.moveCursor(QTextCursor.End)


class LogBridge(QObject):
    message = Signal(int, str)  # levelno, formatted text


class QtSignalLogHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.bridge.message.emit(record.levelno, msg)


class StreamToLogger(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int):
        super().__init__()
        self.logger = logger
        self.level = int(level)
        self._buf = ''

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line:
                self.logger.log(self.level, line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self.logger.log(self.level, self._buf)
        self._buf = ''


def add_console(main_window: QMainWindow, *, logger_name: str = 'aosim') -> logging.Logger:
    """Attach a log console dock to `main_window`.

    Returns the configured logger instance.
    """
    console = ConsoleDock(main_window)
    main_window.addDockWidget(Qt.RightDockWidgetArea, console)
    console.show()

    bridge = LogBridge(main_window)  # parented => won't get GC'd
    handler = QtSignalLogHandler(bridge)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(threadName)-10s | %(message)s',
            '%H:%M:%S',
        )
    )
    logger.addHandler(handler)
    logger.propagate = False

    def on_message(levelno: int, text: str) -> None:
        color = (
            '#ff4dff' if levelno >= logging.CRITICAL else
            '#ff4d4d' if levelno >= logging.ERROR else
            '#ffa500' if levelno >= logging.WARNING else
            '#e6e6e6' if levelno >= logging.INFO else
            '#9aa0a6'
        )
        safe = html.escape(text)
        console.append_html(
            f'<span style="white-space: pre; color:{color}; font-family:monospace;">{safe}</span>'
        )

    bridge.message.connect(on_message, Qt.QueuedConnection)

    # Capture print() and stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # Uncaught exceptions
    def excepthook(exc_type, exc, tb):
        logger.critical('Uncaught exception:\n%s', ''.join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = excepthook

    def thread_excepthook(args):
        logger.critical(
            'Uncaught thread exception (%s):\n%s',
            args.thread.name,
            ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    threading.excepthook = thread_excepthook

    # Keep refs on the window (extra-safe)
    main_window._console_dock = console
    main_window._log_bridge = bridge
    main_window._log_handler = handler

    return logger
