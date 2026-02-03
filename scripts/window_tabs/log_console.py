import sys, io, html, logging, threading, traceback
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QMainWindow, QDockWidget, QTextEdit

from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFontDatabase, QFontMetrics

class ConsoleDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Console", parent)
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.document().setMaximumBlockCount(5000)
        self.setWidget(self.view)

        # monospace + nicer tab width
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        self.view.setFont(font)
        self.view.setTabStopDistance(QFontMetrics(font).horizontalAdvance(" ") * 4)

    def append_text(self, text: str, color: str):
        cursor = self.view.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))

        # insert the text exactly as-is (incl. indentation/newlines)
        cursor.insertText(text, fmt)
        if not text.endswith("\n"):
            cursor.insertText("\n", fmt)

        self.view.setTextCursor(cursor)
        self.view.ensureCursorVisible()

class ConsoleDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Console", parent)
        self.view = QTextEdit()
        self.view.setReadOnly(True)
        self.view.document().setMaximumBlockCount(5000)
        self.setWidget(self.view)
        self.setMinimumHeight(180)  # prevents “collapsed to nothing” look

    def append_html(self, line_html: str):
        self.view.moveCursor(QTextCursor.End)
        self.view.insertHtml(line_html + "<br/>")
        self.view.moveCursor(QTextCursor.End)


class LogBridge(QObject):
    message = Signal(int, str)  # levelno, formatted text


class QtSignalLogHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record):
        msg = self.format(record)
        self.bridge.message.emit(record.levelno, msg)


class StreamToLogger(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int):
        super().__init__()
        self.logger = logger
        self.level = level
        self._buf = ""

    def write(self, s: str):
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.logger.log(self.level, line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.logger.log(self.level, self._buf)
        self._buf = ""


def add_console(main_window: QMainWindow) -> logging.Logger:
    console = ConsoleDock(main_window)
    main_window.addDockWidget(Qt.RightDockWidgetArea, console)
    console.show()

    bridge = LogBridge(main_window)  # parented => won’t get GC’d
    handler = QtSignalLogHandler(bridge)

    logger = logging.getLogger("app")
    logger.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(threadName)-10s | %(message)s",
        "%H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.propagate = False

    # thread-safe UI update
    def on_message(levelno: int, text: str):
        color = (
            "#ff4dff" if levelno >= logging.CRITICAL else
            "#ff4d4d" if levelno >= logging.ERROR else
            "#ffa500" if levelno >= logging.WARNING else
            "#e6e6e6" if levelno >= logging.INFO else
            "#9aa0a6"
        )
        safe = html.escape(text)
        console.append_html(
            f'<span style="white-space: pre; color:{color}; font-family:monospace;">{safe}</span>')

    bridge.message.connect(on_message, Qt.QueuedConnection)

    # capture print() and stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # uncaught exceptions
    def excepthook(exc_type, exc, tb):
        logger.critical("Uncaught exception:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
    sys.excepthook = excepthook

    def thread_excepthook(args):
        logger.critical(
            "Uncaught thread exception (%s):\n%s",
            args.thread.name,
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )
    threading.excepthook = thread_excepthook

    # keep refs on the window (extra-safe)
    main_window._console_dock = console
    main_window._log_bridge = bridge
    main_window._log_handler = handler

    return logger

