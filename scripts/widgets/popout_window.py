from PySide6.QtWidgets import (
    QMainWindow, QWidget, QWidget,
)

class PopoutWindow(QMainWindow):
    """Window that hosts a popped-out page and calls a callback to reattach on close."""
    def __init__(self, page_widget: QWidget, title: str, on_close_callback):
        super().__init__()
        self.setWindowTitle(title)
        self._page = page_widget
        self._on_close = on_close_callback

        self.setCentralWidget(self._page)
        self.resize(self.sizeHint())

        self._page.show()

    def closeEvent(self, event):
        # When the window closes, hand the page widget back to the caller
        central = self.centralWidget()
        if central:
            try:
                central.layout().removeWidget(self._page)
            except Exception:
                pass
        # ensure widget has no parent so caller can reattach
        self._page.setParent(None)

        if callable(self._on_close):
            # pass original title back
            self._on_close(self._page, self.windowTitle())
        event.accept()