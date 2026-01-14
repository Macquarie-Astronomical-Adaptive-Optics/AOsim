from PySide6.QtWidgets import QLabel
from PySide6.QtCore import QTimer, Slot

class DotLoadingLabel(QLabel):
    def __init__(self, base_text="Running", interval_ms=100, parent=None):
        super().__init__(parent)
        self.base_text = base_text
        self.symbols = ['◜','◝', '◞', "◟"]
        self._dots = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(interval_ms)

        self._tick()

    @Slot(str)
    def setBaseText(self, text):
        self.base_text = text

    @Slot(bool)
    def run(self, running):
        if running:
            self.start()
        else:
            self.stop()

    def _tick(self):
        self._dots = (self._dots + 1) % 4
        self.setText(f"{self.base_text} {self.symbols[self._dots]}")

    def start(self):
        self._timer.start()

    def stop(self, final_text=None):
        self._timer.stop()
        if final_text is not None:
            self.setText(final_text)
        else:
            self.setText(self.base_text)
