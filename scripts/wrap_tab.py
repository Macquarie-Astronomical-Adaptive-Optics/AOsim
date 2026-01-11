from PySide6.QtCore import Qt, QSize, QRect, QPoint, Signal, QMargins, Slot
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTabWidget, QTabBar,
    QLayout, QSizePolicy, QStackedWidget, QScrollArea
)
from PySide6.QtGui import QPalette
from PySide6.QtGui import QFontMetrics
from scripts.popout_window import PopoutWindow

# modified code found online
# allows Widgets to wrap if there is not enough space horizontally
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)

        if parent is not None:
            self.setContentsMargins(QMargins(margin, margin, margin, margin))
        if spacing >= 0:
            self.setSpacing(spacing)

        self._item_list = []

    def __del__(self):
        while self.count():
            item = self.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.setParent(None)

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())

        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(),
                      margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only):
        if self.parentWidget() and isinstance(self.parentWidget(), QScrollArea):
            rect = self.parentWidget().parentWidget().viewport().rect()

        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._item_list:
            widget = item.widget()
            if widget is None:
                continue

            style = widget.style()
            layout_spacing_x = style.layoutSpacing(
                QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton,
                Qt.Orientation.Horizontal
            )
            layout_spacing_y = style.layoutSpacing(
                QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton,
                Qt.Orientation.Vertical
            )
            space_x = spacing + layout_spacing_x
            space_y = spacing + layout_spacing_y
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y()

class DetachableTabBar(QTabBar):
    """TabBar that emits a signal when a tab should detach."""
    detach_requested = Signal(int)

    def mouseDoubleClickEvent(self, event):
        idx = self.tabAt(event.pos())
        if idx != -1:
            self.detach_requested.emit(idx)
        else:
            super().mouseDoubleClickEvent(event)

class DetachableTabWidget(QTabWidget):
    """QTabWidget that supports popping out tabs into windows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        bar = DetachableTabBar(self)
        bar.detach_requested.connect(self.detach_tab)
        self.setTabBar(bar)
        self._popouts = {}  # keep track of popped out widgets

    @Slot(int)
    def detach_tab(self, index: int):
        if index < 0 or index >= self.count():
            return

        widget = self.widget(index)
        title = self.tabText(index)

        # remove tab but keep the widget alive
        self.removeTab(index)

        def on_close_callback(returned_widget, returned_title):
            # reattach widget to tab widget
            self.addTab(returned_widget, returned_title)
            self.setCurrentWidget(returned_widget)
            # remove from popout tracking
            self._popouts.pop(returned_widget, None)

        # create popout window
        pop = PopoutWindow(widget, title, on_close_callback)
        self._popouts[widget] = pop
        pop.show()

    def hideEvent(self, event):
        for w in list(self._popouts.values()):
               w.close()
        return super().hideEvent(event)
