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



# Widget denoting each tab in the tab bar
class TabItemWidget(QWidget):
    clicked = Signal()
    closed = Signal()
    popped = Signal()

    def __init__(self, text: str, normal_style, active_style, closable=True):
        super().__init__()
        self.closable = closable

        self._text = text
        self.label = QLabel(text)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)
        layout.addWidget(self.label)

        self._normal_style = normal_style
        self._active_style = active_style
        self.setStyleSheet(self._normal_style)

    def set_active(self, active: bool):
        self.setStyleSheet(self._active_style if active else self._normal_style)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()
        elif (self.closable and e.button() == Qt.RightButton):
            self.closed.emit() 

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.popped.emit()
        

    def sizeHint(self):
        fm = QFontMetrics(self.label.font())
        txtw = fm.horizontalAdvance(self._text)
        width = txtw
        height = max(fm.height() + 12, 28)
        return QSize(width, height)


# ---------- Container for tabs using FlowLayout ----------
class TabBarContainer(QWidget):
    def __init__(self):
        super().__init__()
        self.flow = FlowLayout(self, spacing=0)
        self.setLayout(self.flow)
        self.setStyleSheet("background: transparent;")

    def add_tab_widget(self, widget):
        self.flow.addWidget(widget)
        self.update_height()

    def remove_tab_widget(self, widget):
        self.flow.removeWidget(widget)
        widget.setParent(None)
        self.update_height()

    def update_height(self):
        if self.width() > 0:
            self.setFixedHeight(self.flow.heightForWidth(self.width()))
        
        self.flow.invalidate()
        self.flow.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_height()

    def showEvent(self, event):
        super().showEvent(event)
        self.update_height()


# ---------- WrappedTabs main widget ----------
class WrappedTabs(QWidget):
    tabKilled = Signal(QWidget)

    def __init__(self):
        super().__init__()
        self.stack = QStackedWidget()
        self.tab_bar = TabBarContainer()
        self._popouts = {}
        self._titles = {}

        self.tab_scroll = QScrollArea()
        self.tab_scroll.setWidgetResizable(True)
        self.tab_scroll.setWidget(self.tab_bar)
        self.tab_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tab_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tab_scroll.setFrameShape(QScrollArea.NoFrame)
        self.tab_scroll.setMaximumHeight(60)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self.tab_scroll)
        layout.addWidget(self.stack, 1)

        self._tabs = []

        tab_widget = QTabWidget()

        # Grab palette colors
        pal = tab_widget.palette()
        bg_normal = pal.color(QPalette.Button).name()        # tab background
        bg_active = pal.color(QPalette.Highlight).name()     # selected tab
        border_normal = pal.color(QPalette.Dark).name()      # border for inactive tab
        border_active = pal.color(QPalette.Highlight).name() # border for active tab
        text_color = pal.color(QPalette.ButtonText).name()   # tab text color

        # Build stylesheet using palette colors
        self.normal_style = f"""
        QWidget {{
            background: {bg_normal};
            border: 1px solid {border_normal};
            border-radius: 2px;
            padding-left: 8px;          
            padding-right: 8px; 
            color: {text_color};
        }}
        """

        self.active_style = f"""
        QWidget {{
            background: {bg_active};
            border: 1px solid {border_active};
            border-radius: 2px;
            padding-left: 8px;          
            padding-right: 8px; 
            color: {text_color};
        }}
        """

    def add_tab(self, page_widget: QWidget, closable=True):
        title = page_widget.title

        tab = TabItemWidget(title, self.normal_style, self.active_style, closable)
        self.tab_bar.add_tab_widget(tab)
        self.stack.addWidget(page_widget)
        self._tabs.append((tab, page_widget))
        self._titles[page_widget] = title

        self.tab_scroll.ensureWidgetVisible(tab)

        tab.clicked.connect(lambda: self.set_current(tab))
        tab.closed.connect(lambda: self.close_tab(tab))
        tab.popped.connect(lambda: self.pop_tab(tab, page_widget))

        self.set_current(tab)
        self._refresh()

    def set_current(self, tab):
        for t, w in self._tabs:
            t.set_active(t is tab)
            if hasattr(w, "displayed"):
                setattr(w, "displayed", t is tab)
            if t is tab:
                self.stack.setCurrentWidget(w)

        self._refresh()

    def close_tab(self, tab, delete_page=True):
        for t, w in self._tabs:
            if t is tab:
                self.tab_bar.remove_tab_widget(t)
                self.stack.removeWidget(w)
                if delete_page:
                    w.deleteLater()   # only delete if not popping out
                    self.tabKilled.emit(w)
                
                    if hasattr(w, "kill"):
                        w.kill()
                    if hasattr(w, "cleanup"):
                        w.cleanup()
                
                self._titles.pop(w, None)
                self._tabs.remove((t, w))
                break
        
        if self._tabs:
            self.set_current(self._tabs[0][0])
        
        self._refresh()


    def pop_tab(self, tab, page_widget):
        for t, w in self._tabs:
            if t is tab:
                self.close_tab(t, delete_page=False)  # keep the widget alive
                title = self._titles.get(w, tab.label.text())
                
                pop = PopoutWindow(page_widget, title, self._reattach)
                self._popouts[w] = pop
                pop.show()
                break

    def _reattach(self, widget, title):
        self._popouts.pop(widget, None)
        widget.setParent(None)
        self.add_tab(widget, closable=True)
        
    def _refresh(self):
        self.tab_bar.update()
        self.tab_bar.repaint()
        self.tab_scroll.updateGeometry()
        self.tab_scroll.viewport().updateGeometry()
                

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
