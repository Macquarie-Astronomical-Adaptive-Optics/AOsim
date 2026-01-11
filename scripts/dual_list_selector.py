from PySide6.QtCore import Qt, Signal, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QListWidget,
)

class DualListSelector(QWidget):
    activeChanged = Signal(list)
    def __init__(self, available=None, active=None, text_key=None, parent=None):
        super().__init__(parent)

        available = available or []
        active = active or []

        self.text_key = text_key  # can be a dict key, attribute, or function

        # Lists
        self.available_list = QListWidget()
        self.active_list = QListWidget()

        self.available_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.active_list.setSelectionMode(QListWidget.ExtendedSelection)

        # Buttons
        self.btn_add = QPushButton("→")
        self.btn_remove = QPushButton("←")
        self.btn_add.clicked.connect(self.move_to_active)
        self.btn_remove.clicked.connect(self.move_to_available)

        # Layouts
        button_layout = QVBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.btn_add)
        button_layout.addWidget(self.btn_remove)
        button_layout.addStretch()

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Available"))
        left_layout.addWidget(self.available_list)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Active"))
        right_layout.addWidget(self.active_list)

        main_layout = QHBoxLayout(self)
        main_layout.addLayout(left_layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(right_layout)

        # Populate lists
        for obj in available:
            self._add_item(self.available_list, obj)
        for obj in active:
            self._add_item(self.active_list, obj)

        # Optional: double-click moves
        self.available_list.itemDoubleClicked.connect(lambda _: self.move_to_active())
        self.active_list.itemDoubleClicked.connect(lambda _: self.move_to_available())

    # --- Internal helpers ---
    def _get_text(self, obj):
        """Determine display text for an object."""
        if self.text_key is None:
            return str(obj)
        elif callable(self.text_key):
            return str(self.text_key(obj))
        elif isinstance(obj, dict):
            return str(obj.get(self.text_key, str(obj)))
        else:
            return str(getattr(obj, self.text_key, str(obj)))

    def _add_item(self, list_widget, obj):
        item = QListWidgetItem(self._get_text(obj))
        item.setData(Qt.UserRole, obj)
        list_widget.addItem(item)

    def remove_item(self, obj):
        removed_from_active = False

        for lst in (self.available_list, self.active_list):
            for i in reversed(range(lst.count())):
                item = lst.item(i)
                if item.data(Qt.UserRole) is obj:
                    lst.takeItem(i)
                    if lst is self.active_list:
                        removed_from_active = True

        if removed_from_active:
            self.activeChanged.emit(self.active_items())

    # --- Moving items ---
    def move_to_active(self):
        for item in self.available_list.selectedItems():
            obj = item.data(Qt.UserRole)
            self._add_item(self.active_list, obj)
            self.available_list.takeItem(self.available_list.row(item))

        self.activeChanged.emit(self.active_items())

    def move_to_available(self):
        for item in self.active_list.selectedItems():
            obj = item.data(Qt.UserRole)
            self._add_item(self.available_list, obj)
            self.active_list.takeItem(self.active_list.row(item))

        self.activeChanged.emit(self.active_items())

    # --- Accessor methods ---
    def active_items(self):
        return [self.active_list.item(i).data(Qt.UserRole) for i in range(self.active_list.count())]

    def available_items(self):
        return [self.available_list.item(i).data(Qt.UserRole) for i in range(self.available_list.count())]




