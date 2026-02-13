import json
from typing import Any, Dict, Iterable, List, MutableMapping, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QPushButton,
    QFileDialog, QMessageBox,
)

from data.CONFIG_DTYPES import CONFIG_DTYPES

try:
    # Optional import (keeps widget usable without the central store)
    from scripts.core.config_store import ConfigStore
except Exception:  # pragma: no cover
    ConfigStore = None  # type: ignore


class Config_table(QWidget):
    """
    Simple key/value editor for a dict-like config section.

    - Keys are fixed to `section_key` order.
    - Values are edited as text but converted using CONFIG_DTYPES:
        * bool: accepts true/false/1/0/yes/no
        * list/tuple/dict: accepts JSON (e.g. [1,2], {"a": 1})
        * other types: dtype(value_str)
      If conversion fails, stores the raw string.
    """
    params_changed = Signal(dict)

    def __init__(
        self,
        section_key: Iterable[str],
        config_dict: MutableMapping[str, Any],
        parent=None,
        *,
        config_store: Optional["ConfigStore"] = None,
    ):
        super().__init__(parent)

        self.row_keys: List[str] = list(section_key)
        self.config: MutableMapping[str, Any] = config_dict
        self._store = config_store

        self._loading = False  # guard to ignore itemChanged during programmatic updates
        self._header_map: Dict[str, str] = {}  # readable -> key

        # ---- UI ----
        outer = QVBoxLayout(self)

        self.table = QTableWidget(len(self.row_keys), 2, self)
        self.table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.table.setWordWrap(True)
        self.table.itemChanged.connect(self._on_item_changed)
        outer.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("Save" if self._store is None else "Save All")
        self.btn_load = QPushButton("Load" if self._store is None else "Load All")
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_load)
        outer.addLayout(btn_row)

        self.btn_save.clicked.connect(self.save_file)
        self.btn_load.clicked.connect(self.open_file)

        # If we have a central store, refresh when it changes.
        if self._store is not None:
            try:
                self._store.changed.connect(self.refresh)
            except Exception:
                pass

        self._populate_table()

    def refresh(self) -> None:
        """Refresh the table from the underlying mapping."""
        self._populate_table()

    # -------------------------
    # Table population
    # -------------------------
    def _populate_table(self):
        """Fill/refresh table from self.config for the configured keys."""
        self._loading = True
        self.table.blockSignals(True)

        self._header_map.clear()

        for row, key in enumerate(self.row_keys):
            readable = self._readable_key(key)
            self._header_map[readable] = key

            key_item = QTableWidgetItem(readable)
            key_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)

            val = self.config.get(key, "")
            val_item = QTableWidgetItem(self._format_value(val))

            self.table.setItem(row, 0, key_item)
            self.table.setItem(row, 1, val_item)

        self.table.resizeRowsToContents()

        self.table.blockSignals(False)
        self._loading = False

    @staticmethod
    def _readable_key(key: str) -> str:
        return " ".join(str(key).split("_")).title()

    @staticmethod
    def _format_value(val: Any) -> str:
        # Prefer JSON formatting for containers so users can round-trip edit.
        if isinstance(val, (list, tuple, dict)):
            try:
                return json.dumps(val)
            except Exception:
                return str(val)
        return str(val)

    # -------------------------
    # Editing
    # -------------------------
    def _on_item_changed(self, item: QTableWidgetItem):
        if self._loading:
            return
        if item.column() != 1:
            return

        row = item.row()
        if not (0 <= row < len(self.row_keys)):
            return

        key = self.row_keys[row]
        value_str = item.text()

        value = self._convert_value(key, value_str)
        self.config[key] = value

        # Emit a snapshot of the edited mapping.
        # (Downstream should treat this as read-only; the source of truth is self.config)
        self.params_changed.emit(dict(self.config))

        # Central store changes are already applied in-place by virtue of shared mappings,
        # but we still emit a store-level change event for listeners that rely on it.
        if self._store is not None:
            try:
                self._store.changed.emit()
            except Exception:
                pass

    @staticmethod
    def _convert_value(key: str, value_str: str) -> Any:
        """
        Convert using CONFIG_DTYPES[key] if present.
        Adds JSON parsing for list/tuple/dict automatically.
        """
        dtype = CONFIG_DTYPES.get(key, str)

        s = value_str.strip()

        try:
            # bool
            if dtype is bool:
                return s.lower() in ("true", "1", "yes", "y", "on")

            # JSON containers (common in your turbulence config: wind=[vx,vy])
            if dtype in (list, tuple, dict):
                parsed = json.loads(s)  # expects valid JSON: [..] or {..}
                if dtype is tuple:
                    return tuple(parsed)
                if dtype is list:
                    return list(parsed)
                return dict(parsed)

            # If dtype isn't container but the user typed JSON anyway, optionally accept it.
            # (Handy when CONFIG_DTYPES is incomplete)
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try:
                    return json.loads(s)
                except Exception:
                    pass

            # numeric / other cast
            return dtype(s)
        except Exception:
            # fallback to raw string
            return value_str

    # -------------------------
    # Save / Load
    # -------------------------
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select config file", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        # Centralized: load entire config
        if self._store is not None:
            try:
                self._store.load(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Load Failed", f"Could not load JSON:\n{e}")
                return
            # Store emitted changed; just ensure this section reflects new values.
            self._populate_table()
            self.params_changed.emit(dict(self.config))
            return

        try:
            with open(file_path, "r") as f:
                loaded = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Load Failed", f"Could not load JSON:\n{e}")
            return

        # update only known keys
        for key in self.row_keys:
            if key in loaded:
                self.config[key] = loaded[key]

        self._populate_table()
        self.params_changed.emit(dict(self.config))

    def save_file(self):
        # Centralized: save entire config
        if self._store is not None:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Config", "config.json", "JSON Files (*.json)"
            )
            if not file_path:
                return
            try:
                self._store.save(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Save Failed", f"Could not write file:\n{e}")
                return
            self.params_changed.emit(dict(self.config))
            return

        # build a dict with only keys in this section (preserving row order)
        section_out = {key: self.config.get(key, "") for key in self.row_keys}

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "config.json", "JSON Files (*.json)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "w") as f:
                json.dump(section_out, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Could not write file:\n{e}")
            return

        self.params_changed.emit(dict(self.config))

    # -------------------------
    # Optional debug helper
    # -------------------------
    def show_config_dtypes(self):
        lines = []
        for k in self.row_keys:
            v = self.config.get(k, None)
            lines.append(f"{k}: {type(v).__name__}")
        QMessageBox.information(self, "Config Types", "\n".join(lines))
