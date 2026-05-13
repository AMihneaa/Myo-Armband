import pyqtgraph as pg

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QWidget,
    QPushButton,
)
from PySide6.QtGui import QFont

from recording.session import SessionRecorder, SessionCue


CUE_BACKGROUNDS= {
    "white":  "#ffffff",
    "green":  "#2ecc71",
    "orange": "#e67e22",
    "red":    "#e74c3c",
    "gray":   "#1a1a1a",
}

CUE_TEXT_COLORS= {
    "white":  "#111111",
    "green":  "#111111",
    "orange": "#111111",
    "red":    "#ffffff",
    "gray":   "#aaaaaa",
}


class CueWindow(QMainWindow):
    def __init__(
        self,
        session_recorder: SessionRecorder,
        refresh_ms: int= 33,
    ) -> None:
        super().__init__()

        self._session_recorder= session_recorder

        self.setWindowTitle("Myo Recorder — Cue")
        self.resize(800, 400)

        self._build_ui()
        self._start_timer(refresh_ms)

    def _build_ui(self) -> None:
        root= QWidget()
        layout= QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._cue_widget= QWidget()
        self._cue_widget.setStyleSheet(f"background: {CUE_BACKGROUNDS['gray']};")

        cue_layout= QVBoxLayout(self._cue_widget)
        cue_layout.setContentsMargins(32, 32, 32, 32)
        cue_layout.setSpacing(16)

        self._cue_text= QLabel("")
        text_font= QFont()
        text_font.setPointSize(36)
        text_font.setBold(True)
        self._cue_text.setFont(text_font)

        self._cue_timer= QLabel("")
        timer_font= QFont()
        timer_font.setPointSize(24)
        self._cue_timer.setFont(timer_font)

        self._exit_btn= QPushButton("Exit")
        exit_font= QFont()
        exit_font.setPointSize(18)
        self._exit_btn.setFont(exit_font)
        self._exit_btn.setFixedHeight(60)
        self._exit_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border-radius: 8px; }"
            "QPushButton:hover { background: #e74c3c; }"
        )
        self._exit_btn.clicked.connect(self._on_exit)
        self._exit_btn.setVisible(False)

        cue_layout.addStretch()
        cue_layout.addWidget(self._cue_text)
        cue_layout.addWidget(self._cue_timer)
        cue_layout.addStretch()
        cue_layout.addWidget(self._exit_btn)

        layout.addWidget(self._cue_widget)
        self.setCentralWidget(root)

    def _start_timer(self, refresh_ms: int) -> None:
        self._timer= QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(refresh_ms)

    def _update(self) -> None:
        cue= self._session_recorder.cue

        bg= CUE_BACKGROUNDS.get(cue.color, CUE_BACKGROUNDS["gray"])
        fg= CUE_TEXT_COLORS.get(cue.color, "#ffffff")

        self._cue_widget.setStyleSheet(f"background: {bg};")
        self._cue_text.setStyleSheet(f"color: {fg};")
        self._cue_text.setText(cue.text)

        if cue.timer > 0.0:
            self._cue_timer.setStyleSheet(f"color: {fg};")
            self._cue_timer.setText(f"{cue.timer:.1f}s")
        else:
            self._cue_timer.setText("")

        if cue.text == "Session complete":
            self._exit_btn.setVisible(True)
            self._timer.stop()

    def _on_exit(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.quit()