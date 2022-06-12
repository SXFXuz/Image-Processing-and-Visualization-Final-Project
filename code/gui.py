from PyQt5.QtGui import QImage, QPixmap, QPicture, QBrush, QPen, QPainter, QPolygon, QMouseEvent, QPaintEvent, QKeySequence
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton, QMainWindow, QAction, \
    QSizePolicy, QShortcut
from PyQt5.QtCore import QPoint, Qt, QRect, QPointF, QRectF, QThread, pyqtSignal, QObject
import sys
import cv2
from plot_utils import select_region, mask_img_from_phi
from cv import cv_evolution
from gac import create_g, gac_evolution
from gc_gui import GCGUI
import numpy as np


class ChanVeseThread(QThread):
    progress = pyqtSignal(object)
    msg = pyqtSignal(str)

    def __init__(self, img, parent=None):
        super(ChanVeseThread, self).__init__(parent=parent)
        self.img = img

    def run(self):
        # Parameters
        iteration = 800
        it = 0
        step = 1
        mu = 0.001 * (255 ** 2)
        lambda1 = 1  # outside uniformity
        lambda2 = 1  # inside uniformity
        delta_t = 0.0001
        v = 0
        self.msg.emit(f"Press 'R' to select rectangle.\n Press 'C' to select circle.\n "
                      f"Press 'Q' to finish.")
        phi = select_region(self.img)
        img_grey = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        while it < iteration:
            phi = cv_evolution(phi, img_grey, mu, v, lambda1, lambda2, delta_t, step)
            it += step
            self.progress.emit(mask_img_from_phi(self.img, phi))
            self.msg.emit(f'Iteration {it} of Chan-Vese Method')


class GACThread(QThread):
    progress = pyqtSignal(object)
    msg = pyqtSignal(str)

    def __init__(self, img, parent=None):
        super(GACThread, self).__init__(parent=parent)
        self.img = img

    def run(self):
        # Parameters
        max_iter = 300
        it = 0
        delta_t = 5

        self.msg.emit(f"Press 'R' to select rectangle.\n Press 'C' to select circle.\n "
                      f"Press 'Q' to finish.")
        phi = select_region(self.img)
        img_grey = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        g = create_g(img_grey)
        gy, gx = np.gradient(g)

        while it <= max_iter:
            phi = gac_evolution(phi, g, gx, gy, delta_t)
            it += 1
            self.progress.emit(mask_img_from_phi(self.img, phi))
            self.msg.emit(f'Iteration {it} of Geodesic Active Contour')


class MainGUI(QMainWindow):
    WIN_WIDTH, WIN_HEIGHT = 1500, 700
    IMG_WIDTH, IMG_HEIGHT = 600, 400

    def __init__(self):
        super(MainGUI, self).__init__()

        self.img = None
        self.qImg = None
        self.sImg = None
        self.top_pad, self.bottom_pad, self.left_pad, self.right_pad = None, None, None, None

        self.setGeometry(200, 200, self.WIN_WIDTH, self.WIN_HEIGHT)
        self.setWindowTitle("demo")

        self.background = QLabel(self)
        self.background.setGeometry(QRect(0, 20, self.WIN_WIDTH, self.WIN_HEIGHT))
        bg_img = cv2.imread("./background.jpg")
        bg_img = cv2.resize(bg_img, (self.WIN_WIDTH, self.WIN_HEIGHT), cv2.INTER_LINEAR)
        qImg = QImage(bg_img.data, self.WIN_WIDTH, self.WIN_HEIGHT, self.WIN_WIDTH * 3, QImage.Format_RGB888).rgbSwapped()
        self.background.setPixmap(QPixmap.fromImage(qImg))

        self.label = QLabel(self)
        self.label.setText("Load image from file")
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setGeometry(
            QRect((self.WIN_WIDTH - self.IMG_WIDTH) // 2, (self.WIN_HEIGHT - self.IMG_HEIGHT) // 2 + 30, self.IMG_WIDTH,
                  self.IMG_HEIGHT))
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 1px solid black; background-color: white;")

        self.msg = QLabel(self)
        self.msg.setAlignment(Qt.AlignCenter)
        self.msg.setText("Welcome to the demo!")
        self.msg.setGeometry(
            QRect((self.WIN_WIDTH - self.IMG_WIDTH) // 2 - 100, (self.WIN_HEIGHT - self.IMG_HEIGHT) // 2 - 120,
                  self.IMG_WIDTH + 200,
                  160))
        self.msg.setStyleSheet(
            "border: 1px solid black; background-color: white; font-size: 15pt; font-family: Courier;")

        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('File')
        openAction = QAction('Load Image', self)
        openAction.triggered.connect(self.openImage)
        openAction.setShortcut(QKeySequence("Ctrl+O"))
        fileMenu.addAction(openAction)

        saveAction = QAction('Save', self)
        saveAction.triggered.connect(self.saveImage)
        saveAction.setShortcut(QKeySequence("Ctrl+S"))
        fileMenu.addAction(saveAction)

        closeAction = QAction('Exit', self)
        closeAction.triggered.connect(self.close)
        closeAction.setShortcut(QKeySequence("Ctrl+E"))
        fileMenu.addAction(closeAction)

        self.buttonGC = QPushButton(self)
        self.buttonGC.setText("Graph Cut")
        self.buttonGC.move(self.WIN_WIDTH // 2 - 200, self.WIN_HEIGHT - 100)
        self.buttonGC.clicked.connect(self.doGC)
        self.buttonGC.setGeometry(self.WIN_WIDTH // 2 - 300, self.WIN_HEIGHT - 100, 200, 40)

        self.buttonCV = QPushButton(self)
        self.buttonCV.setText("Chan-Vese Method")
        self.buttonCV.clicked.connect(self.doCV)
        self.buttonCV.setGeometry(self.WIN_WIDTH // 2 - 100, self.WIN_HEIGHT - 100, 200, 40)

        self.buttonGAC = QPushButton(self)
        self.buttonGAC.setText("Geodesic Active Contour")
        self.buttonGAC.clicked.connect(self.doGAC)
        self.buttonGAC.setGeometry(self.WIN_WIDTH // 2 + 100, self.WIN_HEIGHT - 100, 200, 40)

        self.show()

    def fitImage(self, img):
        h, w, c = img.shape
        ratio = min(self.IMG_HEIGHT / h, self.IMG_WIDTH / w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)), cv2.INTER_LINEAR)
        return img

    def openImage(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        self.msg.setText("Welcome to the demo!")

        if filename == '':
            return
        self.img = cv2.imread(filename, -1)

        if self.img.size == 1:
            return

        if self.img.ndim == 2:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        self.img = self.fitImage(self.img)
        self.showImage(self.img)

    def showImage(self, img):
        h, w, c = img.shape
        bytesPerline = 3 * w
        self.sImg = img
        self.qImg = QImage(img.data, w, h, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))

    def saveImage(self):
        if self.sImg is None:
            self.msg.setText('Load an image first!')
            return
        cv2.imwrite('./result.jpg', self.sImg)
        self.msg.setText('Result saved to result.jpg.')

    def doGC(self):
        if self.img is None:
            self.msg.setText('Load an image first!')
            return

        self.msg.setText(f"MouseL for foreground, MouseR for background.\n"
                         f"Press 'C' to clear samples.\n Press 'M' to show masked image.\n "
                         f"Press 'S' to show the sampled image.\n Press 'Q' to finish.")
        gui = GCGUI(self.img)
        self.showImage(gui.run())
        self.msg.setText('Press Ctrl+S to save result.')

    def doCV(self):
        if self.img is None:
            self.msg.setText('Load an image first!')
            return

        self.buttonCV.setEnabled(False)
        cv = ChanVeseThread(self.img, parent=self)
        cv.progress.connect(self.showImage)
        cv.msg.connect(self.msg.setText)
        cv.start()
        cv.finished.connect(
            lambda: self.buttonCV.setEnabled(True)
        )
        self.msg.setText('Press Ctrl+S to save result.')

    def doGAC(self):
        if self.img is None:
            self.msg.setText('Load an image first!')
            return

        self.buttonGAC.setEnabled(False)
        gac = GACThread(self.img, parent=self)
        gac.progress.connect(self.showImage)
        gac.msg.connect(self.msg.setText)
        gac.start()
        gac.finished.connect(
            lambda: self.buttonGAC.setEnabled(True)
        )
        self.msg.setText('Press Ctrl+S to save result.')


def window():
    app = QApplication(sys.argv)
    win = MainGUI()
    sys.exit(app.exec_())


if __name__ == "__main__":
    window()
