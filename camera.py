import threading
import time
import cv2


class Camera:
  def __init__(self, args):
    self.vid = cv2.VideoCapture(args.camera)

    self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, args.resolution)

    self.frame = None
    while self.frame is None:
      _, self.frame = self.vid.read()

    self.width = self.frame.shape[1]
    self.height = self.frame.shape[0]

    self.enabled = False

  def release(self):
    self.stop()
    self.vid.release()

  def start(self):
    if self.enabled:
      return
    self.enabled = True

    self.update_thread = threading.Thread(target=self.update_frame)
    self.update_thread.start()

  def stop(self):
    if not self.enabled:
      return
    self.enabled = False

    self.update_thread.join()

  def countdown_capture(self, delay_seconds):
    for i in range(delay_seconds):
      print(delay_seconds - i)
      time.sleep(1)
    return self.frame

  def update_frame(self):
    cv2.namedWindow("live_preview", cv2.WINDOW_NORMAL)
    while self.enabled:
      _, frame = self.vid.read()
      if frame is not None:
        self.frame = frame
        cv2.imshow("live_preview", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("live_preview")
