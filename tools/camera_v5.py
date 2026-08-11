import time
import ctypes
import subprocess as sp
import multiprocessing as mp
import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from logger import setup_logger
import torch.nn.functional as F
from utils.plots import plot_one_box
from numpy import random
peotable_COLORMAP = [
    [70,70,70],
    [255, 255, 0]

]


def label2image(pred, COLORMAP=peotable_COLORMAP):
    colormap = np.array(COLORMAP, dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]


class Camera(object):
    def __init__(self, weights, camera_ip, rtspurl, log_path):
        self.weights = weights
        self.camera_ip = camera_ip
        self.log_path = log_path

        # Initialize
        self.device = select_device()
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # # Load model
        # self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        # self.stride = int(self.model.stride.max())  # model stride
        # if self.half:
        #     self.model.half()  # to FP16
        self.img_size = 640
        self.conf_thr = 0.3
        self.iou_thr = 0.45
        self.count = 0
        self.timeF = 10
        self.det = None

        size = (1920, 1080)
        fps = 25
        self.command = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '{}x{}'.format(size[0], size[1]),
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-rtsp_transport', 'tcp', '-f', 'rtsp', rtspurl
        ]

        mp.set_start_method(method='spawn')
        self.queue = mp.Queue(maxsize=2)
        self.signal = mp.Value(ctypes.c_bool, True)

    def read_frame(self):
        logger1 = setup_logger('camera', self.log_path, 'read_log.txt')
        cap = cv2.VideoCapture(self.camera_ip)
        while True:
            if cap is None or not cap.isOpened():
                # print('Opening camera is failed.')
                logger1.info('opening camera is failed.')
                time.sleep(5)
                cap = cv2.VideoCapture(self.camera_ip)
                continue
            try:
                success, frame = cap.read()
            except:
                logger1.error('reading frame is wrong. & restarting')
                self.signal.value = False
            else:
                if success and frame is not None:
                    self.queue.put(frame)
                    self.queue.get() if self.queue.qsize() > 1 else time.sleep(0.01)
                else:
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.camera_ip)
                    # raise RuntimeError('Video is not available.')

    def push_frame(self):
        logger2 = setup_logger('camera', self.log_path, 'push_log.txt')
        # print(os.getpid())
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        if self.half:
            model.half()  # to FP16

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # todo 初始化管道
        while True:
            if len(self.command) > 0:
                pipe = sp.Popen(self.command, stdin=sp.PIPE)
                break

        # todo 读取视频并执行检测
        while True:
            if self.queue.empty():
                time.sleep(0.1)
                continue
            try:
                frame0 = self.queue.get(timeout=10)
            except:
                # print(">> except >> frame0 = self.queue.get(timeout=10)")
                logger2.info('getting frame from queue is wrong.')
                continue
            if frame0 is None:
                # continue
                self.signal.value = False

            # todo 等效于每10s 处理一次
            if self.count % self.timeF == 1:
                # process frame
                frame = frame0.copy()
                frame = letterbox(frame, self.img_size, stride=stride)[0]
                frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                frame = np.ascontiguousarray(frame)
                frame = torch.from_numpy(frame).to(self.device)
                frame = frame.half() if self.half else frame.float()  # uint8 to fp16/32
                frame /= 255.0  # 0 - 255 to 0.0 - 1.0
                if frame.ndimension() == 3:
                    frame = frame.unsqueeze(0)

                # Run inference
                if self.device.type != 'cpu':
                    model(torch.zeros(1, 3, self.img_size, self.img_size).to(
                        self.device).type_as(next(model.parameters())))  # run once

                out = model(frame)
                pred = out[0][0]
                seg = out[1]  # [0]
                # Apply NMS
                pred = non_max_suppression(pred, self.conf_thr, self.iou_thr)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    gn = torch.tensor(frame0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], frame0.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, frame0, label=label, color=colors[int(cls)], line_thickness=3)

                    save_path = './demo/'
                    seg = F.interpolate(seg, (frame0.shape[0], frame0.shape[1]), mode='bilinear', align_corners=True)[0]
                    mask = label2image(seg.max(axis=0)[1].cpu().numpy(), peotable_COLORMAP)[:, :, ::-1]
                    dst = cv2.addWeighted(mask, 0.4, frame0, 0.6, 0)

                    cv2.imwrite(save_path + str(self.count) + '.jpg', frame0)
                    cv2.imwrite(save_path + str(self.count) + "_mask" + ".jpg" , mask)
                    cv2.imwrite(save_path + str(self.count) + "_dst" + ".jpg", dst)
            self.count += 1
            """ mine """

            # todo 管道推流
            try:
                pipe.stdin.write(dst.tostring())
            # except BrokenPipeError:
            except:
                pipe = sp.Popen(self.command, stdin=sp.PIPE)

    def run(self):
        # processes = [mp.Process(target=Camera.read_frame, args=(self,)),
        #              mp.Process(target=Camera.push_frame, args=(self,))]
        # [process.start() for process in processes]
        # [process.join() for process in processes]
        p1 = mp.Process(target=Camera.read_frame, args=(self,))
        p2 = mp.Process(target=Camera.push_frame, args=(self,))
        p1.start()
        p2.start()
        while True:
            if not self.signal.value:
                p1.terminate()
                p2.terminate()
                break
        p1.join()
        p2.join()
        # self.logger.info('restarting...')
        # raise RuntimeError('Video is not available.')


if __name__ == '__main__':
    camera = Camera(weights='./runs/train/exp3/weights/best.pt',
                    # camera_ip='rtsp://192.168.50.222:22005/www/mnt/mfs/dl_models/RTSP/cam001',
                    camera_ip='./video/1.mp4',
                    rtspurl='rtsp://192.168.50.222:22005/www1',
                    log_path='./')
    camera.run()
    assert 1 == 2
    print()
