from os import stat
import numpy as np
import cv2

import time
import ctypes

import subprocess as sp
import multiprocessing as mp
from numpy.core.numeric import Inf
import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from logger import setup_logger
import torch.nn.functional as F
from utils.plots import plot_one_box
from numpy import random

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, GLib


# RtspVideo用于获取Rtsp视频流
class RtspVideo(object):
    def __init__(self, address, path, port=8554):
        rtsp_location = ("rtsp://%s:%d%s") % (address, port, path)
        gst_str = ("rtspsrc latency=0 protocols=tcp location=%s" + \
                   " ! rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 " + \
                   "! nvvidconv ! videoconvert ! video/x-raw, format=(string)BGR ! appsink") % (rtsp_location)
        print("opening: ", gst_str)
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        return self.cap.read()
# CameraVideo用于获取摄像头视频
class CameraVideo(object):
    def __init__(self, deviceAddress):
        gst_str = ("v4l2src device=%s"  
            " ! image/jpeg, width=1920,height=1080,framerate=30/1,format=MJPG" + \
            " ! jpegdec" + \
            " ! nvvidconv " + \
            " ! video/x-raw(memory:NVMM) " + \
            ", format=(string)NV12 " + \
            " ! nvvidconv" + \
            " ! video/x-raw, format=(string)BGRx" + \
            " ! videoconvert ! video/x-raw, format=(string)BGR ! appsink") %(deviceAddress)
        print("opening: ", gst_str)
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    
    def isOpened(self):
        return self.cap.isOpened()
    
    def read(self):
        return self.cap.read()

# SensorFactory用于给
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, imgWidth, imgHeight, fps, imageQueue, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.number_frames = 0
        self.imgWidth = imgHeight
        self.imgHeight = imgHeight
        self.fps = fps
        self.duration = 1.0 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = 'appsrc name=source is-live=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1 ' \
                             '! videoconvert ! nvvidconv ' \
                             '! nvv4l2h264enc preset-level=UltraFastPreset maxperf-enable=1 ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' % (imgWidth, imgHeight, fps)
        self.imageQueue = imageQueue

        # self.cap = RtspVideo("172.20.10.9", "/live/test1", 8554)
        
    def on_need_data(self, src, length):
        # print("on call...")
        # ret, self.image = self.cap.read()
        image = self.imageQueue.get()
        # if self.imageQueue.qsize()< 1:
        #     self.imageQueue.put(image)
        if not image is None:
            data = image.tostring()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)
            # print('pushed buffer, frame {}, duration {} ns, durations {} s'.format(self.number_frames,
            #                                                                         self.duration,
            #                                                    self.duration / Gst.SECOND))
            if retval != Gst.FlowReturn.OK:
                print(retval)
        else:
            print("image is None...")

    def do_create_element(self, url):
        print("create element")
        parse_ret = Gst.parse_launch(self.launch_string)
        print("parse launch string: ", parse_ret)
        return  parse_ret

    def do_configure(self, rtsp_media):
        print("configure...")
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

    # def updateMedia(self, frame):
    #     self.image = frame

# RtspServer用于创建Rtsp服务器
class RtspServer(object):
    init_flag = False

    def __new__(cls,*args, **kwargs):
        if not cls.init_flag:
            GObject.threads_init()
            Gst.init(None)
        cls.init_flag = True
        return super().__new__(cls)

    def __init__(self, imgWidth, imgHeight, fps, path, port, imageQueue):
        self.server = GstRtspServer.RTSPServer.new()
        self.server.props.service = '%d' % (port)
        # self.server.set_service('%d' % port)
        self.server.attach(None)

        self.factory = SensorFactory(imgWidth, imgHeight, fps, imageQueue)
        # factory.set_launch(gst_str)
        self.factory.set_shared(True)

        self.server.get_mount_points().add_factory(path, self.factory)
        self.server.attach(None)
    
    def updateMedia(self, frame):
        self.factory.updateMedia(frame)
    
    def run(self):
        # pass
        self.loop = GObject.MainLoop()
        self.loop.run()
        

# Inferencer负责模型推理
class Inferencer(object):
    peotable_COLORMAP = [
            [70, 70, 70],
            [255, 255, 0],
            [0, 85, 255],
            [0, 170, 0],
            [204, 43, 41]
        ]

    def __init__(self, weightFile):
        self.weightFile = weightFile
        self.device = select_device()
        self.half = self.device.type != 'cpu'

        self.img_size = 640
        self.conf_thr = 0.3
        self.iou_thr = 0.45
        self.det = None

        print("loading the model")
        self.model = attempt_load(self.weightFile, map_location=self.device) # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        print("run for the first time...")
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(
                self.device).type_as(next(self.model.parameters())))  # run once
    
    @staticmethod
    def label2image(pred, COLORMAP=peotable_COLORMAP):
        colormap = np.array(COLORMAP, dtype='uint8')
        X = pred.astype('int32')
        return colormap[X, :]

    def inference(self, image):
        if image is None:
            return None
        frame = image.copy()
        frame = letterbox(frame, self.img_size, stride=self.stride)[0]
        frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        frame = np.ascontiguousarray(frame)
        frame = torch.from_numpy(frame).to(self.device)
        frame = frame.half() if self.half else frame.float()  # uint8 to fp16/32
        frame /= 255.0  # 0 - 255 to 0.0 - 1.0
        if frame.ndimension() == 3:
                    frame = frame.unsqueeze(0)

        out = self.model(frame)
        pred = out[0][0]
        seg = out[1]  # [0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thr, self.iou_thr)

        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], image.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)], line_thickness=3)
            
        seg = F.interpolate(seg, (image.shape[0], image.shape[1]), mode='bilinear', align_corners=True)[0]
        mask = Inferencer.label2image(seg.max(axis=0)[1].cpu().numpy(), Inferencer.peotable_COLORMAP)[:, :, ::-1]
        dst = cv2.addWeighted(mask, 0.4, image, 0.6, 0)

        return mask, dst

# RtspVideo类的测试用例
class TestCasesRtspVideo():
    @staticmethod
    def testRtspVideo():
        cap = RtspVideo("172.20.10.9", "/live/test1", 8554)
        writer = cv2.VideoWriter("testrtspvideo.avi", 0 , fps=30,frameSize=(1280,720))
        while(True):
            if not cap.isOpened():
                print("not opened")
                break
            ret, frame = cap.read()
            if ret<0:
                continue
            if not writer.isOpened():
                print("writer is not opened...")
                continue
            writer.write(frame)
            print("writing...")

# RtspServer类的测试用例
class TestCaseRtspServer():
    inputQuueSize = 2
    outputQueueSize = 5

    @staticmethod
    def testRtspServer():
        inputQueue = mp.Queue(maxsize=TestCaseRtspServer.inputQuueSize)
        outputQueue = mp.Queue(maxsize=TestCaseRtspServer.inputQuueSize)

        pread = mp.Process(target=TestCaseRtspServer.readThread, args=(inputQueue,), daemon=True)
        pwrite = mp.Process(target=TestCaseRtspServer.writeThread, args=(outputQueue,), daemon=True)
        pread.start()
        pwrite.start()
        while True:
            img = inputQueue.get()
            if outputQueue.qsize() > TestCaseRtspServer.outputQueueSize-1:
                outputQueue.get()
            
            outputQueue.put(img)
        pread.join()
        pwrite.join()

    @staticmethod
    def readThread(q):
        cap = RtspVideo("172.20.10.9", "/live/test1", 8554)
        while True:
            print("reading thread running ...")
            if not cap.isOpened():
                print("capture is not opened, exit...")
                # exit(-1)
                break
            # print("reading ...")
            ret, img = cap.read()
            if (not ret==0) and (not img is None):
                print("update...")
                q.put(img)
                if q.qsize() > TestCaseRtspServer.inputQuueSize-1:
                    q.get()
            else:
                print("false...")
            print("exit...")
    
    @staticmethod
    def writeThread(imageQueue):
        # cap = RtspVideo("172.20.10.9", "/live/test1", 8554)
        server = RtspServer(1280, 720, 30, "/wosuiyi/shezhide", 8553, imageQueue)
        # for i in range(100):
        #     ret, img = cap.read()
        # img = q.get()
        # server.updateMedia(img)
        server.run()

# Inferencer测试用例
class TestCaseInferencer():
    @staticmethod
    def testInferencer():
        imgFile = "./testImage.jpg"
        image = cv2.imread(imgFile, -1)
        inferencer = Inferencer("./runs/train/exp3/weights/best.pt")

        mask, dst = inferencer.inference(image)
        save_path = './demo/'
        cv2.imwrite(save_path + str("xxxx") + '.jpg', image)
        cv2.imwrite(save_path + str("xxxx") + "_mask" + ".jpg" , mask)
        cv2.imwrite(save_path + str("xxxx") + "_dst" + ".jpg", dst)


# todo 返回的是仅有检测结果的队列
class Executor(object):
    inputQuueSize = 2
    outputQueueSize = 5

    def __init__(self):
        super().__init__()
        self.inputQueue = mp.Queue(maxsize=Executor.inputQuueSize)
        self.outputQueue = mp.Queue(maxsize=Executor.outputQueueSize)
        self.cap = None
        self.pread = None
        self.pinference = None
        self.write = None

    def run(self):
        self.startStreaming()
        self.startInferencing()
        self.startwrite()

    def stop(self):
        self.stopInferencing()
        self.stopStreaming()
        self.stopwrite()

    def getOneResult(self):
        return self.outputQueue.get()

    def startStreaming(self):
        # if self.cap is None or not self.cap.isOpened():
        #    # self.cap = CameraVideo("/dev/video0")
        #     self.cap = RtspVideo("192.168.50.222", "/live/test1", 8554)
        #     # path = "/home/nvidia/Desktop/yolov5-master/video/1.mp4"
        #     # self.cap = CameraVideo(path)
        #     # print(path)
        #     # self.pread = mp.Process(target=Executor.readThread, args=(self.inputQueue, self.cap,), daemon=True)
        #     # self.pread.start()
        #
        # if not self.cap.isOpened():
        #     return False

        if self.pread is None:
            self.pread = mp.Process(target=Executor.readThread, args=(self.inputQueue, self.cap,), daemon=True)
            self.pread.start()

        if not self.pread is None:
            if not self.pread.is_alive:
                self.pread.terminate()
                self.pread = mp.Process(target=Executor.readThread, args=(self.inputQueue, self.cap,), daemon=True)
                self.pread.start()
        # return self.cap.isOpened()

    def stopStreaming(self):
        if not self.pread is None:
            self.pread.terminate()
            self.pread = None

    def startInferencing(self):
        # ctx = mp.get_context('spawn')
        ctx = mp
        if self.pinference is None:
            self.pinference = ctx.Process(target=Executor.inferenceThread,
                                          args=(self.inputQueue, self.outputQueue), daemon=True)
            self.pinference.start()

        if not self.pinference is None:
            if not self.pinference.is_alive:
                self.pinference.terminate()
                self.pinference = ctx.Process(target=Executor.inferenceThread,
                                              args=(self.inputQueue, self.outputQueue), daemon=True)
                self.pinference.start()

    def stopInferencing(self):
        if not self.pinference is None:
            self.pinference.terminate()
            self.pinference = None

    def startwrite(self):

        if self.write is None:
            self.write = mp.Process(target=Executor.writeThread, args=(self.outputQueue,), daemon=True)
            self.write.start()

        if not self.write is None:
            if not self.write.is_alive:
                self.write.terminate()
                self.write = mp.Process(target=Executor.writeThread, args=(self.outputQueue,), daemon=True)
                self.write.start()

    def stopwrite(self):
        if not self.write is None:
            self.write.terminate()
            self.write = None

    @staticmethod
    def readThread(q, fcap):
        if fcap is None or not fcap.isOpened():
            # cap = CameraVideo("/home/nvidia/Desktop/yolov5-master/video/1.mp4")
            print("read the video")
            # todo  rtsp://192.168.50.222:8554/live/test1
            cap = RtspVideo("192.168.50.222", "/live/test1", 8554)
            # cap = CameraVideo("/dev/video0")
            print('read is ok')

        if not cap.isOpened():
            return False
        print("read image")
        while (True):
            # print("reading thread running ...")
            if not cap.isOpened():
                print("capture is not opened, exit...")
                # exit(-1)
                break
            # print("reading ...")
            ret, img = cap.read()
            # print(" get the figure ")
            if (not ret == 0) and (not img is None):
                # print("update...")
                q.put(img)
                if q.full():
                    q.get()
                # if q.qsize() > Executor.inputQuueSize-1:
                #     q.get()
            else:
                print("false...")
            # print("exit...")

    @staticmethod
    def inferenceThread(inQueue, outQueue):
        # todo mul
        inferencer = Inferencer("/home/chenkq/Desktop/multiyolov5-BS2021/runs/train/exp3/weights/best.pt")
        while True:
            img = inQueue.get()
            # print("image shape: ", img.shape)
            _, dst = inferencer.inference(img)
            print("det is finished")
            outQueue.put(dst)
            # outQueue2.put(img)
            if outQueue.qsize() > Executor.outputQueueSize - 1:
                outQueue.get()

    @staticmethod
    def writeThread( imageQueue):
        server = RtspServer(1920, 1080, 30, "/live/detection", 8553, imageQueue)
        server.run()


if __name__ == "__main__":
    c = Executor()
    while True:
        c.run()

# todo Excutor-> Inferencer -> Executor.readThread and Executor.writeThread